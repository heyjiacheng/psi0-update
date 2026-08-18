"""PI-R2's latency-adaptive, per-position inference schedule.

The reference implementation describes time as ``tau`` with zero denoting pure
noise and ``noise_s`` denoting a clean action. Psi0 is trained with the opposite
flow convention: ``sigma=1`` is noise and ``sigma=0`` is clean. This module
keeps the rolling state in Psi0's convention and performs the conversion

    sigma = 1 - tau / noise_s

explicitly.

The schedule is derived from the PI-R2 release in
``pi-r2-flow/learning/Isaac-GR00T`` (Apache-2.0); see ``src/psi_r2/NOTICE``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

VelocityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class PiR2Schedule:
    """A cycle-closing PI-R2 schedule in Psi0 noise-time coordinates.

    Args:
        horizon: Number of positions in the action buffer.
        slide_steps: Number of newly-clean positions emitted and shifted per
            fast-channel update. PI-R2 calls this latency-dependent width ``d``.
        train_timesteps: Scale used by Psi0's timestep embedding.
        noise_s: Clean endpoint used in the reference tau convention.
    """

    horizon: int
    slide_steps: int
    train_timesteps: int = 1000
    noise_s: float = 0.999

    def __post_init__(self) -> None:
        if self.horizon < 3:
            raise ValueError(f"horizon must be at least 3, got {self.horizon}")
        if self.slide_steps < 1:
            raise ValueError(f"slide_steps must be positive, got {self.slide_steps}")
        if 2 * self.slide_steps >= self.horizon:
            raise ValueError(
                "PI-R2 requires a non-empty ramp: "
                f"2 * slide_steps < horizon, got d={self.slide_steps}, "
                f"horizon={self.horizon}"
            )
        if self.train_timesteps <= 0:
            raise ValueError(
                f"train_timesteps must be positive, got {self.train_timesteps}"
            )
        if not 0.0 < self.noise_s <= 1.0:
            raise ValueError(f"noise_s must be in (0, 1], got {self.noise_s}")

    @property
    def ramp_width(self) -> int:
        return self.horizon - 2 * self.slide_steps

    def initial_tau(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return reference PI-R2 clean-time values with shape ``[T]``."""
        positions = torch.arange(self.horizon, device=device, dtype=dtype)
        d = self.slide_steps
        ramp = self.noise_s * (1.0 - (positions - d + 0.5) / float(self.ramp_width))
        clean = torch.as_tensor(self.noise_s, device=device, dtype=dtype)
        noise = torch.as_tensor(0.0, device=device, dtype=dtype)
        return torch.where(
            positions < d,
            clean,
            torch.where(positions >= self.horizon - d, noise, ramp),
        )

    def initial_sigma(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return Psi0 noise-time values with shape ``[T]``.

        The clean prefix is zero, the tail is one, and the interior increases
        monotonically from clean to noisy.
        """
        tau = self.initial_tau(device=device, dtype=dtype)
        return (1.0 - tau / self.noise_s).clamp(0.0, 1.0)

    def target_sigma(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Noise times reached by one fast update, before the FIFO slide."""
        initial = self.initial_sigma(device=device, dtype=dtype)
        return torch.cat(
            [
                torch.zeros(self.slide_steps, device=device, dtype=dtype),
                initial[: self.horizon - self.slide_steps],
            ]
        )

    def model_timesteps(self, sigma: torch.Tensor) -> torch.Tensor:
        """Map continuous Psi0 sigma values to its training timestep scale."""
        return sigma * float(self.train_timesteps)


class PiR2RollingBuffer:
    """Stateful PI-R2 action buffer with manual per-position Euler updates.

    This is deliberately not an ``nn.Module``: the action and time buffers are
    episode runtime state and must never become checkpoint parameters/buffers.
    """

    def __init__(self, schedule: PiR2Schedule) -> None:
        self.schedule = schedule
        self.actions: torch.Tensor | None = None
        self.sigma: torch.Tensor | None = None

    @property
    def seeded(self) -> bool:
        return self.actions is not None and self.sigma is not None

    def reset(self) -> None:
        self.actions = None
        self.sigma = None

    @staticmethod
    def _validate_noise(noise: torch.Tensor, expected: torch.Size, name: str) -> None:
        if noise.shape != expected:
            raise ValueError(
                f"{name} must have shape {tuple(expected)}, got {tuple(noise.shape)}"
            )

    def seed(
        self,
        clean_chunk: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
    ) -> None:
        """Place a clean bootstrap chunk on the per-position flow manifold."""
        if clean_chunk.ndim != 3:
            raise ValueError(
                f"clean_chunk must have shape [B,T,A], got {tuple(clean_chunk.shape)}"
            )
        if clean_chunk.shape[1] != self.schedule.horizon:
            raise ValueError(
                f"clean_chunk horizon {clean_chunk.shape[1]} does not match "
                f"schedule horizon {self.schedule.horizon}"
            )
        if noise is None:
            noise = torch.randn_like(clean_chunk)
        else:
            self._validate_noise(noise, clean_chunk.shape, "noise")
            noise = noise.to(device=clean_chunk.device, dtype=clean_chunk.dtype)

        sigma_1d = self.schedule.initial_sigma(
            device=clean_chunk.device, dtype=clean_chunk.dtype
        )
        sigma = sigma_1d.unsqueeze(0).expand(clean_chunk.shape[0], -1).clone()
        self.actions = (1.0 - sigma.unsqueeze(-1)) * clean_chunk + sigma.unsqueeze(
            -1
        ) * noise
        self.sigma = sigma

    def denoise_and_slide(
        self,
        velocity_fn: VelocityFn,
        *,
        substeps: int = 1,
        tail_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Advance to the cycle target, snapshot, then slide and append noise.

        ``velocity_fn`` receives ``(actions, sigma)`` and must return Psi0's
        flow velocity ``dx/dsigma`` with the same shape as ``actions``.
        """
        if not self.seeded:
            raise RuntimeError("PI-R2 rolling buffer must be seeded before inference")
        if substeps < 1:
            raise ValueError(f"substeps must be positive, got {substeps}")
        assert self.actions is not None and self.sigma is not None

        target_1d = self.schedule.target_sigma(
            device=self.sigma.device, dtype=self.sigma.dtype
        )
        target = target_1d.unsqueeze(0).expand_as(self.sigma)
        total_delta = target - self.sigma
        # In Psi0 coordinates denoising always moves sigma downward. Clamp only
        # tiny positive round-off; a real positive delta means the cycle drifted.
        if torch.any(total_delta > 1e-6):
            raise RuntimeError(
                "PI-R2 schedule drifted toward more noise before the slide"
            )
        delta = total_delta.clamp(max=0.0) / float(substeps)

        for _ in range(substeps):
            velocity = velocity_fn(self.actions, self.sigma)
            if velocity.shape != self.actions.shape:
                raise ValueError(
                    "velocity_fn returned shape "
                    f"{tuple(velocity.shape)}, expected {tuple(self.actions.shape)}"
                )
            self.actions = (
                self.actions + delta.unsqueeze(-1).to(velocity.dtype) * velocity
            )
            self.sigma = self.sigma + delta

        # Avoid accumulated floating-point error across long episodes.
        self.sigma = target.clone()
        snapshot = self.actions.clone()

        batch, _, action_dim = self.actions.shape
        d = self.schedule.slide_steps
        if tail_noise is None:
            tail_noise = torch.randn(
                batch,
                d,
                action_dim,
                device=self.actions.device,
                dtype=self.actions.dtype,
            )
        else:
            expected = torch.Size((batch, d, action_dim))
            self._validate_noise(tail_noise, expected, "tail_noise")
            tail_noise = tail_noise.to(
                device=self.actions.device, dtype=self.actions.dtype
            )

        self.actions = torch.cat([self.actions[:, d:], tail_noise], dim=1)
        self.sigma = torch.cat(
            [
                self.sigma[:, d:],
                torch.ones(batch, d, device=self.sigma.device, dtype=self.sigma.dtype),
            ],
            dim=1,
        )
        expected_sigma = self.schedule.initial_sigma(
            device=self.sigma.device, dtype=self.sigma.dtype
        ).unsqueeze(0)
        if not torch.allclose(
            self.sigma, expected_sigma.expand_as(self.sigma), atol=1e-6
        ):
            raise RuntimeError("PI-R2 schedule failed to close after the FIFO slide")
        return snapshot

    def emitted(self, snapshot: torch.Tensor) -> torch.Tensor:
        """Select the newly-clean ``[d:2d]`` window from a pre-slide snapshot."""
        if snapshot.ndim != 3 or snapshot.shape[1] != self.schedule.horizon:
            raise ValueError(
                f"snapshot must have shape [B,{self.schedule.horizon},A], "
                f"got {tuple(snapshot.shape)}"
            )
        d = self.schedule.slide_steps
        return snapshot[:, d : 2 * d]
