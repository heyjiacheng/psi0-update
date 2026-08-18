"""Modality config for Psi0's SIMPLE G1 whole-body datasets.

Ported from ``src/gr00t/gr00t/configs/modality/g1_locomanip.py`` onto the N1.7 /
πR² types so the same LeRobot datasets that train the GR00T-N1.6 baseline also
train πR².

Loaded by path, not by import — both the trainer
(``--modality-config-path``) and the server (``--modality-config-path``) do
``sys.path.append(parent); import_module(stem)`` and rely on the
``register_modality_config`` call at the bottom as an import side effect.

``DATASET_PATH`` must point at the LeRobot dataset root so the state/action keys
come from the dataset's own ``meta/modality.json`` rather than being duplicated
here. ``ACTION_HORIZON`` (env ``PIR2_ACTION_HORIZON``) is a training choice, not
a property of the data: it is the chunk length ``T`` the flow head predicts, and
πR²'s ``slide_steps`` at deploy time must divide into it.
"""

import json
import os
from pathlib import Path

from pir2.configs.data.embodiment_configs import register_modality_config
from pir2.data.embodiment_tags import EmbodimentTag
from pir2.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


DATASET_PATH = os.environ.get("DATASET_PATH")
if not DATASET_PATH:
    raise RuntimeError("DATASET_PATH must be set to load the SIMPLE meta/modality.json")
META_PATH = Path(DATASET_PATH) / "meta" / "modality.json"
if not META_PATH.exists():
    raise RuntimeError(f"Missing modality.json at {META_PATH}")
try:
    MODALITY_META = json.load(META_PATH.open("r"))
except Exception as exc:
    raise RuntimeError(f"Failed to load modality.json at {META_PATH}") from exc

EXPECTED_STATE_KEYS = [
    "left_hand",
    "right_hand",
    "left_arm",
    "right_arm",
    "rpy",
    "height",
]
EXPECTED_ACTION_KEYS = [
    "left_hand",
    "right_hand",
    "left_arm",
    "right_arm",
    "rpy",
    "height",
    "torso_vx",
    "torso_vy",
    "torso_vyaw",
    "target_yaw",
]

state_keys = list(MODALITY_META.get("state", {}).keys())
action_keys = list(MODALITY_META.get("action", {}).keys())
if set(state_keys) != set(EXPECTED_STATE_KEYS):
    raise RuntimeError(f"modality.json state keys mismatch: {state_keys}")
if set(action_keys) != set(EXPECTED_ACTION_KEYS):
    raise RuntimeError(f"modality.json action keys mismatch: {action_keys}")

# Keep a deterministic order: the deploy server concatenates action groups in this
# order to rebuild psi's flat 36-dim action vector.
state_keys = [k for k in EXPECTED_STATE_KEYS]
action_keys = [k for k in EXPECTED_ACTION_KEYS]

# Chunk length T predicted by the flow head. πR² slides `slide_steps` positions per
# server call, so pick T such that the deployed `--action-exec-horizon` divides it.
ACTION_HORIZON = int(os.environ.get("PIR2_ACTION_HORIZON", "48"))

# Every SIMPLE modality is an absolute joint / command target — no EEF poses, no
# relative conversion. One ActionConfig per action key, as ModalityConfig asserts.
def _absolute() -> ActionConfig:
    return ActionConfig(
        rep=ActionRepresentation.ABSOLUTE,
        type=ActionType.NON_EEF,
        format=ActionFormat.DEFAULT,
    )

simple_g1_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["rs_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=state_keys,
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=action_keys,
        action_configs=[_absolute() for _ in action_keys],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(simple_g1_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
