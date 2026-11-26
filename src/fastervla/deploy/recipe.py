"""Deploy run recipes: nested YAML → argparse dest defaults for `fastervla-deploy run`.

Only keys that the CLI currently wires are accepted. Unknown sections/keys fail hard.
CLI flags always override recipe defaults (argparse set_defaults + parse order).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# (section, key) → argparse dest. Keep in sync with deploy/cli.py `run` args.
_CLI_FIELD_MAP: dict[tuple[str, str], str] = {
    ("run", "mode"): "mode",
    ("run", "duration_s"): "duration_s",
    ("run", "results_dir"): "results_dir",
    ("run", "flush_every_s"): "flush_every_s",
    ("run", "no_trace"): "no_trace",
    ("run", "tag"): "tag",
    ("run", "run_id"): "run_id",
    ("policy", "host"): "host",
    ("policy", "port"): "port",
    ("policy", "prompt"): "prompt",
    ("speed", "kappa"): "kappa",
    ("speed", "kappa_max"): "kappa_max",
    ("speed", "kappa_min"): "kappa_min",
    ("retimer", "type"): "retimer",
    ("retimer", "buffer_window_enabled"): "qp_buffer_window",
    ("retimer", "contact_kappa"): "qp_contact_kappa",
    ("retimer", "contact_window_frames"): "qp_contact_window",
    ("retimer", "contact_window_scale_kappa"): "qp_contact_window_scale_kappa",
    ("retimer", "contact_grip_delta"): "qp_contact_grip_delta",
    ("retimer", "transport_kappa_max"): "qp_transport_kappa_max",
    ("retimer", "dwell_kappa_max"): "qp_dwell_kappa_max",
    ("retimer", "dwell_step_rad"): "qp_dwell_step_rad",
    ("retimer", "prefix_lock"): "qp_prefix_lock",
    ("retimer", "lock_extra_frames"): "qp_lock_extra",
    ("inference", "tracker"): "tracker",
    ("inference", "execution_prefix_frames"): "prefix_frames",
    ("inference", "execution_buffer_frames"): "buffer_frames",
    ("inference", "arm_chunk_filter"): "arm_chunk_filter",
    ("inference", "arm_chunk_filter_cutoff_hz"): "arm_filter_cutoff_hz",
    ("inference", "arm_interpolation"): "arm_interp",
    ("inference", "mpc_preview"): "mpc_preview",
    ("inference", "hold_gripper"): "hold_gripper",
    ("inference", "t_infer_min_s"): "t_infer_min_s",
    ("inference", "t_switch_min_s"): "t_switch_min_s",
    ("lead_tracker", "tau_s"): "lead_tau_s",
    ("lead_tracker", "gain"): "lead_gain",
    ("lead_tracker", "max_lead_rad"): "lead_max_lead_rad",
    ("lead_tracker", "lead_smooth_tau_s"): "lead_smooth_tau_s",
    ("lead_tracker", "hold_during_blend"): "lead_hold_during_blend",
    ("lead_tracker", "closed_grip_max_lead_rad"): "lead_closed_grip_max_lead_rad",
    ("robot", "sdk_root"): "sdk_root",
    ("robot", "sdk_source_ip"): "sdk_source_ip",
    ("robot", "discovery_uri"): "discovery_uri",
    ("robot", "robot_aid"): "robot_aid",
    ("robot", "ignore_right_gripper_health"): "ignore_right_gripper_health",
    ("meta", "robot_aid"): "robot_aid",
}

_ALLOWED_SECTIONS = frozenset(
    {"meta", "run", "policy", "speed", "retimer", "inference", "lead_tracker", "robot"}
)

# Nested keys allowed but not yet mapped to CLI (ignored with no error).
# Keep empty: prefer lean recipes; expand when wiring QP/lead YAML end-to-end.
_IGNORED_KEYS: dict[str, frozenset[str]] = {
    "meta": frozenset({"name", "machine"}),
    "policy": frozenset({"policy_fps", "horizon", "action_dim"}),
}


def _as_mapping(raw: Any, *, path: Path) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return raw


def recipe_to_arg_defaults(raw: dict[str, Any]) -> dict[str, Any]:
    """Flatten a recipe mapping into argparse dest → value."""
    unknown_sections = sorted(set(raw) - _ALLOWED_SECTIONS)
    if unknown_sections:
        raise KeyError(
            "unknown recipe sections "
            f"{unknown_sections}; allowed={sorted(_ALLOWED_SECTIONS)}"
        )

    out: dict[str, Any] = {}
    for section, body in raw.items():
        if body is None:
            continue
        if not isinstance(body, dict):
            raise TypeError(f"section {section!r} must be a mapping, got {type(body).__name__}")
        ignored = _IGNORED_KEYS.get(section, frozenset())
        for key, value in body.items():
            dest = _CLI_FIELD_MAP.get((section, key))
            if dest is None:
                if key in ignored:
                    continue
                raise KeyError(
                    f"unknown or unwired recipe key {section}.{key}; "
                    "remove it or wire it in deploy/recipe.py + cli.py"
                )
            if dest == "results_dir":
                value = Path(value)
            out[dest] = value
    return out


def load_run_recipe(path: str | Path) -> dict[str, Any]:
    """Load YAML recipe → argparse defaults dict."""
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    return recipe_to_arg_defaults(_as_mapping(raw, path=p))


def extract_config_path(argv: list[str] | None) -> Path | None:
    """Find `--config PATH` in argv (does not mutate)."""
    if not argv:
        return None
    for i, tok in enumerate(argv):
        if tok == "--config" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if tok.startswith("--config="):
            return Path(tok.split("=", 1)[1])
    return None
