"""模块: fastervla.deploy.recipe
功能: deploy YAML → argparse defaults；未知键硬失败；CLI 覆盖配方。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from fastervla.deploy.cli import _build_parser, main
from fastervla.deploy.recipe import (
    extract_config_path,
    load_run_recipe,
    recipe_to_arg_defaults,
)


def test_recipe_to_arg_defaults_soft_1x() -> None:
    raw = {
        "meta": {"name": "t", "robot_aid": "AID"},
        "run": {"mode": "shadow", "duration_s": 30.0},
        "policy": {"host": "127.0.0.1", "port": 8013},
        "speed": {"kappa": 1.0, "kappa_max": 2.0, "kappa_min": 1.0},
        "retimer": {"type": "uniform"},
        "inference": {
            "tracker": "lead",
            "execution_prefix_frames": 4,
            "execution_buffer_frames": 4,
            "arm_chunk_filter": "zero_phase_dct",
            "arm_chunk_filter_cutoff_hz": 3.0,
            "arm_interpolation": "pchip",
            "hold_gripper": True,
        },
        "robot": {"sdk_root": "/home/agi/app/gdk"},
    }
    d = recipe_to_arg_defaults(raw)
    assert d["kappa"] == 1.0
    assert d["prefix_frames"] == 4
    assert d["arm_interp"] == "pchip"
    assert d["robot_aid"] == "AID"
    assert d["retimer"] == "uniform"


def test_recipe_rejects_unknown_section() -> None:
    with pytest.raises(KeyError, match="unknown recipe sections"):
        recipe_to_arg_defaults({"safety": {"kappa_cap": 1.2}})


def test_recipe_rejects_unwired_key() -> None:
    with pytest.raises(KeyError, match="unwired"):
        recipe_to_arg_defaults({"retimer": {"type": "qp", "dt_min_s": 0.015}})


def test_load_repo_presets() -> None:
    root = Path(__file__).resolve().parents[2]
    for name in ("h7_soft_1x.yaml", "h7_soft_1p8.yaml", "g2_h7_badminton651.yaml"):
        d = load_run_recipe(root / "configs" / "deploy" / name)
        assert d["prefix_frames"] == 4
        assert d["arm_chunk_filter"] == "zero_phase_dct"
        assert d["arm_interp"] == "pchip"


def test_run_tag_and_run_id_wire_to_cli() -> None:
    d = recipe_to_arg_defaults(
        {"run": {"tag": "R2_lead2", "run_id": "20261125-R2"}}
    )
    assert d["tag"] == "R2_lead2"
    assert d["run_id"] == "20261125-R2"
    args = _build_parser(recipe_defaults=d).parse_args(
        ["run", "--tag", "R3_dwell"]
    )
    assert args.tag == "R3_dwell"  # CLI overrides recipe
    assert args.run_id == "20261125-R2"


def test_lead_tracker_section_wires_to_cli_dests() -> None:
    d = recipe_to_arg_defaults(
        {
            "inference": {"tracker": "lead"},
            "lead_tracker": {
                "tau_s": 0.09,
                "gain": 0.35,
                "max_lead_rad": 0.10,
                "lead_smooth_tau_s": 0.040,
                "hold_during_blend": True,
            },
        }
    )
    assert d["lead_tau_s"] == 0.09
    assert d["lead_gain"] == 0.35
    assert d["lead_max_lead_rad"] == 0.10
    assert d["lead_smooth_tau_s"] == 0.040
    assert d["lead_hold_during_blend"] is True


def test_load_lead2_preset() -> None:
    root = Path(__file__).resolve().parents[2]
    d = load_run_recipe(root / "configs" / "deploy" / "h7_k1p3_qp_phase_lead2.yaml")
    assert d["kappa"] == 1.3
    assert d["tracker"] == "lead"
    assert d["lead_tau_s"] == 0.09
    assert d["lead_gain"] == 0.35
    assert d["lead_max_lead_rad"] == 0.10
    args = _build_parser(recipe_defaults=d).parse_args(
        ["run", "--config", "configs/deploy/h7_k1p3_qp_phase_lead2.yaml"]
    )
    assert args.lead_tau_s == 0.09
    assert args.lead_gain == 0.35
    assert args.lead_max_lead_rad == 0.10
    assert args.lead_hold_during_blend is True


def test_load_lead2_dwell_preset() -> None:
    root = Path(__file__).resolve().parents[2]
    d = load_run_recipe(
        root / "configs" / "deploy" / "h7_k1p3_qp_phase_lead2_dwell.yaml"
    )
    assert d["qp_dwell_kappa_max"] == 2.0
    assert d["qp_dwell_step_rad"] == 0.005
    assert d["lead_tau_s"] == 0.09
    assert d["qp_contact_kappa"] == 1.1
    args = _build_parser(recipe_defaults=d).parse_args(
        ["run", "--config", "configs/deploy/h7_k1p3_qp_phase_lead2_dwell.yaml"]
    )
    assert args.qp_dwell_kappa_max == 2.0
    assert args.qp_transport_kappa_max == 1.5


def test_cli_config_defaults_overridden_by_flags(tmp_path: Path) -> None:
    path = tmp_path / "r.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "meta": {"robot_aid": "G2A0205C800352"},
                "run": {"mode": "shadow", "duration_s": 60.0},
                "speed": {"kappa": 1.0},
                "inference": {
                    "tracker": "lead",
                    "hold_gripper": True,
                    "arm_chunk_filter": "zero_phase_dct",
                    "arm_interpolation": "pchip",
                },
            }
        ),
        encoding="utf-8",
    )
    recipe = load_run_recipe(path)
    args = _build_parser(recipe_defaults=recipe).parse_args(
        ["run", "--config", str(path), "--kappa", "1.8", "--mode", "execute"]
    )
    assert args.kappa == 1.8
    assert args.mode == "execute"
    assert args.duration_s == 60.0
    assert args.tracker == "lead"
    assert args.robot_aid == "G2A0205C800352"
    assert args.arm_interp == "pchip"
    assert args.arm_chunk_filter == "zero_phase_dct"
    assert args.hold_gripper is True


def test_main_loads_config_from_sys_argv_when_argv_none(
    tmp_path: Path, monkeypatch
) -> None:
    """Entry point calls main() with argv=None; --config must still apply."""
    path = tmp_path / "r.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "meta": {"robot_aid": "G2A0205C800352"},
                "speed": {"kappa": 1.5},
                "inference": {"arm_interpolation": "pchip"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["fastervla-deploy", "run", "--config", str(path)],
    )
    argv_list = list(sys.argv[1:])
    assert extract_config_path(None) is None  # naive None must not be used
    recipe = load_run_recipe(extract_config_path(argv_list))
    args = _build_parser(recipe_defaults=recipe).parse_args(argv_list)
    assert args.robot_aid == "G2A0205C800352"
    assert args.kappa == 1.5
    assert args.arm_interp == "pchip"


def test_extract_config_path() -> None:
    assert extract_config_path(["run", "--config", "a.yaml"]) == Path("a.yaml")
    assert extract_config_path(["run", "--config=b.yaml"]) == Path("b.yaml")
    assert extract_config_path(["run", "--kappa", "1"]) is None


def test_main_rejects_bad_config(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"retimer": {"dt_min_s": 0.01}}), encoding="utf-8")
    with pytest.raises(KeyError, match="unwired"):
        main(["run", "--config", str(path)])
