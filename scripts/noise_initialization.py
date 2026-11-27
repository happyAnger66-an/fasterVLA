"""Explicit Gaussian initialization in the action latent; no JAX or robot SDK."""
import copy
import math

import numpy as np

CAPABILITY = "configurable_state_mean_v1"
SOURCES = ("zero", "raw_state", "state_normalized", "action_normalized")


def noise_options(value=None):
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value) - {"source", "scale", "std"}:
        raise ValueError("initial_noise must contain only source, scale and std")
    result = {"source": "zero", "scale": 1.0, "std": 1.0, **value}
    if result["source"] not in SOURCES:
        raise ValueError("Unknown initial-noise mean source")
    for key in ("scale", "std"):
        v = result[key]
        if type(v) not in (int, float) or not math.isfinite(v) or (key == "std" and v < 0):
            raise ValueError(f"Invalid initial-noise {key}")
        result[key] = float(v)
    return result


def noise_mean(state, stats, options, *, horizon=30, normalization="quantile"):
    """Repeat a current-state mean in time, zero-pad ONLY its mean to 32 dims."""
    options = noise_options(options)
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (24,) or not np.isfinite(state).all():
        raise ValueError("Noise mean requires a finite 24-dimensional current state")
    source = options["source"]
    values = state.copy()
    if source == "zero":
        values.fill(0)
    elif source in ("state_normalized", "action_normalized"):
        norm = stats["state" if source == "state_normalized" else "actions"]
        if normalization == "quantile":
            low, high = (np.asarray(norm[k], dtype=np.float32) for k in ("q01", "q99"))
            values = (state - low) / (high - low + np.float32(1e-6)) * 2 - 1
        elif normalization == "standard":
            center, scale = (np.asarray(norm[k], dtype=np.float32) for k in ("mean", "std"))
            values = (state - center) / (scale + np.float32(1e-6))
        else:
            raise ValueError("Unknown action normalization")
    values *= np.float32(options["scale"])
    mean = np.broadcast_to(np.pad(values, (0, 8)), (horizon, 32)).copy()
    if not np.isfinite(mean).all():
        raise ValueError("Nonfinite state-derived noise mean")
    return mean


def add_noise_arguments(parser):
    parser.add_argument("--noise-mean-source", choices=SOURCES, default=None,
                        help="State-mean experiment: zero, raw_state, state_normalized or action_normalized")
    parser.add_argument("--noise-mean-scale", type=float, default=None,
                        help="Multiply the mean curve by this factor, default 1")
    parser.add_argument("--noise-std", type=float, default=None,
                        help="Gaussian standard deviation in all 32 channels, default 1 (zero allowed)")


def apply_noise_overrides(config, *, noise_mean_source=None, noise_mean_scale=None, noise_std=None):
    changes = {key: value for key, value in (("source", noise_mean_source), ("scale", noise_mean_scale),
                                            ("std", noise_std)) if value is not None}
    if "initial_noise" not in config and not changes:
        return config
    if (config.get("inference_task", {}).get("noise_initialization") != CAPABILITY
            or config.get("inference_task", {}).get("noise_mode") != "per_observation_search"):
        raise ValueError("Noise mean arguments require the separate state-mean noise-search profile")
    effective = copy.deepcopy(config)
    effective["initial_noise"] = noise_options({**config.get("initial_noise", {}), **changes})
    return effective
