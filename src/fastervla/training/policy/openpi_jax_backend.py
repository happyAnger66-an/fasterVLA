"""Adapter: run a few openpi JAX train steps (debug configs) with optional velocity skip."""

from __future__ import annotations

import dataclasses
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastervla.training.policy.openpi_bridge import VelocitySkipConfig

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenpiDebugRunResult:
    """Summary of a short openpi debug training run."""

    losses: tuple[float, ...]
    steps: int
    params_updated: bool
    config_name: str


def _require_openpi() -> Any:
    try:
        import openpi  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "openpi is required for run_openpi_debug_train; "
            "install/sync the openpi environment and ensure it is on PYTHONPATH"
        ) from exc
    return openpi


def _import_openpi_train_module() -> Any:
    """Load openpi's ``scripts/train.py`` (not shipped as a package module)."""
    openpi = _require_openpi()
    train_py = Path(openpi.__file__).resolve().parents[2] / "scripts" / "train.py"
    if not train_py.is_file():
        raise ImportError(
            f"openpi is importable but scripts/train.py was not found at {train_py}"
        )
    scripts_dir = str(train_py.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import train as openpi_train  # type: ignore  # noqa: PLC0415

    return openpi_train


def _param_fingerprint(params: Any) -> float:
    import jax
    import jax.numpy as jnp
    import numpy as np

    leaves = jax.tree_util.tree_leaves(params)
    total = 0.0
    for leaf in leaves:
        arr = np.asarray(jax.device_get(leaf), dtype=np.float64)
        if arr.size:
            total += float(np.mean(np.abs(arr)))
    return float(jnp.asarray(total))


def _create_data_loader_with_optional_skip(
    config: Any,
    *,
    skip_cfg: VelocitySkipConfig | None,
    sharding: Any,
) -> Any:
    import jax
    from openpi.training import data_loader as openpi_data_loader

    from fastervla.training.policy.dataset_assembly import wrap_dataset_with_velocity_skip

    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = openpi_data_loader.create_torch_dataset(
        data_config, config.model.action_horizon, config.model
    )
    if skip_cfg is not None:
        dataset = wrap_dataset_with_velocity_skip(dataset, skip_cfg)
    dataset = openpi_data_loader.transform_dataset(
        dataset, data_config, skip_norm_stats=True
    )

    local_batch_size = config.batch_size // jax.process_count()
    torch_loader = openpi_data_loader.TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=sharding,
        shuffle=True,
        num_batches=None,
        num_workers=0,
        seed=config.seed,
        framework="jax",
    )
    return openpi_data_loader.DataLoaderImpl(data_config, torch_loader)


def run_openpi_debug_train(
    *,
    config_name: str = "debug_pi05",
    num_steps: int = 1,
    checkpoint_base_dir: str | Path,
    skip_cfg: VelocitySkipConfig | None = None,
) -> OpenpiDebugRunResult:
    """Run ``num_steps`` of openpi training on a debug config.

    Optionally wraps the dataset with velocity-skip before openpi transforms.
    Prefers CPU when ``JAX_PLATFORMS`` is unset (smoke-friendly).
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")

    _require_openpi()
    if not os.environ.get("JAX_PLATFORMS"):
        os.environ["JAX_PLATFORMS"] = "cpu"

    import functools

    import jax
    import numpy as np
    from openpi.training import config as openpi_config
    from openpi.training import sharding as openpi_sharding

    openpi_train = _import_openpi_train_module()

    checkpoint_base_dir = Path(checkpoint_base_dir)
    checkpoint_base_dir.mkdir(parents=True, exist_ok=True)

    base = openpi_config.get_config(config_name)
    config = dataclasses.replace(
        base,
        checkpoint_base_dir=str(checkpoint_base_dir),
        exp_name="fastervla_debug",
        num_train_steps=int(num_steps),
        overwrite=True,
        resume=False,
        wandb_enabled=False,
        num_workers=0,
    )

    openpi_train.init_logging()
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = openpi_sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS)
    )
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    data_loader = _create_data_loader_with_optional_skip(
        config, skip_cfg=skip_cfg, sharding=data_sharding
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)

    train_state, train_state_sharding = openpi_train.init_train_state(
        config, init_rng, mesh, resume=False
    )
    jax.block_until_ready(train_state)
    before_fp = _param_fingerprint(train_state.params)

    ptrain_step = jax.jit(
        functools.partial(openpi_train.train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    losses: list[float] = []
    for step in range(int(num_steps)):
        with openpi_sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        loss = float(np.asarray(jax.device_get(info["loss"])))
        losses.append(loss)
        _LOG.info("openpi debug step %s loss=%s", step, loss)
        batch = next(data_iter)

    after_fp = _param_fingerprint(train_state.params)
    params_updated = after_fp != before_fp or int(train_state.step) > 0

    return OpenpiDebugRunResult(
        losses=tuple(losses),
        steps=int(num_steps),
        params_updated=bool(params_updated),
        config_name=config_name,
    )
