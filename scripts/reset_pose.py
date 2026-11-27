"""Standalone full 24-joint reset; default is a read-only plan, never inference."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

from robot_control import GdkBackend, NAMES, FINGERS, finite_vector

RESET_NAMES = (*NAMES[:22], *FINGERS)
# rad/s caps: waist / head / 14-arm / grippers. Waist was 0.05 and made long resets
# feel "stuck"; keep below native tracking comfort, still finish in one run.
RESET_RATES = np.asarray([0.15] * 5 + [0.2] * 3 + [0.35] * 14 + [0.15] * 2)
LEAD_MAX_RAD = 0.10  # stay below native ~0.3 tracking reject
DIVERGE_RAD = 0.15


def reset_step(previous, measured, target, dt):
    previous = finite_vector(previous, (24,))
    measured = finite_vector(measured, (24,))
    target = finite_vector(target, (24,))
    if not math.isfinite(dt) or not 0 < dt <= 0.05:
        raise ValueError('Invalid reset control interval')
    # Re-sync on feedback spikes instead of aborting: abort mid-reset forced
    # operators to re-run many times for one pose.
    gap = previous - measured
    if np.max(np.abs(gap)) > DIVERGE_RAD:
        previous = measured + np.clip(gap, -LEAD_MAX_RAD, LEAD_MAX_RAD)
    step_cap = RESET_RATES * min(dt, 0.01)
    candidate = previous + np.clip(target - previous, -step_cap, step_cap)
    # Per-joint lead limit: one lagging joint must not freeze the whole body.
    lead = candidate - measured
    waiting = bool(np.any(np.abs(lead) > LEAD_MAX_RAD + 1e-12))
    candidate = measured + np.clip(lead, -LEAD_MAX_RAD, LEAD_MAX_RAD)
    return candidate, waiting


def check_limits(target, limits):
    target = finite_vector(target, (24,))
    for name, value in zip(RESET_NAMES, target, strict=True):
        interval = limits.get(name, {})
        lo, hi = interval.get("min"), interval.get("max")
        if (type(lo) not in (float, int) or type(hi) not in (float, int) or
                not math.isfinite(lo) or not math.isfinite(hi) or not lo < hi or not lo <= value <= hi):
            raise ValueError(f"Reset target outside/missing SDK limits: {name}: {value}, {interval}")
    return target


def check_other_controller():
    # Also catches controllers started before the shared lock was introduced.
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            words = (entry / 'cmdline').read_bytes().split(b'\0')
        except (OSError, PermissionError):
            continue
        if any(Path(os.fsdecode(w)).name == 'robot_bridge.py' for w in words if w):
            if b'execute' in words or b'--mode=execute' in words:
                raise RuntimeError(f"Stop the running robot execution first (PID {entry.name})")


def initial_reset_command(measured, limits):
    """Allow sub-milliradian feedback rounding, but command strictly inside limits."""
    measured = finite_vector(measured, (24,))
    # Validate every interval before clipping; missing limits must still fail.
    lo = np.asarray([limits.get(n, {}).get('min', np.nan) for n in RESET_NAMES])
    hi = np.asarray([limits.get(n, {}).get('max', np.nan) for n in RESET_NAMES])
    command = check_limits(np.clip(measured, lo, hi), limits)
    if np.max(np.abs(command - measured)) > 0.001:
        raise ValueError('Reset initial feedback exceeds SDK limits by more than 0.001 rad')
    return command


def reset_target(initial, saved_pose, *, arms_only=False):
    target = finite_vector(saved_pose, (24,)).copy()
    if arms_only:
        target[:8] = initial[:8]
        target[22:] = initial[22:]
    return target


def send_reset(backend, target, *, arms_only=False):
    if backend.read_only:
        raise ValueError('Read-only reset cannot send commands')
    target = check_limits(target, backend.limits)
    request = backend.sdk.JointServoControlReq()
    selection = slice(8, 22) if arms_only else slice(None)
    request.joint_names = list(RESET_NAMES[selection])
    request.joint_positions = target[selection].tolist()
    request.joint_velocities = []
    request.control_period = 0.02
    started = time.monotonic()
    result = backend.robot.joint_servo_control(request, False)
    if result != 0 or time.monotonic() - started > 0.05:
        raise RuntimeError('Reset SDK publication failed or blocked; do not retry')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--arms-only', action='store_true', help='Reset only the 14 arm joints')
    parser.add_argument('--config', type=Path, default=Path('configs/toolbox/h6_89999.json'))
    parser.add_argument('--pose', type=Path, default=Path('configs/common/h6_reset_pose.json'))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    target = finite_vector(json.loads(args.pose.read_text())['state24'], (24,))
    if args.execute and not sys.stdin.isatty():
        raise ValueError('Reset execution requires the operator terminal')
    # Match the existing SDK process configuration; no interface or routing changes.
    os.environ['LOCATOR_IP'] = config['sdk_source_ip']
    os.environ['AORTA_DISCOVERY_URI'] = config['discovery_uri']
    sys.path.insert(0, config['sdk_root'] + '/lib')
    with open('/tmp/agibot-pi05-control.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_other_controller()
        backend = GdkBackend(config, read_only=not args.execute)
        try:
            time.sleep(1.0)
            backend.limits = backend.robot.get_joint_limits()
            start = backend.read(require_control_status=True)
            stamp = backend.last_stamp
            wait_until = time.monotonic() + 0.15
            while backend.last_stamp == stamp:
                if time.monotonic() >= wait_until:
                    raise RuntimeError('No advancing feedback before reset')
                time.sleep(0.01)
                start = backend.read(require_control_status=True)
            initial_command = initial_reset_command(start, backend.limits)
            target = reset_target(initial_command, target, arms_only=args.arms_only)
            check_limits(target, backend.limits)
            selection = slice(8, 22) if args.arms_only else slice(None)
            print(json.dumps({'reset_plan': True, 'execute': args.execute,
                              'arms_only': args.arms_only,
                              'joint_names': RESET_NAMES[selection], 'start24': start.tolist(),
                              'target24': target.tolist(), 'command_mode': 'feedback_paced_reset',
                              'rate_limits_rad_s': RESET_RATES.tolist()}), flush=True)
            if not args.execute:
                return
            began = previous = time.monotonic()
            settled_since = None
            last_log = -math.inf
            commands = 0
            command = initial_command.copy()
            # Long bent-arm poses + slow waist need >2 min if rates were conservative.
            while time.monotonic() - began < 180.0:
                time.sleep(max(0.0, previous + 0.01 - time.monotonic()))
                tick = time.monotonic()
                dt = tick - previous
                if not 0 < dt <= 0.05:
                    raise RuntimeError('Reset servo cadence missed; stopping publication')
                measured = backend.read(require_control_status=True)
                command, waiting = reset_step(command, measured, target, dt)
                if time.monotonic() - tick > 0.05:
                    raise RuntimeError('Reset feedback acquisition delayed')
                send_reset(backend, command, arms_only=args.arms_only)
                commands += 1
                previous = tick
                error = np.abs(measured - target)[selection]
                arrived = float(error.max()) <= 0.02 and np.max(np.abs(command - target)[selection]) < 1e-9
                settled_since = (settled_since if settled_since is not None else tick) if arrived else None
                if tick - last_log >= 1.0:
                    worst = int(np.argmax(error))
                    print(json.dumps({'reset_elapsed_s': tick - began,
                                      'worst_joint': RESET_NAMES[selection][worst],
                                      'max_error_rad': float(error[worst]),
                                      'waiting_for_feedback': waiting,
                                      'command_feedback_error_rad': float(np.max(np.abs(command - measured))),
                                      'servo_publications': commands}), flush=True)
                    last_log = tick
                if settled_since is not None and tick - settled_since >= 0.5:
                    print(json.dumps({'reset_complete': True, 'measured_state24': measured.tolist(),
                                      'max_error_rad': float(error.max()),
                                      'servo_publications': commands}), flush=True)
                    return
            raise TimeoutError('Reset did not reach measured tolerance; no further commands')
        finally:
            backend.close()


if __name__ == '__main__':
    main()
