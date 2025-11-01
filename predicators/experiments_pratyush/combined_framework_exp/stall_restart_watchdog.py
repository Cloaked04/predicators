#!/usr/bin/env python3
"""
restart_on_stall.py

Usage:
  python restart_on_stall.py path/to/your_script.py [--stall-secs 40] [--grace-secs 8] [--max-restarts -1] [--pass-args ...]
Examples:
  python restart_on_stall.py myjob.py
  python restart_on_stall.py myjob.py --stall-secs 35 --max-restarts 10 --pass-args -- --foo 123 --bar
"""
import argparse
import asyncio
import os
import sys
import time
import signal
from typing import Optional

IS_WINDOWS = os.name == "nt"

def _now() -> float:
    return time.monotonic()

async def _read_stream(stream: asyncio.StreamReader, prefix: str, update_activity):
    """Read a stream and forward to our stdout, updating last-activity on every chunk."""
    try:
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                return
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()
            update_activity()
    except asyncio.CancelledError:
        pass

async def _kill_process_tree(proc: asyncio.subprocess.Process, grace_secs: float):
    """Terminate child process; escalate to kill if it ignores the gentle ask."""
    try:
        if IS_WINDOWS:
            proc.terminate()
        else:
            proc.send_signal(signal.SIGTERM)
        try:
            await asyncio.wait_for(proc.wait(), timeout=grace_secs)
            return
        except asyncio.TimeoutError:
            pass
        # Escalate
        if IS_WINDOWS:
            proc.kill()
        else:
            proc.send_signal(signal.SIGKILL)
        await proc.wait()
    except ProcessLookupError:
        pass

async def run_with_watchdog(
    target_script: str,
    pass_args: list[str],
    stall_secs: float,
    grace_secs: float,
    max_restarts: int,
):
    """
    Keep restarting the child if output stalls for `stall_secs`.
    Stop permanently if the child exits on its own with exit code 0.
    If it exits non-zero, restart (counts toward max_restarts).
    """
    restarts = 0

    while True:
        # Use unbuffered Python so we can actually see frequent output
        cmd = [sys.executable, "-u", target_script, *pass_args]
        print(f"\n[watchdog] Launching: {' '.join(cmd)} (restart #{restarts})", flush=True)
        last_activity = _now()

        def update_activity():
            nonlocal last_activity
            last_activity = _now()

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=os.environ.copy(),
        )

        reader_task = asyncio.create_task(_read_stream(proc.stdout, "", update_activity))
        stall_triggered = False

        try:
            # Poll for stall or natural exit
            while True:
                # Check natural exit first
                try:
                    rc = proc.returncode
                except ProcessLookupError:
                    rc = None

                if rc is not None:
                    await reader_task
                    print(f"[watchdog] Process exited with code {rc}", flush=True)
                    if rc == 0:
                        print("[watchdog] Child exited cleanly. Stopping watchdog.", flush=True)
                        return 0
                    else:
                        # Non-zero exit -> restart (subject to max_restarts)
                        restarts += 1
                        if 0 <= max_restarts < restarts:
                            print("[watchdog] Reached max restarts. Exiting.", flush=True)
                            return rc
                        print("[watchdog] Non-zero exit; preparing to restart.", flush=True)
                        break

                # Stall check
                if (_now() - last_activity) >= stall_secs:
                    stall_triggered = True
                    print(f"\n[watchdog] No output for {stall_secs:.0f}s — restarting...", flush=True)
                    break

                # Sleep a tick
                await asyncio.sleep(0.5)
        finally:
            if stall_triggered:
                # Kill and restart
                reader_task.cancel()
                await _kill_process_tree(proc, grace_secs)
            else:
                # If we got here because of exit (zero or non-zero), ensure cleanup
                if not reader_task.done():
                    reader_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await reader_task

            if stall_triggered:
                restarts += 1
                if 0 <= max_restarts < restarts:
                    print("[watchdog] Reached max restarts after stall. Exiting.", flush=True)
                    return 1

async def amain():
    parser = argparse.ArgumentParser(description="Restart a Python file if its output stalls.")
    parser.add_argument("script", help="Path to the Python file to run.")
    parser.add_argument("--stall-secs", type=float, default=40.0,
                        help="Seconds of no stdout/stderr activity before restart (default: 40).")
    parser.add_argument("--grace-secs", type=float, default=8.0,
                        help="Seconds to wait after TERM before KILL (default: 8).")
    parser.add_argument("--max-restarts", type=int, default=-1,
                        help="Maximum restarts before giving up (-1 = unlimited).")
    parser.add_argument("--pass-args", nargs=argparse.REMAINDER, default=[],
                        help="Arguments to pass to the child script. Use `--` before them.")
    args = parser.parse_args()

    # If user put a leading `--` for pass-args, argparse includes it; strip it.
    pass_args = args.pass_args
    if pass_args and pass_args[0] == "--":
        pass_args = pass_args[1:]

    # Basic checks
    if not os.path.exists(args.script):
        print(f"Error: {args.script} not found.", file=sys.stderr)
        sys.exit(2)

    rc = await run_with_watchdog(
        target_script=args.script,
        pass_args=pass_args,
        stall_secs=args.stall_secs,
        grace_secs=args.grace_secs,
        max_restarts=args.max_restarts,
    )
    sys.exit(rc if rc is not None else 0)

if __name__ == "__main__":
    import contextlib
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        print("\n[watchdog] Interrupted by user.", flush=True)
