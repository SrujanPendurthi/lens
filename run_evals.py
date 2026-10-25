"""Usage: run_evals.py LOG -- <eval.py args>; stop on 2 GiB swap growth."""
import os
import re
import signal
import sys
import subprocess
import time
from pathlib import Path


def swap_bytes():
    output = subprocess.check_output(["sysctl", "vm.swapusage"], text=True)
    value, unit = re.search(r"used = ([\d.]+)([KMG])", output).groups()
    return float(value) * {"K": 2**10, "M": 2**20, "G": 2**30}[unit]


def run(args, log):
    initial = swap_bytes()
    with Path(log).open("w") as stream:
        process = subprocess.Popen(
            ["../.gpu/gpu-run", "lens", ".venv/bin/python", "eval.py", *args],
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            while process.poll() is None:
                if swap_bytes() - initial > 2 * 2**30:
                    raise RuntimeError(f"swap grew by more than 2 GiB; stopped PID {process.pid}")
                time.sleep(5)
            if process.returncode:
                raise RuntimeError(f"evaluation exited {process.returncode}; see {log}")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[2] != "--":
        raise SystemExit("usage: run_evals.py LOG -- <eval.py args...>")
    run(sys.argv[3:], sys.argv[1])
