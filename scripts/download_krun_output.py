"""Download an existing KRun job with a longer transfer timeout.

Run with the Python interpreter belonging to the installed krun-cli tool.
This command only downloads output; it never submits a kernel.
"""

import argparse
import os
import subprocess
from pathlib import Path

from krun import cli
from krun.errors import CommandError, KrunError
from krun.kaggle import KaggleClient


class TransferClient(KaggleClient):
    def __init__(self, transfer_timeout: int) -> None:
        super().__init__()
        self.transfer_timeout = transfer_timeout

    def download_output(self, kernel: str, destination: Path) -> str:
        destination.mkdir(parents=True, exist_ok=True)
        command = [self.executable, "kernels", "output", kernel,
                   "--path", str(destination), "--force"]
        try:
            result = subprocess.run(
                command, text=True, encoding="utf-8", capture_output=True,
                check=False, timeout=self.transfer_timeout,
                env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
            )
        except subprocess.TimeoutExpired as exc:
            raise CommandError(
                "Output download timed out. Retry this download command with "
                "a larger --transfer-timeout; the remote job is unchanged."
            ) from exc
        if result.returncode:
            raise CommandError(result.stderr.strip() or result.stdout.strip()
                               or "Kaggle output download failed.")
        return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id")
    parser.add_argument("--transfer-timeout", type=int, default=1800)
    parser.add_argument("--wait", action="store_true", help="Wait for the existing job to finish, then download.")
    parser.add_argument("--wait-timeout", type=int, default=43200)
    args = parser.parse_args()
    if args.transfer_timeout < 1:
        parser.error("--transfer-timeout must be positive")
    if args.wait_timeout < 1:
        parser.error("--wait-timeout must be positive")
    root = Path(__file__).resolve().parents[1]
    store = cli._store(root)
    job = store.load(args.job_id)
    client = TransferClient(args.transfer_timeout)
    client.validate_environment()
    try:
        if args.wait:
            print(f"Waiting for {job.job_id}; output will download automatically on completion.", flush=True)
            status, _ = client.wait_for_terminal_status(
                job.kernel, poll_interval=30, timeout=args.wait_timeout,
            )
            job.status = status
            store.save(job)
            if status != "complete":
                raise CommandError(f"Remote job ended with status {status}; inspect its logs before retrying.")
        cli._download(client, store, job, store.directory(job.job_id) / "output")
    except KrunError as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
