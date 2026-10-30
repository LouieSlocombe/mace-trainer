"""Importable worker used to verify spawned process and directory isolation."""

import json
import os
import time
from contextlib import chdir
from pathlib import Path

from mace_trainer.config import Campaign, Condition


def run(
    campaign: Campaign, condition: Condition, _forcefield_hashes: dict[str, str]
) -> None:
    folder = campaign.output_dir / condition.name
    folder.mkdir()
    with chdir(folder):
        (campaign.output_dir / f"{condition.name}.started").touch()
        deadline = time.monotonic() + 10
        while len(list(campaign.output_dir.glob("*.started"))) < 2:
            if time.monotonic() >= deadline:
                raise TimeoutError("The second worker did not start concurrently")
            time.sleep(0.01)
        Path("worker.json").write_text(
            json.dumps({"pid": os.getpid(), "cwd": str(Path.cwd())}),
            encoding="utf-8",
        )
