"""Build Miles task-ID prompts from a local Terminal-Bench 2.1 checkout.

Only instruction.md and task.toml are read; solutions and verifier answers
never become model prompts. Keep the source task directories unchanged.
"""

import hashlib
import json
from pathlib import Path

from tap import Tap


class Args(Tap):
    tasks_dir: Path  # Directory containing one Harbor task directory per task.
    output: Path  # Destination train.jsonl (must not already exist).
    exclude_task: list[str] = []  # Explicit exclusions for provider capability or a held-out split.


def prepare_data(tasks_dir: Path, output: Path, exclude_task: list[str]) -> int:
    task_dirs = sorted(p for p in tasks_dir.iterdir() if p.is_dir() and (p / "task.toml").is_file())
    if not task_dirs:
        raise ValueError(f"No Harbor tasks found directly under {tasks_dir}")
    unknown = set(exclude_task) - {p.name for p in task_dirs}
    if unknown:
        raise ValueError(f"Unknown excluded tasks: {sorted(unknown)}")
    selected = [p for p in task_dirs if p.name not in exclude_task]
    if not selected:
        raise ValueError("Exclusions removed every task")
    records = []
    hashes = {}
    for task in selected:
        instruction = task / "instruction.md"
        if not instruction.is_file() or not instruction.read_text().strip():
            raise ValueError(f"Missing or empty task instruction: {instruction}")
        records.append({"prompt": task.name, "metadata": {"instance_id": task.name, "agent_name": "terminus-2"}})
        for path in (task / "task.toml", instruction):
            hashes[str(path.relative_to(tasks_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    output.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "tasks_dir": str(tasks_dir.resolve()),
                "count": len(records),
                "excluded": sorted(exclude_task),
                "input_sha256": hashes,
            },
            indent=2,
        )
    )
    return len(records)


if __name__ == "__main__":
    args = Args().parse_args()
    print(f"Prepared {prepare_data(args.tasks_dir, args.output, args.exclude_task)} task prompts at {args.output}")
