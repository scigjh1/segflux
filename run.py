import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("stage", choices=["train", "evaluate"])
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    task = config[args.stage]
    module = ".".join(Path(task["script"]).with_suffix("").parts)
    command = [sys.executable, "-m", module]
    for key, value in task["args"].items():
        if value is None or value is False:
            continue
        if value is True:
            command.append("--" + key.replace("_", "-"))
        elif isinstance(value, list):
            command.extend(["--" + key.replace("_", "-"), *map(str, value)])
        else:
            command.extend(["--" + key.replace("_", "-"), str(value)])
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
