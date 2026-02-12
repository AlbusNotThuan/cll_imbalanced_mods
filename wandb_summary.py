"""Utilities for summarizing local W&B runs into a Pandas DataFrame."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


def _parse_cli_args(args: Optional[Iterable[str]]) -> Dict[str, str]:
    """Convert a wandb metadata args list into a dictionary."""
    parsed: Dict[str, str] = {}
    if not args:
        return parsed

    args_list = list(args)
    skip_next = False
    for idx, raw_arg in enumerate(args_list):
        if skip_next:
            skip_next = False
            continue
        if not raw_arg.startswith("--"):
            continue

        body = raw_arg[2:]
        if "=" in body:
            key, value = body.split("=", 1)
            parsed[key.strip()] = value.strip()
            continue

        next_idx = idx + 1
        key = body.strip()
        if next_idx < len(args_list) and not args_list[next_idx].startswith("--"):
            parsed[key] = args_list[next_idx]
            skip_next = True
        else:
            parsed[key] = "True"

    return parsed


def _extract_config_value(config_path: Path, key: str) -> Optional[str]:
    if not config_path.exists():
        return None
    try:
        text = config_path.read_text()
    except OSError:
        return None

    pattern = re.compile(rf"^{re.escape(key)}:\s*\n\s+value:\s*(.+)$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return None
    value = match.group(1).strip()
    return value.strip('"\'')


def _safe_load_json(json_path: Path) -> Dict:
    if not json_path.exists():
        return {}
    try:
        return json.loads(json_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def collect_wandb_runs(wandb_root: str | Path = "wandb") -> pd.DataFrame:
    base_path = Path(wandb_root)
    if not base_path.exists():
        raise FileNotFoundError(f"wandb root '{base_path}' does not exist")

    records: List[Dict[str, Optional[str]]] = []
    for run_dir in sorted(p for p in base_path.iterdir() if p.is_dir() and p.name.startswith("run-")):
        files_dir = run_dir / "files"
        metadata = _safe_load_json(files_dir / "wandb-metadata.json")
        summary = _safe_load_json(files_dir / "wandb-summary.json")
        args_map = _parse_cli_args(metadata.get("args"))

        dataset = args_map.get("dataset_name") or args_map.get("dataset")
        algo = args_map.get("algo") or _extract_config_value(files_dir / "config.yaml", "algo")
        cll_type = args_map.get("cll_type")
        best_acc = summary.get("best_acc")
        step = int(summary.get("_step", 0))
        model = args_map.get("model")
        ord_num = args_map.get("ord_num")

        if step <= 60:
            continue

        records.append(
            {
                "run_id": run_dir.name,
                "dataset": dataset,
                "algo": algo,
                "cll_type": cll_type,
                "best_acc": best_acc,
                "model": model,
                "ord_num": ord_num,
            }
        )

    df = pd.DataFrame.from_records(records)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize local wandb runs into a DataFrame")
    parser.add_argument("--wandb-root", default="wandb", help="Path to the wandb directory")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to save the resulting dataframe as CSV",
    )
    args = parser.parse_args()

    df = collect_wandb_runs(args.wandb_root)
    if df.empty:
        print("No run directories found under", args.wandb_root)
        return

    print(df.to_string(index=False))
    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"Saved summary to {args.output_csv}")


if __name__ == "__main__":
    main()
