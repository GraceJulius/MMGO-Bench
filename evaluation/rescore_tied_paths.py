"""
One-off rescoring pass for the tied-shortest-path grading bug.

Ground truth used to store only one shortest path per query
(nx.dijkstra_path returns a single path even when several tie on total
weight) and evaluate_vlms.py graded "path"-format answers by exact match
against that one path — so a model producing a different, equally-valid
tied path was marked incorrect. dataset/v2/mmrb_v2.0.json has since been
patched to store every tied path per sample under
metadata["all_paths"], and evaluate_vlms.py's grader now checks
membership in that set for any *new* run.

This script re-grades every existing "path"-format result file
(evaluation/results/path/**/*.json) against the patched dataset, using
only data already saved in those files (predicted_answer) — no model is
re-run. Flipped results are marked with "tie_rescored": true so the
correction is traceable in the file itself; summaries are recomputed to
match.

Usage:
  python rescore_tied_paths.py            # apply and overwrite in place
  python rescore_tied_paths.py --dry_run   # report flips without writing
"""

import argparse
import glob
import json
import re
from pathlib import Path

DATASET_PATH = Path("../dataset/v2/mmrb_v2.0.json")
RESULTS_GLOB = "results/path/**/*.json"


def parse_path(s):
    if not isinstance(s, str):
        return None
    tokens = [t.strip() for t in re.split(r"->|→", s)]
    return [t for t in tokens if t]


def recompute_summary(results, model_name, prompt_mode, timestamp):
    total = len(results)
    correct = sum(1 for r in results if r.get("is_correct", False))
    layouts_present = sorted(set(r["layout"] for r in results))
    label_styles_present = sorted(set(r["label_style"] for r in results if r.get("label_style")))

    summary = {
        "model": model_name,
        "prompt_mode": prompt_mode,
        "timestamp": timestamp,
        "total_samples": total,
        "correct": correct,
        "accuracy": round(correct / total * 100, 2) if total > 0 else 0,
        "by_layout": {},
        "by_difficulty": {},
        "by_layout_and_difficulty": {},
        "by_label_style": {},
    }

    for layout in layouts_present:
        lr = [r for r in results if r["layout"] == layout]
        lc = sum(1 for r in lr if r.get("is_correct", False))
        summary["by_layout"][layout] = {
            "total": len(lr), "correct": lc,
            "accuracy": round(lc / len(lr) * 100, 2) if lr else 0
        }

    for diff in ["easy", "medium", "hard"]:
        dr = [r for r in results if r["difficulty"] == diff]
        dc = sum(1 for r in dr if r.get("is_correct", False))
        summary["by_difficulty"][diff] = {
            "total": len(dr), "correct": dc,
            "accuracy": round(dc / len(dr) * 100, 2) if dr else 0
        }

    for layout in layouts_present:
        for diff in ["easy", "medium", "hard"]:
            combo = [r for r in results if r["layout"] == layout and r["difficulty"] == diff]
            cc = sum(1 for r in combo if r.get("is_correct", False))
            key = f"{layout}_{diff}"
            summary["by_layout_and_difficulty"][key] = {
                "total": len(combo), "correct": cc,
                "accuracy": round(cc / len(combo) * 100, 2) if combo else 0
            }

    for label_style in label_styles_present:
        sr = [r for r in results if r.get("label_style") == label_style]
        sc = sum(1 for r in sr if r.get("is_correct", False))
        summary["by_label_style"][label_style] = {
            "total": len(sr), "correct": sc,
            "accuracy": round(sc / len(sr) * 100, 2) if sr else 0
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry_run", action="store_true",
                        help="Report what would change without writing any files.")
    args = parser.parse_args()

    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)
    dataset_by_id = {s["id"]: s for s in dataset}

    files = [f for f in glob.glob(RESULTS_GLOB, recursive=True) if "/logs/" not in f]
    total_flips = 0
    files_changed = 0

    for fp in sorted(files):
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        results = data["results"]
        if not results or results[0].get("answer_format") != "path":
            continue

        file_flips = 0
        for r in results:
            if r.get("is_correct"):
                continue
            sample = dataset_by_id.get(r["sample_id"])
            if sample is None:
                continue
            valid_paths = sample["metadata"].get("all_paths", [sample["metadata"]["path"]])
            if len(valid_paths) < 2:
                continue
            predicted_list = parse_path(r.get("predicted_answer", ""))
            if predicted_list is not None and predicted_list in valid_paths:
                r["is_correct"] = True
                r["tie_rescored"] = True
                file_flips += 1

        if file_flips == 0:
            continue

        total_flips += file_flips
        files_changed += 1
        old_acc = data["summary"]["accuracy"]
        new_summary = recompute_summary(
            results, data["summary"]["model"], data["summary"]["prompt_mode"],
            data["summary"]["timestamp"])
        new_summary["tie_rescore_flips"] = file_flips
        print(f"{fp}: {file_flips} flip(s), accuracy {old_acc}% -> {new_summary['accuracy']}%")

        if not args.dry_run:
            data["summary"] = new_summary
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}"
          f"Total: {total_flips} flip(s) across {files_changed} file(s) "
          f"(of {len(files)} path-format files scanned).")


if __name__ == "__main__":
    main()
