"""
Computes the per-layout x per-difficulty accuracy breakdown and McNemar's
paired-significance tests for MMGO-Bench's full-results reporting. Reads
directly from the per-condition summary blocks already saved in each
result JSON (by_layout, by_difficulty, by_layout_and_difficulty) -- no
re-scoring, no re-derivation of ground truth, just aggregation and the
paired significance test over the saved per-sample is_correct values.

Scope (documented, not arbitrary): label_style=uppercase (the primary
track; a secondary same-graph numeric-label comparison is also in
evaluation/results/ for all 5 models), description=long,
prompt_mode=zero_shot_cot as the headline reasoning condition, both
answer_format tracks (numeric and path) shown separately since they are
deliberately independent prompt conditions.

Run from the evaluation/ directory: python3 ../analysis/analyze_results.py
"""
import json
import glob
from scipy.stats import binomtest

MODELS = ["llava", "gemma3_4b", "minicpm-v", "qwen2vl", "claude-haiku"]
LAYOUTS = ["spring", "circular", "kamada_kawai", "random"]
DIFFICULTIES = ["easy", "medium", "hard"]


def find_result(model, answer_format, prompt_mode, description="long", label_style="uppercase"):
    pattern = f"results/{answer_format}/{label_style}/multimodal/eval_{model}_{prompt_mode}_{answer_format}_{description}desc_{label_style}*.json"
    matches = [p for p in glob.glob(pattern) if "logs" not in p]
    corrected = [p for p in matches if "_corrected_" in p]
    chosen = sorted(corrected)[-1] if corrected else (sorted(matches)[-1] if matches else None)
    if chosen is None:
        return None
    return json.load(open(chosen))


def mcnemar(pairs):
    """pairs: list of (correct_a, correct_b) bools for matched samples.
    Exact two-sided binomial test on the discordant pairs."""
    b = sum(1 for a, c in pairs if a and not c)
    c = sum(1 for a, cc in pairs if not a and cc)
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue
    return b, c, p


def main():
    print("=== Per-model, per-answer-format accuracy (zero_shot_cot, long desc, uppercase) ===\n")
    table_rows = {}
    for model in MODELS:
        for af in ["numeric", "path"]:
            d = find_result(model, af, "zero_shot_cot")
            if d is None:
                print(f"MISSING: {model} {af} zero_shot_cot")
                continue
            s = d["summary"]
            table_rows[(model, af)] = s
            print(f"{model:14s} {af:8s} overall={s['accuracy']:5.2f}%  n={s['total_samples']}")

    print("\n=== by_layout_and_difficulty (numeric answer_format) ===\n")
    for model in MODELS:
        s = table_rows.get((model, "numeric"))
        if not s:
            continue
        print(f"-- {model} --")
        bld = s["by_layout_and_difficulty"]
        for layout in LAYOUTS:
            row = []
            for diff in DIFFICULTIES:
                key = f"{layout}_{diff}"
                cell = bld.get(key, {})
                row.append(f"{diff}={cell.get('accuracy', float('nan')):.1f}%")
            print(f"  {layout:14s} " + "  ".join(row))

    print("\n=== by_layout_and_difficulty (path answer_format) ===\n")
    for model in MODELS:
        s = table_rows.get((model, "path"))
        if not s:
            continue
        print(f"-- {model} --")
        bld = s["by_layout_and_difficulty"]
        for layout in LAYOUTS:
            row = []
            for diff in DIFFICULTIES:
                key = f"{layout}_{diff}"
                cell = bld.get(key, {})
                row.append(f"{diff}={cell.get('accuracy', float('nan')):.1f}%")
            print(f"  {layout:14s} " + "  ".join(row))

    print("\n=== McNemar's test: direct vs zero_shot_cot (numeric answer_format) ===\n")
    for model in MODELS:
        d_direct = find_result(model, "numeric", "direct")
        d_zscot = find_result(model, "numeric", "zero_shot_cot")
        if d_direct is None or d_zscot is None:
            print(f"{model}: MISSING one of the two conditions")
            continue
        correct_direct = {r["sample_id"]: r["is_correct"] for r in d_direct["results"]}
        correct_zscot = {r["sample_id"]: r["is_correct"] for r in d_zscot["results"]}
        shared_ids = sorted(set(correct_direct) & set(correct_zscot))
        pairs = [(correct_direct[i], correct_zscot[i]) for i in shared_ids]
        b, c, p = mcnemar(pairs)
        acc_direct = 100 * sum(correct_direct[i] for i in shared_ids) / len(shared_ids)
        acc_zscot = 100 * sum(correct_zscot[i] for i in shared_ids) / len(shared_ids)
        sig = "*" if p < 0.05 else ""
        print(f"{model:14s} direct={acc_direct:5.2f}%  zero_shot_cot={acc_zscot:5.2f}%  "
              f"discordant(b={b},c={c})  p={p:.4f}{sig}  n={len(shared_ids)}")

    print("\n=== McNemar's test: kamada_kawai vs circular layout (zero_shot_cot, numeric) ===\n")
    for model in MODELS:
        d = find_result(model, "numeric", "zero_shot_cot")
        if d is None:
            continue
        kk_by_matrix = {r["matrix_id"]: r["is_correct"] for r in d["results"] if r["layout"] == "kamada_kawai"}
        ci_by_matrix = {r["matrix_id"]: r["is_correct"] for r in d["results"] if r["layout"] == "circular"}
        shared = sorted(set(kk_by_matrix) & set(ci_by_matrix))
        if not shared:
            print(f"{model}: no shared matrix_ids between layouts (unexpected)")
            continue
        pairs = [(kk_by_matrix[m], ci_by_matrix[m]) for m in shared]
        b, c, p = mcnemar(pairs)
        acc_kk = 100 * sum(kk_by_matrix[m] for m in shared) / len(shared)
        acc_ci = 100 * sum(ci_by_matrix[m] for m in shared) / len(shared)
        sig = "*" if p < 0.05 else ""
        print(f"{model:14s} kamada_kawai={acc_kk:5.2f}%  circular={acc_ci:5.2f}%  "
              f"discordant(b={b},c={c})  p={p:.4f}{sig}  n={len(shared)}")


if __name__ == "__main__":
    main()
