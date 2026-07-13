"""
Uniform retry-on-format-failure correction for the numeric track.

For every sample that came back with a non-numeric answer (a path like
"A -> F" instead of the requested weight), sends ONE corrective follow-up
turn in the same conversation and swaps in the corrected answer. The
original first-turn response is reused as-is (already saved from the
original run) — no redundant first-turn API/model call.

Applied uniformly across models so no single model gets a fairness
advantage from a second chance the others don't get. This is the
alternative to prompt-hardening: instead of tuning the prompt (which
would only ever apply to whichever model we were debugging), every model
gets the identical original prompt AND the identical correction mechanism.

Usage:
  python fix_format_failures.py --model claude-haiku
  python fix_format_failures.py --model gemma3_4b --prompt_mode few_shot_cot --description long
  python fix_format_failures.py --model claude-haiku --label_style numeric
  (omit --prompt_mode/--description to process all 6 combos for that model)

Writes new files (never overwrites the original):
  results/numeric/<label_style>/<modality>/eval_<model>_<mode>_numeric_<desc>desc_<label_style>[_textonly]_corrected_<ts>.json

--modality text_only corrects text-only-ablation runs (no image sent, ever
— including in the reconstructed prior-turn context for the correction
call) instead of the main multimodal track.
"""

import argparse
import glob
import re
import sys
from functools import partial
from pathlib import Path

from dotenv import load_dotenv

import evaluate_vlms as ev

RESULTS_NUMERIC_DIR = Path("results/numeric")

CORRECTION_MESSAGE = (
    "Your last answer wasn't a number. Respond again with ONLY:\n"
    "ANSWER: <the total weight of the shortest path as a single number, no other text>"
)


def is_clean_numeric(pred):
    return bool(re.fullmatch(r"\d+", (pred or "").strip()))


def call_claude_correction(image_b64, question, model_id, prior_response):
    """image_b64=None reconstructs the prior turn as text-only, matching a
    --modality text_only original run exactly (that run never sent an
    image either, so the correction call shouldn't invent one)."""
    import anthropic

    client = anthropic.Anthropic()
    if image_b64 is not None:
        content = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": image_b64},
            },
            {"type": "text", "text": question},
        ]
    else:
        content = question
    response = client.messages.create(
        model=model_id,
        max_tokens=2048,
        system=ev.SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": content},
            {"role": "assistant", "content": prior_response},
            {"role": "user", "content": CORRECTION_MESSAGE},
        ],
    )
    return response.content[0].text


def call_ollama_correction(image_b64, question, model_id, prior_response):
    """image_b64=None reconstructs the prior turn as text-only — see
    call_claude_correction."""
    import httpx

    if image_b64 is not None:
        first_turn = {"role": "user", "content": question, "images": [image_b64]}
    else:
        first_turn = {"role": "user", "content": question}

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": ev.SYSTEM_PROMPT},
            first_turn,
            {"role": "assistant", "content": prior_response},
            {"role": "user", "content": CORRECTION_MESSAGE},
        ],
        "stream": False,
        "options": {"temperature": 0.1},
    }
    response = httpx.post("http://localhost:11434/api/chat", json=payload, timeout=120.0)
    if response.status_code == 404:
        raise ValueError(f"Model '{model_id}' not found in Ollama. Run: ollama pull {model_id}")
    response.raise_for_status()
    return response.json()["message"]["content"]


def find_original_result_files(model, label_style, modality="multimodal", prompt_mode=None,
                               description=None):
    mode_pattern = prompt_mode or "*"
    desc_pattern = f"{description}desc" if description else "*desc"
    modality_suffix = "_textonly" if modality == "text_only" else ""
    pattern = str(RESULTS_NUMERIC_DIR / label_style / modality /
                  f"eval_{model}_{mode_pattern}_numeric_{desc_pattern}_{label_style}"
                  f"{modality_suffix}_*.json")
    matches = sorted(glob.glob(pattern))
    # Never re-correct an already-corrected file.
    return [m for m in matches if "_corrected" not in m]


def correct_one_combo(model, prompt_mode, description, label_style, orig_file, caller_func,
                      model_id, modality="multimodal", max_samples=None):
    import json

    d = json.load(open(orig_file, encoding="utf-8"))
    results = d["results"]
    flagged_idx = [i for i, r in enumerate(results)
                   if not is_clean_numeric(r["predicted_answer"])]
    total_flagged = len(flagged_idx)
    if max_samples:
        flagged_idx = flagged_idx[:max_samples]

    print(f"\n{'='*60}\n{model} / {prompt_mode} / {description}desc / {label_style} / {modality}")
    print(f"{orig_file}")
    if max_samples:
        print(f"Flagged: {total_flagged}/{len(results)} (testing on first {len(flagged_idx)})")
    else:
        print(f"Flagged: {total_flagged}/{len(results)}")
    print(f"{'='*60}")

    if not flagged_idx:
        print("Nothing to correct.")
        return

    ev.SYSTEM_PROMPT = ev.PROMPT_VARIANTS["numeric"][prompt_mode]

    with open(ev.DATASET_PATH, encoding="utf-8") as f:
        full_dataset = json.load(f)
    dataset_by_id = {s["id"]: s for s in full_dataset}

    fixed_count = 0
    for n, i in enumerate(flagged_idx, 1):
        r = results[i]
        sample = dataset_by_id.get(r["sample_id"])
        if sample is None:
            print(f"  [{n}/{len(flagged_idx)}] SKIP {r['sample_id']} — not in current dataset")
            continue

        from pathlib import PureWindowsPath
        # Same modality-specific description preference as evaluate_vlms.py
        # — the correction retry reconstructs the same first-turn text, so
        # it must match what the original run actually sent.
        desc_key = f"description_{description}_file"
        if modality == "text_only":
            textonly_key = f"description_{description}_textonly_file"
            if sample.get(textonly_key):
                desc_key = textonly_key
        desc_path = ev.DATASET_ROOT / PureWindowsPath(sample[desc_key]).as_posix()
        question_text = f"{desc_path.read_text(encoding='utf-8')}\n\n{sample['question']}"
        if modality == "multimodal":
            image_path = ev.DATASET_ROOT / PureWindowsPath(sample["image_file"]).as_posix()
            image_b64 = ev.encode_image_base64(image_path)
        else:
            image_b64 = None

        caller = partial(caller_func, prior_response=r["full_response"])
        try:
            corrected_response = ev.call_with_retries(
                caller, image_b64, question_text, model_id, label=r["sample_id"])
        except Exception as e:
            print(f"  [{n}/{len(flagged_idx)}] ERROR {r['sample_id']}: {e}")
            continue

        new_predicted = ev.extract_answer(corrected_response)
        now_numeric = is_clean_numeric(new_predicted)
        is_correct = new_predicted.strip() == str(r["ground_truth"]).strip()

        r["original_predicted_answer"] = r["predicted_answer"]
        r["original_full_response"] = r["full_response"]
        r["predicted_answer"] = new_predicted
        r["full_response"] = corrected_response
        r["model_reasoning"] = ev.extract_reasoning(corrected_response)
        r["is_correct"] = is_correct
        r["format_corrected"] = True

        if now_numeric:
            fixed_count += 1
        status = "+" if is_correct else "X"
        fmt = "num" if now_numeric else "STILL non-numeric"
        print(f"  [{n}/{len(flagged_idx)}] {status} {r['sample_id']} "
              f"GT={r['ground_truth']} PRED={new_predicted} ({fmt})")

    print(f"\nFixed format: {fixed_count}/{len(flagged_idx)} "
          f"({100 * fixed_count / len(flagged_idx):.1f}%)")

    if max_samples:
        print("(--max_samples set — test run only, not saving a partial 'corrected' file)")
        return

    modality_suffix = "_textonly" if modality == "text_only" else ""
    save_name = f"{model}_{prompt_mode}_numeric_{description}desc_{label_style}{modality_suffix}_corrected"
    filepath, summary = ev.save_results(results, save_name, prompt_mode=prompt_mode,
                                        answer_format="numeric", label_style=label_style,
                                        modality=modality)
    print(f"Corrected overall accuracy: {summary['accuracy']}%")
    print(f"Saved to: {filepath}")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        choices=["claude-haiku", "gemma3_4b", "qwen2vl", "llava"])
    parser.add_argument("--prompt_mode", default=None,
                        choices=["direct", "zero_shot_cot", "few_shot_cot"])
    parser.add_argument("--description", default=None, choices=["short", "long"])
    parser.add_argument("--label_style", default="uppercase", choices=["uppercase", "numeric"])
    parser.add_argument("--modality", default="multimodal", choices=["multimodal", "text_only"],
                        help="'multimodal' (default) corrects the main image+text track; "
                             "'text_only' corrects a --modality text_only ablation run — "
                             "no image is sent, including in the reconstructed prior turn.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap flagged samples per combo (testing only — skips saving)")
    args = parser.parse_args()

    if args.model in ev.MODEL_TAGS:
        caller_func = call_ollama_correction
        model_id = ev.MODEL_TAGS[args.model]
    elif args.model in ev.CLAUDE_MODELS:
        caller_func = call_claude_correction
        model_id = ev.CLAUDE_MODELS[args.model]
    else:
        print(f"ERROR: Unknown model '{args.model}'")
        sys.exit(1)

    orig_files = find_original_result_files(args.model, args.label_style, args.modality,
                                            args.prompt_mode, args.description)
    if not orig_files:
        print("No matching original result files found.")
        sys.exit(1)

    modality_suffix = "_textonly" if args.modality == "text_only" else ""
    mode_re = re.compile(r"eval_.+?_(direct|zero_shot_cot|few_shot_cot)_numeric_"
                          rf"(short|long)desc_{args.label_style}{modality_suffix}_")
    for orig_file in orig_files:
        m = mode_re.search(orig_file)
        prompt_mode, description = m.group(1), m.group(2)
        correct_one_combo(args.model, prompt_mode, description, args.label_style, orig_file,
                          caller_func, model_id, modality=args.modality,
                          max_samples=args.max_samples)


if __name__ == "__main__":
    main()
