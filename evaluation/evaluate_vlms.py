"""
MMGO-Bench v1.0 - Vision-Language Model Evaluation Pipeline
==============================================================
Evaluates VLMs on shortest-path reasoning over graph images. The main
track sends an image AND a text description together on every query;
--modality text_only drops the image and sends only the description +
question, as a perception-vs-reasoning ablation.

Three independent choices per run:
  --description  short|long   which graph_description text file accompanies
                               the image (required)
  --answer_format path|numeric  ask for the full node sequence (graded by
                               exact path match) or just the total path
                               weight (graded against the precomputed
                               shortest-path length) (required)
  --modality  multimodal|text_only  image+description (default) or
                               description-only, no image sent at all
Both description and answer_format are kept on the same dataset
(dataset/v1/mmgo_bench_v1.0.json stores the full path AND its weight on
every sample) so either can be run, or both compared, without
regenerating anything.

Most models run through Ollama (local weights or Ollama Cloud) — no
external API keys required. Claude is also supported as a paid-API
comparison point; needs ANTHROPIC_API_KEY set in .env.

Usage:
  python evaluate_vlms.py --model gemma3_4b --description short --answer_format path
  python evaluate_vlms.py --model claude-haiku --description short --answer_format numeric --max_samples 10

Author: Grace Julius
Mentor: Dr. Mina Samizadeh
Institution: Lincoln University
"""

import sys
import json
import base64
import time
import argparse
import re
from pathlib import Path, PureWindowsPath
from datetime import datetime
from dotenv import load_dotenv

# ─── Configuration ──────────────────────────────────────────────────────────
DATASET_ROOT = Path("../dataset")
DATASET_PATH = DATASET_ROOT / "v1/mmgo_bench_v1.0.json"
IMAGE_DIR = DATASET_ROOT / "v1/images"
DESCRIPTION_DIR = DATASET_ROOT / "v1/graph_description"
RESULTS_DIR = Path("results")

# Model name -> Ollama tag. Local tags must already be pulled
# (`ollama pull <tag>`); cloud tags need `ollama signin` and, depending on
# the model, an Ollama Cloud subscription.
MODEL_TAGS = {
    "llava": "llava",
    "qwen2vl": "qwen2.5vl:7b",
    "gemma3_4b": "gemma3:4b",
    "minicpm-v": "minicpm-v",
}

# Model name -> Anthropic model ID. Needs ANTHROPIC_API_KEY in .env and
# billing/credits on the account. Pinned to the exact dated snapshot
# (confirmed 2026-07-05 via the API's own `response.model` field, which
# resolves the "claude-haiku-4-5" alias to this exact string) rather than
# the floating alias, so future runs can't silently land on a different
# checkpoint if Anthropic ever repoints the bare name.
CLAUDE_MODELS = {
    "claude-haiku": "claude-haiku-4-5-20251001",
}

RATE_LIMITS = {
    "llava": 30,               # local — no API limit
    "qwen2vl": 30,              # local — no API limit
    "gemma3_4b": 30,             # local — no API limit
    "minicpm-v": 30,            # local — no API limit
    "claude-haiku": 40,          # paid API — well under Claude's tier-1 rate limit
}

# ─── Prompts ─────────────────────────────────────────────────────────────────
# Every prompt assumes an image PLUS an accompanying text description in the
# same query (--description short|long picks which description file). Two
# independent answer formats, each with its own direct/zero-shot-CoT/
# few-shot-CoT trio:
#   "path"    — ANSWER is the ordered node sequence, graded by exact match
#               against the true path (can't be gamed by a right total via
#               a nonexistent path, unlike numeric grading).
#   "numeric" — ANSWER is the total path weight, graded against the
#               precomputed shortest-path length. Simpler for a model to
#               produce; doesn't verify *which* path it found.
# zero_shot_cot deliberately says just "think step by step" (Kojima et al.
# 2022) rather than spelling out the shortest-path algorithm as a numbered
# recipe — an earlier draft's step list ("read edge weights, find the
# shortest path, add up weights") handed the model the method instead of
# testing whether it could work it out, which isn't genuine CoT.

PATH_DIRECT_PROMPT = """Respond with ONLY the final answer. Do not explain your reasoning.

Respond in this exact format:
ANSWER: <ordered sequence of node labels separated by "->", e.g. A -> C -> B>"""

PATH_ZERO_SHOT_COT_PROMPT = """Think step by step, then give your final answer.

Respond in this exact format:
REASONING: <your step-by-step reasoning>
ANSWER: <ordered sequence of node labels separated by "->", e.g. A -> C -> B>"""

PATH_FEW_SHOT_COT_PROMPT = """Here is a worked example showing the reasoning format (this example graph is not the one you
will be shown — it illustrates the process only):

Example graph: node W connects to node X with weight 3; node X connects to node Y with weight 2;
node W connects directly to node Y with weight 8.
Example question: What is the shortest path from W to Y considering edge weights?
REASONING: Two paths exist from W to Y: the direct edge W-Y with weight 8, and the path
W-X-Y with weight 3+2=5. Comparing totals, 5 is less than 8, so the shortest path is W-X-Y.
ANSWER: W -> X -> Y

Now apply the same process to the graph actually shown below.

Respond in this exact format:
REASONING: <your step-by-step reasoning>
ANSWER: <ordered sequence of node labels separated by "->", e.g. A -> C -> B>"""

NUMERIC_DIRECT_PROMPT = """Respond with ONLY the final answer. Do not explain your reasoning.

Respond in this exact format:
ANSWER: <the total weight of the shortest path as a single number>"""

NUMERIC_ZERO_SHOT_COT_PROMPT = """Think step by step, then give your final answer.

Respond in this exact format:
REASONING: <your step-by-step reasoning>
ANSWER: <the total weight of the shortest path as a single number>"""

NUMERIC_FEW_SHOT_COT_PROMPT = """Here is a worked example showing the reasoning format (this example graph is not the one you
will be shown — it illustrates the process only):

Example graph: node W connects to node X with weight 3; node X connects to node Y with weight 2;
node W connects directly to node Y with weight 8.
Example question: What is the shortest distance from W to Y considering edge weights?
REASONING: Two paths exist from W to Y: the direct edge W-Y with weight 8, and the path
W-X-Y with weight 3+2=5. Comparing totals, 5 is less than 8, so the shortest path is W-X-Y.
ANSWER: 5

Now apply the same process to the graph actually shown below.

Respond in this exact format:
REASONING: <your step-by-step reasoning>
ANSWER: <the total weight of the shortest path as a single number>"""

PROMPT_VARIANTS = {
    "path": {
        "direct": PATH_DIRECT_PROMPT,
        "zero_shot_cot": PATH_ZERO_SHOT_COT_PROMPT,
        "few_shot_cot": PATH_FEW_SHOT_COT_PROMPT,
    },
    "numeric": {
        "direct": NUMERIC_DIRECT_PROMPT,
        "zero_shot_cot": NUMERIC_ZERO_SHOT_COT_PROMPT,
        "few_shot_cot": NUMERIC_FEW_SHOT_COT_PROMPT,
    },
}

# Mutated in main() based on --answer_format/--prompt_mode; call_ollama/
# call_claude read this module-level name at call time, so reassigning it
# before evaluate_model() runs is sufficient — no need to thread a
# parameter through every caller.
SYSTEM_PROMPT = PATH_ZERO_SHOT_COT_PROMPT


# ─── Custom Exceptions ───────────────────────────────────────────────────────

class RateLimitError(Exception):
    def __init__(self, message, retry_after=60):
        super().__init__(message)
        self.retry_after = retry_after


# ─── Image Encoding ──────────────────────────────────────────────────────────

def encode_image_base64(image_path):
    """Read and encode an image file as base64."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


# ─── API Caller ────────────────────────────────────────────────────────────────

def call_ollama(image_b64, question, model_id="llava", temperature=0.1):
    """Call a vision model via Ollama (local weights or Ollama Cloud).

    `image_b64=None` sends a text-only message (the --modality text_only
    arm) — the image_b64-provided branch below is unchanged.
    """
    import httpx

    if image_b64 is not None:
        user_message = {"role": "user", "content": question, "images": [image_b64]}
    else:
        user_message = {"role": "user", "content": question}

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            user_message,
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
        }
    }

    response = httpx.post(
        "http://localhost:11434/api/chat",
        json=payload,
        timeout=120.0
    )
    if response.status_code == 404:
        raise ValueError(
            f"Model '{model_id}' not found in Ollama. "
            f"Run: ollama pull {model_id}"
        )
    response.raise_for_status()
    return response.json()["message"]["content"]


def call_claude(image_b64, question, model_id="claude-haiku-4-5"):
    """Call an Anthropic Claude model. Needs ANTHROPIC_API_KEY in .env.

    `image_b64=None` sends a text-only message (the --modality text_only
    arm) — the image_b64-provided branch below is unchanged.
    """
    import anthropic

    client = anthropic.Anthropic()
    if image_b64 is not None:
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_b64,
                },
            },
            {"type": "text", "text": question},
        ]
    else:
        content = question
    response = client.messages.create(
        model=model_id,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    return response.content[0].text


# ─── Response Parsing ─────────────────────────────────────────────────────────

def extract_answer(response_text):
    """Extract the numeric answer from model response."""
    text = response_text.strip()

    answer_match = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if answer_match:
        raw_answer = answer_match.group(1).strip()
    else:
        raw_answer = text.split('\n')[-1].strip()

    raw_answer = raw_answer.strip('.')
    nums = re.findall(r'\d+', raw_answer)
    if nums:
        return nums[0]
    return raw_answer


def extract_path_answer(response_text):
    """Extract an ordered node sequence from an ANSWER line. Node tokens
    are separated by "->" or "→" in the prompted format; falls back to the
    last non-empty line if no ANSWER: tag is present, matching
    extract_answer()'s fallback behavior."""
    text = response_text.strip()

    answer_match = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    raw_answer = answer_match.group(1).strip() if answer_match else text.split('\n')[-1].strip()

    raw_answer = raw_answer.strip('.')
    tokens = [t.strip() for t in re.split(r'->|→', raw_answer)]
    return [t for t in tokens if t]


def extract_reasoning(response_text):
    """Extract the reasoning portion from model response."""
    match = re.search(r'REASONING:\s*(.+?)(?:ANSWER:|$)', response_text,
                      re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return response_text.strip()


def call_with_retries(caller_func, image_b64, question, model_id, label="", max_attempts=5):
    """One model call with retry/backoff on failure."""
    response = None
    for attempt in range(max_attempts):
        try:
            response = caller_func(image_b64, question, model_id)
            break
        except RateLimitError as rate_err:
            wait = rate_err.retry_after
            if attempt < max_attempts - 1:
                print(f"    429 rate limit — waiting {wait}s "
                      f"(attempt {attempt+1}/{max_attempts})...")
                time.sleep(wait)
            else:
                raise rate_err
        except Exception as retry_err:
            if attempt < max_attempts - 1:
                wait = 10 * (2 ** attempt)
                print(f"    Retry {attempt+1}/{max_attempts} for "
                      f"{label} (wait {wait}s)...")
                time.sleep(wait)
            else:
                raise retry_err
    if response is None:
        raise ValueError("Empty response from API")
    return response


# ─── Evaluation Engine ────────────────────────────────────────────────────────

def evaluate_model(model_name, caller_func, model_id, dataset, max_samples=None,
                   prompt_mode="zero_shot_cot", description="short", answer_format="path",
                   modality="multimodal"):
    """
    Run evaluation for a single model across the dataset. Default query is
    image + text description + question (see module docstring);
    modality="text_only" drops the image and sends only the description +
    question, to separate perception failures from reasoning failures.

    Args:
        model_name: display name for the model
        caller_func: call_ollama or call_claude
        model_id: Ollama tag or Anthropic model ID to call
        dataset: list of MMGO-Bench samples
        max_samples: optional limit for testing
        prompt_mode: "direct" | "zero_shot_cot" | "few_shot_cot" — recorded
            on each result so runs across conditions can be told apart later
        description: "short" or "long" — which graph_description text file
            accompanies the image in the query.
        answer_format: "path" (exact-match on the full node sequence) or
            "numeric" (match against the precomputed shortest-path weight).
        modality: "multimodal" (default, image + text) or "text_only"
            (description + question only, no image sent at all).
    """
    rate_limit = RATE_LIMITS.get(model_name, 10)
    delay = 60.0 / rate_limit

    if modality == "multimodal":
        # Fail fast with a diagnosable absolute path instead of grinding
        # through the whole dataset only to discover every sample was
        # skipped (this is exactly what happened twice on 2026-07-04 — this
        # check turns a multi-minute silent grind into an instant,
        # actionable error). Not applicable to modality="text_only", which
        # never touches IMAGE_DIR at all.
        resolved_image_dir = IMAGE_DIR.resolve()
        if not IMAGE_DIR.is_dir():
            raise RuntimeError(
                f"Image directory not found: {resolved_image_dir} "
                f"(cwd={Path.cwd()}). Check you're running from the "
                f"evaluation/ directory and that dataset/v1/images/ hasn't "
                f"been moved, unmounted, or is mid-sync (e.g. iCloud Desktop "
                f"sync)."
            )
        # rglob, not glob — images nest under images/uppercase/ and
        # images/numeric/.
        image_count = len(list(IMAGE_DIR.rglob("*.png")))
        if image_count == 0:
            raise RuntimeError(
                f"Image directory {resolved_image_dir} exists but contains no "
                f".png files (cwd={Path.cwd()}). It was non-empty moments ago "
                f"if checked separately — this points to a transient sync/mount "
                f"issue rather than a missing dataset."
            )
        print(f"Image directory OK: {resolved_image_dir} ({image_count} .png files)")

    samples = dataset[:max_samples] if max_samples else dataset
    results = []
    correct = 0
    total = 0
    skipped = 0
    extract_fn = extract_path_answer if answer_format == "path" else extract_answer

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name} on {len(samples)} samples "
          f"(description={description}, answer_format={answer_format}, "
          f"modality={modality})")
    print(f"Rate limit: {rate_limit} req/min (delay: {delay:.1f}s)")
    print(f"{'='*60}\n")

    for i, sample in enumerate(samples):
        sample_id = sample["id"]

        if modality == "multimodal":
            # Resolve via the full relative path (as_posix(), not .name) —
            # images nest under images/uppercase/ or images/numeric/, so
            # truncating to the bare filename would silently look in the wrong
            # (flat) location. PureWindowsPath still normalizes backslash
            # separators first regardless of which OS wrote image_file into the
            # dataset JSON (Path(...) on macOS/Linux does NOT split on "\\" —
            # this is exactly what happened on 2026-07-04).
            image_path = DATASET_ROOT / PureWindowsPath(sample["image_file"]).as_posix()

            if not image_path.exists():
                print(f"  [{i+1}] SKIP {sample_id} - Image not found")
                skipped += 1
                continue

        try:
            if modality == "multimodal":
                image_b64 = encode_image_base64(image_path)
            else:
                image_b64 = None
            # For text_only, prefer a modality-specific description if the
            # dataset has one (e.g. description_long_textonly_file strips
            # "This is an image in <layout> format." — nonsensical when no
            # image is sent). Falls back to the regular description file
            # for any (description, label_style) combo that doesn't have a
            # text-only variant generated yet.
            desc_key = f"description_{description}_file"
            if modality == "text_only":
                textonly_key = f"description_{description}_textonly_file"
                if sample.get(textonly_key):
                    desc_key = textonly_key
            desc_path = DATASET_ROOT / PureWindowsPath(sample[desc_key]).as_posix()
            question_text = f"{desc_path.read_text(encoding='utf-8')}\n\n{sample['question']}"

            start_time = time.time()
            response = call_with_retries(caller_func, image_b64, question_text,
                                          model_id, label=sample_id)
            reasoning = extract_reasoning(response)
            elapsed = time.time() - start_time

            if answer_format == "path":
                predicted_list = extract_fn(response)
                predicted = " -> ".join(predicted_list)
                # Some graphs have >1 shortest path tied on total weight;
                # nx.dijkstra_path only ever returns one of them, so
                # "metadata.path" alone would false-negative a model that
                # found a different, equally-valid tied path. "all_paths"
                # (added retroactively — see dataset patch) holds every
                # tied path; fall back to the single path for any sample
                # generated before that field existed.
                valid_paths = sample["metadata"].get("all_paths", [sample["metadata"]["path"]])
                is_correct = predicted_list in valid_paths
                ground_truth = sample["answer"]
            else:
                predicted = extract_fn(response)
                is_correct = predicted.strip() == str(sample["path_length"]).strip()
                ground_truth = str(sample["path_length"])

            if is_correct:
                correct += 1
            total += 1

            result = {
                "sample_id": sample_id,
                "matrix_id": sample["matrix_id"],
                "model": model_name,
                "prompt_mode": prompt_mode,
                "description": description,
                "answer_format": answer_format,
                "modality": modality,
                "label_style": sample.get("label_style"),
                "layout": sample["layout"],
                "difficulty": sample["difficulty"],
                "question": sample["question"],
                "ground_truth": ground_truth,
                "predicted_answer": predicted,
                "is_correct": is_correct,
                "response_time_sec": round(elapsed, 2),
                "timestamp": datetime.now().isoformat(),
                "model_reasoning": reasoning,
                "full_response": response,
            }
            results.append(result)

            status = "+" if is_correct else "X"
            running_acc = correct / total * 100
            print(f"  [{i+1}/{len(samples)}] {status} {sample_id} "
                  f"({sample['layout']}/{sample['difficulty']}) "
                  f"GT={ground_truth} PRED={predicted} "
                  f"[{elapsed:.1f}s] Acc: {running_acc:.1f}%")

            time.sleep(delay)

        except Exception as e:
            print(f"  [{i+1}] ERROR {sample_id}: {e}")
            results.append({
                "sample_id": sample_id,
                "matrix_id": sample["matrix_id"],
                "model": model_name,
                "prompt_mode": prompt_mode,
                "description": description,
                "answer_format": answer_format,
                "modality": modality,
                "label_style": sample.get("label_style"),
                "layout": sample["layout"],
                "difficulty": sample["difficulty"],
                "question": sample["question"],
                "ground_truth": sample["answer"],
                "predicted_answer": "ERROR",
                "is_correct": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            total += 1
            time.sleep(delay)

    if total == 0:
        # Every sample was skipped or errored — fail loudly instead of
        # silently saving a "successful" 0-sample/0%-accuracy file.
        if skipped > 0:
            raise RuntimeError(
                f"{model_name} ({prompt_mode}): 0/{len(samples)} samples "
                f"processed — all {skipped} were skipped because their "
                f"image file was not found under {IMAGE_DIR}. This is not "
                f"a real 0% accuracy result; check the path still exists "
                f"and isn't a transient sync/mount issue before re-running."
            )
        raise RuntimeError(
            f"{model_name} ({prompt_mode}): 0/{len(samples)} samples "
            f"processed and none were skipped — every sample raised an "
            f"error (see ERROR lines above). This is not a real 0% "
            f"accuracy result; fix the underlying error before re-running."
        )

    if skipped > 0:
        print(f"  WARNING: {skipped}/{len(samples)} samples were skipped "
              f"(image not found) — results below only cover the remaining "
              f"{total} samples, not the full set.")

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"RESULTS: {model_name}")
    print(f"  Overall Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print(f"{'='*60}")

    return results


# ─── Save Results ─────────────────────────────────────────────────────────────

def save_results(results, model_name, prompt_mode, answer_format, label_style="uppercase",
                 modality="multimodal"):
    """Save evaluation results with layout-stratified summary.

    `prompt_mode` is passed explicitly by the caller (the mode actually run)
    rather than inferred from results[0] — inferring it was fragile: an
    empty `results` list (e.g. every sample skipped) silently defaulted to
    "zero_shot_cot" regardless of what mode was actually requested, so a
    file named eval_<model>_direct_<ts>.json could contain
    "prompt_mode": "zero_shot_cot" internally. Passing it explicitly makes
    the filename and the recorded prompt_mode incapable of disagreeing.

    `answer_format` ("path" or "numeric") and `label_style` ("uppercase" or
    "numeric") together select the output subdirectory
    (results/<answer_format>/<label_style>/) — with both a numeric answer
    format and a numeric label style in play, a flat folder distinguished
    only by filename would produce names like
    "..._numeric_shortdesc_numeric_...json", where the two "numeric"s mean
    different things. Separate directories keep that unambiguous.

    Each modality gets its own subdirectory —
    results/<format>/<label_style>/multimodal/ or .../text_only/ — so the
    two are never mixed and analysis code doesn't need to filter one back
    out of the other.
    """
    results_dir = RESULTS_DIR / answer_format / label_style / modality
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = results_dir / f"eval_{model_name}_{timestamp}.json"

    total = len(results)
    correct = sum(1 for r in results if r.get("is_correct", False))

    # Layouts derived from the actual results rather than hardcoded — v2.0
    # dropped "shell" (degenerate with "circular", REJECTION_RISKS.md §6b),
    # and a hardcoded list would silently produce an empty/misleading entry
    # for it or miss a layout added later.
    layouts_present = sorted(set(r["layout"] for r in results))
    label_styles_present = sorted(set(r["label_style"] for r in results if r.get("label_style")))

    summary = {
        "model": model_name,
        "prompt_mode": prompt_mode,
        "modality": modality,
        "timestamp": timestamp,
        "total_samples": total,
        "correct": correct,
        "accuracy": round(correct / total * 100, 2) if total > 0 else 0,
        "by_layout": {},
        "by_difficulty": {},
        "by_layout_and_difficulty": {},
        "by_label_style": {},
    }

    # Accuracy per layout
    for layout in layouts_present:
        lr = [r for r in results if r["layout"] == layout]
        lc = sum(1 for r in lr if r.get("is_correct", False))
        summary["by_layout"][layout] = {
            "total": len(lr), "correct": lc,
            "accuracy": round(lc / len(lr) * 100, 2) if lr else 0
        }

    # Accuracy per difficulty
    for diff in ["easy", "medium", "hard"]:
        dr = [r for r in results if r["difficulty"] == diff]
        dc = sum(1 for r in dr if r.get("is_correct", False))
        summary["by_difficulty"][diff] = {
            "total": len(dr), "correct": dc,
            "accuracy": round(dc / len(dr) * 100, 2) if dr else 0
        }

    # Accuracy per layout x difficulty combo
    for layout in layouts_present:
        for diff in ["easy", "medium", "hard"]:
            combo = [r for r in results
                    if r["layout"] == layout and r["difficulty"] == diff]
            cc = sum(1 for r in combo if r.get("is_correct", False))
            key = f"{layout}_{diff}"
            summary["by_layout_and_difficulty"][key] = {
                "total": len(combo), "correct": cc,
                "accuracy": round(cc / len(combo) * 100, 2) if combo else 0
            }

    # Accuracy per label style (uppercase-vs-numeric axis)
    for label_style in label_styles_present:
        sr = [r for r in results if r.get("label_style") == label_style]
        sc = sum(1 for r in sr if r.get("is_correct", False))
        summary["by_label_style"][label_style] = {
            "total": len(sr), "correct": sc,
            "accuracy": round(sc / len(sr) * 100, 2) if sr else 0
        }

    output = {"summary": summary, "results": results}

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {filepath}")
    return filepath, summary


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="MMGO-Bench v1.0 Evaluation Pipeline")
    parser.add_argument("--model", type=str, required=True,
                       help=f"Which model to evaluate. Ollama: {list(MODEL_TAGS.keys())}. "
                            f"Claude (paid API): {list(CLAUDE_MODELS.keys())}.")
    parser.add_argument("--model_id", type=str, default=None,
                       help="Override the Ollama tag / Anthropic model ID for --model")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Max samples to evaluate (for testing)")
    parser.add_argument("--dataset", type=str, default=str(DATASET_PATH))
    parser.add_argument("--prompt_mode", type=str, default="zero_shot_cot",
                       help="Prompting condition(s), comma-separated to run several "
                            "back-to-back in one invocation, e.g. "
                            "'direct,zero_shot_cot,few_shot_cot'. Choices: "
                            f"{list(PROMPT_VARIANTS['path'].keys())}")
    parser.add_argument("--difficulty", type=str, default=None,
                       choices=["easy", "medium", "hard"],
                       help="Filter the dataset to a single difficulty tier "
                            "(e.g. for a scoped ablation run). Default: all.")
    parser.add_argument("--description", type=str, required=True,
                       choices=["short", "long"],
                       help="Which graph_description text file to send alongside "
                            "the image in the same query (or alone, if "
                            "--modality text_only).")
    parser.add_argument("--modality", type=str, default="multimodal",
                       choices=["multimodal", "text_only"],
                       help="'multimodal' (default) sends the image AND the "
                            "description together — the main track. "
                            "'text_only' sends only the description + "
                            "question, no image at all, to separate "
                            "perception failures from reasoning failures.")
    parser.add_argument("--answer_format", type=str, required=True,
                       choices=["path", "numeric"],
                       help="'path' asks for the full ordered node sequence, "
                            "graded by exact match against the true path. "
                            "'numeric' asks for the total path weight, graded "
                            "against the precomputed shortest-path length. "
                            "Kept as two deliberately separate prompt styles "
                            "(not one prompt asking for both) so each can be "
                            "run and compared independently.")
    parser.add_argument("--label_style", type=str, default="uppercase",
                       choices=["uppercase", "numeric"],
                       help="Which node-labeling rendering to evaluate. Default "
                            "'uppercase' is the main track; 'numeric' is the "
                            "secondary same-graph comparison check.")

    args = parser.parse_args()

    modes = [m.strip() for m in args.prompt_mode.split(",") if m.strip()]
    for mode in modes:
        if mode not in PROMPT_VARIANTS["path"]:
            print(f"ERROR: Unknown prompt_mode '{mode}'")
            print(f"Available prompt modes: {list(PROMPT_VARIANTS['path'].keys())}")
            sys.exit(1)

    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        sys.exit(1)

    with open(dataset_path, encoding='utf-8') as f:
        dataset = json.load(f)
    print(f"Loaded dataset: {len(dataset)} samples")

    if dataset and "label_style" in dataset[0]:
        dataset = [s for s in dataset if s["label_style"] == args.label_style]
        print(f"Filtered to label_style={args.label_style}: {len(dataset)} samples")

    if args.difficulty:
        dataset = [s for s in dataset if s["difficulty"] == args.difficulty]
        print(f"Filtered to difficulty={args.difficulty}: {len(dataset)} samples")

    if args.model in MODEL_TAGS:
        caller_func = call_ollama
        model_id = args.model_id or MODEL_TAGS[args.model]
    elif args.model in CLAUDE_MODELS:
        caller_func = call_claude
        model_id = args.model_id or CLAUDE_MODELS[args.model]
    else:
        print(f"ERROR: Unknown model '{args.model}'")
        print(f"Available Ollama models: {list(MODEL_TAGS.keys())}")
        print(f"Available Claude models: {list(CLAUDE_MODELS.keys())}")
        sys.exit(1)

    global SYSTEM_PROMPT
    for mode in modes:
        if len(modes) > 1:
            print(f"\n{'#'*60}\n# prompt_mode: {mode}\n{'#'*60}")
        SYSTEM_PROMPT = PROMPT_VARIANTS[args.answer_format][mode]

        results = evaluate_model(args.model, caller_func, model_id, dataset,
                                args.max_samples, prompt_mode=mode,
                                description=args.description,
                                answer_format=args.answer_format,
                                modality=args.modality)

        # Suffix the saved model name so different conditions never collide.
        save_name = f"{args.model}_{mode}_{args.answer_format}_{args.description}desc_{args.label_style}"
        if args.modality == "text_only":
            save_name += "_textonly"
        if args.difficulty:
            save_name += f"_{args.difficulty}"
        filepath, summary = save_results(results, save_name, prompt_mode=mode,
                                        answer_format=args.answer_format,
                                        label_style=args.label_style,
                                        modality=args.modality)

        # Print layout-stratified results
        print(f"\nOverall Accuracy: {summary['accuracy']}%")
        print(f"\nAccuracy by Layout:")
        for layout, stats in summary['by_layout'].items():
            print(f"  {layout:15s}: {stats['accuracy']}%")
        print(f"\nAccuracy by Difficulty:")
        for diff, stats in summary['by_difficulty'].items():
            print(f"  {diff:10s}: {stats['accuracy']}%")
        if summary['by_label_style']:
            print(f"\nAccuracy by Label Style:")
            for ls, stats in summary['by_label_style'].items():
                print(f"  {ls:10s}: {stats['accuracy']}%")


if __name__ == "__main__":
    main()
