# MMGO-Bench: A Multi-Modal Graph Optimization Benchmark

A benchmark for evaluating vision-language models (VLMs) on shortest-path
reasoning over rendered graph images.

**Author:** Grace Julius
**Supervisor:** Dr. Mina Samizadeh
**Institution:** Lincoln University

## Overview

MMGO-Bench asks a VLM to find the shortest weighted path between two
labeled nodes in a graph, given only a rendered image of the graph (plus
an accompanying text description). It is designed to isolate several
variables that are usually left uncontrolled in similar benchmarks:

- **Layout**: the same underlying graph is rendered under 4 different
  node-link layout algorithms (spring, Kamada-Kawai, circular, random),
  so layout's effect on accuracy can be measured on a same-graph basis
  rather than confounded with graph difficulty.
- **Node label style**: every graph is rendered under both uppercase
  letters (A, B, C...) and digits (1, 2, 3...) as node labels — the same
  graph, same layout, same query, only the label style differs — to
  check for interference between numeric labels and numeric answers.
- **Description length**: each image is paired with both a short
  description (layout + node/edge count only) and a long description
  (adds direct edge weights, but not all-pairs shortest distances, which
  would leak the answer) — an image+text pairing, not text replacing the
  image.
- **Answer format**: models are asked for either the full ordered node
  sequence (path format, exact-match graded, tie-aware) or just the
  total path weight (numeric format) — independent prompt conditions,
  not one prompt asking for both.
- **Modality**: an optional `--modality text_only` ablation drops the
  image entirely, to separate perception failures from reasoning
  failures.

Ground truth is computed via Dijkstra's algorithm; where a graph has
multiple shortest paths tied on total weight, all of them are stored and
any one counts as correct.

## Dataset

- 90 base adjacency matrices (30 easy, 30 medium, 30 hard)
- Each matrix rendered under 4 layouts × 2 label styles = 8 images
- 720 total samples, each with paired short/long text descriptions
- Ground truth: full shortest-path node sequence (with tied-path
  awareness) and total path weight
- Random seed: 67

See `dataset/v1/mmgo_bench_v1.0_metadata.json` for the full generation
parameters (per-difficulty node/edge-probability/weight ranges, exact
distribution counts).

## Project Structure

```
MMGO-Bench/
├── dataset/
│   ├── generate_dataset.py           # Dataset generation pipeline
│   └── v1/
│       ├── mmgo_bench_v1.0.json      # Full dataset (720 samples)
│       ├── mmgo_bench_v1.0_metadata.json
│       ├── matrices/                 # Raw adjacency matrices (JSON)
│       ├── images/{uppercase,numeric}/       # Graph images (PNG)
│       └── graph_description/{uppercase,numeric}/  # Short/long text descriptions
├── evaluation/
│   ├── evaluate_vlms.py              # VLM evaluation pipeline
│   ├── fix_format_failures.py        # One-turn retry for non-compliant numeric answers
│   ├── rescore_tied_paths.py         # Re-grades path answers against all tied-optimal paths
│   └── results/                      # Evaluation outputs, by answer_format/label_style/modality
├── analysis/
│   └── analyze_results.py            # Accuracy breakdowns + McNemar's significance tests
└── .gitignore
```

## Requirements

- Python 3.13
- `pip install -r requirements.txt` (NetworkX, NumPy, Matplotlib, SciPy, httpx, python-dotenv, anthropic)
- Ollama (for local model inference)
- Claude Haiku needs `ANTHROPIC_API_KEY` in a local `.env` with billing/credits enabled

## Usage

Regenerate the dataset (not required — it's already included under `dataset/v1/`):
```
cd dataset
python generate_dataset.py
```

Run an evaluation (from `evaluation/`):
```
python evaluate_vlms.py --model gemma3_4b --description short --answer_format path
python evaluate_vlms.py --model claude-haiku --description long --answer_format numeric --prompt_mode direct,zero_shot_cot,few_shot_cot
python evaluate_vlms.py --model gemma3_4b --description long --answer_format path --modality text_only
```

Apply the format-correction retry pass (numeric answer format only — retries any response that
didn't parse as a clean integer, reusing the original first-turn response):
```
python fix_format_failures.py --model claude-haiku --label_style uppercase
```

Re-grade path-format results against the full tied-shortest-path set (no model re-run):
```
python rescore_tied_paths.py
```

Aggregate results and run significance tests (from `evaluation/`):
```
python ../analysis/analyze_results.py
```

## Model roster

| Model | `--model` key | Ollama tag / Anthropic ID |
|---|---|---|
| LLaVA | `llava` | `llava` |
| Gemma 3 4B | `gemma3_4b` | `gemma3:4b` |
| MiniCPM-V | `minicpm-v` | `minicpm-v` |
| Qwen2.5-VL 7B | `qwen2vl` | `qwen2.5vl:7b` |
| Claude Haiku 4.5 | `claude-haiku` | `claude-haiku-4-5-20251001` (pinned to the exact dated snapshot, not the floating alias) |

Claude Haiku is Anthropic's budget tier, not their frontier model — treat
any "commercial model" comparison here as scoped to that tier.

## Results (primary track, uppercase labels)

Pooled accuracy across both answer formats, all prompt modes and description lengths (n=4,320 per model):

| Model | Accuracy |
|---|---|
| Claude Haiku 4.5 | 73.7% |
| Gemma 3 4B | 45.6% |
| Qwen2.5-VL 7B | 36.6% |
| LLaVA | 7.7% |
| MiniCPM-V | 5.8% |

Full per-condition breakdowns (by layout, difficulty, label style, and
the text-only ablation) are in `evaluation/results/` and reproducible
via `analysis/analyze_results.py`.

## Citation

*Citation details to be added on publication.*
