"""
MMGO-Bench v1.0 - Dataset Generator
=====================================
Generates the MMGO-Bench dataset: rendered graph images paired with a
shortest-path query, for evaluating vision-language models on multi-modal
graph optimization (weighted shortest-path) reasoning.

Design notes:

  1. Layouts: four node-link layouts — spring, kamada_kawai, circular,
     random. (A fifth candidate, shell layout, was dropped: with no
     explicit `nlist` it degenerates to circular_layout — verified
     identical output on a sample of matrices.)
  2. Node labels are a controlled variable, not a random per-matrix choice.
     Every one of the 90 base graphs is rendered under BOTH "uppercase" and
     "numeric" labels (same underlying adjacency matrix, same query pair,
     same path indices, same node positions — only the node names differ),
     kept in separate images/<style>/ and graph_description/<style>/
     subfolders so the two tracks (main uppercase battery vs. secondary
     numeric comparison) stay visually organized.
  3. Each (matrix, layout, label_style) triple gets paired short/long text
     description files, meant to accompany the image in one query, not
     replace it. "Long" lists only direct edge weights (NOT all-pairs shortest distances,
     which would leak the answer for the queried pair).
  4. Ground truth is the full shortest-path node sequence, not just its
     total weight — grading (in evaluate_vlms.py) is exact-match on the
     path itself (with tie-awareness: see metadata["all_paths"]).
  5. Layout quality gate: for "spring" and "random" (the two seeded,
     non-deterministic layouts), multiple seeds are tried and scored by
     edge_node_occlusion_score() — picks the seed where no unrelated node
     sits almost exactly on top of another edge's line. This matters
     concretely: on one hard-difficulty matrix, two nodes connected
     directly happened to have a third, unconnected node sit almost
     exactly on their connecting line under a naive fixed seed, making
     that edge and an unrelated edge visually merge into what looked like
     a single doubled/thicker line. A fixed global seed can't avoid this
     since it's a per-graph geometric coincidence.

Naming:
  Matrix files:       v1/matrices/MAT_001_easy.json (shared, label-agnostic)
  Image files:        v1/images/uppercase/MAT_001_easy_spring.png
                       v1/images/numeric/MAT_001_easy_spring.png
  Description files:  v1/graph_description/uppercase/MAT_001_easy_spring_short_description.txt
  Sample IDs:         MAT_001_easy_spring_uppercase

Author: Grace Julius
Supervisor: Dr. Mina Samizadeh
Institution: Lincoln University
"""

import json
import random
import string
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ─── Configuration ──────────────────────────────────────────────────────────
RANDOM_SEED = 67
OUT_ROOT = Path("v1")
IMAGE_DIR = OUT_ROOT / "images"
MATRIX_DIR = OUT_ROOT / "matrices"
DESCRIPTION_DIR = OUT_ROOT / "graph_description"
OUTPUT_JSON = OUT_ROOT / "mmgo_bench_v1.0.json"
METADATA_FILE = OUT_ROOT / "mmgo_bench_v1.0_metadata.json"
QUALITY_REPORT_FILE = OUT_ROOT / "layout_quality_report.txt"

DIFFICULTY_CONFIGS = {
    "easy": {"min_nodes": 4, "max_nodes": 6, "edge_prob": 0.5, "weight_range": (1, 5), "count": 30},
    "medium": {"min_nodes": 7, "max_nodes": 10, "edge_prob": 0.35, "weight_range": (1, 10), "count": 30},
    "hard": {"min_nodes": 11, "max_nodes": 15, "edge_prob": 0.25, "weight_range": (1, 15), "count": 30},
}

GRAPH_LAYOUTS = ["spring", "kamada_kawai", "circular", "random"]
LAYOUT_DISPLAY = {
    "spring": "spring",
    "kamada_kawai": "kamada-kawai",
    "circular": "circular",
    "random": "random",
}
SEED_SEARCH_ATTEMPTS = 25       # spring/random: cheap to evaluate, search widely
LAYOUT_SEARCH_ATTEMPTS = 15     # kamada_kawai/circular: pricier per attempt
OCCLUSION_T_RANGE = (0.15, 0.85)  # ignore near-endpoint "closeness" as unremarkable
OCCLUSION_DIST_THRESHOLD = 0.06   # in layout's normalized coordinate space

# Controlled axis, not a random per-matrix choice. Every matrix is
# rendered in both styles so "does label style affect accuracy" is a
# same-graph comparison.
LABEL_STYLES = ["uppercase", "numeric"]


def get_node_labels(n, style):
    if style == "uppercase":
        return list(string.ascii_uppercase[:n])
    elif style == "numeric":
        return [str(i) for i in range(1, n + 1)]
    raise ValueError(f"Unknown label style: {style}")


def generate_adjacency_matrix(n_nodes, edge_prob, weight_range):
    w_min, w_max = weight_range
    weights = np.random.randint(w_min, w_max + 1, size=(n_nodes, n_nodes))
    mask = np.random.random((n_nodes, n_nodes)) < edge_prob
    matrix = weights * mask
    matrix = np.triu(matrix, k=1)
    matrix = matrix + matrix.T
    np.fill_diagonal(matrix, 0)
    return matrix.astype(int)


def compute_shortest_paths_indexed(matrix):
    """Compute all-pairs shortest paths on the raw (unlabeled, 0..n-1)
    graph. Path indices are invariant under node relabeling, so this is
    computed once per matrix and reused for both label styles."""
    G = nx.from_numpy_array(matrix)
    n = G.number_of_nodes()
    valid_pairs = []
    paths = {}
    lengths = {}
    for i in range(n):
        for j in range(i + 1, n):
            try:
                path = nx.dijkstra_path(G, i, j, weight='weight')
                length = nx.dijkstra_path_length(G, i, j, weight='weight')
                paths[(i, j)] = path
                lengths[(i, j)] = length
                valid_pairs.append((i, j))
            except nx.NetworkXNoPath:
                pass
    return G, valid_pairs, paths, lengths


def edge_node_occlusion_score(pos, edges):
    """Badness score for a layout: penalizes any node that sits almost
    exactly on another edge's line segment (excluding that edge's own
    endpoints), which visually reads as a doubled/merged edge or an edge
    that appears to touch a node it doesn't actually connect to. Zero means
    no such coincidence was found."""
    score = 0.0
    nodes = list(pos.keys())
    lo, hi = OCCLUSION_T_RANGE
    for (u, v) in edges:
        pu, pv = np.array(pos[u]), np.array(pos[v])
        seg = pv - pu
        seg_len2 = float(seg @ seg)
        if seg_len2 < 1e-12:
            continue
        for w in nodes:
            if w == u or w == v:
                continue
            pw = np.array(pos[w])
            t = float(np.dot(pw - pu, seg) / seg_len2)
            if t < lo or t > hi:
                continue
            closest = pu + t * seg
            dist = float(np.linalg.norm(pw - closest))
            if dist < OCCLUSION_DIST_THRESHOLD:
                score += (OCCLUSION_DIST_THRESHOLD - dist)
    return score


def _circular_pos(order):
    n = len(order)
    return {node: np.array([np.cos(2 * np.pi * i / n), np.sin(2 * np.pi * i / n)])
            for i, node in enumerate(order)}


def compute_best_layout(G, layout_name, base_seed=RANDOM_SEED):
    """Returns (pos, seed_used_or_None, occlusion_score) for a graph keyed
    by its *indexed* (0..n-1) node labels. All four layouts get a search
    over multiple candidates, scored by edge_node_occlusion_score(), and
    the cleanest one wins (early-exits on a score of 0):
      - spring/random: vary the RNG seed directly.
      - kamada_kawai: vary the initial position guess it refines from
        (deterministic otherwise — no seed parameter exists).
      - circular: vary the node visiting order around the circle (which
        edges become short arcs vs. long chords through the middle is an
        ordering choice, not a graph-structure choice, so this is a
        legitimate drawing decision, not "cheating" the graph)."""
    edges = list(G.edges())
    nodes = list(G.nodes())
    n = G.number_of_nodes()

    if layout_name == "spring":
        k = 3.0 / (n ** 0.5)
        make = lambda seed: nx.spring_layout(G, seed=seed, k=k, weight='weight')
        attempts = SEED_SEARCH_ATTEMPTS
    elif layout_name == "random":
        make = lambda seed: nx.random_layout(G, seed=seed)
        attempts = SEED_SEARCH_ATTEMPTS
    elif layout_name == "kamada_kawai":
        def make(seed):
            rng = np.random.RandomState(seed)
            init_pos = {node: rng.uniform(-1, 1, 2) for node in nodes}
            return nx.kamada_kawai_layout(G, pos=init_pos, weight='weight')
        attempts = LAYOUT_SEARCH_ATTEMPTS
    elif layout_name == "circular":
        def make(seed):
            order = nodes.copy()
            np.random.RandomState(seed).shuffle(order)
            return _circular_pos(order)
        attempts = LAYOUT_SEARCH_ATTEMPTS
    else:
        raise ValueError(f"Unknown layout: {layout_name}")

    best_seed, best_score, best_pos = base_seed, None, None
    for i in range(attempts):
        seed = base_seed + i
        pos = make(seed)
        score = edge_node_occlusion_score(pos, edges)
        if best_score is None or score < best_score:
            best_score, best_seed, best_pos = score, seed, pos
        if best_score == 0:
            break
    return best_pos, best_seed, best_score


def render_graph_image(G, filepath, layout_name, pos):
    n = G.number_of_nodes()
    m = G.number_of_edges()

    # Scale canvas size and edge-label font size with node/edge count —
    # a fixed canvas/font size packs too many edge-weight labels into
    # dense hard-difficulty graphs (11-15 nodes), causing visible overlap.
    if n <= 6:
        figsize = (8, 6)
    elif n <= 10:
        figsize = (10, 7.5)
    else:
        figsize = (13, 10)

    if m <= 10:
        edge_font_size = 9
    elif m <= 20:
        edge_font_size = 8
    else:
        edge_font_size = 7

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=100)

    color_palette = ['#4ECDC4', '#FF6B6B', '#45B7D1', '#96CEB4', '#FFEAA7',
                     '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
                     '#F1948A', '#82E0AA', '#F8C471', '#AED6F1', '#D2B4DE']
    node_colors = [color_palette[i % len(color_palette)] for i in range(n)]

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#555555', width=2.0, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=700 if n <= 8 else (500 if n <= 12 else 400),
                           edgecolors='#333333', linewidths=2.0)
    nx.draw_networkx_labels(G, pos, ax=ax,
                            font_size=12 if n <= 8 else (9 if n <= 12 else 8),
                            font_weight='bold', font_color='#1a1a1a')

    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                  font_size=edge_font_size, font_color='#cc0000',
                                  rotate=False,
                                  bbox=dict(boxstyle='round,pad=0.15',
                                           facecolor='white', edgecolor='#cccccc',
                                           alpha=0.9))

    ax.set_title(f"Layout: {layout_name}", fontsize=10, color='#666666')
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('#ffffff')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def make_descriptions(layout_name, n_nodes, edges):
    """Short: layout + node/edge count. Long: short + direct edge weights
    only (confirmed with mentor 2026-07-05 — NOT all-pairs shortest
    distances, which would leak the answer for the queried pair)."""
    layout_display = LAYOUT_DISPLAY[layout_name]
    short = f"This is an image in {layout_display} format. It has {n_nodes} nodes and {len(edges)} edges."

    if edges:
        clauses = [f"The distance between {edges[0]['source']} and {edges[0]['target']} is {edges[0]['weight']}"]
        clauses += [f"the distance between {e['source']} and {e['target']} is {e['weight']}" for e in edges[1:]]
        long = short + " " + ", ".join(clauses) + "."
    else:
        long = short

    return short, long


QUESTION_TEMPLATES = [
    "What is the shortest path from node {s} to node {t}? State the path as an ordered sequence of nodes.",
    "Find the minimum cost path between {s} and {t} in this weighted graph. Give the path as a sequence of nodes (e.g., X -> Y -> Z).",
    "Using the edge weights shown, what is the cheapest route from {s} to {t}? Answer with the ordered sequence of nodes visited.",
    "What sequence of nodes forms the shortest (lowest total weight) path from {s} to {t}?",
]


def generate_dataset():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    for label_style in LABEL_STYLES:
        (IMAGE_DIR / label_style).mkdir(parents=True, exist_ok=True)
        (DESCRIPTION_DIR / label_style).mkdir(parents=True, exist_ok=True)
    MATRIX_DIR.mkdir(parents=True, exist_ok=True)

    dataset = []
    matrix_counter = 0
    skipped_no_path = 0
    quality_report_lines = []
    flagged_count = 0

    print("=" * 60)
    print("MMGO-Bench v1.0 — Dataset Generation")
    print("=" * 60)

    for difficulty, config in DIFFICULTY_CONFIGS.items():
        print(f"\n{difficulty}: generating {config['count']} matrices...")
        for _ in range(config["count"]):
            matrix_counter += 1
            n_nodes = random.randint(config["min_nodes"], config["max_nodes"])
            matrix = generate_adjacency_matrix(n_nodes, config["edge_prob"], config["weight_range"])
            matrix_id = f"MAT_{matrix_counter:03d}_{difficulty}"

            G_indexed, valid_pairs, paths, lengths = compute_shortest_paths_indexed(matrix)
            if not valid_pairs:
                skipped_no_path += 1
                continue

            s_idx, t_idx = random.choice(valid_pairs)
            path_idx = paths[(s_idx, t_idx)]
            length = lengths[(s_idx, t_idx)]
            question_template = random.choice(QUESTION_TEMPLATES)

            # Resolve one layout (best seed + positions, by index) per
            # layout algorithm — shared across both label styles below, so
            # uppercase and numeric renderings of the same matrix are
            # visually identical apart from the node text.
            layout_seeds = {}
            layout_scores = {}
            positions_indexed = {}
            for layout_name in GRAPH_LAYOUTS:
                pos, seed, score = compute_best_layout(G_indexed, layout_name)
                positions_indexed[layout_name] = pos
                layout_seeds[layout_name] = seed
                layout_scores[layout_name] = round(score, 4)
                if score > 0:
                    flagged_count += 1
                    quality_report_lines.append(
                        f"{matrix_id} / {layout_name}: residual occlusion score "
                        f"{score:.4f} (seed={seed}, searched "
                        f"{SEED_SEARCH_ATTEMPTS if layout_name in ('spring', 'random') else LAYOUT_SEARCH_ATTEMPTS} option(s))"
                    )

            matrix_file = MATRIX_DIR / f"{matrix_id}.json"
            with open(matrix_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "matrix_id": matrix_id,
                    "difficulty": difficulty,
                    "n_nodes": n_nodes,
                    "adjacency_matrix": matrix.tolist(),
                    "query_indices": [s_idx, t_idx],
                    "path_indices": path_idx,
                    "path_length": length,
                    "layout_seeds": layout_seeds,
                    "layout_occlusion_scores": layout_scores,
                }, f, indent=2)

            for label_style in LABEL_STYLES:
                labels = get_node_labels(n_nodes, label_style)
                mapping = {i: labels[i] for i in range(n_nodes)}
                G = nx.relabel_nodes(G_indexed, mapping)

                source, target = labels[s_idx], labels[t_idx]
                path = [labels[k] for k in path_idx]
                path_str = " -> ".join(path)

                edges = [{"source": str(u), "target": str(v), "weight": G[u][v].get('weight', None)}
                         for u, v in G.edges()]
                graph_info = {
                    "num_nodes": G.number_of_nodes(),
                    "num_edges": G.number_of_edges(),
                    "is_connected": nx.is_connected(G),
                    "node_labels": list(G.nodes()),
                    "label_style": label_style,
                    "adjacency_matrix": matrix.tolist(),
                    "edges": edges,
                }

                for layout_name in GRAPH_LAYOUTS:
                    uid = f"{matrix_id}_{layout_name}_{label_style}"
                    pos = {mapping[i]: p for i, p in positions_indexed[layout_name].items()}

                    img_path = IMAGE_DIR / label_style / f"{matrix_id}_{layout_name}.png"
                    render_graph_image(G, img_path, layout_name, pos)

                    short_desc, long_desc = make_descriptions(layout_name, n_nodes, edges)
                    short_path = DESCRIPTION_DIR / label_style / f"{matrix_id}_{layout_name}_short_description.txt"
                    long_path = DESCRIPTION_DIR / label_style / f"{matrix_id}_{layout_name}_long_description.txt"
                    short_path.write_text(short_desc, encoding='utf-8')
                    long_path.write_text(long_desc, encoding='utf-8')

                    question = question_template.format(s=source, t=target)

                    dataset.append({
                        "id": uid,
                        "matrix_id": matrix_id,
                        "layout": layout_name,
                        "label_style": label_style,
                        "difficulty": difficulty,
                        "image_file": img_path.as_posix(),
                        "description_short_file": short_path.as_posix(),
                        "description_long_file": long_path.as_posix(),
                        "graph_info": graph_info,
                        "question": question,
                        "answer": path_str,
                        "answer_type": "path",
                        "path_length": length,
                        "explanation": f"The shortest path from {source} to {target} is: {path_str} (total weight: {length}).",
                        "layout_occlusion_score": layout_scores[layout_name],
                        "metadata": {
                            "query_nodes": [source, target],
                            "path": path,
                        },
                    })

        print(f"  {difficulty}: done")

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    with open(QUALITY_REPORT_FILE, 'w', encoding='utf-8') as f:
        if quality_report_lines:
            f.write(f"{len(quality_report_lines)} matrix/layout combos with residual "
                     "edge-node occlusion after seed search (0 = clean):\n\n")
            f.write("\n".join(quality_report_lines))
        else:
            f.write("No residual edge-node occlusion detected in any matrix/layout combo.\n")

    metadata = {
        "benchmark_name": "MMGO-Bench v1.0",
        "full_name": "MMGO-Bench: A Multi-Modal Graph Optimization Benchmark",
        "version": "1.0",
        "created_date": datetime.now().isoformat(),
        "author": "Grace Julius",
        "supervisor": "Dr. Mina Samizadeh",
        "institution": "Lincoln University",
        "description": (
            "Shortest-path reasoning over rendered graph images: 4 node-link "
            "layouts (spring, kamada-kawai, circular, random), node label "
            "style (uppercase/numeric) as a controlled same-graph comparison "
            "rather than a random per-matrix choice, paired short/long text "
            "descriptions to accompany (not replace) the image, full "
            "shortest-path exact-match as the grading metric instead of "
            "total-weight match, and a layout-occlusion seed search so no "
            "unrelated node visually sits on top of another edge's line."
        ),
        "total_base_matrices": matrix_counter - skipped_no_path,
        "total_samples": len(dataset),
        "samples_per_matrix": len(GRAPH_LAYOUTS) * len(LABEL_STYLES),
        "layouts": GRAPH_LAYOUTS,
        "label_styles": LABEL_STYLES,
        "difficulty_distribution": {
            diff: sum(1 for s in dataset if s["difficulty"] == diff) for diff in DIFFICULTY_CONFIGS
        },
        "layout_distribution": {
            lay: sum(1 for s in dataset if s["layout"] == lay) for lay in GRAPH_LAYOUTS
        },
        "label_style_distribution": {
            ls: sum(1 for s in dataset if s["label_style"] == ls) for ls in LABEL_STYLES
        },
        "graph_parameters": {k: {kk: vv for kk, vv in v.items() if kk != "count"}
                            for k, v in DIFFICULTY_CONFIGS.items()},
        "random_seed": RANDOM_SEED,
        "layout_quality_flagged_combos": flagged_count,
    }
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print("MMGO-Bench v1.0 Dataset Generation Complete")
    print(f"{'='*60}")
    print(f"Base matrices: {metadata['total_base_matrices']} (skipped {skipped_no_path} with no valid path)")
    print(f"Total samples: {len(dataset)}")
    for diff, n in metadata["difficulty_distribution"].items():
        print(f"  {diff:10s}: {n}")
    print(f"\nBy layout:")
    for lay, n in metadata["layout_distribution"].items():
        print(f"  {lay:15s}: {n}")
    print(f"\nBy label style:")
    for ls, n in metadata["label_style_distribution"].items():
        print(f"  {ls:15s}: {n}")
    print(f"\nLayout quality: {flagged_count} matrix/layout combos with residual "
          f"occlusion after seed search — see {QUALITY_REPORT_FILE}")
    print(f"\nFiles saved under: {OUT_ROOT}/")

    return dataset, metadata


if __name__ == "__main__":
    generate_dataset()
