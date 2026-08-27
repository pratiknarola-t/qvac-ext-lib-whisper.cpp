#!/usr/bin/env python3
"""Compare per-AR-iteration dumps from two `mm3-replay --dump-iters` runs.

`--dump-iters N [--dump-dir DIR]` writes, for iteration i in 0..N-1, five raw
little-endian f32 files with no header: `ar-iter-<i>-last-hidden.f32` (2 x
hidden), `ar-iter-<i>-sem-logits.f32` (2 x semantic_vocab: conditional half
then unconditional half), `ar-iter-<i>-guided.f32` (CFG-guided logits,
semantic_vocab floats, EOS slot already dropped), `ar-iter-<i>-feedback.f32`
(2 x hidden), and `ar-iter-<i>-depth-hidden.f32` (depth-decoder hiddens for
the frame).

In `--mode replay`, the LM and depth decoder still run their full forward
pass every iteration and are only teacher-forced on the final token choice
(see mm3_ar_choose_semantic/mm3_ar_decode_depth in mm3-ar-loop.h), so two
dump directories captured from the same replay inputs at different quant
levels or on different backends have the same iteration count and are
directly comparable position by position. A mismatched iteration count
between the two directories means the inputs weren't the same replay run,
not that quantization changed anything, so it is treated as an error rather
than a truncated comparison.

usage:
  compare_ar_dumps.py REFERENCE_DIR CANDIDATE_DIR
      [--min-cosine X] [--min-argmax-agree PCT] [--json]

Exit 0 if the comparison ran clean and any given gates pass, 1 otherwise
(including on a directory that can't actually be compared -- a gate that
returns 0 without having compared anything is worse than no gate).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

FIELDS = ["last-hidden", "sem-logits", "guided", "feedback", "depth-hidden"]
ARGMAX_FIELDS = ["guided", "sem-logits"]
TOP_K = 8


def die(message):
    print(f"compare_ar_dumps: {message}", file=sys.stderr)
    raise SystemExit(1)


def dump_path(dump_dir, iteration, field):
    return dump_dir / f"ar-iter-{iteration}-{field}.f32"


def load_field(dump_dir, iteration, field):
    path = dump_path(dump_dir, iteration, field)
    if not path.exists():
        die(f"{path}: missing")
    data = np.fromfile(path, dtype="<f4")
    if data.size == 0:
        die(f"{path}: empty")
    return data


def count_iterations(dump_dir):
    count = 0
    while dump_path(dump_dir, count, "sem-logits").exists():
        count += 1
    if count == 0:
        die(f"{dump_dir}: no ar-iter-0-sem-logits.f32 (not a --dump-iters output directory)")
    return count


def cosine(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0.0 and norm_b == 0.0:
        return 1.0
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(a @ b / (norm_a * norm_b))


def conditional_half(sem_logits):
    if sem_logits.size % 2 != 0:
        die("sem-logits length is odd; expected equal-sized [conditional, unconditional] halves")
    half = sem_logits.size // 2
    return sem_logits[:half]


def top_k_overlap(ref_vec, cand_vec, k):
    k = min(k, ref_vec.size)
    ref_top = set(np.argsort(ref_vec)[-k:].tolist())
    cand_top = set(np.argsort(cand_vec)[-k:].tolist())
    return len(ref_top & cand_top)


def argmax_stat(ref_vec, cand_vec):
    return {
        "agree": int(np.argmax(ref_vec)) == int(np.argmax(cand_vec)),
        "top_k_overlap": top_k_overlap(ref_vec, cand_vec, TOP_K),
    }


def compare_field_at_iteration(ref_dir, cand_dir, iteration, field):
    ref = load_field(ref_dir, iteration, field)
    cand = load_field(cand_dir, iteration, field)
    if ref.size != cand.size:
        die(f"iteration {iteration} {field}: size mismatch ({ref.size} vs {cand.size})")
    return ref, cand


def compare_iteration(ref_dir, cand_dir, iteration, field_cosines, argmax_accum):
    per_field_cosine = {}
    for field in FIELDS:
        ref, cand = compare_field_at_iteration(ref_dir, cand_dir, iteration, field)
        value = cosine(ref, cand)
        field_cosines[field].append(value)
        per_field_cosine[field] = value

        if field == "guided":
            argmax_accum["guided"].append(argmax_stat(ref, cand))
        elif field == "sem-logits":
            argmax_accum["sem-logits"].append(argmax_stat(conditional_half(ref), conditional_half(cand)))

    return min(per_field_cosine.values())


def compare(ref_dir, cand_dir, n_iterations):
    field_cosines = {field: [] for field in FIELDS}
    argmax_accum = {name: [] for name in ARGMAX_FIELDS}
    per_iteration_min = []

    for iteration in range(n_iterations):
        per_iteration_min.append(compare_iteration(ref_dir, cand_dir, iteration, field_cosines, argmax_accum))

    return field_cosines, argmax_accum, per_iteration_min


def summarize_field(cosines):
    return {
        "n": len(cosines),
        "mean_cosine": float(np.mean(cosines)),
        "min_cosine": float(np.min(cosines)),
        "worst_iteration": int(np.argmin(cosines)),
    }


def summarize_argmax(entries):
    n = len(entries)
    agree = sum(1 for e in entries if e["agree"])
    overlap_sum = sum(e["top_k_overlap"] for e in entries)
    return {
        "n": n,
        "argmax_agree_pct": 100.0 * agree / n if n else 0.0,
        "mean_top_k_overlap": overlap_sum / n if n else 0.0,
        "top_k": TOP_K,
    }


def build_summary(field_cosines, argmax_accum, per_iteration_min):
    return {
        "fields": {field: summarize_field(cosines) for field, cosines in field_cosines.items()},
        "argmax": {name: summarize_argmax(entries) for name, entries in argmax_accum.items()},
        "overall_worst_iteration": int(np.argmin(per_iteration_min)),
        "overall_worst_min_cosine": float(np.min(per_iteration_min)),
    }


def apply_gates(summary, min_cosine, min_argmax_agree):
    violations = []
    if min_cosine is not None:
        for field, stats in summary["fields"].items():
            if stats["min_cosine"] < min_cosine:
                violations.append(
                    f"{field}: min_cosine {stats['min_cosine']:.6f} < {min_cosine} "
                    f"(worst at iteration {stats['worst_iteration']})"
                )
    if min_argmax_agree is not None:
        for name, stats in summary["argmax"].items():
            if stats["argmax_agree_pct"] < min_argmax_agree:
                violations.append(
                    f"{name}: argmax_agree_pct {stats['argmax_agree_pct']:.2f} < {min_argmax_agree}"
                )
    return violations


def print_summary(summary, n_iterations):
    print(f"compared {n_iterations} iteration(s)\n")
    print(f"{'field':<14}{'n':>6}{'mean_cosine':>14}{'min_cosine':>14}{'worst_iter':>12}")
    for field, stats in summary["fields"].items():
        print(f"{field:<14}{stats['n']:>6}{stats['mean_cosine']:>14.6f}{stats['min_cosine']:>14.6f}"
              f"{stats['worst_iteration']:>12}")

    print(f"\n{'argmax':<26}{'n':>6}{'agree_pct':>12}{'mean_top8_overlap':>20}")
    for name, stats in summary["argmax"].items():
        print(f"{name:<26}{stats['n']:>6}{stats['argmax_agree_pct']:>11.2f}%{stats['mean_top_k_overlap']:>19.2f}")

    print(f"\noverall worst iteration: {summary['overall_worst_iteration']} "
          f"(min cosine {summary['overall_worst_min_cosine']:.6f} across all fields at that iteration)")

    if summary["violations"]:
        print(f"\nGATE: FAIL ({len(summary['violations'])} violation(s))")
        for violation in summary["violations"]:
            print(f"  {violation}")
    elif summary.get("gated"):
        print("\nGATE: PASS")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference_dir")
    parser.add_argument("candidate_dir")
    parser.add_argument("--min-cosine", type=float, default=None,
                        help="every field's min_cosine across iterations must be >= this")
    parser.add_argument("--min-argmax-agree", type=float, default=None,
                        help="percent (0-100); both guided and sem-logits argmax_agree_pct must be >= this")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ref_dir, cand_dir = Path(args.reference_dir), Path(args.candidate_dir)
    for directory in (ref_dir, cand_dir):
        if not directory.is_dir():
            die(f"{directory}: not a directory")

    n_ref, n_cand = count_iterations(ref_dir), count_iterations(cand_dir)
    if n_ref != n_cand:
        die(f"iteration count mismatch: {ref_dir} has {n_ref}, {cand_dir} has {n_cand} "
            f"-- replay mode teacher-forces both runs to the same length, so a mismatch "
            f"means these two dump directories are not from the same replay inputs")

    field_cosines, argmax_accum, per_iteration_min = compare(ref_dir, cand_dir, n_ref)
    summary = build_summary(field_cosines, argmax_accum, per_iteration_min)
    summary["gated"] = args.min_cosine is not None or args.min_argmax_agree is not None
    summary["violations"] = apply_gates(summary, args.min_cosine, args.min_argmax_agree)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_summary(summary, n_ref)

    return 1 if summary["violations"] else 0


if __name__ == "__main__":
    sys.exit(main())
