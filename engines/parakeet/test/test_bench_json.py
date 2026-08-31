#!/usr/bin/env python3
"""Assert --bench --bench-json writes a well-formed report in every CLI mode.

Regression gate for the modes that used to return before the bench block and
silently ignore the flag: Sortformer offline, Sortformer streaming, and
speaker-attributed transcription.

usage: test_bench_json.py <parakeet-cli> <mode> <model> [diarization-model] <wav>
       mode = asr | sortformer | sortformer-stream | attributed
"""
import json
import os
import subprocess
import sys
import tempfile

RUNS = 1
STAGE_KEYS = ("mel_ms", "encoder_ms", "decode_ms")
EXPECTED_COUNTS = {
    "asr":               ["tokens"],
    "sortformer":        ["segments", "num_spks"],
    "sortformer-stream": ["segments", "chunk_ms", "history_ms"],
    "attributed":        ["asr_calls", "diar_segments", "merged_segments"],
}
SPLIT_STAGES = ("asr", "sortformer")


def build_argv(cli, mode, models, wav, out):
    argv = [cli, "--model", models[0], "--wav", wav, "--threads", "2",
            "--bench", "--bench-warmup", "0", "--bench-runs", str(RUNS),
            "--bench-json", out]
    if mode == "attributed":
        argv += ["--diarization-model", models[1]]
    if mode == "sortformer-stream":
        argv += ["--stream", "--stream-chunk-ms", "2000"]
    return argv


def run_cli(argv):
    proc = subprocess.run(argv, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit("FAIL: %s\n  rc=%d\n%s" % (" ".join(argv), proc.returncode, proc.stderr[-2000:]))


def check_report(path, mode):
    if not os.path.exists(path):
        sys.exit("FAIL: %s: no bench JSON written (the flag was silently ignored)" % mode)
    with open(path) as fh:
        report = json.load(fh)
    if report.get("model_type") != mode:
        sys.exit("FAIL: %s: model_type=%r" % (mode, report.get("model_type")))
    samples = report.get("inference_ms", {}).get("samples")
    if samples is None or len(samples) != RUNS:
        sys.exit("FAIL: %s: inference_ms.samples=%r, expected %d entries" % (mode, samples, RUNS))
    if not report.get("backend", "").startswith("ggml-"):
        sys.exit("FAIL: %s: backend=%r" % (mode, report.get("backend")))
    if report.get("audio_seconds", 0) <= 0:
        sys.exit("FAIL: %s: audio_seconds=%r" % (mode, report.get("audio_seconds")))
    for key in EXPECTED_COUNTS[mode]:
        if key not in report:
            sys.exit("FAIL: %s: missing count %r" % (mode, key))
    split = mode in SPLIT_STAGES
    for key in STAGE_KEYS:
        if split and key not in report:
            sys.exit("FAIL: %s: missing stage %r" % (mode, key))
        if not split and key in report:
            sys.exit("FAIL: %s: stage %r reported but stages are not separable" % (mode, key))
    print("OK %s: rtf_median=%.6f %s" % (mode, report["rtf_median"],
          " ".join("%s=%s" % (k, report[k]) for k in EXPECTED_COUNTS[mode])))


def main():
    cli, mode = sys.argv[1], sys.argv[2]
    models, wav = sys.argv[3:-1], sys.argv[-1]
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "bench.json")
        run_cli(build_argv(cli, mode, models, wav, out))
        check_report(out, mode)


if __name__ == "__main__":
    main()
