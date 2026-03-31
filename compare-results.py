#!/usr/bin/env python3
"""Compare Rust Lambda peak outputs against reference peaks.db files.

Reads from /tmp/sherlock-test-results/ and produces a comparison report.
"""
import json
import sqlite3
import math
import os
from pathlib import Path

RESULTS_DIR = Path("/tmp/sherlock-test-results")
TOLERANCE = 1.5  # Da — match tolerance for nearest-neighbor


def load_peaks_db(db_path: str) -> list[dict]:
    """Load peaks from a SQLite peaks.db file."""
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT swim_mass, tof_mass, amplitude FROM peaks")
    peaks = [{"swim_mass": r[0], "tof_mass": r[1], "amplitude": r[2]} for r in cursor]
    conn.close()
    return peaks


def load_peaks_json(json_path: str) -> list[dict]:
    """Load peaks from our peaks.json output."""
    with open(json_path) as f:
        data = json.load(f)
    return data.get("peaks", [])


def compare_peaks(ref_peaks: list[dict], our_peaks: list[dict], tolerance: float) -> dict:
    """Nearest-neighbor comparison of two peak lists."""
    n_ref = len(ref_peaks)
    n_our = len(our_peaks)

    if n_ref == 0 or n_our == 0:
        return {
            "ref_count": n_ref, "our_count": n_our,
            "matched": 0, "precision": 0, "recall": 0, "f1": 0,
            "mean_dist": 0,
        }

    # Sort our peaks by tof for binary search
    our_sorted = sorted(range(n_our), key=lambda i: our_peaks[i]["tof_mass"])
    our_tof = [our_peaks[i]["tof_mass"] for i in our_sorted]

    import bisect
    matched_ref = [False] * n_ref
    matched_our = [False] * n_our
    distances = []

    for ri in range(n_ref):
        rt = ref_peaks[ri]["tof_mass"]
        rs = ref_peaks[ri]["swim_mass"]

        lo = bisect.bisect_left(our_tof, rt - tolerance)
        hi = bisect.bisect_right(our_tof, rt + tolerance)

        best_dist = float("inf")
        best_idx = None

        for j in range(lo, hi):
            oi = our_sorted[j]
            dt = rt - our_peaks[oi]["tof_mass"]
            ds = rs - our_peaks[oi]["swim_mass"]
            dist = math.sqrt(dt * dt + ds * ds)
            if dist < tolerance and dist < best_dist:
                best_dist = dist
                best_idx = oi

        if best_idx is not None:
            matched_ref[ri] = True
            matched_our[best_idx] = True
            distances.append(best_dist)

    n_matched = sum(matched_ref)
    precision = n_matched / n_our if n_our > 0 else 0
    recall = n_matched / n_ref if n_ref > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    mean_dist = sum(distances) / len(distances) if distances else 0

    return {
        "ref_count": n_ref,
        "our_count": n_our,
        "matched": n_matched,
        "ref_only": n_ref - n_matched,
        "our_only": n_our - sum(matched_our),
        "precision": round(precision * 100, 1),
        "recall": round(recall * 100, 1),
        "f1": round(f1 * 100, 1),
        "mean_dist": round(mean_dist, 4),
    }


def main():
    results = []

    # Find all test runs
    response_files = sorted(RESULTS_DIR.glob("test-*-response.json"))

    print(f"{'Sample':<50} {'Ref':>5} {'Ours':>5} {'Match':>5} {'P%':>6} {'R%':>6} {'F1%':>6} {'Time':>7} {'Status':<10}")
    print("=" * 140)

    for resp_file in response_files:
        run_id = resp_file.stem.replace("-response", "")

        # Load Lambda response
        with open(resp_file) as f:
            resp = json.load(f)

        # Check for error
        if "errorMessage" in resp:
            sample = run_id
            print(f"{sample:<50} {'':>5} {'':>5} {'':>5} {'':>6} {'':>6} {'':>6} {'':>7} {'FAILED':<10}")
            results.append({"run_id": run_id, "status": "FAILED", "error": resp["errorMessage"]})
            continue

        duration_ms = resp.get("duration_ms", 0)
        peak_count = resp.get("peak_count", 0)
        status = resp.get("status", "unknown")

        # Load reference peaks
        ref_db = RESULTS_DIR / f"{run_id}-ref-peaks.db"
        our_json = RESULTS_DIR / f"{run_id}-peaks.json"

        if not ref_db.exists() or not our_json.exists():
            sample = run_id
            print(f"{sample:<50} {'':>5} {peak_count:>5} {'':>5} {'':>6} {'':>6} {'':>6} {duration_ms:>6}ms {'NO REF':<10}")
            results.append({"run_id": run_id, "status": "NO_REF", "peak_count": peak_count})
            continue

        ref_peaks = load_peaks_db(str(ref_db))
        our_peaks = load_peaks_json(str(our_json))

        comparison = compare_peaks(ref_peaks, our_peaks, TOLERANCE)

        sample = run_id
        print(
            f"{sample:<50} "
            f"{comparison['ref_count']:>5} "
            f"{comparison['our_count']:>5} "
            f"{comparison['matched']:>5} "
            f"{comparison['precision']:>5.1f}% "
            f"{comparison['recall']:>5.1f}% "
            f"{comparison['f1']:>5.1f}% "
            f"{duration_ms:>6}ms "
            f"{status:<10}"
        )

        results.append({
            "run_id": run_id,
            "status": status,
            "duration_ms": duration_ms,
            **comparison,
        })

    # Summary
    completed = [r for r in results if r.get("status") == "completed" and "f1" in r]
    if completed:
        print("\n" + "=" * 140)
        avg_f1 = sum(r["f1"] for r in completed) / len(completed)
        avg_recall = sum(r["recall"] for r in completed) / len(completed)
        avg_precision = sum(r["precision"] for r in completed) / len(completed)
        avg_duration = sum(r["duration_ms"] for r in completed) / len(completed)
        print(f"\nSummary ({len(completed)} completed samples):")
        print(f"  Average F1:        {avg_f1:.1f}%")
        print(f"  Average Precision: {avg_precision:.1f}%")
        print(f"  Average Recall:    {avg_recall:.1f}%")
        print(f"  Average Duration:  {avg_duration:.0f}ms")

    # Save results
    report_path = RESULTS_DIR / "comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull report saved to: {report_path}")


if __name__ == "__main__":
    main()
