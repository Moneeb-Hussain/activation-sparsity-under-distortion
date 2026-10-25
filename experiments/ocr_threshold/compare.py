"""Offline comparison of OCR thresholds 8, 4, and 2.

Reads committed textScore values only. Does not rerun OCR, call a vision model,
or write committed artifacts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.frame_select import MIN_TEXT_SCORE, passes_text_gate  # noqa: E402

ANALYSIS_PATH = REPO_ROOT / "data" / "demo" / "GDm_uH6VxPY" / "frame_analysis.json"
REPORT_PATH = Path(__file__).resolve().parent / "report.md"

# Preliminary labels for frames the default gate (score >= 8) does not select.
# Not ground truth, and not a label of the frames the default gate already kept.
USEFUL_DIAGRAMS = {
    "frame-0004", "frame-0010", "frame-0011", "frame-0012",
    "frame-0039", "frame-0040", "frame-0041", "frame-0055",
}
SOFTWARE_DEMO = {"frame-0071"}
IRRELEVANT = {
    "frame-0001", "frame-0003", "frame-0037", "frame-0044",
    "frame-0058", "frame-0087", "frame-0093",
}
THRESHOLDS = (8, 4, 2)


def ids_at(frames: list[dict], threshold: int) -> list[str]:
    return [frame["id"] for frame in frames if passes_text_gate(int(frame["textScore"]), threshold)]


def main() -> None:
    analysis = json.loads(ANALYSIS_PATH.read_text(encoding="utf-8"))
    frames = analysis["frames"]
    if analysis["params"]["minTextScore"] != MIN_TEXT_SCORE:
        raise SystemExit("committed analysis is not the default threshold")
    labeled = USEFUL_DIAGRAMS | SOFTWARE_DEMO | IRRELEVANT
    missed_by_default = {frame["id"] for frame in frames if not passes_text_gate(int(frame["textScore"]), 8)}
    if not labeled <= missed_by_default:
        raise SystemExit("a preliminary label sits inside the default candidate set")

    rows = []
    baseline = set(ids_at(frames, 8))
    for threshold in THRESHOLDS:
        chosen = set(ids_at(frames, threshold))
        added = chosen - baseline
        rows.append({
            "threshold": threshold,
            "candidates": len(chosen),
            "added_calls": len(added),
            "diagrams": sorted(added & USEFUL_DIAGRAMS),
            "irrelevant": sorted(added & IRRELEVANT),
            "demo": sorted(added & SOFTWARE_DEMO),
        })

    lines = [
        "# OCR threshold comparison",
        "",
        "Reproduce: `python experiments/ocr_threshold/compare.py`",
        "",
        "Single video, preliminary labels, candidate selection only. No vision call and no claim about generated-note quality.",
        "",
        f"Source: `data/demo/GDm_uH6VxPY/frame_analysis.json` ({len(frames)} representative frames). "
        "The default gate is OCR score >= 8.",
        "",
        "The 8 diagrams and 7 irrelevant frames below were identified only among frames that gate misses. "
        "They are not the full set of useful or useless frames in the video. "
        "The fraction of those 8 that a lower threshold selects is not overall recall.",
        "",
        "| Threshold | Candidates | Additional vision calls | Labeled diagrams from the missed set | Labeled irrelevant frames | Software demo |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['threshold']} | {row['candidates']} | {row['added_calls']} | "
            f"{len(row['diagrams'])}/{len(USEFUL_DIAGRAMS)} | "
            f"{len(row['irrelevant'])}/{len(IRRELEVANT)} | {len(row['demo'])}/{len(SOFTWARE_DEMO)} |"
        )
    lines.extend([
        "",
        "## Which missed frames each threshold adds",
        "",
        f"- Threshold 4 diagrams: {', '.join(rows[1]['diagrams']) or 'none'}",
        f"- Threshold 4 irrelevant: {', '.join(rows[1]['irrelevant']) or 'none'}",
        f"- Threshold 4 software demo: {', '.join(rows[1]['demo']) or 'none'}",
        f"- Threshold 2 diagrams: {', '.join(rows[2]['diagrams']) or 'none'}",
        f"- Threshold 2 irrelevant: {', '.join(rows[2]['irrelevant']) or 'none'}",
        f"- Threshold 2 software demo: {', '.join(rows[2]['demo']) or 'none'}",
        "",
        "Representative ids, timestamps, and image paths are unchanged. "
        "Selecting a frame here means it would be sent to vision. It does not mean the resulting note is better.",
        "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    for row in rows:
        print(
            f"threshold {row['threshold']}: candidates={row['candidates']} "
            f"added={row['added_calls']} diagrams={len(row['diagrams'])}/8 "
            f"irrelevant={len(row['irrelevant'])}/7 demo={len(row['demo'])}/1"
        )
    print("wrote", REPORT_PATH)


if __name__ == "__main__":
    main()
