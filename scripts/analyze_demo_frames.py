"""Step 4A: the deterministic text GATE over the committed change-aware representatives.

Step 6 moved frame SELECTION (dense -> cropped-pHash segment -> OCR-superset representative) into
``scripts/create_demo_artifacts.py``, which commits the representatives + their per-frame signals
(phash / ocrText / textScore / segment provenance) in ``manifest.json``. This step is what's left
of 4A: apply the on-screen-text gate to each representative and emit the committed artifact

    data/demo/GDm_uH6VxPY/frame_analysis.json

that Step 4B reads to know WHICH representatives are worth a vision call (and where to seek on
verify). NO LLM, NO network, NO re-OCR -- it reads the signals the selector already computed.
Every committed frame IS a representative (selection already deduped); the only decision here is
the gate: textScore >= min_text_score (default MIN_TEXT_SCORE, which is 8).

Run from the repo root:  python scripts/analyze_demo_frames.py [--min-text-score 8]
Omit the flag to use selection.minTextScore from the manifest when that key is present, else 8.
A non-default threshold will not overwrite the committed analysis; pass --output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.frame_select import (  # noqa: E402
    MIN_TEXT_SCORE,
    passes_text_gate,
    require_min_text_score,
    resolve_min_text_score,
)

VIDEO_ID = "GDm_uH6VxPY"
DATA_DIR = REPO_ROOT / "data" / "demo" / VIDEO_ID
MANIFEST_PATH = DATA_DIR / "manifest.json"
ANALYSIS_PATH = DATA_DIR / "frame_analysis.json"


def positive_int(value: str) -> int:
    try:
        return require_min_text_score(int(value))
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("min text score must be a positive integer") from None


def mmss(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"


def gate(text_score: int, min_text_score: int = MIN_TEXT_SCORE) -> tuple[bool, str]:
    floor = require_min_text_score(min_text_score)
    if passes_text_gate(text_score, floor):
        return True, (f"representative with {text_score} confident on-screen words "
                      f"(>= {floor}); vision candidate")
    return False, (f"representative but only {text_score} confident words "
                   f"(< {floor}); below text gate, skipped")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply the text gate to committed representatives.")
    parser.add_argument(
        "--min-text-score", type=positive_int, default=None,
        help=f"confident-word threshold (default {MIN_TEXT_SCORE}, or the manifest's recorded value)",
    )
    parser.add_argument(
        "--output", type=Path, default=ANALYSIS_PATH,
        help="where to write the analysis JSON (default: committed frame_analysis.json)",
    )
    return parser


def resolve_output(min_text_score: int, output: Path) -> Path:
    """Return the write path. Refuse to replace the committed analysis at another threshold."""
    dest = Path(output).resolve()
    if int(min_text_score) != MIN_TEXT_SCORE and dest == ANALYSIS_PATH.resolve():
        raise SystemExit(
            "refusing to overwrite committed frame_analysis.json at a non-default threshold; pass --output"
        )
    return dest


def main() -> None:
    args = build_parser().parse_args()
    if not MANIFEST_PATH.exists():
        raise SystemExit("missing manifest -- run scripts/create_demo_artifacts.py first")
    manifest = json.loads(MANIFEST_PATH.read_text())
    min_text_score = resolve_min_text_score(
        args.min_text_score, manifest.get("selection", {}).get("minTextScore")
    )
    output_path = resolve_output(min_text_score, args.output)

    frames = []
    for f in manifest["frames"]:
        if not (REPO_ROOT / f["imagePath"].lstrip("/")).is_file():
            raise SystemExit(f"frame_analysis references missing file: {f['imagePath']}")
        is_cand, reason = gate(int(f["textScore"]), min_text_score)
        frames.append({
            "id": f["id"], "timestampSec": f["timestampSec"], "imagePath": f["imagePath"],
            "phash": f["phash"], "segmentId": f["segmentId"],
            "selectionReason": f["selectionReason"], "segMemberTimestamps": f["segMemberTimestamps"],
            "isRepresentative": True,          # every committed frame is a selected representative
            "ocrText": f["ocrText"], "wordCount": f["wordCount"], "textLength": f["textLength"],
            "textScore": f["textScore"], "ocrConfidence": f["ocrConfidence"],
            "isVisionCandidate": is_cand, "candidateReason": reason,
            "suggestedVerifyTimestampSec": max(float(f["timestampSec"]) - 3.0, 0.0),
        })

    candidates = [f for f in frames if f["isVisionCandidate"]]
    text_positive = [f for f in frames if f["textScore"] > 0]
    out = {
        "videoId": VIDEO_ID, "source": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "params": {
            "minTextScore": min_text_score,
            "segThreshold": manifest["selection"]["segThreshold"],
            "slideRegionFrac": manifest["selection"]["slideRegionFrac"],
            "adjacentOverlap": manifest["selection"]["adjacentOverlap"],
        },
        "selection": manifest["selection"],
        "counts": {
            "totalFrames": len(frames), "representatives": len(frames),
            "textPositiveFrames": len(text_positive), "visionCandidates": len(candidates),
        },
        "frames": frames,
    }
    output_path.write_text(json.dumps(out, indent=2) + "\n")

    print("=== Step 4A text gate ===")
    print(f"  min text score:   {min_text_score}")
    print(f"  representatives:  {len(frames)}")
    print(f"  text-positive:    {len(text_positive)}")
    print(f"  vision candidates:{len(candidates)}")
    try:
        shown = output_path.relative_to(REPO_ROOT)
    except ValueError:
        shown = output_path
    print(f"  wrote:            {shown}")
    print(f"\n  candidate ids: {[f['id'] for f in candidates]}")


if __name__ == "__main__":
    main()
