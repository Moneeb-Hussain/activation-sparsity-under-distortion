"""Text-gate threshold tests. No OCR, no image decode, no network.

The pipeline default stays at 8. Alternate floors change only the candidate
decision; committed frame identity is left as stored.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from app.frame_select import MIN_TEXT_SCORE, passes_text_gate, require_min_text_score, resolve_min_text_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import build_visual_blocks as bvb  # noqa: E402
import create_demo_artifacts as cda  # noqa: E402
ANALYSIS_PATH = REPO_ROOT / "data" / "demo" / "GDm_uH6VxPY" / "frame_analysis.json"
MANIFEST_PATH = REPO_ROOT / "data" / "demo" / "GDm_uH6VxPY" / "manifest.json"
IDENTITY_KEYS = ("id", "timestampSec", "imagePath", "segmentId", "selectionReason")


def _load_analyze():
    path = REPO_ROOT / "scripts" / "analyze_demo_frames.py"
    spec = importlib.util.spec_from_file_location("analyze_demo_frames_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_threshold_remains_8():
    assert MIN_TEXT_SCORE == 8
    module = _load_analyze()
    args = module.build_parser().parse_args([])
    assert args.min_text_score is None
    assert resolve_min_text_score(args.min_text_score) == 8
    assert args.output == module.ANALYSIS_PATH
    assert module.resolve_output(8, module.ANALYSIS_PATH) == module.ANALYSIS_PATH.resolve()
    accepted, reason = module.gate(8)
    assert accepted is True and ">= 8" in reason


def test_score_6_rejected_at_8_accepted_at_4_and_2():
    assert passes_text_gate(6) is False
    assert passes_text_gate(6, 8) is False
    assert passes_text_gate(6, 4) is True
    assert passes_text_gate(6, 2) is True
    module = _load_analyze()
    assert module.gate(6)[0] is False and "< 8" in module.gate(6)[1]
    assert module.gate(6, 4)[0] is True and ">= 4" in module.gate(6, 4)[1]
    assert module.gate(6, 2)[0] is True and ">= 2" in module.gate(6, 2)[1]


def test_score_1_rejected_at_threshold_2():
    assert passes_text_gate(1, 2) is False
    assert passes_text_gate(1, 4) is False
    assert passes_text_gate(1) is False
    module = _load_analyze()
    accepted, reason = module.gate(1, 2)
    assert accepted is False and "< 2" in reason


def test_nondefault_threshold_does_not_overwrite_committed_analysis(tmp_path):
    module = _load_analyze()
    with pytest.raises(SystemExit):
        module.resolve_output(4, module.ANALYSIS_PATH)
    with pytest.raises(SystemExit):
        module.resolve_output(2, module.ANALYSIS_PATH)
    dest = tmp_path / "frame_analysis.json"
    assert module.resolve_output(4, dest) == dest.resolve()


def test_committed_identity_unchanged_when_regating():
    analysis = json.loads(ANALYSIS_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    frames = analysis["frames"]
    manifest_frames = manifest["frames"]

    assert analysis["params"]["minTextScore"] == 8
    assert len(frames) == len(manifest_frames) == 93
    assert [frame["id"] for frame in frames] == [frame["id"] for frame in manifest_frames]

    for frame, source in zip(frames, manifest_frames):
        for key in IDENTITY_KEYS:
            assert frame[key] == source[key]
        assert frame["isRepresentative"] is True
        assert frame["imagePath"].startswith("/static/demo/GDm_uH6VxPY/frames/")
        assert frame["isVisionCandidate"] is passes_text_gate(frame["textScore"], 8)

    before = [{key: frame[key] for key in IDENTITY_KEYS} for frame in frames]
    for threshold in (8, 4, 2):
        for frame in frames:
            passes_text_gate(frame["textScore"], threshold)
    assert before == [{key: frame[key] for key in IDENTITY_KEYS} for frame in frames]

    frame_0012 = next(frame for frame in frames if frame["id"] == "frame-0012")
    assert frame_0012["timestampSec"] == 75.0
    assert frame_0012["textScore"] == 6
    assert frame_0012["isVisionCandidate"] is False
    assert passes_text_gate(frame_0012["textScore"], 4) is True
    assert passes_text_gate(frame_0012["textScore"], 2) is True

    score_one = [frame for frame in frames if int(frame["textScore"]) == 1]
    assert len(score_one) == 1
    assert passes_text_gate(score_one[0]["textScore"], 2) is False

    def ids_at(threshold: int) -> list[str]:
        return [frame["id"] for frame in frames if passes_text_gate(frame["textScore"], threshold)]

    at_8, at_4, at_2 = ids_at(8), ids_at(4), ids_at(2)
    assert (len(at_8), len(at_4), len(at_2)) == (36, 45, 52)
    assert set(at_8) <= set(at_4) <= set(at_2)
    assert [frame["id"] for frame in frames if frame["isVisionCandidate"]] == at_8
    added_at_4 = [frame for frame in frames if frame["id"] in set(at_4) - set(at_8)]
    added_at_2 = [frame for frame in frames if frame["id"] in set(at_2) - set(at_4)]
    assert len(added_at_4) == 9 and all(4 <= int(frame["textScore"]) <= 7 for frame in added_at_4)
    assert len(added_at_2) == 7 and all(2 <= int(frame["textScore"]) <= 3 for frame in added_at_2)


def test_threshold_must_be_a_positive_integer():
    for bad in (0, -1, True):
        with pytest.raises(ValueError):
            require_min_text_score(bad)
    assert require_min_text_score(1) == 1
    assert passes_text_gate(8, 8) and not passes_text_gate(7, 8)
    assert passes_text_gate(4, 4) and not passes_text_gate(3, 4)
    assert passes_text_gate(2, 2) and not passes_text_gate(1, 2)
    assert passes_text_gate(1, 1) and not passes_text_gate(0, 1)
    module = _load_analyze()
    for raw in ("0", "-2", "2.5", "nope"):
        with pytest.raises(SystemExit):
            module.build_parser().parse_args(["--min-text-score", raw])
        with pytest.raises(SystemExit):
            bvb.build_parser().parse_args(["--min-text-score", raw])
        with pytest.raises(SystemExit):
            cda.build_parser().parse_args(["--min-text-score", raw])
    assert module.build_parser().parse_args(["--min-text-score", "4"]).min_text_score == 4


def test_threshold_propagates_into_vision_candidate_selection(tmp_path):
    assert resolve_min_text_score(None, None) == 8
    assert resolve_min_text_score(None, 4) == 4
    assert resolve_min_text_score(2, 4) == 2

    analysis = json.loads(ANALYSIS_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    before = [(frame["id"], frame["timestampSec"], frame["imagePath"]) for frame in analysis["frames"]]
    assert before == [(frame["id"], frame["timestampSec"], frame["imagePath"]) for frame in manifest["frames"]]

    at_default = bvb.select_candidates(analysis)
    at_4 = bvb.select_candidates(analysis, 4)
    at_2 = bvb.select_candidates(analysis, 2)
    flagged = [frame["id"] for frame in analysis["frames"] if frame["isVisionCandidate"]]
    assert [frame["id"] for frame in at_default] == flagged
    assert (len(at_default), len(at_4), len(at_2)) == (36, 45, 52)
    assert [frame["id"] for frame in at_default] == [frame["id"] for frame in bvb.select_candidates(analysis, 8)]

    recorded = {"params": {"minTextScore": 4}, "frames": analysis["frames"]}
    assert [frame["id"] for frame in bvb.select_candidates(recorded)] == [frame["id"] for frame in at_4]
    assert before == [(frame["id"], frame["timestampSec"], frame["imagePath"]) for frame in analysis["frames"]]

    reps = [
        {"id": "low", "textScore": 3, "timestampSec": 1.0, "imagePath": "/low.jpg"},
        {"id": "high", "textScore": 9, "timestampSec": 2.0, "imagePath": "/high.jpg"},
    ]
    assert [rep["id"] for rep in cda.text_gate_candidates(reps, 8)] == ["high"]
    assert [rep["id"] for rep in cda.text_gate_candidates(reps, 2)] == ["low", "high"]
    assert [(rep["id"], rep["timestampSec"], rep["imagePath"]) for rep in reps] == [
        ("low", 1.0, "/low.jpg"), ("high", 2.0, "/high.jpg"),
    ]

    with pytest.raises(SystemExit):
        bvb.resolve_blocks_output(4, None)
    assert bvb.resolve_blocks_output(8, None) == bvb.BLOCKS_PATH.resolve()
    dest = tmp_path / "visual_blocks.json"
    assert bvb.resolve_blocks_output(2, dest) == dest.resolve()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert bvb.resolve_vision_gate(manifest, analysis, None) == 8
    assert bvb.resolve_vision_gate(
        {"selection": {"minTextScore": 4}}, {"params": {"minTextScore": 8}, "frames": []}, 2
    ) == 2
    with pytest.raises(SystemExit):
        bvb.resolve_vision_gate(
            {"selection": {"minTextScore": 4}},
            {"params": {"minTextScore": 8}, "frames": analysis["frames"]},
            None,
        )
    stale = {
        "params": {"minTextScore": 8},
        "frames": [{
            "id": "frame-x", "textScore": 9, "isRepresentative": True,
            "isVisionCandidate": False, "timestampSec": 1.0, "imagePath": "/x.jpg",
        }],
    }
    with pytest.raises(SystemExit):
        bvb.resolve_vision_gate({"selection": {}}, stale, None)
    assert bvb.gate_reason(stale["frames"][0], 8).startswith("representative with 9")
    assert stale["frames"][0]["id"] == "frame-x"
    assert stale["frames"][0]["imagePath"] == "/x.jpg"
