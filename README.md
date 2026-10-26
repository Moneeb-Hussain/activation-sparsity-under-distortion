# Sparse-Text Frame Recovery

Evaluating and improving the selection of low-text visual frames in AI-generated video notes.

## Project origin

This repository is a fork of [yuningzhang520/ai-video-notes](https://github.com/yuningzhang520/ai-video-notes).

The original system is by Yuning Zhang. It turns a visually dense video into topic-segmented notes, extracts on-screen code, tables, diagrams, and text, and links each grounded point to a source frame or transcript timestamp.

This fork contains independent improvements to the OCR text gate, plus an offline evaluation of candidate-frame coverage and processing cost. The original system remains credited to Yuning Zhang.

The evaluation uses the original project's committed AI agent design patterns demo (GDm_uH6VxPY), containing 93 representative frames.

## Problem

The original pipeline samples frames, collapses them to one representative per on-screen state, and then applies an OCR text gate. The score is the number of Tesseract words whose confidence is at least 40. The default gate keeps a representative only when that score is at least 8.

That rule skips many talking-head frames. It can also skip a diagram, title, or sparse slide that has fewer than 8 confident words. These frames are excluded from visual extraction, potentially leaving important diagram content unavailable to the generated notes.

On the committed demo pack (`data/demo/GDm_uH6VxPY/frame_analysis.json`), `frame-0012` is a representative at 75.0 seconds. Its OCR score is 6, so the default gate marks it `isVisionCandidate: false` with the reason "only 6 confident words (< 8)". A preliminary review of frames missed by that gate labeled `frame-0012` as a useful diagram. Representative ids, timestamps, image paths, and deduplication are unchanged by the work below. Only the candidate decision changes.

## My contributions

- A configurable OCR threshold. The default remains 8, so an unmodified run matches the original candidate set. `--min-text-score` accepts a positive integer.
- The same threshold is used by `scripts/create_demo_artifacts.py`, `scripts/analyze_demo_frames.py`, and `scripts/build_visual_blocks.py`. Lowering it does not drop representatives. It only changes which representatives are vision candidates.
- Stale-configuration checks in the vision step. With no flag, `scripts/build_visual_blocks.py` stops if `manifest.json` and `frame_analysis.json` record different thresholds, or if a stored `isVisionCandidate` flag disagrees with the stored score at the recorded gate. An explicit `--min-text-score` overrides the recorded value and re-selects from the stored scores. A non-default run will not overwrite the committed `frame_analysis.json` or `visual_blocks.json` unless `--output` is set. The vision script reads the committed `frame_analysis.json`, not an analyze `--output` side file.
- Automated tests for the default, the boundaries at 8, 4, 2, and 1, invalid values, and agreement between the stored flags and `textScore`.
- An offline comparison of thresholds 8, 4, and 2 that reads stored scores only.
- A documented OpenCV diagram-detector experiment. It was removed from the production path, tests, CLI, and `requirements.txt` after evaluation. The result is `experiments/diagram_fallback/report.md`.

## Experimental results

Source: the 93 representatives in `data/demo/GDm_uH6VxPY/frame_analysis.json`. Counts below are vision candidates, which means additional model calls if those frames are not already cached. They are not a measure of note quality.

| Threshold | Candidates | Additional vision calls |
|---:|---:|---:|
| 8 | 36 | 0 |
| 4 | 45 | +9 |
| 2 | 52 | +16 |

A preliminary review labeled 16 frames that the default gate misses: 8 useful diagrams, 7 irrelevant frames (titles, transitions, blanks, or end cards), and 1 possible software demo. The other frames, including the original 36 candidates, were not labeled.

| Threshold | Labeled diagrams from the missed set | Labeled irrelevant frames | Possible software demo |
|---:|---:|---:|---:|
| 8 | 0/8 | 0/7 | 0/1 |
| 4 | 4/8 | 4/7 | 1/1 |
| 2 | 8/8 | 7/7 | 1/1 |

These labels come from one video. The diagram fraction is only the share of those 8 missed frames, not overall recall, and not evidence that the generated note is better. Selecting a frame means it would be sent to vision.

The diagram detector was applied only to scores 2 through 7, on top of the default gate. It produced 38 candidates (+2), selected 2 of the 8 labeled diagrams (`frame-0011`, `frame-0012`), and selected none of the 7 labeled irrelevant frames. That tradeoff was not good enough to keep. Feature rows are in `experiments/diagram_fallback/sparse_frame_features.csv`.

Regenerate the threshold table with:

```bash
python experiments/ocr_threshold/compare.py
```

That command rewrites `experiments/ocr_threshold/report.md` from the committed scores. It does not call a paid API.

## Reproducibility

Offline evaluation and the test suite do not need an Anthropic key, a YouTube download, or a vision call. They do need the Python packages in `requirements.txt`, because `app/frame_select.py` imports them. The Tesseract binary is not required for the comparison script: it reads stored `textScore` values.

From the repository root:

```bash
python -m pip install -r requirements.txt
python experiments/ocr_threshold/compare.py
python -m pytest
```

`python -m pytest` collects the modules under `tests/`, including `tests/test_text_gate_threshold.py`.

To record or apply a non-default gate without replacing committed artifacts:

```bash
python scripts/analyze_demo_frames.py --min-text-score 4 --output work/frame_analysis_t4.json
```

`work/` is gitignored. The command reads stored `textScore` values from `data/demo/GDm_uH6VxPY/manifest.json`, checks that each frame file exists, and writes a new analysis file. It does not rerun OCR and does not call a vision API.

The candidate counts in the table come from `experiments/ocr_threshold/compare.py`. These two commands were not run for that table. The first reuses a video already in `work/` and otherwise downloads it, then reruns sampling and OCR and replaces the committed representative frames. The second calls the vision API for candidates that are not already cached:

```bash
python scripts/create_demo_artifacts.py --min-text-score 4
python scripts/build_visual_blocks.py --min-text-score 4 --output work/visual_blocks_t4.json
```

Pass the same `--min-text-score` to each stage you actually run. The vision script does not read `work/frame_analysis_t4.json`.

The rejected diagram-detector experiment is documented in experiments/diagram_fallback/report.md, with supporting measurements in experiments/diagram_fallback/sparse_frame_features.csv. The experimental implementation is not included in this fork.

## Original application

The sections below describe Yuning Zhang's system. The threshold work above does not change the reader, the committed demo note, or the default gate of 8.

Most transcript-first summarizers drop what the screen shows: code, table numbers, and diagram structure. This application extracts that on-screen content and makes each grounded point click-to-verify in an embedded player. A point with no source is shown, but not presented as verified. Low-confidence extractions are flagged for review.

Yuning Zhang's README describes the running demo as a slide-dense Google Cloud talk on AI agent design patterns. The committed pack for that reader is `GDm_uH6VxPY`: 93 representative frames selected from a 250-frame sample, with 36 vision candidates at the default gate. The embedded player in `static/app.js` uses this same ID.

### Pipeline

1. Ingest the transcript and densely sample frames.
2. Select one representative per on-screen state with a cropped perceptual hash. No model chooses which frames to inspect.
3. Gate on on-screen text. The default keeps representatives with at least 8 confident words.
4. Run vision on each gated frame. At most one block per frame: code, table, diagram, or text.
5. Structure the extracted blocks into topic sections. Bullets keep their frame or transcript source.
6. Render a hierarchical reader. A click seeks the embedded player to the cited moment.

Dense samples are compared with a cropped perceptual hash so letterboxing and player chrome do not define a scene change. Adjacent frames whose OCR text is contained in a more complete capture are collapsed. The text gate then decides which remaining representatives go to vision.

Design notes, including the rejection of an LLM frame router and the datacenter YouTube fetch limit, are in [APPROACH.md](APPROACH.md). The build log is in [PLAN.md](PLAN.md).

One committed frame shows why the source image stays available: OCR read a class name as `Parallet-Agent`, and the vision model read it as `ParallelAgent`.

Stack: Python, FastAPI, Claude for build-time vision and structuring, a plain HTML/CSS/JS reader, and ffmpeg, perceptual hashing, and OCR for frame selection. The running server reads committed artifacts and does not call Anthropic.

### Run the original reader

Docker, from the repository root:

```bash
docker compose up
```

Then open http://localhost:8000. The service definition is `docker-compose.yml`. Runtime needs no API key.

Locally:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

The image and the local app expect ffmpeg and Tesseract when you re-run ingest or OCR. See the `Dockerfile`. Optional build-time reruns of vision and structuring read `ANTHROPIC_API_KEY` from the environment or `.env`. Copy `.env.example` if you do that. Do not commit `.env`.

## Limitations and future work

- The evaluation is a single-video pilot, `GDm_uH6VxPY`.
- The 8 diagrams, 7 irrelevant frames, and 1 possible software demo are preliminary labels, not validated ground truth. The original 36 candidates were not labeled.
- Every extra candidate can add a build-time vision call. Threshold 4 adds 9. Threshold 2 adds 16.
- Generated notes were not rebuilt or compared, so candidate counts do not show a better note.
- The OpenCV diagram detector recovered 2 of the 8 labeled diagrams and was removed. A later detector would need evidence on more than this video before it belongs in the default path.
- A useful next measurement is a multi-video labeled set, plus a comparison of the notes actually produced at thresholds 8, 4, and 2.
