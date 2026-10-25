# Diagram-fallback evaluation

> Status: rejected and removed from the production path, CLI, tests, and requirements. This report and `sparse_frame_features.csv` are the preserved result. Re-run with `python experiments/diagram_fallback/evaluate.py` only if `opencv-python-headless` is installed locally; it is not a project dependency, and this checked-in report is the result. The "Recall" column is only the share of 8 preliminarily labeled diagrams among frames the original gate missed. It is not overall recall and it is not a measure of generated-note quality.

Source: committed `frame_analysis.json` (93 representatives) and the local JPEGs. No OCR rerun and no vision call.

Preliminary labels cover only the 16 frames with OCR scores 2-7. They are not validated ground truth. Recall is the fraction of those 8 labeled useful diagrams that a gate selects. False positives are labeled titles, transitions, blank screens, or ending cards that the gate adds. The software-demo frame is counted separately. The original 36 OCR >= 8 frames are unlabeled, so they sit in every candidate total and are not in the recall denominator.

The detector did not run on the 36 frames already selected by OCR >= 8.

## Design

The detector crops the slide with the existing slide-region fraction, then:

1. Canny edges.
2. Keeps only long horizontal and vertical strokes.
3. Counts rectangular boxes made by those strokes.
4. Also records line segments, connected components, and edge density.

It selects a sparse frame when it finds at least two medium rectangular nodes and the frame is not blank. One box, a title, a logo, or a blank screen is not enough. No frame id is special-cased.

## Comparison

| Approach | Candidates | Added vision calls | Useful diagrams | Recall | False positives | Software demo |
|---|---:|---:|---:|---:|---:|---:|
| A. OCR >= 8 | 36 | 0 | 0/8 | 0.00 | 0 | 0 |
| B. OCR >= 4 | 45 | 9 | 4/8 | 0.50 | 4 | 1 |
| C. OCR >= 2 | 52 | 16 | 8/8 | 1.00 | 7 | 1 |
| D. OCR >= 8 + diagram fallback on scores 2-7 | 38 | 2 | 2/8 | 0.25 | 0 | 0 |

## Approach D details

- Useful diagrams hit: frame-0011, frame-0012
- Useful diagrams missed: frame-0004, frame-0010, frame-0039, frame-0040, frame-0041, frame-0055
- False positives: none
- Software demo added: none

OCR >= 4 hits: frame-0004, frame-0012, frame-0041, frame-0055. OCR >= 4 false positives: frame-0001, frame-0003, frame-0044, frame-0093.

## Measured features on scores 2-7

| Frame | Time | OCR | Preliminary label | Diagram? | Rectangles | Lines | Structural components | Edge density |
|---|---:|---:|---|---|---:|---:|---:|---:|
| frame-0001 | 3.0 | 4 | not_useful | False | 0 | 0 | 0 | 0.0219 |
| frame-0003 | 15.0 | 4 | not_useful | False | 0 | 4 | 0 | 0.025 |
| frame-0004 | 41.0 | 4 | useful_diagram | False | 0 | 13 | 0 | 0.0571 |
| frame-0010 | 67.0 | 2 | useful_diagram | False | 1 | 8 | 0 | 0.0289 |
| frame-0011 | 69.0 | 2 | useful_diagram | True | 3 | 22 | 0 | 0.0489 |
| frame-0012 | 75.0 | 6 | useful_diagram | True | 3 | 25 | 0 | 0.0528 |
| frame-0037 | 189.0 | 2 | not_useful | False | 0 | 5 | 0 | 0.0331 |
| frame-0039 | 195.0 | 2 | useful_diagram | False | 0 | 7 | 0 | 0.0259 |
| frame-0040 | 209.0 | 3 | useful_diagram | False | 1 | 23 | 0 | 0.0495 |
| frame-0041 | 217.0 | 5 | useful_diagram | False | 1 | 32 | 0 | 0.0715 |
| frame-0044 | 243.0 | 6 | not_useful | False | 0 | 1 | 0 | 0.0097 |
| frame-0055 | 285.0 | 6 | useful_diagram | False | 1 | 8 | 1 | 0.0408 |
| frame-0058 | 305.0 | 3 | not_useful | False | 0 | 5 | 0 | 0.0364 |
| frame-0071 | 375.0 | 4 | software_demo | False | 0 | 1 | 0 | 0.0135 |
| frame-0087 | 473.0 | 2 | not_useful | False | 0 | 4 | 0 | 0.0216 |
| frame-0093 | 499.0 | 6 | not_useful | False | 0 | 1 | 0 | 0.0276 |

## Verdict

Not worth enabling. The fallback is precise on this set (2 labeled diagrams, 0 labeled false positives, 2 added calls) but recovers only 2/8 labeled diagrams. OCR >= 4 recovers 4/8, including multi-box slides whose rectangle count stayed below 2. Line counts are higher on several of those misses than on the labeled non-diagrams; retuning the rule to that gap would be a fit to this one video. The rectangle counter is too brittle to carry the fallback. Leave the default at OCR >= 8.

Default behavior is unchanged: OCR >= 8, diagram fallback off.
