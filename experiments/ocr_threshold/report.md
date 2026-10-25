# OCR threshold comparison

Reproduce: `python experiments/ocr_threshold/compare.py`

Single video, preliminary labels, candidate selection only. No vision call and no claim about generated-note quality.

Source: `data/demo/GDm_uH6VxPY/frame_analysis.json` (93 representative frames). The default gate is OCR score >= 8.

The 8 diagrams and 7 irrelevant frames below were identified only among frames that gate misses. They are not the full set of useful or useless frames in the video. The fraction of those 8 that a lower threshold selects is not overall recall.

| Threshold | Candidates | Additional vision calls | Labeled diagrams from the missed set | Labeled irrelevant frames | Software demo |
|---:|---:|---:|---:|---:|---:|
| 8 | 36 | 0 | 0/8 | 0/7 | 0/1 |
| 4 | 45 | 9 | 4/8 | 4/7 | 1/1 |
| 2 | 52 | 16 | 8/8 | 7/7 | 1/1 |

## Which missed frames each threshold adds

- Threshold 4 diagrams: frame-0004, frame-0012, frame-0041, frame-0055
- Threshold 4 irrelevant: frame-0001, frame-0003, frame-0044, frame-0093
- Threshold 4 software demo: frame-0071
- Threshold 2 diagrams: frame-0004, frame-0010, frame-0011, frame-0012, frame-0039, frame-0040, frame-0041, frame-0055
- Threshold 2 irrelevant: frame-0001, frame-0003, frame-0037, frame-0044, frame-0058, frame-0087, frame-0093
- Threshold 2 software demo: frame-0071

Representative ids, timestamps, and image paths are unchanged. Selecting a frame here means it would be sent to vision. It does not mean the resulting note is better.
