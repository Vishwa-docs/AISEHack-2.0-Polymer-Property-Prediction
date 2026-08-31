# Round 2 competition details

Evidence captured through authenticated read-only Kaggle APIs on
2026-08-03 Asia/Kolkata. No notebook was started and no file was submitted.

## Identity and schedule

- Title: AISEHack 2.0 Polymer Property Prediction: Round 2
- URL: https://www.kaggle.com/competitions/ppp-round-2
- Competition ID: 157637
- Host shown by the live API: VIJITH P
- Enabled: 2026-07-29 18:31:46 UTC
- Deadline: 2026-08-12 18:30:00 UTC / 2026-08-13 00:00:00 IST
- Maximum team size: 5
- Final submissions: 2

The static Timeline page still contains a placeholder and disagrees with the
live API. Use the live deadline above, then refresh it before packaging.

## Task and metric

Predict one target value for every row of `test.csv`, where `target_type` is one
of `tg`, `egc`, `egb`, `ei`, `eea`, `nc`, or `eps`.

The score is the arithmetic mean of seven independently calculated R² values:

```text
(R2_tg + R2_egc + R2_egb + R2_ei + R2_eea + R2_nc + R2_eps) / 7
```

The output columns are `id,target`; a real Kaggle file must be named
`submission.csv`.

## Official files

| File | Rows | Role |
|---|---:|---|
| `train.csv` | 7,409 | Current labeled training rows for seven targets |
| `test.csv` | 4,940 | Current prediction rows |
| `PI1M.csv` | 995,799 | Official unlabeled auxiliary polymer SMILES |
| `archive/train.csv` | 6,171 | Official bundled Round 1 labels for Tg/Egc |
| `archive/test.csv` | 4,115 | Official bundled Round 1 unlabeled rows |
| `archive/sample_submission.csv` | 10 | Illustrative format only |
| `archive/base_line_model.ipynb` | — | Organizer baseline |

The Data page incorrectly says the test has 4,497 rows. The downloaded file has
4,940 unique IDs and controls candidate construction.

## Specific rule posture

- Official competition data only.
- No external data, including data attached to a notebook or generated outside
  notebook execution.
- No pretrained weights, checkpoints, embeddings, caches, or processed inputs.
- Public code may be used when it brings no prohibited artifact and the pipeline
  executes reproducibly in the notebook.
- Notebook/code-only: loading, splitting, preprocessing, model initialization,
  training, inference, and CSV creation must happen in one run.
- Each real submission description must link the exact generating notebook.
- The exact default/pinned generating version must be shared with the hosts and
  reproduce the result after the competition.
- Fixed seeds and end-to-end reproducibility are mandatory.

The rules page says three daily submissions; the authenticated live control says
five and currently reports five available. No submission is authorized from this
workspace, so use zero unless the user later directs an exact upload.

## Host-sharing list copied from the rules page

- Rohit Batra IITM
- Rahulsundar
- LaksmanN
- VIJITH P
- shreyasri0301

Refresh this list before any future authorized upload.

## Access notes

The competition is private/invite-gated. Official data was downloaded through
Kaggle CLI OAuth into `ppp-round-2/`. The OAuth credential is not a project
artifact and must never be printed or copied here.

