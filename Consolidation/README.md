# Consolidation

Every artifact this project ever produced, organised. **An archive, not a working tree** — see
`AGENTS.md` for the map and the rules, and `MANIFEST.md` for what came from where.

```
00_competition/   the official brief + the ONE canonical dataset copy
01_round1/        Round 1
02_round2/        Round 2, including the submissions and the research log
03_round3_working_repo/  the Round-3 working repo as it stood at consolidation
04_phases/        phase 2, 3, 4, 5, 5A
05_submissions/   every submission CSV + provenance
06_oracle_QUARANTINE/  held-out verification data — read its README first
07_gpu_reference/ paths and a connection recipe; no bulk copies
08_research/      research notes and the score-discrepancy analysis
09_handoff/       the consolidation plan's research documents and progress log
```

**Quick answers**

* *"Where is the dataset?"* → `00_competition/dataset/` (everything else symlinks to it).
* *"Where is the submitted file?"* → the codebase root; the archived copy with provenance is
  `05_submissions/`.
* *"What did we try in Round 2?"* → `02_round2/` and `08_research/round2_research-log.md`.
* *"Why did the private score drop?"* → `08_research/score_discrepancy/`.
* *"Where did the verified EDA numbers come from?"* → `09_handoff/research/EDA_VERIFIED_FACTS.md`.
