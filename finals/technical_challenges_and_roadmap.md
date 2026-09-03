## 3. Technical Challenges and Pivots

The first challenge was that the seven targets do not have the same data regime. Tg has many labels, while the electronic and optical properties have far fewer examples but equal weight in the score. We addressed this with canonical-SMILES `GroupKFold`, target-specific learned lanes, and a shared representation that carries both molecular and polymer-specific features. This avoids allowing the largest target to decide the whole system.

The second challenge was deciding when chemistry should guide a model and when it should be left alone. We tested each physical route instead of adding every plausible formula. The simple Egc = Ei−Eea relation was retained because a learned correction added noise on the small co-measured subset; the Egb correction was retained because its ExtraTrees residual improved validation. This selective approach keeps the pipeline interpretable without forcing a physical relation where the data do not support it.

## 4. Final Sprint Roadmap

We will finalise the reproducible release package: the pipeline, evidence tables, model outputs and dashboard. The dashboard will show a polymer structure, its prediction, nearest analogue, applicability tier and explanation, including a same-polymer SMILES comparison for the invariance demonstration. Alongside the full pipeline, we are testing a transparent pure-ML ExtraTrees alternative; its current grouped-CV mean R² is **0.816344**. The final refinement is to organise explanations into chemical concepts—such as rigidity, polarity and local functional groups—and use a small literature-based polymer panel to demonstrate where the model agrees with experiment and where it needs experimental follow-up.
