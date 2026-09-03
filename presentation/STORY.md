Slide 1.
Good morning. I am Vishwa Kumaresh from Team Sandman, and our project is on polymer property prediction from repeat-unit SMILES. The main idea is simple: if someone gives us a proposed repeat unit, can we turn that string into a useful screening decision, not just a number? Our final selected record is 0.907551 mean R2 on the local held-out verification panel and 0.920 on the selected public leaderboard record. But the important part of this talk is not only the score. It is what made the score, where the model is reliable, and what chemical signals the pipeline is using.

Slide 2.
The problem has three gaps. First, a repeat-unit SMILES is not a full material: it does not include molecular weight, tacticity, processing, packing or morphology. Second, the same polymer repeat can be written in multiple ways, so a model that reads only the spelling can be unstable. Third, a prediction is not enough for screening. If we want to propose a new polymer for a target Tg, bandgap, refractive index or dielectric constant, we need the estimate, the reason, and the operating boundary. That is the design requirement for the rest of the pipeline.

Slide 3.
Before fixing the architecture, we ran experiments across chemistry, representation, learning and assembly. Some things worked, and some did not. The band-edge identity Egc equals Ei minus Eea is very strong on co-measured polymers, but a learned residual on top of it gave negative leave-one-out R2, so we removed that residual. Circular or non-cross-fitted stacking looked strong internally and then transferred badly, so we moved to out-of-fold routing. Optical decomposition helped, so we kept it. The pattern is: a component stays only when the validation route supports it.

Slide 4.
The EDA is what decided the architecture. We have 7,409 training rows and 4,940 test queries, but the targets are not balanced. Tg has 4,143 labels, while Ei, Eea, nc and eps have roughly 221 to 229 labels. The metric gives every property equal weight, so one small-target mistake matters much more than one Tg row. Also, Tg dominates pooled variance, so a pooled loss would mostly optimize Tg and hide weaker electronic or optical predictions. This is why the pipeline cannot be one generic regressor. It has to be target-aware.

The other important EDA finding is structure overlap. Several small DFT targets often share canonical polymer structures with other properties in train, even when the exact same property is not available. That makes the DFT side closer to cross-property imputation, while Tg is more structure-to-property extrapolation. This is also why we use canonical-SMILES GroupKFold for validation: if the same polymer family appears on both sides of a split, the validation can become too easy.

Slide 5.
This is the overall architecture. There are five stages: ingest, represent, learn, constrain and assemble. Ingest creates the clean structure key. Represent creates multiple chemical views. Learn chooses the right model family for each target. Constrain adds chemistry only where measured relations support it. Assemble combines out-of-fold predictions and calibrates the final distribution. I will walk through those stages one by one.

Slide 6.
In ingestion, we start with four official sources: train, test, PI1M and the auxiliary SMILES file. The labelled train has 7,409 rows, test has 4,940 queries, PI1M gives about one million unlabelled polymer strings, and the larger auxiliary SMILES file was tested but rejected because it did not improve the route we needed. Every train and test string parsed with RDKit and had two repeat-unit endpoints. We canonicalize structures and use that canonical form as the validation key. That gives us a way to prevent the same polymer from validating against itself.

Slide 7.
In representation, we use five views because each view captures a different part of the chemistry. RDKit descriptors give size, polarity, surface and shape, which are useful for bulk-like properties. Morgan fingerprints capture local chemical environments and functional groups; they are standard in cheminformatics, from Rogers and Hahn's ECFP work. Polymer Genome-style descriptors give repeat-unit context, and in our experiments they improved Egb and nc. Tanimoto similarity helps in the small-data regime because it behaves like interpolation over known chemistry. PI1M SVD gives a label-free polymer sequence prior fitted inside the run, not a pretrained external model. The reason this combination is useful is that we are not betting everything on one representation.

Slide 8.
The learning stage is target-specific. Tg has enough labels, so boosted trees and ExtraTrees are strong. Egc has enough data for boosting plus a residual representation from PI1M. Egb has only 337 labels, but it has a useful relation to Egc, so we use an affine route and an ExtraTrees residual. Ei and Eea have around 222 labels, so high-capacity deep models are risky; we use a mix of MLP, Gaussian process and Tanimoto kernel ridge for diversity. For nc and eps, the optical structure is helpful: eps can be decomposed into n squared plus an ionic or polar remainder, so the model learns the better-conditioned remainder instead of the full noisy target. This also matches the general lesson from Grinsztajn et al. that tree models remain very strong on small tabular datasets.

Slide 9.
The physical-route slide is where we show that we did not just add equations because they sound scientific. Egc equals Ei minus Eea has R2 0.9716 on the co-measured subset, so the direct identity is kept. But the learned correction on top of it was worse, so that correction was removed. For eps, decomposing into n squared plus an ionic part gave a non-negative remainder on the audited subset and made the learning problem better conditioned. For Egb, the affine Egc route plus ExtraTrees residual improved validation, so that route stayed. This is the chemistry layer: measured relations are useful, but only if the held-out test supports them.

Slide 10.
Assembly is out-of-fold. That means each component prediction used for blending comes from a model that did not train on that row. We use non-negative least squares because it gives an auditable blend and avoids negative cancellation weights that can fit fold noise. The GINE graph model is not a pretrained foundation model. It is trained on the official labelled data with structure-grouped folds and is added because it makes different errors from the tabular stack. The main idea here is complementarity: not every model has to be the best alone; it has to add a useful error pattern to the final assembly.

Slide 11.
Now we move from quantitative score to qualitative evidence. The judges asked for invariance, explainability, generalization and robustness. We handle these separately because each one means something different. Invariance asks whether equivalent spellings change the input representation. Explainability asks whether the important features are actually load-bearing. Generalization asks what happens as the input moves away from familiar chemistry. Robustness asks whether the model communicates its own boundary.

Slide 12.
For invariance, we tested a concrete repeat-unit case. An oligomer is a short chain made from repeated units, so dimer and trimer are the two-repeat and three-repeat cases. The same PEO repeat can be written as the primitive form, a translated cut-point, a dimer or a trimer. In the code, those four supported terminal-star linear PSMILES forms normalize to one primitive repeat, *CCO*, before graph construction and inference. That gives 4 out of 4 normalization success and zero prediction range across all seven targets in the tested panel. The strong claim is not that every possible polymer grammar is solved, and not that real finite oligomers with end groups have identical material properties. The strong claim is that for the supported repeat grammar shown in the demo, monomer, dimer, trimer and cut-point spellings collapse before prediction. That directly addresses the oligomer and dimer invariance requirement in a scoped, testable way.

Slide 13.
For explainability, we use a proxy tree ensemble because the full final assembly is too complex for one clean attribution method. The proxy shares the feature space with the production system, so it is useful for understanding which feature families carry signal. We do not just show SHAP plots; we test whether the explanation matters. When the top 10 percent SHAP-ranked features are masked, R2 drops much more than when the same number of random features are masked. That is the evidence that the explanation is load-bearing for the proxy model. It is still not a claim of chemical causality, and if asked, I will say exactly that.

Slide 14.
For generalization and robustness, we show the ladder. Random validation is easiest. Canonical group validation is harder. Scaffold and low-similarity validation are harder still. As similarity falls, performance falls, especially for Tg. That is not something we hide; it becomes a feature of the dashboard. The dashboard shows a nearest-neighbour similarity and an applicability tier, so a user knows whether the model is operating near familiar chemistry or extrapolating. This is what makes the output a screening decision rather than just a number.

Slide 15.
The result slide combines the two scoring records: 0.907551 mean R2 on the local held-out verification panel and 0.920 on the selected public leaderboard record. The parity plots are there because a single mean R2 can hide target behavior. The metric averages seven target-wise R2 values equally, so we inspect all properties, not just Tg. This matters because electronic and optical properties have far fewer labels but the same metric weight.

Slide 16.
This slide shows the failures that shaped the final system. The band-edge residual failed and was removed. The D-MPNN route on Ei failed because the data was too small for that model family. Circular stacking looked good internally and transferred poorly, so we fixed the routing with cross-fitting. Explanation ranking agreement did not meet our threshold, so we report it as a limitation and keep the explanation claim narrower. These failures are not side notes; they are why the final pipeline is more disciplined.

Slide 17.
Now I will switch from the slide deck to the live dashboard. The demo flow is: choose the PEO strict invariance example, run inference, then compare primitive, translated, dimer and trimer forms. The point is to show the same normalized representation and the same prediction. Then I can enter or choose another polymer, show the predicted property, the interval, the nearest training analogue, the applicability tier and the feature chart. If a judge gives a distant or unusual SMILES, I will not oversell it. I will show the applicability warning and explain that this is a screening estimate.

Slide 18.
The research contribution is the loop: propose a repeat unit, predict the target property, inspect the model's chemical basis, check applicability, and decide the next measurement. The immediate future work is to extend repeat normalization beyond the simple linear grammar, improve uncertainty for scarce electronic and optical targets, and build a literature-based external panel. I also have a research-paper draft direction around making these polymer-informatics claims auditable against pretrained-model controls, but that is future work, not part of the submitted score.

Closing.
The main takeaway is: a score is useful only if we know what made it. Our pipeline is not just a leaderboard file. It is a target-aware architecture with chemical features, validated physical routes, out-of-fold assembly, invariance checks, explanation fidelity and an applicability boundary. That is the system I would want if I were using polymer property prediction to guide the next experiment.

If they ask about external datasets: the public submission uses official competition files and PI1M label-free representation only. No external labels or pretrained weights are used in the final submission.

If they ask about the dashboard model: the live dashboard runs the compact portable predictor for arbitrary inputs. The full fixed-panel assembly is documented separately and is not silently substituted into the live demo.

If they ask whether SHAP proves chemistry: no. SHAP is a model-attribution method. Our claim is feature-fidelity evidence for the proxy, not causal chemistry.

If they ask whether repeat normalization is universal: no. The implemented proof is for supported terminal-star linear PSMILES, demonstrated with PEO primitive, translated, dimer and trimer forms. Unsupported inputs still get ordinary inference and an applicability tier.
