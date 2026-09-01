# **Final Submission: AI for Science & Engineering**

**Project Title:** Flood prediction (ANRF AISEHack Phase 2 — Theme 1\)

**Team Name:** Megalodon 

**GitHub Repository:https://github.com/itikelabhaskar/AISE\_PHASE2.git**

**Model Weights:** https://huggingface.co/buckets/itikelabhaskar/AISE\_PHASE2

**What we did:** Binary flood masks from 512×512 satellite tiles (SAR \+ optical \+ extra maps). Labels are scarce (\~69 train patches, 3 classes: no flood / flood / water). Kaggle still wants one binary flood mask.

**Why not one big model?** With so little data, one network overfits or gets stuck. **Five different models** each vote “flood or not”; we take **3-of-5** at each pixel. That’s how we got **\~0.227** public LB (vs **\~0.207** best single model, **\~0.221** with four voters). More voters only help if they **disagree in useful ways**.

**Why these five?** Mix of ideas so mistakes don’t line up: Siamese nets (SAR and side channels **separate**—SAR and hills/water maps behave differently), KD from **different** teachers (cheap extra signal without only copying one style), one net with **3 classes** so “flood” and “permanent water” aren’t forced into the same bucket, one simpler 21-channel U-Net \+ Otsu, B7 next to B4 because B7 alone was weaker but **adds another angle** for the vote.

**Why those training choices?** Mixed precision hurt us here → **FP32**. Flood is rare → **class weights** and IoU-style losses. Edges matter for maps → **edge term**. Test tiles looked **brighter/different** than train → **per-patch normalization** and threshold **0.25** on the ConvNeXt flood channel. We did **not** put a physics PDE in the loss—we’re doing **sensor → mask**, not fluid simulation.

**Local check (10 test patches, not Kaggle):** mIoU **0.4209**, flood IoU **0.2239**, pixel acc **0.6558**, boundary mIoU **0.2722** — see `artifacts/metrics/p4f_02273_test_ama.json`.

---

### 2\. Problem (plain)

Satellites see **wet ground** and **open water** in similar tones; SAR is **noisy**. Small label set \+ **shift** between train and competition tiles broke naive setups. We leaned on **aux maps**, **multi-class head for one model**, and **ensemble** to survive that—not on embedding PDEs in the network.

---

### 3\. What shipped

Five fixed models in `config/models_manifest.json`: Siamese B4, KD B4 (t=0.5), Siamese B7, 21-ch \+ Otsu, ConvNeXt 3-class (flood at τ=0.25). **Vote:** 3-of-5.

---

### 4\. Numbers

| What | \~Public LB |
| :---- | :---- |
| Best single (KD) | \~0.207 |
| 4-model vote | \~0.221 |
| **5-model 3-of-5** | **\~0.227** |

Local 10-patch AMA metrics (same script \+ `water_union` GT): **mIoU 0.4209**, **IoU flood 0.2239**, **pixel acc 0.6558**, **boundary mIoU 0.2722**, **F1 0.5649** (full JSON in repo).

**Figures:** 2–3 patches: image, label, prediction, error.

---

### 5\. Visuals

Show where the model confuses **flood vs standing water**. Optional: errors on **DEM / static water** — reads as “where it’s wrong in the real world,” not fake physics.

---

### 6\. What we tried that didn’t pay off

*(from `docs/APPENDIX_FAILED_STRATEGIES.md` — intuition in one line each)*

- **FP16** — unstable; **FP32** won.  
- **B7 alone** — overfit; **still worth it inside the vote**.  
- **Huge loss stacks / Lovász-heavy** — simpler **BCE \+ IoU \+ edge** did better.  
- **KD only from clones of the same idea** — **diverse** teachers helped more.  
- **Flip TTA on SAR** — often **worse**.  
- **Fix brightness on frozen BatchNorm without retrain** — hurt; **retrain or match norm** (e.g. per-patch).  
- **7–10 similar models** — **no gain** vs **five picked** models.  
- **Swapping in weak/correlated members** — **LB dropped**.

No “remove physics loss” number — we **never** used a PDE loss.

---

**Flood vs water** is the hard part; **aux data \+ 3-class path \+ vote** are there for that. **Ensemble** catches **speckle** mistakes one model repeats.

---

### 8\. Robustness & speed

Biggest real issue: **train vs test look different**. We didn’t fully stress-test **other floods/regions**. Inference \= **five forwards** — fine for batches, heavy for real-time unless you distill to one net later.

---

### 9\. Limits & “if we had time”

**Little data**, **speckle**, **shorelines**, **mis-alignment**. B7 checkpoint in the bundle is **epoch 35** (regenerated; not old epoch-20) — see manifest.

**Next:** careful self-training, one small student for deploy, optional FM baseline (e.g. Prithvi) if the track asks for it.

---

### 10\. Individual Contributions & References

**Team Roles:**

* **Itikela Bhaskar:** Spearheaded the Siamese Encoders, loss-function engineering (Tversky \+ Border weighting), and ensemble threshold validation strategies.  
* **Vijay Aravynthan:** Drove the ConvNeXt baseline iterations, Pseudo-labeling architecture, domain shift analytics, and Topological Feature Engineering.

**References:**

1. Bonafilia, D. et al., *Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1* (CVPRW 2020\) — Primary defense for CNN deployments on uni-temporal SAR environments.   
2. Kashtan, V. et al., *Deep learning-based segmentation of multi-temporal satellite imagery for flood detection* (2023) — Architectural guidance for multimodal aggregation.  
3. Prithvi EO-2 Framework Research (NASA/IBM) — Benchmarks highlighting domain constraints and ViT failure modes on radar.  
4. Attentive Dual Stream Siamese U-Net for Flood Detection on Multi-Temporal Sentinel-1 Data (IEEE 2020\) (https://ieeexplore.ieee.org/document/9883132)  
     
   

---

### Checklist

- [ ] PDF ≤3 pages  
- [x] Repo: [https://github.com/itikelabhaskar/AISE\_PHASE2](https://github.com/itikelabhaskar/AISE_PHASE2)  
- [x] Weights: [https://huggingface.co/buckets/itikelabhaskar/AISE\_PHASE2](https://huggingface.co/buckets/itikelabhaskar/AISE_PHASE2)  
- [x] Appendix: `docs/APPENDIX_FAILED_STRATEGIES.md`

