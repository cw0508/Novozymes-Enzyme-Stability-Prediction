# Silver Medal Solution for Novozymes Enzyme Stability Prediction (Kaggle, 2023)

**Authors:** Chuanwan(Christine) Wu, Haoyuan Liu

**Result:** Silver Medal — Top 1% (24/2482 teams)

---

## Abstract
We present our solution to the Novozymes Enzyme Stability Prediction Kaggle competition (2023), which challenged participants to predict enzyme thermostability, measured by melting temperature, from amino acid sequences. The dataset comprised ~48,000 natural and engineered sequences with metadata (pH, data_source). The official metric was Spearman’s rank correlation. We combined established models (gradient boosting and transformer-based features) with sequence embeddings, careful preprocessing, and ensembling. To address family-wise distribution shifts, we used stratified cross-validation grouped by enzyme family. This achieved a top 1% performance (24/2482; Silver Medal). While not methodologically novel, the work shows how rigorous validation and systematic application of existing methods yield strong results in computational biology.

---

## 1. Problem & Task
- **Goal:** Predict enzyme thermostability (melting temperature) from sequence.
- **Relevance:** Thermostability is key for protein engineering and industrial enzymes.
- **Metric:** **Spearman’s rank correlation** (ranking emphasis).

## 2. Data
- **Train/Test:** Provided by competition (~48k sequences).  
- **Features:** amino-acid sequence; metadata: **pH**, **data_source**.  
- **Sequence length:** mostly 221 AA (some 220 due to deletions).  
- **Challenges:** family-wise correlation, class imbalance, potential leakage.

## 3. Methods
- **Feature engineering (final pipeline):**
  - Normalized counts for 20 amino acids (A…Y) + grouped properties (hydrophobic, negative, positive, special, polar).
  - Optional structural signals at mutation site (SASA at edit index, BLOSUM-weighted SASA).
  - Optional pooled protein embeddings (ESM/ProtBERT) when available.
- **Models:** XGBoost/LightGBM baseline, simple MLP (ablations), and ensemble (weighted average).
- **Validation:** **GroupKFold (k=5)** grouped by enzyme family (or closest proxy) with **Spearman** as CV metric.
- **Hyperparameters:** see `src/models.py` (frozen in repo).

## 4. Results
| Model                     | CV Spearman (mean ± sd) | Public LB | Private LB |
|--------------------------|--------------------------|-----------|------------|
| GBM (AA20 + groups)      | 0.XXX ± 0.XXX           | 0.XXX     | 0.XXX      |
| + SASA/BL62 at edit idx  | 0.XXX ± 0.XXX           | 0.XXX     | 0.XXX      |
| + Pooled ESM embedding   | 0.XXX ± 0.XXX           | 0.XXX     | 0.XXX      |
| Final weighted ensemble  | **0.XXX ± 0.XXX**       | 0.XXX     | 0.XXX      |

Leaderboard: **24 / 2482 (Silver)**

## 5. Discussion
- Grouped CV prevented leakage; naive random splits overestimate performance.
- AA composition + grouped properties are strong, cheap baselines; structural site features add stability.
- ESM features help modestly; cost–benefit depends on hardware.
- Limitations: competition-tuned; limited interpretability; no explicit structure input in final model.

## 6. Conclusion & Future Work
- Strong results are attainable with careful validation and ensembling.
- Future: add structure (3D voxels/graphs), richer domain splits, and interpretability on mutational effects.

## 7. Reproducibility
- **Environment:** `requirements.txt`
- **Data:** Place competition files under `data/` (see `README.md`).
- **Train:** `python -m src.train`
- **Predict:** `python -m src.predict`
- **Outputs:** models → `outputs/models/`, submissions → `outputs/submissions/`, CV logs → `outputs/cv_logs/`.

## 8. Repository Map

```
.
├─ README.md
├─ report.md
├─ requirements.txt
├─ src/
│  ├─ data.py
│  ├─ features.py
│  ├─ models.py
│  ├─ validate.py
│  ├─ train.py
│  └─ predict.py
├─ data/              # (gitignored)
└─ outputs/
   ├─ models/
   ├─ cv_logs/
   └─ submissions/
```
