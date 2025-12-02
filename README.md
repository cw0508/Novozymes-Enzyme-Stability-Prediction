# Novozymes Enzyme Stability Prediction

- **Author:** [Chuanwan(Christine) Wu, Haoyuan Liu]
- **Submission Date:** [Jan, 2023]
- **Competition:** [Novozymes Enzyme Stability Prediction](https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction)
---

![Novozymes](https://img.shields.io/badge/Competition-Novozymes-blue?logo=kaggle)
![Kaggle](https://img.shields.io/badge/Platform-Kaggle-20beff?logo=kaggle)
![Python](https://img.shields.io/badge/Language-Python-3776AB?logo=python)
![Status](https://img.shields.io/badge/Status-Completed-success)


This repository contains a **silver medal (24th place)** solution for the **Novozymes Enzyme Stability Prediction** competition on Kaggle. The project focuses on predicting the thermal stability (Tm value) of enzyme variants using a combination of **sequence-based**, **structural**, and **evolutionary** features, along with a **multi-model ensemble strategy**.

### 2. **Feature Engineering**

#### 🧬 Sequence-Level Features
- Amino acid composition  
- Physicochemical descriptors  
- Mutation deltas (hydrophobicity, charge, volume)  

#### 🧱 Structural Features
- **B-factor difference:**  
  - *Input:* PDB (WT & Mutant) 3D coordinates  
  - *Output:* `ΔB = B(WT) – B(Mutant)`  

- **Residue-level SASA:**  
  - Computed from predicted structures  
  - **Normalized using BLOSUM62** to incorporate evolutionary plausibility  

- **Residue contact & geometry**  
  - ESM-2 embeddings and distance-based contacts  

#### 🧠 3D CNN Features
- Voxelization of the 3D neighborhood around each mutated residue  
- CNN outputs:  
  - `dT` (predicted ΔTm)  
  - `ddG` (predicted stability energy change)  

---

## 🤖 Models Used

- **B-factor regression model**  
- **SASA + BLOSUM62 hybrid structural model**  
- **3D CNN** (voxel-based geometric deep learning)

---

## 🧮 Ensemble Strategy

Final predictions are produced by a **weighted rank ensemble** integrating all model outputs:
Final = X% * B-factor
+ Y% * normalized SASA
+ Z% * CNN (dT, ddG)
+ additional rank-averaged model contributions

This approach improves **Spearman correlation**, reduces bias, and stabilizes leaderboard performance.

---

## 📈 Results

- Structural (B-factor, SASA) and evolutionary (BLOSUM62) signals provide complementary boosts  
- 3D CNN introduces localized geometric awareness  
- Rank-based ensemble achieves the final lift to **silver medal** performance  

---

## 🚀 Future Work

- Integrate **secondary structure** and **torsion angle** features  
- Explore **meta-learning / stacking** for ensembling  
- Automated hyperparameter optimization
- Experiment with **GNNs** and **transformers** on 3D structures  

---
