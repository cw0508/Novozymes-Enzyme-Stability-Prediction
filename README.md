# Novozymes Enzyme Stability Prediction – README

This repository contains a **silver medal (24th place)** solution for the **Novozymes Enzyme Stability Prediction** competition on Kaggle. The project focuses on predicting the thermal stability (Tm value) of enzyme variants using a combination of **sequence-based**, **structural**, and **evolutionary** features, along with a **multi-model ensemble strategy**.

## 🔍 Overview

The solution integrates:
- **XGBoost** as the primary regression model  
- **Structural biophysics features** such as B-factors and SASA  
- **Deep learning embeddings** via ESM-2 contact maps  
- **Evolutionary information** from BLOSUM62 and sequence conservation  
- A **rank-based ensemble** to combine model outputs robustly

## ⚙️ Main Components

### **1. Data Preprocessing**
- Mutation extraction and sequence alignment  
- Structural parsing of PDB files  
- Levenshtein-based sequence grouping  

### **2. Feature Engineering**
- **Sequence features:** composition, physicochemical groups, mutation descriptors  
- **Structural features:** B-factors, solvent accessibility (SASA), residue contacts  
- **Evolutionary features:** BLOSUM62 scores, conservation, similarity measures  

### **3. Models Used**
- **XGBoost Regressor** (primary model)  
- **ESM-2-based contact map model**  
- **B-factor model**  
- **SASA + BLOSUM hybrid model**

### **4. Ensemble Method**
Predictions are combined using a **weighted rank averaging** strategy to improve leaderboard stability and reduce model bias.

## 📈 Results
- Strong performance driven by feature-rich XGBoost model  
- Structural + evolutionary cues provide complementary predictive power  
- Rank-based ensembling boosts Spearman correlation and leaderboard robustness  

## 🚀 Future Improvements
- Incorporate secondary structure and dihedral angle features  
- Explore meta-learning ensembling (stacking)  
- Apply automated hyperparameter optimization  

---
