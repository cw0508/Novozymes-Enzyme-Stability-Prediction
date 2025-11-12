# Novozymes Enzyme Stability Prediction

![Novozymes](https://img.shields.io/badge/Competition-Novozymes-blue?logo=kaggle)
![Kaggle](https://img.shields.io/badge/Platform-Kaggle-20beff?logo=kaggle)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?logo=xgboost)
![Python](https://img.shields.io/badge/Language-Python-3776AB?logo=python)

## Table of Contents
- [About](#about)  
- [Summary of Results](#summary-of-results)  
- [Approach Overview](#approach-overview)  
- [Feature Engineering](#feature-engineering)  
- [Models](#models)  
- [Ensemble Strategy](#ensemble-strategy)  
- [Validation & Results](#validation--results)  
- [Challenges & Limitations](#challenges--limitations)  
- [Future Work](#future-work)  

---

## About

This project addresses the **Novozymes Enzyme Stability Prediction** Kaggle competition, predicting thermal stability (melting temperature Tm) of enzyme variants from protein sequences and structural data.

## Summary of Results

The solution combined multiple approaches:
- **Primary Model**: XGBoost with comprehensive feature engineering  
- **Structural Features**: B-factor and SASA from PDB files  
- **Simple Ensembling**: Rank-based blending of different prediction strategies  
- **Baseline Methods**: ESM-2 contact maps and pure structural approaches  

Best submission achieved competitive performance through careful feature engineering and strategic model combination.

---

## Approach Overview

```mermaid
graph TD
    A[Raw Data] --> B[Feature Engineering];
    B --> C[XGBoost Model];
    B --> D[Structural Features];
    B --> E[ESM-2 Analysis];
    C --> F[Predictions];
    D --> F;
    E --> F;
    F --> G[Rank Ensemble];
    G --> H[Final Submission];

## About

This project addresses the **Novozymes Enzyme Stability Prediction** Kaggle competition, predicting thermal stability (melting temperature Tm) of enzyme variants from protein sequences and structural data.

## Summary of Results

The solution combined multiple approaches:
- **Primary Model**: XGBoost with comprehensive feature engineering  
- **Structural Features**: B-factor and SASA from PDB files  
- **Simple Ensembling**: Rank-based blending of different prediction strategies  
- **Baseline Methods**: ESM-2 contact maps and pure structural approaches  

---

## Approach Overview

```mermaid
graph TD
    A[Raw Data] --> B[Feature Engineering];
    B --> C[XGBoost Model];
    B --> D[Structural Features];
    B --> E[ESM-2 Analysis];
    C --> F[Predictions];
    D --> F;
    E --> F;
    F --> G[Rank Ensemble];
    G --> H[Final Submission];
```

---

## Feature Engineering

### Sequence-Based Features
- **Basic Statistics:** Sequence length, unique amino acid count  
- **Amino Acid Composition:** Percentage of each amino acid type  
- **Chemical Groups:** Hydrophobic, polar, charged, special amino acid proportions  
- **Mutation Identification:** Position and type of mutations relative to wildtype  

### Structural Features
- **B-factor:** Atomic displacement parameters from PDB files  
- **SASA:** Solvent Accessible Surface Area using Shrake-Rupley algorithm  
- **PDB Processing:** Automated extraction from mutant structure files  

### External Features
- **Levenshtein Distance:** Sequence similarity grouping  
- **BLOSUM62 Scores:** Substitution matrix values for mutations  

---

## Models

### 1. XGBoost Regressor (Primary)

```python
model = xgb.XGBRegressor(n_estimators=140, max_depth=4)
```

- Handled mixed feature types effectively  
- Provided robust baseline performance  
- Enabled feature importance analysis  

### 2. Neural Network (Exploratory)
- Simple fully-connected architecture  
- Limited success compared to XGBoost  
- Used primarily for comparison  

### 3. ESM-2 Contact Maps
- Protein language model for evolutionary information  
- Contact map analysis for stability prediction  
- Provided complementary signals  

### 4. Pure Structural Approaches
- Direct use of B-factor differences  
- SASA-based predictions  
- Served as baseline methods  

---

## Ensemble Strategy

Final submission used **rank-based ensembling**:

```python
# Weighted combination of ranked predictions
final_rank = (0.15 * rankdata(structural_predictions) +
              0.78 * rankdata(xgboost_predictions) +
              0.07 * rankdata(other_predictions)) / total_samples
```

This approach leveraged **relative ordering** rather than absolute values from different methods, improving leaderboard robustness.

---

## Validation & Results

### Validation Strategy
- **Train-Test Split:** 80-20 random split  
- **Metrics:** Spearman correlation, MAE  
- **Cross-Validation:** Basic k-fold validation  

### Key Findings
- XGBoost with comprehensive features provided strongest individual performance  
- Structural features (B-factor, SASA) offered valuable complementary information  
- Simple ensembling improved robustness over single models  
- Sequence-based features were more reliable than structure-only approaches  

---

## Challenges & Limitations

### Data Challenges
- **Limited Training Data:** ~2,400 labeled variants  
- **Structural Coverage:** Not all mutants had reliable PDB structures  
- **Experimental Noise:** Variability in Tm measurements  

### Technical Limitations
- **Computational Constraints:** Limited molecular dynamics simulations  
- **PDB Quality:** Variable quality in predicted structures  
- **Feature Engineering:** Manual feature design vs. learned representations  

### Model Limitations
- Limited success with neural networks  
- ESM-2 underutilized due to computational constraints  
- ThermoNet imported but not fully implemented  

---

## Future Work

### Immediate Improvements
- **Better Feature Engineering:** Dihedral angles, secondary structure  
- **Advanced Ensembling:** Neural network meta-learners  
- **Hyperparameter Optimization:** Systematic tuning of all models  

### Technical Enhancements
- **Full ESM-2 Fine-tuning:** Domain adaptation for stability prediction  
- **Graph Neural Networks:** Direct protein structure processing  
- **AlphaFold2 Integration:** Improved structural features  

### Methodological Improvements
- **Cross-Validation Strategy:** Grouped k-fold by protein families  
- **Uncertainty Quantification:** Bayesian methods for confidence intervals  
- **Transfer Learning:** Pre-training on larger protein datasets  

---
