# Novozymes Enzyme Stability Prediction

![Novozymes](https://img.shields.io/badge/Competition-Novozymes-blue?logo=kaggle)
![Kaggle](https://img.shields.io/badge/Platform-Kaggle-20beff?logo=kaggle)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?logo=xgboost)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c?logo=pytorch)

## Table of Contents
- [About](#about)  
- [Summary of Results](#summary-of-results)  
- [Codeflow](#codeflow)  
- [Workspace & Environment](#workspace--environment)  
- [Dataset](#dataset)  
- [Reproducing the Pipeline](#reproducing-the-pipeline)  
- [Feature Representations](#feature-representations)  
- [Models](#models)  
- [Ensembling Strategy](#ensembling-strategy)  
- [Validation & Metrics](#validation--metrics)  
- [Error Analysis](#error-analysis)  
- [Limitations](#limitations)  
- [Future Work](#future-work)  
- [References](#references)  

---

## About

This project tackles the **Novozymes Enzyme Stability Prediction** Kaggle competition, which aims to predict the thermal stability of enzyme variants. Enhanced enzyme stability can significantly improve industrial processes by enabling enzymes to function under harsh conditions. The challenge involves predicting melting temperatures (Tm) from protein sequence data and structural information.

## Summary of Results

The final solution employs a multi-faceted approach combining:
- **XGBoost regression** with engineered sequence features
- **Structural features** including B-factor and SASA (Solvent Accessible Surface Area)
- **Ensemble methods** blending multiple prediction strategies
- **ESM-2 protein language model** for contact map analysis

Best submission achieved competitive performance on the private leaderboard through careful feature engineering and model ensembling.

## Codeflow

```mermaid
graph TD
    A[Data Acquisition] --> B[Preprocessing];
    B --> C[Feature Engineering];
    C --> D[Model Training];
    D --> E[Prediction];
    E --> F[Ensembling];
    F --> G[Submission];
    
    B --> B1[Sequence Alignment];
    B --> B2[PDB Processing];
    B --> B3[Levenshtein Grouping];
    
    C --> C1[Sequence Features];
    C --> C2[Structural Features];
    C --> C3[ESM Embeddings];
    
    D --> D1[XGBoost];
    D --> D2[Neural Networks];
    D --> D3[ThermoNet];
