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
This repository contains a complete, reproducible solution for the **Novozymes Enzyme Stability Prediction** task.  
The approach follows the principle that **representation quality drives performance**, combining:
1. structure-/sequence-based **thermostability surrogates**,  
2. **pretrained protein embeddings**, and  
3. **gradient boosting** with a simple **stacked ensemble**.
