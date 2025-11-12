# Novozymes Enzyme Stability Prediction — REPORT

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
- [Ablations](#ablations)  
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
