# PBiLoss

This repository contains the source code and experiment setup of ***PBiLoss*** paper:

> **[PBiLoss: Popularity-Aware Regularization to Improve Fairness in Graph-Based Recommender Systems](https://arxiv.org/abs/2507.19067)**

## 📌 Overview

Recommender systems often suffer from *popularity bias*, leading to unfair exposure for less popular items and less personalization. In this work, we propose a novel loss function called *Popularity Bias Loss* (**PBiLoss**) to reduce popularity bias and improve fairness in recommender systems. We integrate PBiLoss into several graph-based recommendation models like LightGCN.

## 🛠️ Setup

### Enviroment requirements

```
pip install -r requirements.txt
```
### Run code

```
python main.py --dataset="Epinions" --PBiLoss="PopNeg" --PBiLoss_weight=0.005 --pop_threshold=30
```
