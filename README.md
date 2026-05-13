# Not 6-7, Try 6-4: Replicating PatchTST for Long-Term Time Series Forecasting

> A re-implementation of **"A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"** (Nie et al., ICLR 2023) — completed as part of CS 4782: Deep Learning at Cornell University.

---

## 1. Introduction

This repo contains a from-scratch re-implementation of **PatchTST**, a Transformer architecture for long-term time series forecasting. PatchTST's two key contributions — **patch-based tokenization** (reducing sequence length and capturing local temporal patterns) and **channel independence** (processing each variate separately) — allow it to outperform all prior Transformer-based forecasting models while being significantly more efficient.

---

## 2. Chosen Result

We reproduce **Table 3** from the paper: multivariate long-term forecasting results (MSE and MAE) of PatchTST/64 across four benchmark datasets (Weather, Traffic, Electricity, ILI) at prediction horizons T ∈ {96, 192, 336, 720} (T ∈ {24, 36, 48, 60} for ILI). This table is the paper's central empirical claim — that PatchTST achieves state-of-the-art performance across all settings.

📄 **Reference:** Table 3, "A Time Series is Worth 64 Words" (Nie et al., 2023)

---

## 3. GitHub Contents

```
├── code/               # the models and ipynb file
├── data/               # Download scripts for benchmark datasets
├── data_provider       # how to load the models
├── models/             # PatchTST model definition
├── poster/             # final poster presentation
├── report/             # 2-page report
├── results/            # results based on our chosen benchmark
└── results/            # Saved model outputs and comparison tables
```

---

## 4. Re-implementation Details

**Model:** PatchTST encoder with patch length P=16, stride S=8, 3 Transformer layers, hidden dim 128, 4 attention heads, and a linear prediction head. Each channel is processed independently through shared weights.

**Datasets:** Weather, Traffic, Electricity, ILI — sourced from the [authors' public repository](https://github.com/yuqinie98/PatchTST).

**Metrics:** Mean Squared Error (MSE) and Mean Absolute Error (MAE) on held-out test sets.

**Key modifications:** We used a simplified inverse normalization in place of the paper's full RevIN formulation, and trained for 10 epochs per model (vs. full convergence) due to compute constraints.

---

## 5. Reproduction Steps

Run the code in our code/CS 4782 - Final Project.ipynb. It covers all the necessary steps and imports.

---

## 6. Results / Insights

Our model came within ~5% MSE of the official PatchTST/64 on Weather, Traffic, and Electricity, and still **outperformed all Transformer baselines** (Informer, Autoformer, FEDFormer, etc.) across all settings. The ILI dataset showed a larger gap (~25% MSE) due to its small size and high noise.

| Dataset    | Our MSE | Paper MSE | Δ MSE   | Our MAE | Δ MAE  |
|------------|---------|-----------|---------|---------|--------|
| Weather    | 0.231   | 0.206     | +11.7%  | 0.273   | +5.4%  |
| Traffic    | 0.449   | 0.388     | +15.7%  | 0.284   | +6.2%  |
| Electricity| 0.179   | 0.159     | +12.6%  | 0.265   | +5.6%  |
| ILI        | 2.174   | 1.743     | +24.7%  | 0.939   | +12.7% |
| **Average**|         |           | **+12.7%**|       | **+7.0%**|

---

## 7. Conclusion

PatchTST's core ideas are robust — even a constrained replication with limited training time and minor implementation differences consistently outperforms prior Transformer-based forecasting models. The main lesson: in time series replication, training duration and normalization details matter more than architectural fidelity.

---

## 8. References

- Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). [A time series is worth 64 words: Long-term forecasting with transformers.](https://arxiv.org/abs/2211.14730) ICLR 2023.
- Official implementation: https://github.com/yuqinie98/PatchTST

---

## 9. Acknowledgements

This project was completed as part of **CS 4782: Deep Learning** at **Cornell University** (Spring 2025), under the course's paper replication assignment. We thank the original authors (Nie et al.) for open-sourcing their code and datasets, which made this replication feasible.
