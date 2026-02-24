# Quadeye Market Data Prediction Challenge

**Author:** Saurabh Kaushik  
**Competition:** Quadeye Market Data Prediction Challenge (Round 2 – Convolve 4.0 Hackathon)  
**Evaluation Metric:** Root Mean Squared Error (RMSE)

---

## 1. Problem Overview

This project was developed for the Quadeye Market Data Prediction Challenge, designed to simulate real-world quantitative research workflows.

Participants were given structured market-style data and asked to predict a target variable `y`.

The dataset reflects real quantitative challenges:

- High noise-to-signal ratio  
- Subtle cross-sectional structure  
- Nonlinear interactions  
- Regime shifts and non-stationarity  
- Strict chronological constraints  

Performance metric:

RMSE = sqrt( (1/N) * Σ (ŷᵢ − yᵢ)² )

where:
- ŷᵢ = predicted value  
- yᵢ = true value  
- N = number of observations  

---

## 2. Production Constraint (No Look-Ahead Bias)

For any prediction at time (dᵢ, tᵢ), only observations satisfying:

dⱼ ≤ dᵢ  AND  tⱼ ≤ tᵢ  

may be used.

This ensures strict chronological integrity and mirrors real trading conditions where future information is unavailable.

---

## 3. Modeling Philosophy

The approach prioritizes:

- Robustness under noise  
- Stationarity transformation  
- Cross-sectional consistency  
- Controlled model complexity  
- Risk-aware output calibration  

Instead of modeling raw magnitudes, the focus is on modeling relative structure across symbols.

---

## 4. Feature Engineering

### 4.1 Cross-Sectional Rank Transformation

For each feature fₖ at timestamp (d, t), the raw value fₖᵢ is converted to a percentile rank:

Rₖᵢ = rank(fₖᵢ) / N(d,t)

where:
- N(d,t) = number of symbols at that timestamp  

This removes scale dependence and preserves relative ordering across symbols.

---

### 4.2 Rank Centering

Ranks are centered:

R̃ₖᵢ = Rₖᵢ − 0.5

This produces features approximately in [-0.5, 0.5], improving symmetry and stability for tree-based models.

---

### 4.3 Volatility Regime Feature

Cross-sectional dispersion at each timestamp is computed:

σ(d,t) = sqrt( (1/N) * Σ (fᵢ − f̄)² )

where:
- f̄ = cross-sectional mean  

This feature captures market regime (calm vs volatile) and allows the model to adapt its predictions accordingly.

---

## 5. Target Stabilization

Financial targets often exhibit heavy-tailed distributions.

Extreme values are clipped:

y* =  
- y_low   if y < y_low  
- y_high  if y > y_high  
- y       otherwise  

This stabilizes gradient updates and reduces sensitivity to rare extreme events.

---

## 6. Validation Strategy

Chronological split:

- First 90% of time → Training  
- Last 10% → Validation  

Ensures:

Training time < Validation time  

K-fold cross-validation is avoided due to temporal leakage risk.

---

## 7. CatBoost Regression Model

### 7.1 Why CatBoost?

CatBoost is a gradient boosting framework based on decision trees, designed to reduce prediction shift and improve robustness.

Reasons for selection:

- Strong performance in noisy environments  
- Effective modeling of nonlinear interactions  
- Built-in regularization mechanisms  
- Ordered boosting to reduce target leakage  

---

### 7.2 Gradient Boosting Formulation

The model prediction can be written as:

ŷ(x) = Σₘ γₘ Tₘ(x)

where:
- Tₘ(x) = decision tree m  
- γₘ = weight of tree m  
- M = number of boosting iterations  

Each tree is trained sequentially to minimize residual error:

Residualₘ = y − ŷₘ₋₁

The model iteratively improves prediction by fitting trees to the residuals.

---

### 7.3 Ordered Boosting

Standard gradient boosting may introduce prediction shift due to target leakage within boosting iterations.

CatBoost addresses this by constructing trees using ordered permutations of the data, ensuring that each prediction is computed without using future target information.

This is particularly important in time-dependent financial datasets.

---

### 7.4 Regularization

Regularization mechanisms include:

- Limited tree depth  
- L2 penalty on leaf weights  
- Learning rate control  
- Early stopping based on validation RMSE  

These controls help balance bias and variance in a low signal-to-noise setting.

---

## 8. Post-Processing and Risk Control

### 8.1 Market Neutralization

Predictions are centered cross-sectionally:

ŷᵢ(neutral) = ŷᵢ − μ(d,t)

where:

μ(d,t) = (1/N) * Σ ŷᵢ

This enforces:

Σ ŷᵢ(neutral) = 0

ensuring the model captures relative alpha rather than systematic market exposure.

---

### 8.2 Prediction Shrinkage

Final predictions are scaled:

ŷᵢ(final) = α * ŷᵢ(neutral)

where 0 < α < 1

Shrinkage reduces overconfidence and improves robustness under noisy conditions.

---

### 8.3 Final Clipping

Predictions are clipped to training distribution bounds to limit tail risk and extreme RMSE penalties.

---

## 9. End-to-End Pipeline

1. Enforce chronological ordering  
2. Apply cross-sectional ranking  
3. Center features  
4. Add volatility regime signal  
5. Stabilize target  
6. Chronological train–validation split  
7. Train CatBoost regression model  
8. Generate predictions  
9. Apply cross-sectional neutralization  
10. Apply shrinkage  
11. Export submission  

---
