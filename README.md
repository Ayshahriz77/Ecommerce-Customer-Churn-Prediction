# 📉 E-Commerce Customer Churn Prediction

A machine learning project predicting customer churn for an e-commerce platform, comparing four classification models and translating the results into concrete, evidence-based retention strategies.

**Course:** BA 360 — Business Data Mining
**Authors:** Aicha Hriz, Malek Omri, Sirine Othmene
**Supervisor:** Pr. El Moubarki Lassad
**Institution:** Tunis Business School, University of Tunis
**Date:** April 2026

---

## 📌 Project Overview

Customer churn is costly for e-commerce businesses, losing a customer means lost revenue and higher acquisition costs to replace them. This project builds a full machine learning pipeline to **predict which customers are likely to churn** and **identify the key behavioral drivers behind it**, enabling targeted, cost-effective retention strategies before customers are lost.

## 📊 Dataset

- **Source:** [E-Commerce Customer Churn Dataset on Kaggle](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction) (CC0 Public Domain)
- **Size:** 5,630 customers, 20 features + target variable
- **Target:** `Churn` (0 = Retained, 1 = Churned)
- **Class balance:** ~16.8% churned vs. 83.2% retained (≈5:1 imbalance)
- **Features cover:** engagement/behavior (tenure, orders, app usage, cashback), demographics (gender, marital status, city tier), and platform preferences (login device, payment mode, order category)

## 🧠 Methodology

The pipeline follows a structured, end-to-end ML workflow:

1. **Data Loading & Summary** — shape, types, nulls, class imbalance check
2. **Exploratory Data Analysis** — 8-panel dashboard + full correlation heatmap
3. **Preprocessing & Feature Engineering**
   - Missingness-indicator flags created *before* imputation (missing ≠ zero — e.g. a missing "days since last order" likely means no order was ever placed)
   - Median imputation on 7 numeric columns
   - Label encoding of categorical features
4. **Train/Test Split & Scaling** — stratified 80/20 split; `RobustScaler` (fit on training data only, used only for Logistic Regression) to handle outliers
5. **Baseline Model Training** — Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
6. **Model Evaluation** — accuracy, ROC-AUC, F1, precision, recall, 5-fold cross-validation, ROC/PR curves, confusion matrices
7. **Hyperparameter Tuning** — exhaustive `GridSearchCV` (5-fold CV, optimized for ROC-AUC) across all four models
8. **Feature Importance** — consensus ranking averaged across tuned Random Forest and Gradient Boosting
9. **Business Insights** — translating statistical signal into retention actions

## 🏆 Results

**Post-tuning model comparison:**

| Model | Accuracy | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression | ~90% | ~0.88 | ~0.73 | ~0.79 | ~0.68 |
| Decision Tree | ~93% | ~0.94 | ~0.86 | ~0.87 | ~0.85 |
| Random Forest | ~97% | ~0.98 | ~0.93 | ~0.96 | ~0.91 |
| **Gradient Boosting** | **~97%** | **~0.98** | **~0.93** | **~0.96** | **~0.91** |

**Winning model:** Gradient Boosting (Random Forest performs comparably) — both substantially outperform the linear baseline, confirming churn behavior is driven by non-linear feature interactions.

**Top predictive features (consensus ranking):**
1. **Tenure** — how long the customer has been on the platform
2. **CashbackAmount** — average cashback received
3. **Complain** — whether the customer filed a complaint
4. NumberOfAddress, WarehouseToHome, DaySinceLastOrder, SatisfactionScore...

## 🔍 Key Insights

- **Early lifecycle is the danger zone** — customers with 0–3 months tenure churn at nearly double the average rate. This is the most critical retention window.
- **Complaints are a near-perfect churn signal** — complainers churn at 31.7% vs. 10.9% for non-complainers.
- **Cashback drives loyalty** — churned customers cluster at lower cashback levels.
- **Platform embedding matters** — customers with more saved addresses/devices churn less.
- **Mobile users churn more** — likely due to UX friction on phone vs. desktop.
- **Tier 3 (smaller) cities churn more** — likely tied to longer delivery distances and fewer promotions.

## ✅ Retention Recommendations

| Recommendation | Target Segment | Signal |
|---|---|---|
| 90-Day Onboarding Rescue Program | Tenure < 3 months | Tenure (#1) |
| Fast-Track Complaint Resolution SLA | Any complaint filed | Complain (#2) |
| Personalized Cashback Boost Campaign | Below-median cashback recipients | CashbackAmount (#3) |
| Dormancy Re-Engagement Trigger | 7+ days since last order | DaySinceLastOrder |
| Mobile UX Improvement Initiative | Phone-login users | EDA signal |
| Tier 3 City Logistics Partnership | Tier 3 customers, high delivery distance | CityTier + WarehouseToHome |
| Platform Embedding Incentive Program | Single saved address | NumberOfAddress |

*(Full detail on each recommendation's actions is in the report.)*

## 🛠️ Tech Stack

- **Language:** Python
- **Data handling:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **ML:** scikit-learn (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, GridSearchCV, StratifiedKFold, RobustScaler, LabelEncoder)
- **Environment:** Google Colab

## ⚠️ Limitations

- Class imbalance not directly addressed (no SMOTE/ADASYN/class-weighting applied)
- Label encoding imposes implicit ordinality on nominal features
- No temporal/time-series context — single observational snapshot
- No SHAP/LIME explainability layer for per-customer explanations
- Results based on a single anonymized platform's data — generalizability untested

## 🔮 Future Work

- Apply SMOTE or class-weight adjustments to improve recall on churned customers
- Integrate SHAP for individual customer-level churn explanations
- Explore XGBoost / LightGBM
- Build a deployment-ready prediction API
- Run a controlled retention experiment to measure real causal impact

## 📄 Full Report

See [`Report.pdf`](./Report.pdf) for the complete write-up, including all code, visualizations, and detailed business analysis.

## 📚 Key References

- Burez & Van den Poel (2009) — class imbalance in churn prediction
- Ori & Hasan (2024) — SHAP-based churn prediction using this same dataset (benchmark: ROC-AUC 0.96–0.98)

---

*This project was completed as part of the Business Data Mining coursework at Tunis Business School.*
