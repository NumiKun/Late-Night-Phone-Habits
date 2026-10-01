# Late-Night Phone Habits and Sleep Debt Analysis

Predictive Modeling of Next-Day Fatigue Scores Using Support Vector Regression (SVR) and Bayesian Hyperparameter Optimization.

---

## Executive Summary

The proliferation of mobile devices has introduced pervasive bedtime screen habits, disrupting circadian biology, delaying sleep onset, and exacerbating cumulative sleep debt. This project presents an end-to-end Machine Learning pipeline that quantifies the relationship between pre-sleep digital behavior, physiological sleep architecture, and daytime impairment.

Using an empirical dataset of 8,500 individuals, we build and validate a **Support Vector Regression (SVR)** model with a Radial Basis Function (RBF) kernel to predict the **Next-Day Fatigue Score** (continuous scale from 1.0 to 10.0). Through Bayesian hyperparameter optimization (Optuna), robust cross-validation, and permutation feature importance, the final pipeline achieves an **R-squared of 0.9541 (+/- 0.0034)** across 10-fold cross-validation and an **RMSE of 0.5863** on an independent hold-out test set.

---

## Research Questions and Objectives

1. **Predictive Capability**: Can next-day subjective fatigue be accurately predicted from pre-bed screen habits, evening caffeine intake, and subsequent sleep metrics?
2. **Path of Impact**: Does late-night screentime act directly on fatigue, or is its effect mediated through prolonged sleep latency and sleep duration compression?
3. **Behavioral Attribution**: Which factors (e.g., app type, screen brightness, blue light filtering, or screen duration) exert the greatest influence on next-day alertness?

---

## Dataset Overview

The dataset (`bedtime_screentime_sleep_debt.csv`) comprises 8,500 individual records across 18 behavioral, demographic, and physiological features.

### Feature Dictionary

| Category | Feature Name | Data Type | Description |
|---|---|---|---|
| Demographic | `user_id` | String | Unique participant identifier (dropped during preprocessing) |
| Demographic | `age` | Integer | Participant age (18 - 65 years) |
| Demographic | `gender` | Categorical | Male, Female, Non-Binary |
| Demographic | `occupation_type` | Categorical | Corporate 9-to-5, Student, Remote Tech, Healthcare / Shift Worker, Freelance / Creative |
| Circadian Profile | `chronotype` | Ordinal | Circadian preference: Morning Lark, Intermediate, Night Owl |
| Screen Habits | `bedtime_phone_minutes` | Integer | Duration of smartphone use immediately prior to sleep (minutes) |
| Screen Habits | `primary_bedtime_app` | Categorical | TikTok / Reels, Instagram / Reddit, YouTube, Streaming, Messaging / Chat, News / Reading |
| Screen Habits | `screen_brightness_pct` | Integer | Display brightness setting (0 - 100%) |
| Screen Habits | `blue_light_filter_active`| Binary | Whether night shift / blue light filter was enabled (0 = No, 1 = Yes) |
| Lifestyle | `caffeine_post_5pm_mg` | Integer | Total caffeine consumed after 5:00 PM (mg) |
| Lifestyle | `physical_activity_min` | Integer | Moderate-to-vigorous physical activity during the day (minutes) |
| Sleep Metrics | `sleep_latency_min` | Float | Time required to transition from full wakefulness to sleep (minutes) |
| Sleep Metrics | `total_sleep_hours` | Float | Net nocturnal sleep duration (hours) |
| Sleep Architecture | `deep_sleep_pct` | Float | Percentage of total sleep spent in slow-wave deep sleep (N3) |
| Sleep Architecture | `rem_sleep_pct` | Float | Percentage of total sleep spent in Rapid Eye Movement (REM) sleep |
| Awakening Metric | `morning_alarm_snoozes` | Integer | Number of consecutive alarm snooze cycles used upon waking |
| Target Variable | `next_day_fatigue_score` | Float | Validated subjective daytime fatigue score (1.0 = Fully alert, 10.0 = Severe exhaustion) |
| Contextual Label | `sleep_debt_category` | Categorical | Stratified debt status (Optimal Recovery, Mild Deficit, Moderate Debt, Severe Sleep Debt) |

---

## Exploratory Data Analysis and Key Findings

### 1. The Sleep Debt Mediation Mechanism
Exploratory correlation analysis demonstrates that pre-sleep smartphone usage does not affect next-day fatigue in isolation. Instead, it triggers a cascade of physiological disturbances:
- **`bedtime_phone_minutes` exhibits a strong positive correlation with `sleep_latency_min` (r = +0.71)**. Increased screen engagement stimulates cognitive arousal and delays melatonin release, requiring users to lie awake significantly longer before entering stage 1 sleep.
- **`sleep_latency_min` negatively correlates with `total_sleep_hours` (r = -0.53)**. Because morning awakening times are constrained by fixed work/study schedules, delayed sleep onset directly truncates total sleep duration.
- **`total_sleep_hours` exhibits the strongest negative correlation with `next_day_fatigue_score` (r = -0.88)**, while `morning_alarm_snoozes` correlates positively at **r = +0.88**.

### 2. Digital Habits: Duration vs Content
- **Duration is decisive**: Each additional 30 minutes of bedtime screen usage is associated with a 12 to 18 minute increase in sleep latency and a noticeable drop in slow-wave deep sleep percentage (`deep_sleep_pct`).
- **Interactive video content exacerbates latency**: Users whose primary app was `TikTok / Reels` or `Instagram / Reddit` experienced higher median latency (48.3 minutes) compared to users engaging in `News / Reading` (27.6 minutes) or `Messaging / Chat` (31.2 minutes).
- **Blue light filtering provides marginal relief**: Enabling the blue light filter is associated with a modest reduction in sleep latency (~4.2 minutes on average), but it fails to compensate for high screentime (>60 minutes).

---

## Machine Learning Pipeline Architecture

The end-to-end modeling pipeline is structured into reproducible, modular components:

```
[Raw CSV Dataset]
       |
       v
[Data Cleaning & Deduplication]
       |-- Duplicate removal
       |-- Drop identifier (`user_id`)
       |-- 3x IQR extreme outlier filtering (8,411 valid records retained)
       v
[Feature & Target Partitioning]
       |-- Target: next_day_fatigue_score
       |-- Exclusion: sleep_debt_category (prevents data leakage)
       |-- 80/20 Stratified Train-Test Split (Train: 6,728 | Test: 1,683)
       v
[Scikit-Learn ColumnTransformer]
       |-- Numerical features (11): StandardScaler()
       |-- Ordinal feature (`chronotype`): OrdinalEncoder(explicit hierarchy)
       |-- Nominal features (3): OrdinalEncoder(unknown_value handling)
       v
[Optuna Bayesian Hyperparameter Optimization]
       |-- Algorithm: Tree-structured Parzen Estimator (TPE)
       |-- Objective: Minimize 5-Fold Cross-Validation RMSE
       |-- Search space: C [0.1, 200.0], epsilon [0.01, 2.0], gamma [1e-4, 1.0]
       v
[Final Support Vector Regression Model Fitting]
       |-- Algorithm: SVR(kernel='rbf')
       |-- Complete pipeline fit on full training partition
       v
[Comprehensive Evaluation & Model Serialization]
       |-- Hold-out test set metrics (RMSE, MAE, R-squared)
       |-- 10-fold cross-validation on full dataset
       |-- Permutation feature importance (20 repeats)
       |-- Serialization: svr_fatigue_model.pkl & model_metadata.json
```

---

## Why Support Vector Regression (SVR)?

Support Vector Regression with a Radial Basis Function (RBF) kernel was selected over linear models and decision tree ensembles based on three methodological justifications:

1. **Non-Linear Manifold Modeling**: The interaction between sleep latency, screentime, and fatigue is fundamentally non-linear. The RBF kernel maps features into an infinite-dimensional Hilbert space, capturing complex interactions without manual polynomial expansion.
2. **Epsilon-Insensitive Loss Function**: SVR uses Vapnik's epsilon-insensitive tube. Errors smaller than epsilon are assigned zero loss, providing intrinsic regularization against micro-variations and subjective reporting noise in survey-based fatigue scores.
3. **Structural Risk Minimization**: Rather than minimizing empirical squared error alone, SVR minimizes a combination of training error and model complexity (norm of the weight vector in RKHS), ensuring superior generalization on moderate-sized datasets (8,000+ samples).

---

## Hyperparameter Optimization Results

Hyperparameter optimization was executed using Optuna across 50 trials with 5-fold cross-validation.

| Hyperparameter | Search Range | Distribution | Optimal Value Found | Description |
|---|---|---|---|---|
| `C` | 0.1 - 200.0 | Log-uniform | **4.989305** | Regularization penalty; balances margin size against slack violations |
| `epsilon` | 0.01 - 2.0 | Log-uniform | **0.041146** | Half-width of the loss-free error tolerance zone |
| `gamma` | 0.0001 - 1.0 | Log-uniform | **0.017112** | Kernel bandwidth; defines the sphere of influence for support vectors |

---

## Experimental Evaluation and Results

The optimized pipeline was evaluated on both the independent hold-out test set (20% of data, n = 1,683) and via 10-fold cross-validation across the entire dataset.

### Quantitative Performance Metrics

| Evaluation Metric | Hold-Out Test Set (n = 1,683) | 10-Fold Cross-Validation (Full Data) |
|---|---|---|
| **Root Mean Squared Error (RMSE)** | **0.5863** | **0.5712 (+/- 0.0116)** |
| **Mean Absolute Error (MAE)** | **0.4366** | **0.4279 (+/- 0.0097)** |
| **Coefficient of Determination (R-squared)** | **0.9505** | **0.9541 (+/- 0.0034)** |

### Diagnostic Observations
- **High Variance Explained**: The model captures over 95.4% of the total variance in next-day fatigue scores.
- **Narrow Generalization Gap**: The cross-validation RMSE (0.5712) and the hold-out test RMSE (0.5863) differ by less than 0.015 points, indicating negligible overfitting.
- **Well-Behaved Residuals**: Residual diagnostic plots confirm homoscedasticity across the prediction range (1.0 to 10.0), with residuals centered symmetrically around zero and exhibiting a Gaussian distribution.

---

## Feature Attribution (Permutation Importance)

Because the RBF kernel operates in dual representation without direct linear coefficients, **Permutation Feature Importance** (20 iterations, scoring on negative RMSE) was conducted on the hold-out test set.

| Rank | Feature | Mean RMSE Increase | Standard Deviation | Interpretation |
|---|---|---|---|---|
| 1 | `total_sleep_hours` | **1.5086** | 0.0275 | Primary physical driver; sleep duration loss has the largest impact |
| 2 | `morning_alarm_snoozes` | **0.4802** | 0.0129 | Strong behavioral signal of unrefreshed sleep and sleep inertia |
| 3 | `sleep_latency_min` | **0.3894** | 0.0117 | Direct consequence of screen arousal; prolongs wakefulness |
| 4 | `bedtime_phone_minutes` | **0.0826** | 0.0070 | Primary upstream behavioral habit initiating the deficit cycle |
| 5 | `caffeine_post_5pm_mg` | **0.0699** | 0.0053 | Adenosine receptor antagonist; exacerbates sleep onset difficulty |
| 6 | `deep_sleep_pct` | **0.0566** | 0.0052 | Critical restorative sleep stage; lower percentages increase fatigue |
| 7 | `screen_brightness_pct` | **0.0261** | 0.0044 | Secondary optical stimulant contributing to circadian phase delay |
| 8 | `blue_light_filter_active` | **0.0110** | 0.0033 | Minor protective factor |
| 9 | `rem_sleep_pct` | **0.0035** | 0.0015 | Minor direct influence on next-day physical fatigue |
| 10 | `occupation_type` | **0.0020** | 0.0015 | Shift workers show higher baseline fatigue vulnerabilities |

---

## Actionable Recommendations

Based on empirical model interpretations:
1. **The 30-Minute Threshold**: Limiting pre-bedtime phone exposure to under 30 minutes prevents the steepest rise in sleep latency. Beyond 60 minutes, fatigue scores escalate rapidly.
2. **Prioritize Total Duration Over Filters**: While blue light filters and lower brightness provide marginal benefits, they do not counteract the sleep latency caused by cognitively engaging media (short-form video feeds).
3. **Caffeine Cutoff**: Post-5:00 PM caffeine consumption above 50 mg consistently shifts sleep latency upwards, amplifying next-day morning snooze frequency.

---

## Repository Structure

```
Late-Night-Phone-Habits/
├── Dataset/
│   └── bedtime_screentime_sleep_debt.csv    # Raw dataset (8,500 rows, 18 columns)
├── Model/
│   ├── svr_fatigue_prediction.ipynb         # Fully executed Jupyter notebook with visualizations
│   └── model_artifacts/
│       ├── svr_fatigue_model.pkl            # Serialized Scikit-Learn Pipeline (ready for production)
│       ├── model_metadata.json              # Complete training run configuration, params, and metrics
│       ├── optuna_trials.csv                # Tabular records of all 50 Bayesian optimization trials
│       ├── permutation_importance.csv       # Feature importance rankings with standard deviations
│       ├── target_distribution.png          # Target histogram and boxplot visualization
│       ├── correlation_heatmap.png          # Correlation matrix of numerical variables
│       ├── scatter_feature_vs_target.png    # Scatter and regression trendlines for key features
│       ├── categorical_vs_target.png        # Categorical variable distribution boxplots
│       ├── optuna_optimization_history.png  # Convergence trajectory across 50 optimization trials
│       ├── evaluation_diagnostics.png       # Residual, Q-Q, actual vs predicted, and CV diagnostics
│       └── permutation_importance.png       # Horizontal bar chart of permutation feature impact
├── LICENSE                                  # MIT License
├── README.md                                # Comprehensive project documentation
└── requirements.txt                         # Pinned dependency environment specification
```

---

## Installation and Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/NumiKun/Late-Night-Phone-Habits.git
cd Late-Night-Phone-Habits
```

### 2. Create and Activate a Virtual Environment
```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Pipeline in Jupyter
```bash
jupyter notebook Model/svr_fatigue_prediction.ipynb
```

---

## Model Inference Example

To run inference on new, unobserved participant data using the persisted model pipeline:

```python
import joblib
import pandas as pd

# Load the trained pipeline (preprocessor + tuned SVR)
model_path = "Model/model_artifacts/svr_fatigue_model.pkl"
pipeline = joblib.load(model_path)

# Define input features conforming to the schema
new_observation = pd.DataFrame([{
    "age": 28,
    "gender": "Female",
    "occupation_type": "Corporate 9-to-5",
    "chronotype": "Night Owl",
    "bedtime_phone_minutes": 120,
    "primary_bedtime_app": "TikTok / Reels",
    "screen_brightness_pct": 80,
    "blue_light_filter_active": 0,
    "caffeine_post_5pm_mg": 85,
    "physical_activity_min": 15,
    "sleep_latency_min": 55.0,
    "total_sleep_hours": 4.5,
    "deep_sleep_pct": 18.0,
    "rem_sleep_pct": 16.0,
    "morning_alarm_snoozes": 5
}])

# Generate predicted fatigue score
predicted_score = pipeline.predict(new_observation)[0]
print(f"Predicted Next-Day Fatigue Score: {predicted_score:.2f} / 10.0")
# Output: Predicted Next-Day Fatigue Score: 8.14 / 10.0
```

---

## Technical Specifications and Environment

- **Python Version**: 3.11+
- **Core ML Framework**: Scikit-Learn 1.9.0
- **Hyperparameter Optimization**: Optuna 4.9.0
- **Data Manipulation**: Pandas 3.0.5, NumPy 2.5.1
- **Visualization Suite**: Matplotlib 3.11.1, Seaborn 0.13.2
- **Model Storage**: Joblib 1.5.3

---

## License

This project is licensed under the terms of the [MIT License](LICENSE).
