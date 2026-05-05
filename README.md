# EPL City Wealth Analysis

## Overview
This project analyzes whether socioeconomic conditions of cities (such as income, cost of living, and affordability) have any relationship with football match outcomes in the English Premier League (EPL).

The main research question is:

> Do wealth-related variables influence the probability of a home team winning?

---

## Dataset
The dataset combines:

- EPL match data (2010–2025 seasons)
- UK city-level socioeconomic data

Key variables include:

- `home_win` (binary target variable: 1 = home team wins, 0 = otherwise)
- `real_wealth`
- `affordability_ratio`
- `rent_to_income_pct`
- `fulltimehomegoals`, `fulltimeawaygoals`

---

## Methodology

### 1. Data Processing
- Data cleaning and merging from multiple sources
- Feature engineering (creation of real_wealth and ratios)
- Handling missing values

### 2. Exploratory Data Analysis (EDA)
- Distribution of match outcomes
- City-level performance comparison
- Visualization of wealth variables
- Seasonal trends in home win rates

### 3. Statistical Analysis
- **Point-biserial correlation** is used to analyze the relationship between:
  - Binary variable (`home_win`)
  - Continuous socioeconomic variables

This method is chosen because Pearson correlation is not appropriate for binary dependent variables.

### 4. Machine Learning Models
The following models are applied:

- **Logistic Regression**
- **Random Forest Classifier**

Purpose:
- Predict whether the home team wins
- Evaluate predictive power of socioeconomic variables

Evaluation metrics:
- Accuracy
- Confusion Matrix
- Classification Report

---

## Key Findings

- Socioeconomic variables show **weak correlation** with match outcomes
- Point-biserial tests indicate **no strong statistical significance**
- Machine learning models achieve **limited predictive accuracy**

### Conclusion
Football match outcomes are **not strongly driven by city-level wealth**.

Other factors such as:
- Team quality
- Player performance
- Tactics
- Injuries

are likely much more important.

---

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
