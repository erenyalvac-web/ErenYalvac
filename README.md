# ErenYalvac
DSA210 Data Science Project – Data analysis, hypothesis testing, and machine learning
# EPL City Wealth Analysis

This project analyzes the relationship between city wealth and home team performance in the English Premier League.

## Dataset

The dataset used in this project combines:
- Premier League match data
- UK city-level economic indicators (salary and cost of living)

The football data includes **performance data from the 2010/2011 season to the 2024/2025 season**.

A cleaned and feature-engineered dataset is provided as:
- `final_cleaned_dataset.csv`

## Features

A key variable created in this project is:

- `real_wealth` = median_salary_gross_gbp_monthly / cost_of_living_index

Other important variables include:
- `home_win` (1 if home team wins, 0 otherwise)
- `fulltimehomegoals`
- `fulltimeawaygoals`

## Methods

The project includes:
- Data cleaning and merging
- Feature engineering
- Exploratory Data Analysis (EDA)
- Correlation analysis
- Hypothesis testing using Pearson correlation

## Results

The analysis explores whether wealthier cities tend to have:
- Higher home win rates
- Higher goal-scoring performance

Results include:
- Scatter plots
- Distribution plots
- City-level summaries

## Files

- `analysis.py` → Main analysis code
- `final_cleaned_dataset.csv` → Cleaned dataset
- `city_summary.csv` → Aggregated city-level results
- `plot_*.png` → Visualizations

## Notes

The dataset has been filtered to include only matches between **2010 and 2025 seasons**, ensuring consistency with the available economic data.
