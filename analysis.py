import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pointbiserialr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================
# 1) DATA LOADING
# =========================
df = pd.read_csv("final_cleaned_dataset.csv")

print("\n=========================")
print("DATA LOADED")
print("=========================")
print(df.head())
print("\nShape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# =========================
# 2) DATA OVERVIEW
# =========================
print("\n=========================")
print("DATA OVERVIEW")
print("=========================")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDescriptive statistics:")
print(df.describe())


# =========================
# 3) FEATURE CHECK
# =========================
wealth_cols = [
    "real_wealth",
    "affordability_ratio",
    "rent_to_income_pct"
]

print("\n=========================")
print("WEALTH VARIABLES")
print("=========================")
print(df[wealth_cols].describe())


# =========================
# 4) MATCH RESULT DISTRIBUTION
# =========================
print("\n=========================")
print("MATCH RESULT DISTRIBUTION")
print("=========================")
print(df["fulltimeresult"].value_counts())

print("\nHome win distribution:")
print(df["home_win"].value_counts())

overall_home_win = df["home_win"].mean()
print("\nOverall home win rate:", overall_home_win)


# =========================
# 5) CITY-LEVEL ANALYSIS
# =========================
print("\n=========================")
print("CITY-LEVEL HOME WIN RATE")
print("=========================")
city_home_win = df.groupby("city")["home_win"].mean().sort_values(ascending=False)
print(city_home_win)

print("\n=========================")
print("CITY-LEVEL HOME GOALS")
print("=========================")
city_home_goals = df.groupby("city")["fulltimehomegoals"].mean().sort_values(ascending=False)
print(city_home_goals)


# =========================
# 6) CORRELATION MATRIX
# =========================
print("\n=========================")
print("CORRELATION MATRIX")
print("=========================")

corr_matrix = df[
    [
        "real_wealth",
        "affordability_ratio",
        "rent_to_income_pct",
        "fulltimehomegoals",
        "fulltimeawaygoals",
        "home_win"
    ]
].corr()

print(corr_matrix)


# =========================
# 7) POINT-BISERIAL CORRELATION
# =========================
print("\n=========================")
print("POINT-BISERIAL CORRELATION")
print("Socioeconomic variables vs binary home_win")
print("=========================")

for col in wealth_cols:
    corr, p_value = pointbiserialr(df["home_win"], df[col])

    print(f"\nVariable: {col}")
    print("Point-biserial correlation:", corr)
    print("P-value:", p_value)

    if p_value < 0.05:
        print("Result: Statistically significant relationship.")
    else:
        print("Result: Not statistically significant.")


# =========================
# 8) CITY SUMMARY TABLE
# =========================
print("\n=========================")
print("CITY SUMMARY")
print("=========================")

city_summary = df.groupby("city").agg(
    matches=("home_win", "count"),
    avg_home_win=("home_win", "mean"),
    avg_home_goals=("fulltimehomegoals", "mean"),
    avg_real_wealth=("real_wealth", "mean"),
    avg_affordability=("affordability_ratio", "mean"),
    avg_rent_to_income=("rent_to_income_pct", "mean")
).sort_values(by="avg_home_win", ascending=False)

print(city_summary)


# =========================
# 9) EDA VISUALIZATIONS
# =========================

plt.figure(figsize=(8, 5))
df["home_win"].value_counts().sort_index().plot(kind="bar")
plt.xlabel("Home Win")
plt.ylabel("Number of Matches")
plt.title("Home Win Distribution")
plt.tight_layout()
plt.savefig("plot_home_win_distribution.png")
plt.show()


plt.figure(figsize=(10, 5))
df.groupby("season")["home_win"].mean().plot(marker="o")
plt.xlabel("Season")
plt.ylabel("Home Win Rate")
plt.title("Home Win Rate by Season")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plot_home_win_rate_by_season.png")
plt.show()


plt.figure(figsize=(8, 5))
plt.scatter(df["real_wealth"], df["home_win"])
plt.xlabel("Real Wealth")
plt.ylabel("Home Win")
plt.title("Real Wealth vs Home Win")
plt.tight_layout()
plt.savefig("plot_real_wealth_vs_home_win.png")
plt.show()


plt.figure(figsize=(8, 5))
plt.hist(df["real_wealth"], bins=20)
plt.xlabel("Real Wealth")
plt.ylabel("Frequency")
plt.title("Distribution of Real Wealth")
plt.tight_layout()
plt.savefig("plot_real_wealth_distribution.png")
plt.show()


plt.figure(figsize=(8, 5))
df.boxplot(column="real_wealth", by="home_win")
plt.xlabel("Home Win")
plt.ylabel("Real Wealth")
plt.title("Real Wealth by Home Win")
plt.suptitle("")
plt.tight_layout()
plt.savefig("plot_real_wealth_by_home_win.png")
plt.show()


plt.figure(figsize=(8, 5))
df.boxplot(column="affordability_ratio", by="home_win")
plt.xlabel("Home Win")
plt.ylabel("Affordability Ratio")
plt.title("Affordability Ratio by Home Win")
plt.suptitle("")
plt.tight_layout()
plt.savefig("plot_affordability_by_home_win.png")
plt.show()


plt.figure(figsize=(8, 5))
df.boxplot(column="rent_to_income_pct", by="home_win")
plt.xlabel("Home Win")
plt.ylabel("Rent to Income Percentage")
plt.title("Rent to Income Percentage by Home Win")
plt.suptitle("")
plt.tight_layout()
plt.savefig("plot_rent_to_income_by_home_win.png")
plt.show()


# =========================
# 10) MACHINE LEARNING MODELS
# =========================
print("\n=========================")
print("MACHINE LEARNING MODELS")
print("=========================")

features = [
    "real_wealth",
    "affordability_ratio",
    "rent_to_income_pct"
]

model_data = df[features + ["home_win"]].dropna()

X = model_data[features]
y = model_data["home_win"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

print("\nLOGISTIC REGRESSION RESULTS")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))
print("Classification Report:")
print(classification_report(y_test, y_pred_log, zero_division=0))

log_coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficient": log_model.coef_[0]
})

print("\nLogistic Regression Coefficients:")
print(log_coefficients)


# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\nRANDOM FOREST RESULTS")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
print("Classification Report:")
print(classification_report(y_test, y_pred_rf, zero_division=0))

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nRandom Forest Feature Importance:")
print(feature_importance)


plt.figure(figsize=(8, 5))
plt.bar(feature_importance["Feature"], feature_importance["Importance"])
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("plot_random_forest_feature_importance.png")
plt.show()


# =========================
# 11) OUTPUT FILES
# =========================
city_summary.to_csv("city_summary.csv")

print("\n=========================")
print("FILES CREATED")
print("=========================")
print("1) plot_home_win_distribution.png")
print("2) plot_home_win_rate_by_season.png")
print("3) plot_real_wealth_vs_home_win.png")
print("4) plot_real_wealth_distribution.png")
print("5) plot_real_wealth_by_home_win.png")
print("6) plot_affordability_by_home_win.png")
print("7) plot_rent_to_income_by_home_win.png")
print("8) plot_random_forest_feature_importance.png")
print("9) city_summary.csv")


# =========================
# 12) INTERPRETATION
# =========================
print("\n=========================")
print("INTERPRETATION")
print("=========================")

print(
    "Point-biserial correlation is used because home_win is a binary variable. "
    "This is more appropriate than Pearson correlation for testing the relationship "
    "between a binary outcome and continuous socioeconomic variables."
)

print(
    "The machine learning models use socioeconomic indicators to predict whether "
    "the home team wins. Logistic Regression provides a simple baseline model, "
    "while Random Forest can capture more complex non-linear patterns."
)

print(
    "If the correlations are weak and model accuracy is limited, this suggests that "
    "socioeconomic variables alone are not strong predictors of football match outcomes. "
    "Football results are also affected by team quality, injuries, tactics, player form, "
    "and other sporting factors."
)
