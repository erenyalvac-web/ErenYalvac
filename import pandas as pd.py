import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# =========================
# 1) DATA YÜKLE
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
# 2) GENEL BİLGİ
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
# 3) SONUÇ DAĞILIMI
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
# 4) ŞEHİR BAZLI ANALİZ
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
# 5) WEALTH ANALYSIS
# =========================
print("\n=========================")
print("WEALTH VARIABLES")
print("=========================")
wealth_cols = ["real_wealth", "affordability_ratio", "rent_to_income_pct"]
print(df[wealth_cols].describe())

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
# 6) HYPOTHESIS TESTS
# =========================
print("\n=========================")
print("HYPOTHESIS TEST 1")
print("real_wealth vs home_win")
print("=========================")
corr1, p1 = pearsonr(df["real_wealth"], df["home_win"])
print("Correlation:", corr1)
print("P-value:", p1)

if p1 < 0.05:
    print("Result: Statistically significant relationship.")
else:
    print("Result: Not statistically significant.")

print("\n=========================")
print("HYPOTHESIS TEST 2")
print("real_wealth vs fulltimehomegoals")
print("=========================")
corr2, p2 = pearsonr(df["real_wealth"], df["fulltimehomegoals"])
print("Correlation:", corr2)
print("P-value:", p2)

if p2 < 0.05:
    print("Result: Statistically significant relationship.")
else:
    print("Result: Not statistically significant.")


# =========================
# 7) CITY SUMMARY TABLE
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
# 8) GRAFİKLER
# =========================
plt.figure(figsize=(8, 5))
plt.scatter(df["real_wealth"], df["home_win"])
plt.xlabel("Real Wealth")
plt.ylabel("Home Win")
plt.title("Real Wealth vs Home Win")
plt.tight_layout()
plt.savefig("plot_real_wealth_vs_home_win.png")
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(df["real_wealth"], df["fulltimehomegoals"])
plt.xlabel("Real Wealth")
plt.ylabel("Full Time Home Goals")
plt.title("Real Wealth vs Full Time Home Goals")
plt.tight_layout()
plt.savefig("plot_real_wealth_vs_home_goals.png")
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(df["real_wealth"], bins=20)
plt.xlabel("Real Wealth")
plt.ylabel("Frequency")
plt.title("Distribution of Real Wealth")
plt.tight_layout()
plt.savefig("plot_real_wealth_distribution.png")
plt.show()


# =========================
# 9) OPTIONAL: CSV OUTPUT
# =========================
city_summary.to_csv("city_summary.csv")

print("\n=========================")
print("FILES CREATED")
print("=========================")
print("1) plot_real_wealth_vs_home_win.png")
print("2) plot_real_wealth_vs_home_goals.png")
print("3) plot_real_wealth_distribution.png")
print("4) city_summary.csv")
