import pandas as pd


epl = pd.read_csv("epl_final.csv")
uk = pd.read_csv("uk_col_salary_longitudinal_2010_2024.csv")

epl.columns = epl.columns.str.lower().str.strip().str.replace(" ", "_")
uk.columns = uk.columns.str.lower().str.strip().str.replace(" ", "_")

print("EPL kolonları:")
print(epl.columns.tolist())
print()

print("UK kolonları:")
print(uk.columns.tolist())
print()


epl["season"] = epl["season"].astype(str).str.strip()
epl["season_start"] = epl["season"].str[:4].astype(int)


epl = epl[(epl["season_start"] >= 2010) & (epl["season_start"] <= 2024)]


epl["year"] = epl["season_start"]


epl["hometeam"] = epl["hometeam"].astype(str).str.lower().str.strip()


team_to_city = {
    "arsenal": "london",
    "chelsea": "london",
    "tottenham": "london",
    "tottenham hotspur": "london",
    "west ham": "london",
    "west ham united": "london",
    "crystal palace": "london",
    "brentford": "london",
    "fulham": "london",
    "queens park rangers": "london",
    "qpr": "london",

    "manchester united": "manchester",
    "manchester city": "manchester",

    "liverpool": "liverpool",
    "everton": "liverpool",

    "newcastle": "newcastle",
    "newcastle united": "newcastle",

    "aston villa": "birmingham",
    "birmingham city": "birmingham",

    "wolverhampton wanderers": "wolverhampton",
    "wolves": "wolverhampton",

    "leeds": "leeds",
    "leeds united": "leeds",

    "brighton": "brighton",
    "brighton & hove albion": "brighton",

    "southampton": "southampton",
    "leicester city": "leicester",
    "nottingham forest": "nottingham",
    "ipswich town": "ipswich",
    "bournemouth": "bournemouth",
    "afc bournemouth": "bournemouth",
    "burnley": "burnley",
    "stoke city": "stoke-on-trent",
    "sunderland": "sunderland",
    "blackburn rovers": "blackburn",
    "bolton wanderers": "bolton",
    "wigan athletic": "wigan",
    "portsmouth": "portsmouth",
    "hull city": "hull",
    "blackpool": "blackpool",
    "swansea city": "swansea",
    "cardiff city": "cardiff",
    "middlesbrough": "middlesbrough",
    "norwich city": "norwich",
    "watford": "watford",
    "reading": "reading",
    "west brom": "west_bromwich",
    "west bromwich albion": "west_bromwich",
    "sheffield united": "sheffield",
    "huddersfield town": "huddersfield",
    "luton town": "luton"
}

epl["city"] = epl["hometeam"].map(team_to_city)


epl = epl.dropna(subset=["city"])


epl = epl[
    [
        "season",
        "year",
        "city",
        "fulltimehomegoals",
        "fulltimeawaygoals",
        "fulltimeresult"
    ]
]


epl["home_win"] = (epl["fulltimeresult"] == "H").astype(int)

epl = epl.drop_duplicates()

print("Filtre sonrası EPL shape:", epl.shape)
print("Kalan sezonlar:")
print(sorted(epl["season"].unique())[:3], "...")
print(sorted(epl["season"].unique())[-3:])
print()

# =========================
# 4) UK DATA TEMİZLE
# =========================
salary_col = "median_salary_gross_gbp_monthly"
cost_col = "cost_of_living_index"
city_col = "city"
year_col = "year"

uk[salary_col] = pd.to_numeric(uk[salary_col], errors="coerce")
uk[cost_col] = pd.to_numeric(uk[cost_col], errors="coerce")
uk[year_col] = pd.to_numeric(uk[year_col], errors="coerce")

uk[city_col] = uk[city_col].astype(str).str.lower().str.strip()


uk = uk[
    [
        year_col,
        city_col,
        salary_col,
        cost_col,
        "affordability_ratio",
        "rent_to_income_pct",
        "region",
        "population_approx"
    ]
]


uk = uk.dropna(subset=[year_col, city_col, salary_col, cost_col])


uk = uk.drop_duplicates()


uk["real_wealth"] = uk[salary_col] / uk[cost_col]

print("Temizlenmiş UK shape:", uk.shape)
print()


df = pd.merge(epl, uk, on=["city", "year"], how="inner")


df = df.dropna()
df = df.drop_duplicates()

print("Final shape:", df.shape)
print(df.head())
print()


df.to_csv("final_cleaned_dataset.csv", index=False)

print("final_cleaned_dataset.csv kaydedildi.")