from APP.functions import get_season_results

df = get_season_results(2020)
print(df.head())
print(df.raceName.unique()[:5])