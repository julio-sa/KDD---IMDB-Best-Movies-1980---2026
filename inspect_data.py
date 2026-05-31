import pandas as pd

df = pd.read_csv("data/raw/imdb_top_movies_1980_2026.csv")
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst rows:")
print(df.head(3))
print("\nNulls per column:")
print(df.isnull().sum())
print("\nDtypes:")
print(df.dtypes)

print("Sample genre values:")
print(df["genres"].head(10).to_list())
print("\nUnique year range:", df["year"].min(), "to", df["year"].max())
print("Rating range:", df["average_rating"].min(), "to", df["average_rating"].max())
print("Votes range:", df["num_votes"].min(), "to", df["num_votes"].max())