import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# ============================================================
# 1.1 LOAD
# ============================================================
df = pd.read_csv('data/raw/imdb_top_movies_1980_2026.csv')
print(f'Initial shape: {df.shape}')

# ============================================================
# 1.2 DROP USELESS COLUMNS (unique per row or redundant)
# ============================================================
df = df.drop(columns=['imdb_id', 'original_title', 'imdb_url'])

# ============================================================
# 1.3 DROP ROWS WITH MISSING ESSENTIAL VALUES
# ============================================================
df = df.dropna(subset=['genres', 'runtime_minutes'])
print(f'After dropping nulls: {df.shape}')

# ============================================================
# 1.4 ENGINEER AUDIENCE-ENGAGEMENT FEATURES
# These directly express our research question
# ============================================================
df['log_votes'] = np.log1p(df['num_votes'])
df['rating_pct'] = df['average_rating'].rank(pct=True)
df['votes_pct'] = df['num_votes'].rank(pct=True)

# love_score: HIGH when bth rating AND popularity are high (mass favorites)
df['love_score'] = df['rating_pct'] * df['votes_pct']

# cult_factor: positive = loved than watched (cult)
#              negative = watched more than loved (mainstream popcorn)
df['cult_factor'] = df['rating_pct'] - df['votes_pct']

# decade for grouping
df['decade'] = (df['year'] // 10 * 10).astype(int)

# ============================================================
# 1.5 MULTI-HOT ENCODE GENRES
# A movie tagged "Drama, Romance" counts toward both
# ============================================================
df['genres_clean'] = df['genres'].astype(str).str.split(',').apply(
    lambda lst: [g.strip() for g in lst]
)
genre_dummies = df['genres_clean'].str.join('|').str.get_dummies()
genre_dummies.columns =[f'genre_{g}' for g in genre_dummies.columns]
df = pd.concat([df.drop(columns=['genres_clean']), genre_dummies], axis=1)
print(f'Created {len(genre_dummies.columns)} genre columns: '
      f'{list(genre_dummies.columns)}')

# ============================================================
# 1.6 HISTOGRAMS
# ============================================================
numeric_cols = ['average_rating', 'num_votes', 'log_votes',
                'love_score', 'cult_factor', 'year', 'runtime_minutes']
df[numeric_cols].hist(bins=30, figsize=(15, 10), color='steelblue', edgecolor='white')
plt.tight_layout()
plt.savefig('plots/histograms.png', dpi=120, bbox_inches='tight')
plt.close()

# Boxplot BEFORE normalization (shows scale problem)
plt.figure(figsize=(14, 6))
df[numeric_cols].boxplot()
plt.xticks(rotation=45, ha='right')
plt.title('Boxplot - raw scale (before normalization)')
plt.tight_layout()
plt.savefig('plots/boxplot_raw.png', dpi=120, bbox_inches='tight')
plt.close()

# ============================================================
# 1.7 DISCRETIZE for association rules and reporting
# ============================================================
df['rating_cat'] = pd.qcut(df['average_rating'], 4,
                           labels = ['Low', 'Medium', 'High', 'Top'])
df['votes_cat'] = pd.qcut(df['num_votes'], 4,
                          labels = ['Few', 'Some', 'Many', 'Massive'])
df['runtime_cat'] = pd.qcut(df['runtime_minutes'], 4,
                            labels = ['Short', 'Standard', 'Long', 'Epic'])

# ============================================================
# 1.8 BUILD ENCODED + NORMALIZED VERSIONS
# ============================================================
df_clean = df.copy() # readable version

df_enc = df.copy()
# Drop label/text columns from the encoded version
df_enc = df_enc.drop(columns=['title', 'genres'])

# LabelEncode the discretized categorical columns
for col in df_enc.select_dtypes(include=['object', 'category']).columns:
    df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))

scaler = MinMaxScaler()
df_norm = pd.DataFrame(scaler.fit_transform(df_enc),
                       columns=df_enc.columns,
                       index=df_enc.index)

# Boxplot AFTER normalization
plt.figure(figsize=(14, 6))
df_norm[numeric_cols].boxplot()
plt.xticks(rotation=45, ha='right')
plt.title('Boxplot - after MinMax Normalization')
plt.tight_layout()
plt.savefig('plots/boxplot_normalized.png', dpi=120, bbox_inches='tight')
plt.close()

# ============================================================
# 1.9 SAVE OUTPUTS
# ============================================================
df_clean.to_csv('data/processed/df_clean.csv', index=False)
df_enc.to_csv('data/processed/df_enc.csv', index=False)
df_norm.to_csv('data/processed/df_normalized.csv', index=False)

print('\n✅ Preprocessing complete')
print(f'   Rows: {len(df_clean)}')
print(f'   Numeric features: {numeric_cols}')
print(f'   Genre features: {len(genre_dummies.columns)}')
print(f'   Files saved to data/processed/')