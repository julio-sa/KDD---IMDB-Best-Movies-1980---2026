import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# ============================================================
# 2.1 LOAD
# ============================================================
df_norm = pd.read_csv('data/processed/df_normalized.csv')
df_clean = pd.read_csv('data/processed/df_clean.csv')
print(f'Loaded {len(df_clean)} movies')

# ============================================================
# 2.2 CORRELATION MATRIX (core features only, for readability)
# ============================================================
core_cols = ['average_rating', 'num_votes', 'log_votes', 'love_score',
             'cult_factor', 'year', 'runtime_minutes',
             'rating_pct', 'votes_pct']

plt.figure(figsize=(12, 9))
sns.heatmap(df_norm[core_cols].corr(),
            annot=True, cmap='coolwarm', fmt='.2f', center=0,
            linewidths=0.5)
plt.tight_layout()
plt.savefig('plots/correlation.png', dpi=120, bbox_inches='tight')
plt.close()
print(f'✅ Saved correlation heatmap')

# ============================================================
# 2.3 PICK THE CLUSTERING FEATURES
# We cluster ONLY on engagement signals — that's what defines audience type
# ============================================================
cluster_features = ['average_rating', 'log_votes', 'love_score', 'cult_factor']
X_cluster = df_norm[cluster_features]
print(f'Cluestering on: {cluster_features}')

# ============================================================
# 2.4 ELBOW METHOD (k = 1..15)
# ============================================================
sse = []
K_range = range(1, 16)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_cluster)
    sse.append(km.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(K_range, sse, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of clusters (k)')
plt.ylabel('SSE (inertia)')
plt.title('Elbow method - Finding the right k')
plt.grid(True, alpha=0.3)
plt.xticks(K_range)
plt.tight_layout()
plt.savefig('plots/elbow.png', dpi=120, bbox_inches='tight')
plt.close()
print('✅ Saved elbow plot - look at plots/elbow.png to choose k')
print(f'   SSE values: {[round(s, 1) for s in sse]}')

# 🚨 STOP HERE THE FIRST TIME YOU RUN: open plots/elbow.png and pick k
# Then set OPTIMAL_K below and re-run
OPTIMAL_K = 4

# ============================================================
# 2.5 FINAL K-MEANS
# ============================================================
kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
df_clean['Cluster'] = kmeans.fit_predict(X_cluster)

print(f'\nCluster sizes (k={OPTIMAL_K}):')
print(df_clean['Cluster'].value_counts().sort_index())
print('\nCluster percentages:')
print((df_clean['Cluster'].value_counts(normalize=True) * 100).round(2).sort_index())

# ============================================================
# 2.6 PROFILE EACH CLUSTER (this tells what to NAME them)
# ============================================================
profile = df_clean.groupby("Cluster").agg(
    rating_mean=("average_rating", "mean"),
    rating_median=("average_rating", "median"),
    votes_mean=("num_votes", "mean"),
    votes_median=("num_votes", "median"),
    log_votes_mean=("log_votes", "mean"),
    love_score_mean=("love_score", "mean"),
    cult_factor_mean=("cult_factor", "mean"),
    year_mean=("year", "mean"),
    runtime_mean=("runtime_minutes", "mean"),
    size=("Cluster", "count"),
).round(2)

print('\n' + '=' * 70)
print('CLUSTER PROFILES (use this to name your clusters):')
print('=' * 70)
print(profile.to_string())
profile.to_csv('data/processed/cluster_profiles.csv')

# ============================================================
# 2.7 NAME THE CLUSTERS
# After the FIRST run, look at the profile table above and update this dict
# Heuristics:
#   - High rating + high log_votes + cult_factor near 0  → Mass Favorites
#   - High rating + lower votes + positive cult_factor   → Cult Darlings
#   - Mid-rating  + high votes + negative cult_factor    → Mainstream Popcorn
#   - Low rating  + low votes                            → Forgotten Films
# ============================================================
CLUSTER_NAMES = {
    0: 'Forgotten Films',
    1: 'Mass Favorites',
    2: 'Cult Darlings',
    3: 'Mainstream Popcorn'
}
df_clean['ClusterName'] = df_clean['Cluster'].map(CLUSTER_NAMES)

# ============================================================
# 2.8 GENRE COMPOSITION HEATMAP — ⭐ the headline chart ⭐
# ============================================================
genre_cols = [c for c in df_clean.columns if c.startswith('genre_')]
cluster_genre_pct = df_clean.groupby('ClusterName')[genre_cols].mean() * 100

# Show only the top 15 most common genres for readability
top_genres = cluster_genre_pct.sum().nlargest(15).index
cluster_genre_pct = cluster_genre_pct[top_genres]
cluster_genre_pct.columns = [c.replace('genre_', '') for c in cluster_genre_pct.columns]

# Order rows by yout conceptual ordering
ordered_names = [n for n in ['Mass Favorites', 'Cult Darlings',
                             'Mainstream Popcorn', 'Forgotten Films']
                 if n in cluster_genre_pct.index]
cluster_genre_pct = cluster_genre_pct.loc[ordered_names]

plt.figure(figsize=(16, 5))
sns.heatmap(cluster_genre_pct, annot=True, fmt='.0f',
            cmap='YlOrRd', cbar_kws={'label': '% of cluster'},
            linewidths=0.5)
plt.title('Genre composition by audience type (% of movies in each cluster)')
plt.ylabel('')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('plots/genre_by_cluster.png', dpi=120, bbox_inches='tight')
plt.close()
print('\n✅ Saved genre heatmap')

# ============================================================
# 2.9 TOP 10 MOVIES PER CLUSTER (headline tables)
# ============================================================
print('\n' + '=' * 70)
print('TOP 10 MOVIES PER CLUSTER')
print('=' * 70)

with open('data/processed/top_movies_per_cluster.txt', 'w', encoding='utf-8') as f:
    for cid in sorted(df_clean['Cluster'].unique()):
        name = CLUSTER_NAMES.get(cid, f'Cluster {cid}')
        subset = df_clean[df_clean['Cluster'] == cid]
        # Best representatives: highest love_score within the cluster
        top = subset.nlargest(10, 'love_score')[
            ['title', 'year', 'genres', 'average_rating', 'num_votes']
        ]
        header= f'\n{'=' * 70}\nCLUSTER {cid}: {name}   ({len(subset)}movies)\n{'=' * 70}\n'
        print(header)
        print(top.to_string(index=False))
        f.write(header)
        f.write(top.to_string(index=False))
        f.write('\n')

print('\n✅ Saved top movies to data/processed/top_movies_per_cluster.txt')

# ============================================================
# 2.10 PER-CLUSTER FEATURE DISTRIBUTIONS (one figure per cluster)
# ============================================================
for cid in sorted(df_clean['Cluster'].unique()):
    subset = df_clean[df_clean['Cluster'] == cid]
    name = CLUSTER_NAMES.get(cid, f'Cluster {cid}')

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(f'Cluster {cid}: {name} ({len(subset)}  movies)',
                 fontsize=14, fontweight='bold')

    # Numeric histograms
    num_plots = [
        ('average_rating', 'Rating'),
        ('num_votes', 'Votes'),
        ('log_votes', 'Log Votes'),
        ('love_score', 'Love Score'),
        ('cult_factor', 'Cult Factor'),
        ('year', 'Year'),
        ('runtime_minutes', 'Runtime (min)'),
    ]
    for ax, (col, label) in zip(axes.ravel(), num_plots):
        ax.hist(subset[col], bins=20, color='steelblue', edgecolor='white')
        ax.set_title(label)

    # Top genres for this cluster (bar chart in the last subplot)
    ax = axes.ravel()[7]
    top_g = (subset[genre_cols].sum().nlargest(8))
    top_g.index = [g.replace('genre_', '') for g in top_g.index]
    top_g.plot(kind='bar', ax=ax, color='darkorange')
    ax.set_title('Top genres')
    ax.invert_yaxis()

    plt.tight_layout()
    safe_name = name.replace(' ', '_').replace('/', '_')
    plt.savefig(f'plots/cluster_{cid}_{safe_name}.png',
                dpi=120, bbox_inches='tight')
    plt.close()

print(f'✅ Saved per-cluster distribution charts ({OPTIMAL_K} files)')

# ============================================================
# 2.11 SAVE FOR NEXT STEPS
# ============================================================
df_clean.to_csv('data/processed/df_with_clusters.csv', index=False)
print('\n✅ Saved data/processed/df_with_clusters.csv')
print('✅ Clustering step complete')
