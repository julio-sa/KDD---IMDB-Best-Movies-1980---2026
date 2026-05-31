import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules

# ============================================================
# 4.1 LOAD
# ============================================================
df = pd.read_csv("data/processed/df_with_clusters.csv")
print(f"Loaded {len(df)} movies")

# ============================================================
# 4.2 BUILD THE TRANSACTION FRAME
# ============================================================
# Multi-hot genre columns (already boolean-ish, just convert to bool)
genre_cols = [c for c in df.columns if c.startswith("genre_")]
genre_bool = df[genre_cols].astype(bool)

# Discretized features + cluster name (one-hot encode them)
other = df[["rating_cat", "votes_cat", "decade",
            "runtime_cat", "ClusterName"]].astype(str)
other_dummies = pd.get_dummies(other)

df_onehot = pd.concat([genre_bool, other_dummies], axis=1)
print(f"Transaction frame shape: {df_onehot.shape}")
print(f"  Genre columns: {len(genre_cols)}")
print(f"  Other dummy columns: {other_dummies.shape[1]}")

# ============================================================
# 4.3 APRIORI — FIND FREQUENT ITEMSETS
# ============================================================
print("\n" + "=" * 70)
print("APRIORI - finding frequent itemsets")
print("=" * 70)

# min_support=0.05 means itemset must appear in at least 5% of movies (~810 rows)
frequent = apriori(df_onehot, min_support=0.05,
                   use_colnames=True, max_len=4)
print(f"Found {len(frequent)} frequent itemsets")

# ============================================================
# 4.4 GENERATE ALL RULES
# ============================================================
rules = association_rules(frequent, metric="lift", min_threshold=1.2)
print(f"Generated {len(rules)} raw rules with lift ≥ 1.2")

# Format for readability
def fmt(itemset):
    return ", ".join(sorted(str(i) for i in itemset))

rules["antecedents_str"] = rules["antecedents"].apply(fmt)
rules["consequents_str"] = rules["consequents"].apply(fmt)

# ============================================================
# 4.5 FILTER FOR THE RESEARCH QUESTION
# Rules where the CONSEQUENT is a cluster + ANTECEDENT contains a genre
# ============================================================
def has_cluster(itemset):
    return any("ClusterName_" in str(i) for i in itemset)

def has_genre(itemset):
    return any("genre_" in str(i) for i in itemset)

def only_cluster(itemset):
    # Consequent should be ONLY a cluster (single-item), for clean rules
    items = list(itemset)
    return len(items) == 1 and "ClusterName_" in str(items[0])

genre_to_cluster = rules[
    rules["consequents"].apply(only_cluster) &
    rules["antecedents"].apply(has_genre)
].copy()

# Apply meaningful thresholds
genre_to_cluster = genre_to_cluster[
    (genre_to_cluster["confidence"] >= 0.4) &
    (genre_to_cluster["lift"] >= 1.3) &
    (genre_to_cluster["support"] >= 0.02)
].sort_values("lift", ascending=False)

print(f"\n{len(genre_to_cluster)} genre→audience rules pass thresholds")

# ============================================================
# 4.6 TOP RULES PER CLUSTER (so each audience type is covered)
# ============================================================
print("\n" + "=" * 70)
print("TOP GENRE→AUDIENCE RULES PER CLUSTER")
print("=" * 70)

top_per_cluster = []
for cluster_label in genre_to_cluster["consequents_str"].unique():
    subset = genre_to_cluster[
        genre_to_cluster["consequents_str"] == cluster_label
    ].head(8)
    top_per_cluster.append(subset)
    cluster_name = cluster_label.replace("ClusterName_", "")
    print(f"\n--- {cluster_name} ---")
    for _, row in subset.iterrows():
        print(f"  {row['antecedents_str']:60s} "
              f"→ {cluster_name:15s}  "
              f"sup={row['support']:.3f}  "
              f"conf={row['confidence']:.2f}  "
              f"lift={row['lift']:.2f}")

if top_per_cluster:
    top_rules = pd.concat(top_per_cluster, ignore_index=True)
    cols_to_save = ["antecedents_str", "consequents_str", "support",
                    "confidence", "lift"]
    top_rules[cols_to_save].to_csv(
        "data/processed/top_genre_to_cluster_rules.csv", index=False
    )
    print(f"\n✅ Saved {len(top_rules)} rules to "
          f"data/processed/top_genre_to_cluster_rules.csv")

# Save the full filtered set too
genre_to_cluster[
    ["antecedents_str", "consequents_str", "support",
     "confidence", "lift"]
].to_csv("data/processed/all_genre_to_cluster_rules.csv", index=False)

# ============================================================
# 4.7 VISUALIZATION — LIFT HEATMAP (single-genre → cluster)
# Shows which single genre most predicts each cluster
# ============================================================
single_genre_rules = genre_to_cluster[
    genre_to_cluster["antecedents"].apply(
        lambda x: len(x) == 1 and "genre_" in str(list(x)[0])
    )
].copy()

if len(single_genre_rules) > 0:
    single_genre_rules["genre"] = single_genre_rules["antecedents_str"].str.replace("genre_", "")
    single_genre_rules["cluster"] = single_genre_rules["consequents_str"].str.replace("ClusterName_", "")

    pivot = single_genre_rules.pivot_table(
        index="genre", columns="cluster",
        values="lift", aggfunc="max"
    ).fillna(1.0)

    # Order columns conceptually
    desired_order = ["Mass Favorites", "Cult Darlings",
                     "Mainstream Popcorn", "Forgotten Films"]
    pivot = pivot[[c for c in desired_order if c in pivot.columns]]

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt=".2f",
                cmap="RdYlGn", center=1.0,
                cbar_kws={"label": "Lift"},
                linewidths=0.5)
    plt.title("Single-genre → audience type association (lift)\n"
              "Lift > 1 = positive association | < 1 = negative association")
    plt.tight_layout()
    plt.savefig("plots/genre_lift_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("✅ Saved plots/genre_lift_heatmap.png")

# ============================================================
# 4.7b ENHANCED LIFT HEATMAP — uses ALL genre-containing rules
# Aggregates by genre (max lift across all rules containing that genre)
# ============================================================
def extract_genres(itemset):
    """Get all genres mentioned in an itemset."""
    return [str(i).replace("genre_", "")
            for i in itemset if "genre_" in str(i)]

# Build a long-format table: one row per (genre, cluster, lift)
records = []
for _, row in genre_to_cluster.iterrows():
    genres = extract_genres(row["antecedents"])
    cluster = row["consequents_str"].replace("ClusterName_", "")
    for g in genres:
        records.append({
            "genre": g, "cluster": cluster,
            "lift": row["lift"],
            "confidence": row["confidence"],
            "support": row["support"],
        })

long_df = pd.DataFrame(records)

if len(long_df) > 0:
    # Best lift per (genre, cluster)
    pivot_lift = long_df.pivot_table(
        index="genre", columns="cluster",
        values="lift", aggfunc="max"
    )

    desired_order = ["Mass Favorites", "Cult Darlings",
                     "Mainstream Popcorn", "Forgotten Films"]
    pivot_lift = pivot_lift[[c for c in desired_order
                              if c in pivot_lift.columns]]

    # Sort genres by max lift across clusters (most informative on top)
    pivot_lift = pivot_lift.loc[
        pivot_lift.max(axis=1).sort_values(ascending=False).index
    ]

    plt.figure(figsize=(10, max(4, len(pivot_lift) * 0.6)))
    sns.heatmap(pivot_lift, annot=True, fmt=".2f",
                cmap="RdYlGn", center=1.0,
                cbar_kws={"label": "Max lift"},
                linewidths=0.5,
                mask=pivot_lift.isna())
    plt.title("Genre → Audience Type — strongest lift across all rules\n"
              "(higher = stronger predictive association)")
    plt.tight_layout()
    plt.savefig("plots/genre_lift_heatmap_enhanced.png",
                dpi=120, bbox_inches="tight")
    plt.close()
    print("✅ Saved plots/genre_lift_heatmap_enhanced.png")

# ============================================================
# 4.8 INTERESTING NON-CLUSTER RULES
# Things like {genre_Horror, decade_2020} → {votes_cat_Few}
# ============================================================
exploratory = rules[
    ~rules["consequents"].apply(has_cluster) &
    rules["antecedents"].apply(has_genre)
].copy()

exploratory = exploratory[
    (exploratory["confidence"] >= 0.7) &
    (exploratory["lift"] >= 1.4) &
    (exploratory["support"] >= 0.05)
].sort_values("lift", ascending=False)

print(f"\n{len(exploratory)} exploratory rules")
if len(exploratory) > 0:
    print("\nTop 15 exploratory rules:")
    for _, row in exploratory.head(15).iterrows():
        print(f"  {row['antecedents_str']:55s} → "
              f"{row['consequents_str']:35s}  "
              f"sup={row['support']:.3f}  "
              f"conf={row['confidence']:.2f}  "
              f"lift={row['lift']:.2f}")
    exploratory[
        ["antecedents_str", "consequents_str", "support",
         "confidence", "lift"]
    ].head(30).to_csv(
        "data/processed/exploratory_rules.csv", index=False
    )

# ============================================================
# 4.9 RULE STATISTICS SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)
print(f"Total frequent itemsets:     {len(frequent)}")
print(f"Total raw rules:             {len(rules)}")
print(f"Genre→cluster rules (kept):  {len(genre_to_cluster)}")
print(f"Exploratory rules (kept):    {len(exploratory)}")

print("\n" + "=" * 70)
print("✅ ASSOCIATION STEP COMPLETE")
print("=" * 70)