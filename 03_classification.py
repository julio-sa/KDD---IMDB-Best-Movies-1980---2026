import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import (SelectKBest, f_classif, mutual_info_classif, RFE)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix)
from imblearn.over_sampling import SMOTE

# ============================================================
# 3.1 LOAD
# ============================================================
df_clusters = pd.read_csv('data/processed/df_with_clusters.csv')
df_enc = pd.read_csv('data/processed/df_enc.csv')
df_enc['Cluster'] = df_clusters['Cluster']

print(f'Loaded {len(df_enc)} movies')

# ============================================================
# 3.2 PREVENT DATA LEAKAGE
# Drop features that were used to BUILD the clusters
# (they would trivially predict the cluster label)
# ============================================================
leak_cols = ['love_score', 'cult_factor', 'rating_pct', 'votes_pct']
features = [c for c in df_enc.columns
            if c not in leak_cols + ['Cluster']]
X = df_enc[features]
y = df_enc['Cluster']

print(f'Using {len(features)} features')
print(f'Class distribution:\n{y.value_counts().sort_index()}')

# ============================================================
# 3.3 FEATURE SELECTION — 4 ALGORITHMS (assignment requirement)
# ============================================================
K = 10 # How many features each algorithm picks
results_fs = {}

print('\n' + '=' * 70)
print('FEATURE SELECTION')
print('=' * 70)

# ANOVA F-value
sel = SelectKBest(f_classif, k='all').fit(X, y)
results_fs["ANOVA"] = pd.Series(sel.scores_, index=X.columns).nlargest(K).index.tolist()

# Mutual Information
sel = SelectKBest(mutual_info_classif, k='all').fit(X, y)
results_fs["MutualInfo"] = pd.Series(sel.scores_, index=X.columns).nlargest(K).index.tolist()

# Random Forest importance
rf = RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=1).fit(X, y)
results_fs['RandomForest'] = pd.Series(rf.feature_importances_, index=X.columns).nlargest(K).index.tolist()

# RFE with decision tree
rfe = RFE(DecisionTreeClassifier(random_state=42), n_features_to_select=K).fit(X, y)
results_fs['RFE'] = X.columns[rfe.support_].tolist()

print('\nTop features per algorithm:')
for name, feats in results_fs.items():
    print(f'\n {name}:')
    for f in feats:
        print(f'    - {f}')

# Majority vote
vote = Counter([f for feats in results_fs.values() for f in feats])
selected_features = [f for f, count in vote.most_common() if count >= 3]
print(f'\n→ Features selected by majority (≥3 of 4 methods): {len(selected_features)}')
for f in selected_features:
    print(f'    {f} (voted by {vote[f]}/4')

# Save the feature selection comparision
fs_df = pd.DataFrame(
    {k: pd.Series(v) for k, v in results_fs.items()}
)
fs_df.to_csv('data/processed/feature_selection.csv', index=False)

# ============================================================
# 3.4 ALGORITHM COMPARISON — FULL vs REDUCED (assignment requirement)
# ============================================================
print('\n' + '=' * 70)
print('ALGORITHM COMPARISON (5-fold CV accuracy)')
print('=' * 70)

algorithms = {
    'KNN': KNeighborsClassifier(),
    'Neural Network': MLPClassifier(max_iter=500, random_state=42),
    'SVM': SVC(random_state=42),
    'Naive Bayes': GaussianNB(),
    'J48 (Decision Tree)': DecisionTreeClassifier(random_state=42),
}

comparison = []
for name, model in algorithms.items():
    print(f'\n  Running {name}...')
    full = cross_val_score(model, X, y, cv=5, n_jobs=1).mean()
    reduced = cross_val_score(model, X[selected_features], y,
                              cv=5, n_jobs=1).mean()
    comparison.append({
        'Algorithm': name,
        'Full Features': round(full, 4),
        'Selected Features': round(reduced, 4),
        'Difference': round(reduced - full, 4),
    })
    print(f'    Full: {full:.4f}   |   Selected: {reduced:.4f}')

comp_df = pd.DataFrame(comparison)
print('\n' + comp_df.to_string(index=False))
comp_df.to_csv('data/processed/algorithm_comparison.csv', index=False)

# Decide which feature set to use for the tree deep-dive
mean_full = comp_df['Full Features'].mean()
mean_reduced = comp_df['Selected Features'].mean()
use_full = mean_full > mean_reduced + 0.01 # require meaningful gap
X_use = X if use_full else X[selected_features]
chosen_features = list(X_use.columns)
print(f'\n→ Using {'FULL' if use_full else 'SELECTED'} feature set '
      f'({len(chosen_features)} features) for decision tree deep-dive')

# ============================================================
# 3.5 HANDLE CLASS IMBALANCE
# ============================================================
print('\n' + '=' * 70)
print('CLASS BALANCE')
print('=' * 70)
balance_ratio = y.value_counts().min() / y.value_counts().max()
print(f'Min/Max class ratio: {balance_ratio:.3f}')

if balance_ratio < 0.5:
    print('→ Applying SMOTE to balance classes')
    sm = SMOTE(random_state=42)
    X_use, y_use = sm.fit_resample(X_use, y)
    print(f'After SMOTE:zn{pd.Series(y_use).value_counts().sort_index()}')
else:
    print('→ Classes reasonably balanced - no SMOTE needed')
    y_use = y

# ============================================================
# 3.6 DECISION TREE — GRID OF EXPERIMENTS (assignment requirement)
# ============================================================
print('\n' + '=' * 70)
print('DECISION TREE EXPERIMENTS')
print('=' * 70)

experiments = []
for depth in [3, 5, 7, 10, None]:
    for min_split in [2, 10, 20, 50]:
        tree = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_split=min_split,
            random_state=42,
        )
        # Percentage split 80/20
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_use, y_use, test_size=0.2,
            random_state=42, stratify=y_use,
        )
        tree.fit(X_tr, y_tr)
        acc_split = accuracy_score(y_te, tree.predict(X_te))
        # 5-fold CV
        acc_cv = cross_val_score(tree, X_use, y_use,
                                 cv=5, n_jobs=-1).mean()
        experiments.append({
            'max_depth': str(depth),
            'min_samples_split': min_split,
            'accuracy_split_80_20': round(acc_split, 4),
            'accuracy_cv_5fold': round(acc_cv, 4),
        })

exp_df = pd.DataFrame(experiments).sort_values(
    'accuracy_cv_5fold', ascending=False
)
print('\nTop 10 configurations:')
print(exp_df.head(10).to_string(index=False))
exp_df.to_csv('data/processed/tree_experiments.csv', index=False)

# ============================================================
# 3.7 FINAL TREE — TRAIN, EVALUATE, EXPORT
# ============================================================
print('\n' + '=' * 70)
print('FINAL DECISION TREE')
print('=' * 70)

best = exp_df.iloc[0]
best_depth = None if best['max_depth'] == 'None' else int(best['max_depth'])
best_split = int(best['min_samples_split'])
print(f'Best config: max_depth={best_depth}, min_samples_split={best_split}')
print(f'CV accuracy: {best['accuracy_cv_5fold']}')

# Train on 80/20 for the confusion matrix
X_tr, X_te, y_tr, y_te = train_test_split(
    X_use, y_use, test_size=0.2, random_state=42, stratify=y_use,
)
final_tree_eval = DecisionTreeClassifier(
    max_depth=best_depth,
    min_samples_split=best_split,
    random_state=42,
).fit(X_tr, y_tr)
y_pred = final_tree_eval.predict(X_te)

print(f'\nTest accuracy: {accuracy_score(y_te, y_pred):.4f}')
print('\nClassification report:')
print(classification_report(y_te, y_pred))

# Map cluster IDs to human names for the confusion matrix
CLUSTER_NAMES = {
    0: 'Forgotten',
    1: 'MassFav',
    2: 'Cult',
    3: 'Popcorn',
}
class_names = [CLUSTER_NAMES[c] for c in sorted(np.unique(y_use))]

cm = confusion_matrix(y_te, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Final Decision Tree')
plt.tight_layout()
plt.savefig('plots/confusion_matrix.png', dpi=120, bbox_inches='tight')
plt.close()
print('✅ Saved plots/confusion_matrix.png')

# Train the FINAL tree on ALL data (for visualization + rules)
final_tree = DecisionTreeClassifier(
    max_depth=best_depth,
    min_samples_split=best_split,
    random_state=42,
).fit(X_use, y_use)

# Visualize the tree
plt.figure(figsize=(28, 14))
plot_tree(
    final_tree,
    feature_names=chosen_features,
    class_names=class_names,
    filled=True, rounded=True, fontsize=9,
    max_depth=4, # limit VIEW depth for readability
)
plt.title(f'Decision Tree (max_depth={best_depth}), '
          f'min_samples_split={best_split}) - showing first 4 levels',
          fontsize=14)
plt.savefig('plots/decision_tree.png', dpi=120, bbox_inches='tight')
plt.close()
print('✅ Saved plots/decision_tree.png')

# Export text rules - easier to interpret than the visual
rules_text = export_text(
    final_tree,
    feature_names=chosen_features,
    max_depth=5,
)
with open('data/processed/tree_rules.txt', 'w', encoding='utf-8') as f:
    f.write(f'Decision Tree Rules\n')
    f.write(f'max_depth={best_depth}, min_samples_split={best_split}\n')
    f.write(f'Classes: {dict(zip(sorted(np.unique(y_use)), class_names))}\n')
    f.write('=' * 70 + '\n\n')
    f.write(rules_text)

print('✅ Saved data/processed/tree_rules.txt')
print('\nFirst 50 lines of rules:')
print('\n'.join(rules_text.split('\n')[:50]))

# ============================================================
# 3.8 FEATURE IMPORTANCE CHART
# ============================================================
importance = pd.Series(
    final_tree.feature_importances_, index=chosen_features
).sort_values(ascending=False).tail(15)

plt.figure(figsize=(10, 7))
importance.plot(kind='barh', color='steelblue')
plt.xlabel('Importance')
plt.title('Top 15 features driving cluster classification')
plt.tight_layout()
plt.savefig('plots/feature_importance.png', dpi=120, bbox_inches='tight')
plt.close()
print('✅ Saved plots/feature_importance.png')

# ============================================================
# 3.9 BONUS — GENRE-ONLY TREE
# A second tree using ONLY content features (genres + year + runtime)
# Reveals which genres CHARACTERIZE each audience type
# ============================================================
print('\n' + '=' * 70)
print('BONUS: GENRE-ONLY DECISION TREE')
print('Using only content features (no rating, no votes)')
print('=' * 70)

# Built content-only feature set
genre_cols = [c for c in df_enc.columns if c.startswith('genre_')]
content_features = genre_cols + ['year', 'runtime_minutes', 'decade']
content_features = [f for f in content_features if f in df_enc.columns]

X_content = df_enc[content_features]
y_original = df_enc['Cluster']

print(f'Content features used: {len(content_features)}')

# Quick grid for genre tree
genre_experiments = []
for depth in [3, 5, 7, 10]:
    for min_split in [2, 20, 50]:
        tree = DecisionTreeClassifier(
            max_depth=depth, min_samples_split=min_split, random_state=42
        )
        acc_cv = cross_val_score(tree, X_content, y_original,
                                 cv=5, n_jobs=-1).mean()
        genre_experiments.append({
            'max_depth': depth,
            'min_samples_split': min_split,
            'accuracy_cv': round(acc_cv, 4),
        })

genre_exp_df = pd.DataFrame(genre_experiments).sort_values(
    'accuracy_cv', ascending=False
)
print('\nGenre-only tree experiments:')
print(genre_exp_df.to_string(index=False))
genre_exp_df.to_csv('data/processed/genre_tree_experiments.csv', index=False)

# Train teh best genre tree
best_g = genre_exp_df.iloc[0]
genre_tree = DecisionTreeClassifier(
    max_depth=int(best_g['max_depth']),
    min_samples_split=int(best_g['min_samples_split']),
    random_state=42,
).fit(X_content, y_original)

print(f'\nBest genre-only tree: depth={best_g['max_depth']}, '
      f'min_split={int(best_g['min_samples_split'])}, '
      f'CV accuracy={best_g['accuracy_cv']}')

# Visualize
plt.figure(figsize=(28, 14))
plot_tree(
    genre_tree,
    feature_names=content_features,
    class_names=class_names,
    filled=True, rounded=True, fontsize=9,
    max_depth=4,
)
plt.title(f'Genre-Only Decision Tree (depth={int(best_g['max_depth'])}) - showing first 4 levels',
          fontsize=14)
plt.savefig('plots/decision_tree_genre_only.png', dpi=120, bbox_inches='tight')
plt.close()
print('✅ Saved plots/decision_tree_genre_only.png')

# Export rules
genre_rules = export_text(
    genre_tree, feature_names=content_features, max_depth=5
)
with open('data/processed/tree_rules_genre_only.txt', 'w', encoding='utf-8') as f:
    f.write(f'Genre-Only Decision Tree Rules\n')
    f.write(f'max_depth={int(best_g['max_depth'])}, '
            f'min_samples_split={int(best_g['min_samples_split'])}\n')
    f.write(f'CV Accuracy: {best_g['accuracy_cv']}\n')
    f.write(f'Classes: {dict(zip(sorted(np.unique(y_original)), class_names))}\n')
    f.write('=' * 70 + '\n\n')
    f.write(genre_rules)
print('✅ Saved data/processed/tree_rules_genre_only.txt')

# Genre feature importance
g_importance = pd.Series(
    genre_tree.feature_importances_, index=content_features
).sort_values(ascending=True).tail(15)

plt.figure(figsize=(10, 7))
g_importance.plot(kind='barh', color='darkorange')
plt.xlabel('Importance')
plt.title('Top 15 content features predicting audience type')
plt.tight_layout()
plt.savefig('plots/feature_importance_genre_only.png', dpi=120, bbox_inches='tight')
plt.close()
print('✅ Saved plots/feature_importance_genre_only.png')

print('\n BONUS step complete')

print('\n' + '=' * 70)
print('✅ CLASSIFICATION STEP COMPLETE')
print('=' * 70)
