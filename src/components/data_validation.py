import pandas as pd
import ast
import re
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

# 1. Ingest data
df = pd.read_csv(r'C:\Users\anike\OneDrive\Desktop\MLProjects\Spotify_data\Data_files\artists.csv')

# 2. Remove rows with missing/empty genres and non-positive followers/popularity
df = df[df['genres'].notnull() & (df['genres'] != '[]')]
df = df[(df['followers'] > 0) & (df['popularity'] > 0)].copy()

# 3. Parse genres from string to list
def parse_genres(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    elif isinstance(x, list):
        return x
    else:
        return []
df['genres'] = df['genres'].apply(parse_genres)

# 4. Clean artist names (keep only readable characters)
df['name_clean'] = df['name'].apply(lambda x: re.sub(r'[^A-Za-z0-9 &]+', '', str(x)))

# 5. Feature engineering
df['name_length'] = df['name_clean'].apply(len)
df['genre_count'] = df['genres'].apply(len)
df['log_followers'] = np.log1p(df['followers'])
df['popularity_bin'] = pd.cut(
    df['popularity'],
    bins=[-1, 40, 70, 100],  # adjust bins as needed
    labels=['Low', 'Medium', 'High']
)

# 6. One-hot encode genres per artist, SPARSE
mlb = MultiLabelBinarizer(sparse_output=True)
genre_sparse = mlb.fit_transform(df['genres'])
print("Genre sparse matrix shape:", genre_sparse.shape)
print("Number of unique genres:", len(mlb.classes_))

# 7. (Optional) Save outputs
import scipy.sparse
scipy.sparse.save_npz(r'C:\Users\anike\OneDrive\Desktop\MLProjects\Spotify_data\Data_files\genre_sparse.npz', genre_sparse)
with open(r'C:\Users\anike\OneDrive\Desktop\MLProjects\Spotify_data\Data_files\genre_labels.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(mlb.classes_))
df.to_csv(r'C:\Users\anike\OneDrive\Desktop\MLProjects\Spotify_data\Data_files\artists_cleaned1.csv', index=False)

# 8. Robust Data Validation
print("\n=== DATA VALIDATION REPORT ===")

print("\nColumn data types:\n", df.dtypes)
print("\nSample rows:\n", df.head(3))
print("\nMissing values per column:\n", df.isnull().sum())
print("\nNumeric column summary:\n", df.describe())

# Categorical/engineered features
if 'popularity_bin' in df.columns:
    print("\n'popularity_bin' value counts:\n", df['popularity_bin'].value_counts(dropna=False))

if 'genre_count' in df.columns:
    print(f"\nGenre count per artist: min={df['genre_count'].min()}, max={df['genre_count'].max()}")
else:
    print("Column 'genre_count' not found in DataFrame.")

if 'name_length' in df.columns:
    print(f"Name length: min={df['name_length'].min()}, max={df['name_length'].max()}")

if 'log_followers' in df.columns:
    print(f"Log followers: min={df['log_followers'].min()}, max={df['log_followers'].max()}")

# Range and uniqueness checks
print(f"\nFollowers: min={df['followers'].min()}, max={df['followers'].max()}")
print(f"Popularity: min={df['popularity'].min()}, max={df['popularity'].max()}")
print(f"Unique artist IDs: {df['id'].nunique()}  (total rows: {len(df)})")

# Genre sparse matrix checks
print(f"\nGenre sparse matrix shape: {genre_sparse.shape}")
total_genre_assignments = genre_sparse.sum()
sum_genre_count = df['genre_count'].sum() if 'genre_count' in df.columns else 'N/A'
print(f"Total genre assignments (one-hot sum): {total_genre_assignments}")
print(f"Total genre_count column sum: {sum_genre_count}")

if sum_genre_count != 'N/A' and total_genre_assignments == sum_genre_count:
    print("✔ Genre assignments in matrix match 'genre_count' in DataFrame.")
else:
    print("✗ Mismatch! (Check your genre parsing and encoding pipeline.)")

# Spot check: Do genres match one-hot encoding? (first 3 artists)
for idx in range(min(3, len(df))):
    artist_name = df.iloc[idx]['name']
    genres_actual = df.iloc[idx]['genres']
    onehot_indices = genre_sparse[idx,:].nonzero()[1]
    genres_from_onehot = [mlb.classes_[i] for i in onehot_indices]
    print(f"\nArtist: {artist_name}")
    print(" - Genres from list:   ", genres_actual)
    print(" - Genres from one-hot:", genres_from_onehot)
    if set(genres_actual) == set(genres_from_onehot):
        print("   ✔ Match!")
    else:
        print("   ✗ Mismatch! (Check this row.)")

print("\n=== END OF DATA VALIDATION ===")
