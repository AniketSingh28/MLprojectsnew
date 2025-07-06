# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# import networkx as nx
# import itertools
# from collections import Counter

# df = pd.read_csv((r'C:\Users\anike\OneDrive\Desktop\MLProjects\Spotify_data\Data_files\artists.csv'))

# # ----- 1. Popularity vs. Followers -----
# plt.figure(figsize=(8,5))
# plt.scatter(df['followers'], df['popularity'], alpha=0.3)
# plt.title("Popularity vs. Followers (All Genres)")
# plt.xlabel("Followers")
# plt.ylabel("Popularity")
# plt.xscale('log')
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.show()

# # Top 3 genres for highlight
# top_3_genres = df['genres'].value_counts().head(3).index

# plt.figure(figsize=(9,6))
# for genre in top_3_genres:
#     subset = df[df['genres'] == genre]
#     plt.scatter(subset['followers'], subset['popularity'], label=genre, alpha=0.4)
# plt.title("Popularity vs. Followers by Top 3 Genres")
# plt.xlabel("Followers (log scale)")
# plt.ylabel("Popularity")
# plt.xscale('log')
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.show()

# for genre in top_3_genres:
#     subset = df[df['genres'] == genre]
#     corr = subset['followers'].corr(subset['popularity'])
#     print(f"Correlation (Followers vs. Popularity) for {genre}: {corr:.2f}")

# # ----- 2. KMeans Clustering -----
# # Prepare clustering data
# clustering_df = df[['followers', 'popularity']].dropna().copy()
# clustering_df['log_followers'] = np.log1p(clustering_df['followers'])

# scaler = StandardScaler()
# X = scaler.fit_transform(clustering_df[['log_followers', 'popularity']])

# k = 4
# kmeans = KMeans(n_clusters=k, random_state=0)
# clustering_df['cluster'] = kmeans.fit_predict(X)

# plt.figure(figsize=(8,6))
# for cluster in range(k):
#     subset = clustering_df[clustering_df['cluster'] == cluster]
#     plt.scatter(subset['log_followers'], subset['popularity'], label=f'Cluster {cluster}', alpha=0.6)
# plt.title('KMeans Clusters: Popularity vs. Log(Followers)')
# plt.xlabel('Log(Followers)')
# plt.ylabel('Popularity')
# plt.legend()
# plt.show()

# centers = scaler.inverse_transform(kmeans.cluster_centers_)
# centers[:, 0] = np.expm1(centers[:, 0])
# print("Cluster Centers (Followers, Popularity):\n", centers)

# # ----- 3. Genre Co-occurrence Network -----
# # Collapse back to list of genres per artist
# artist_genres = df.groupby('id')['genres'].apply(list)

# cooccurrence = Counter()
# for genres in artist_genres:
#     unique_genres = list(set(genres))
#     for g1, g2 in itertools.combinations(sorted(unique_genres), 2):
#         cooccurrence[(g1, g2)] += 1

# G = nx.Graph()
# threshold = 100  # You can adjust this!

# for (g1, g2), count in cooccurrence.items():
#     if count >= threshold:
#         G.add_edge(g1, g2, weight=count)

# plt.figure(figsize=(12, 8))
# pos = nx.spring_layout(G, k=0.5)
# edges = G.edges()
# nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
# nx.draw_networkx_edges(G, pos, edgelist=edges, width=[G[u][v]['weight']/threshold for u,v in edges], alpha=0.6)
# nx.draw_networkx_labels(G, pos, font_size=10)
# plt.title("Genre Co-occurrence Network (edges show frequent genre pairs)")
# plt.axis('off')
# plt.show()


# import pandas as pd
# import ast
# import re

# # 1. Load your data
# df = pd.read_csv((r'C:\Users\anike\OneDrive\Desktop\MLProjects\Spotify_data\Data_files\artists.csv'))

# # 2. Remove rows with missing or '[]' genres (keeps only artists with actual genres)
# df = df[df['genres'].notnull() & (df['genres'] != '[]')].copy()

# # 3. Remove rows with missing or non-positive followers or popularity
# df = df[df['followers'].notnull() & df['popularity'].notnull()]
# df = df[(df['followers'] > 0) & (df['popularity'] > 0)]

# # 4. Parse genres from string to Python list
# def parse_genres(x):
#     if isinstance(x, str):
#         try:
#             return ast.literal_eval(x)
#         except Exception:
#             return []
#     elif isinstance(x, list):
#         return x
#     else:
#         return []

# df['genres'] = df['genres'].apply(parse_genres)

# # 5. Clean artist names: keep only letters, numbers, spaces, and "&"
# df['name_clean'] = df['name'].apply(lambda x: re.sub(r'[^A-Za-z0-9 &]+', '', str(x)))

# # 6. (Optional) Remove any now-empty rows (in case names were only symbols)
# df = df[df['name_clean'].str.strip().astype(bool)]

# # 7. Reset index
# df = df.reset_index(drop=True)

# # 8. Quick summary
# print(f"Data shape after cleaning: {df.shape}")
# print("\nSample cleaned data:")
# print(df.head())

# # 9. Save cleaned data (optional)
# df.to_csv('artists_cleaned.csv', index=False)

# import pandas as pd
# import ast
# import re
# import numpy as np
# from sklearn.preprocessing import MultiLabelBinarizer

# # 1. Ingest data
# df = pd.read_csv(r'C:\Users\anike\OneDrive\Desktop\MLProjects\Spotify_data\Data_files\artists.csv')

# # 2. Remove rows with missing/empty genres and non-positive followers/popularity
# df = df[df['genres'].notnull() & (df['genres'] != '[]')]
# df = df[(df['followers'] > 0) & (df['popularity'] > 0)].copy()

# # 3. Parse genres from string to list
# def parse_genres(x):
#     if isinstance(x, str):
#         try:
#             return ast.literal_eval(x)
#         except Exception:
#             return []
#     elif isinstance(x, list):
#         return x
#     else:
#         return []
# df['genres'] = df['genres'].apply(parse_genres)

# # 4. Clean artist names (keep only readable characters)
# df['name_clean'] = df['name'].apply(lambda x: re.sub(r'[^A-Za-z0-9 &]+', '', str(x)))

# # 5. Feature engineering
# df['name_length'] = df['name_clean'].apply(len)
# df['genre_count'] = df['genres'].apply(len)
# df['log_followers'] = np.log1p(df['followers'])  # log scale, avoids log(0)
# df['popularity_bin'] = pd.cut(
#     df['popularity'],
#     bins=[-1, 40, 70, 100],  # adjust as needed for your data
#     labels=['Low', 'Medium', 'High']
# )

# # 6. One-hot encode genres per artist (not exploded)
# mlb = MultiLabelBinarizer()
# genre_dummies = pd.DataFrame(mlb.fit_transform(df['genres']), columns=mlb.classes_, index=df.index)
# # Optionally, you can concatenate with df:
# df_final = pd.concat([df, genre_dummies], axis=1)

# # 7. (Optional) Save the transformed data for ML/EDA
# df_final.to_csv(r'C:\Users\anike\OneDrive\Desktop\MLProjects\Spotify_data\Data_files\artists_transformed.csv', index=False)

# # 8. Show sample output and shape
# print("Transformed Data Sample:\n", df_final.head())
# print("\nShape of final data (rows, columns):", df_final.shape)
# print("One-hot genres columns:", list(genre_dummies.columns[:10]), "...")  # Show first 10 genre columns

# # 9. (Optional) For memory-intensive data: use sparse matrix
# mlb = MultiLabelBinarizer(sparse_output=True)
# genre_sparse = mlb.fit_transform(df['genres'])
# print("Sparse matrix shape:", genre_sparse.shape)

# # 10. If you want only top N genres (e.g., 50 most common), use:
# all_genres_flat = [g for sublist in df['genres'] for g in sublist]
# topN = pd.Series(all_genres_flat).value_counts().nlargest(50).index
# df['genres_topN'] = df['genres'].apply(lambda x: [g for g in x if g in topN])
# mlb_topN = MultiLabelBinarizer()
# genre_dummies_topN = pd.DataFrame(mlb_topN.fit_transform(df['genres_topN']), columns=mlb_topN.classes_, index=df.index)




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

# 6. One-hot encode genres per artist, SPARSE (no dense concat!)
mlb = MultiLabelBinarizer(sparse_output=True)
genre_sparse = mlb.fit_transform(df['genres'])  # SAFE for large data!
print("Genre sparse matrix shape:", genre_sparse.shape)
print("Number of unique genres:", len(mlb.classes_))

# 7. (Optional) Inspect a small dense sample
sample = pd.DataFrame(genre_sparse[:5,:].toarray(), columns=mlb.classes_)
print("\nSample genre one-hot (first 5 rows):\n", sample.head())

# 8. Save sparse genre matrix and index mapping (for ML use)
import scipy.sparse
scipy.sparse.save_npz(r'C:\Users\anike\OneDrive\Desktop\MLProjects\Spotify_data\Data_files\genre_sparse.npz', genre_sparse)
# Save genre label mapping (column names)
with open(r'C:\Users\anike\OneDrive\Desktop\MLProjects\Spotify_data\Data_files\genre_labels.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(mlb.classes_))
# Save cleaned DataFrame without genre dummies (for joining later)
df.to_csv(r'C:\Users\anike\OneDrive\Desktop\MLProjects\Spotify_data\Data_files\artists_cleaned1.csv', index=False)

# 9. (Optional) Use genre_sparse directly in ML or clustering pipelines
# Example: KMeans clustering (sklearn >=1.2 supports sparse)
# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=10, random_state=42)
# labels = kmeans.fit_predict(genre_sparse)

print("\nData preparation and sparse one-hot genre encoding complete!")
