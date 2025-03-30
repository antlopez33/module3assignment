import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("netflix_titles.csv")
df['listed_in'] = df['listed_in'].fillna('')
df['type'] = df['type'].fillna('')
df['rating'] = df['rating'].fillna('')
df['country'] = df['country'].fillna('')
df['cast'] = df['cast'].fillna('')

df['features'] = df.apply(lambda row: f"{row['listed_in']} {row['type']} {row['rating']} {row['country']} {row['cast']}", axis=1)

features = list(set(word for features in df['features'] for word in features.split()))
index = {feature: idx for idx, feature in enumerate(features)}

bow = []
for features in df['features']:
    row = [0] * len(index)
    for feature in features.split():
        if feature in index:
            row[index[feature]] = 1
    bow.append(row)

cosine_sim = cosine_similarity(bow, bow)

titles = ["Naruto", "Narcos", "Breaking Bad"]
query = df[df['title'].isin(titles)].index

for query_idx in query:
    query_distances = []
    for i in range(len(df)):
        similarity = cosine_sim[query_idx][i]
        query_distances.append((i, similarity))

    query_distances = sorted(query_distances, key=lambda x: x[1], reverse=True)
    print(f"\nTop 10 titles similar to '{df.iloc[query_idx]['title']}':")
    for similar, score in query_distances[1:11]:
        print(f"{df.iloc[similar]['title']} - {score:.4f}")