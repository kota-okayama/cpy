import pickle

# キャッシュを読み込む
with open('.cache/embeddings_cache.pkl', 'rb') as f:
    cache = pickle.load(f)

# キャッシュの内容を確認
print(f"キャッシュに保存されている類似度ペア数: {len(cache)}")

# 類似度の分布を調べる
similarity_values = list(cache.values())
print(f"平均類似度: {sum(similarity_values) / len(similarity_values)}")
print(f"最大類似度: {max(similarity_values)}")
print(f"最小類似度: {min(similarity_values)}")