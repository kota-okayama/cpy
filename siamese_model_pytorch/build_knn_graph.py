import faiss
import numpy as np
import pickle
import os
import json
import time
import sys

# プロジェクトルートをPythonパスに追加 (他モジュールをインポートするため)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- 設定値 ---
K = 10
VECTORIZED_DATA_PATH = "siamese_model_pytorch/vectorized_data"
EMBEDDINGS_FILENAME = "record_embeddings.npy"
IDS_FILENAME = "record_ids.pkl"
OUTPUT_GRAPH_PATH = "siamese_model_pytorch/knn_graph"
OUTPUT_GRAPH_FILENAME = f"knn_graph_k{K}.json"


def build_faiss_knn_graph():
    print(f"Starting K-NN graph construction with K={K} using Faiss...")
    start_time = time.time()

    # --- データのロード ---
    print("\nStep 1: Loading record embeddings and IDs...")
    embeddings_path = os.path.join(VECTORIZED_DATA_PATH, EMBEDDINGS_FILENAME)
    ids_path = os.path.join(VECTORIZED_DATA_PATH, IDS_FILENAME)

    if not os.path.exists(embeddings_path) or not os.path.exists(ids_path):
        print("Error: Embeddings or IDs file not found. Please run vectorize_records.py first.")
        return

    try:
        record_embeddings = np.load(embeddings_path)
        with open(ids_path, "rb") as f:
            record_ids_ordered = pickle.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if record_embeddings.ndim == 1:  # Handle potential 1D array if only one record was processed
        if record_embeddings.shape[0] > 0:
            record_embeddings = record_embeddings.reshape(1, -1)
        else:
            print("Error: Embeddings array is empty or malformed after loading.")
            return
    elif record_embeddings.ndim != 2:
        print(f"Error: Embeddings array has unexpected dimension: {record_embeddings.ndim}. Expected 2D array.")
        return

    num_records, dimension = record_embeddings.shape
    print(f"Loaded {num_records} records with embedding dimension {dimension}.")
    if len(record_ids_ordered) != num_records:
        print("Error: Number of embeddings does not match number of IDs.")
        return

    if K >= num_records:
        print(
            f"Warning: K ({K}) is greater than or equal to the number of records ({num_records-1} other records). Adjusting K or expect fewer neighbors for some."
        )
        # Kを調整するか、エラー処理をする。ここでは警告にとどめる。

    # --- Faissインデックスの構築 ---
    print("\nStep 2: Building Faiss index...")
    try:
        index = faiss.IndexFlatL2(dimension)  # L2距離（ユークリッド距離）
        index.add(record_embeddings)  # NumPy配列をそのまま渡せる
        print(f"Faiss index built. Total vectors in index: {index.ntotal}")
    except Exception as e:
        print(f"Error building Faiss index: {e}")
        print("Please ensure faiss-cpu or faiss-gpu is installed correctly.")
        print("Try: pip install faiss-cpu")
        return

    # --- K近傍探索の実行 ---
    # K+1 で検索 (自身を含むため)。結果から自身を除く
    # ただし、レコード数がK+1未満の場合は、存在するレコード数までしか返らない
    num_neighbors_to_search = min(K + 1, num_records)
    print(f"\nStep 3: Searching for {K} nearest neighbors (requesting {num_neighbors_to_search})...")
    # D: 距離の配列, I: インデックスの配列
    distances, indices = index.search(record_embeddings, num_neighbors_to_search)
    print("Search completed.")

    # --- 結果の処理とグラフ構築 ---
    print("\nStep 4: Processing search results and building graph...")
    knn_graph = {}
    for i in range(num_records):
        source_record_id = record_ids_ordered[i]
        neighbor_ids_for_source = []
        for j in range(num_neighbors_to_search):  # indices[i] は i番目のレコードの近傍のインデックス配列
            neighbor_original_index = indices[i][j]
            if neighbor_original_index == i:  # 自身はスキップ (通常はj=0が自身のはず)
                continue
            if (
                neighbor_original_index == -1
            ):  # faissが-1を返す場合 (検索数に満たないなど。通常IndexFlatL2では起きにくい)
                continue
            neighbor_ids_for_source.append(record_ids_ordered[neighbor_original_index])
            if len(neighbor_ids_for_source) == K:  # K個見つかったら終了
                break
        knn_graph[source_record_id] = neighbor_ids_for_source
    print(f"K-NN graph constructed with {len(knn_graph)} nodes.")

    # --- グラフの保存 ---
    print("\nStep 5: Saving K-NN graph...")
    os.makedirs(OUTPUT_GRAPH_PATH, exist_ok=True)
    output_file_path = os.path.join(OUTPUT_GRAPH_PATH, OUTPUT_GRAPH_FILENAME)
    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(knn_graph, f, indent=4, ensure_ascii=False)
        print(f"K-NN graph saved to {output_file_path}")
    except Exception as e:
        print(f"Error saving graph to JSON: {e}")

    total_time = time.time() - start_time
    print(f"\nK-NN graph construction finished in {total_time // 60:.0f}m {total_time % 60:.0f}s.")


if __name__ == "__main__":
    build_faiss_knn_graph()
