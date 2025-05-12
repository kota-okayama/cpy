import torch
import numpy as np
import os
import sys
import pickle  # For saving record_ids list
import time

# プロジェクトルートをPythonパスに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from data_processing.load_yaml_data import load_bibliographic_data
from data_processing.feature_extraction import load_fasttext_model, get_text_representation
from .network import BaseNetwork  # Assuming network.py is in the same directory

# --- 設定値 (train.pyや他の設定と合わせる) ---
EMBEDDING_DIM = 128  # BaseNetworkの出力次元 (学習時と同じ)
FASTTEXT_LANG = "ja"
INPUT_DIM_FASTTEXT = 300  # fastTextの出力次元 (BaseNetworkの入力次元)
TRAINED_MODEL_FILENAME = f"base_network_emb{EMBEDDING_DIM}_epoch10.pth"  # 学習済みモデルのファイル名
MODEL_LOAD_PATH = "siamese_model_pytorch/saved_models"
OUTPUT_PATH = "siamese_model_pytorch/vectorized_data"


def vectorize_all_records():
    print("Starting process to vectorize all records...")
    start_time = time.time()

    # --- デバイス設定 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 学習済みBaseNetworkモデルのロード ---
    print("\nStep 1: Loading trained BaseNetwork model...")
    model_path = os.path.join(MODEL_LOAD_PATH, TRAINED_MODEL_FILENAME)
    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at {model_path}")
        print("Please ensure the model was trained and saved correctly.")
        return

    # BaseNetworkの初期化 (入力次元はfastTextの次元)
    # train.pyでのDROPOUT_RATEも合わせる必要があるが、推論時はdropoutは無効化されるので必須ではない
    # ただし、構造を完全に一致させるならDROPOUT_RATEも読み込むか指定する
    base_network = BaseNetwork(input_dim=INPUT_DIM_FASTTEXT, embedding_dim=EMBEDDING_DIM)
    base_network.load_state_dict(torch.load(model_path, map_location=device))  # map_locationでデバイス指定
    base_network.to(device)
    base_network.eval()  # モデルを評価モードに設定 (dropoutなどが無効になる)
    print(f"Trained BaseNetwork model loaded from {model_path}")

    # --- fastTextモデルのロード ---
    print("\nStep 2: Loading fastText model...")
    ft_model = load_fasttext_model(lang=FASTTEXT_LANG)
    if not ft_model:
        print("Failed to load fastText model. Exiting.")
        return
    # fastTextの入力次元とBaseNetworkの入力次元が一致するか確認
    assert (
        ft_model.get_dimension() == INPUT_DIM_FASTTEXT
    ), f"fastText dimension ({ft_model.get_dimension()}) does not match BaseNetwork input_dim ({INPUT_DIM_FASTTEXT})"

    # --- 書誌データのロード ---
    print("\nStep 3: Loading all bibliographic records...")
    yaml_path = "benchmark/bib_japan_20241024/1k/record.yml"
    all_records_list = load_bibliographic_data(yaml_path)
    if not all_records_list:
        print("Failed to load bibliographic records. Exiting.")
        return
    print(f"Loaded {len(all_records_list)} records.")

    # --- 全レコードのベクトル化 ---
    print("\nStep 4: Vectorizing all records...")
    record_embeddings = []
    record_ids_ordered = []

    processed_count = 0
    for record_info in all_records_list:
        record_id = record_info.get("record_id")
        record_data = record_info.get("data", {})

        text_repr = get_text_representation(record_data)  # default fields: title, author
        if not text_repr:
            print(f"Warning: No text representation for record_id {record_id}. Skipping.")
            continue

        fasttext_vector_np = ft_model.get_sentence_vector(text_repr)
        fasttext_vector_torch = (
            torch.from_numpy(fasttext_vector_np.astype(np.float32)).unsqueeze(0).to(device)
        )  # バッチ次元追加

        with torch.no_grad():  # 勾配計算は不要
            embedding = base_network(fasttext_vector_torch)

        record_embeddings.append(embedding.squeeze(0).cpu().numpy())  # バッチ次元削除、CPUに戻してNumPyに
        record_ids_ordered.append(record_id)
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"  Processed {processed_count}/{len(all_records_list)} records...")

    print(f"Finished vectorizing {processed_count} records.")

    if not record_embeddings:
        print("No records were vectorized. Exiting.")
        return

    # NumPy配列に変換
    embeddings_array = np.array(record_embeddings, dtype=np.float32)

    # --- 保存処理 ---
    print("\nStep 5: Saving vectorized data...")
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    embeddings_file = os.path.join(OUTPUT_PATH, "record_embeddings.npy")
    np.save(embeddings_file, embeddings_array)
    print(f"Record embeddings saved to {embeddings_file} (shape: {embeddings_array.shape})")

    ids_file = os.path.join(OUTPUT_PATH, "record_ids.pkl")
    with open(ids_file, "wb") as f:
        pickle.dump(record_ids_ordered, f)
    print(f"Corresponding record IDs saved to {ids_file} (count: {len(record_ids_ordered)})")

    total_time = time.time() - start_time
    print(f"\nVectorization process finished in {total_time // 60:.0f}m {total_time % 60:.0f}s.")


if __name__ == "__main__":
    vectorize_all_records()
