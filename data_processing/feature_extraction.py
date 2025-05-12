import fasttext
import numpy as np
import os

# data_processingディレクトリ内のload_yaml_data.pyから関数をインポート
# このファイルから直接load_bibliographic_dataを呼び出すことはないが、
# 関連モジュールとして構造を明示するためにコメントで残しても良い
# from .load_yaml_data import load_bibliographic_data

FASTTEXT_JA_MODEL_PATH = "fasttext_models/cc.ja.300.bin"
FASTTEXT_EN_MODEL_PATH = "fasttext_models/cc.en.300.bin"  # 必要に応じて

# グローバル変数としてモデルを保持（初回ロード後に再利用するため）
loaded_models = {}


def load_fasttext_model(lang="ja"):
    """
    指定された言語のfastTextモデルをロードします。
    既にロードされていればキャッシュから返します。
    """
    global loaded_models
    model_path = ""
    if lang == "ja":
        model_path = FASTTEXT_JA_MODEL_PATH
    elif lang == "en":
        model_path = FASTTEXT_EN_MODEL_PATH
    else:
        raise ValueError(f"Unsupported language: {lang}. Supported: 'ja', 'en'.")

    if lang in loaded_models:
        # print(f"Returning cached fastText model for {lang}.")
        return loaded_models[lang]

    if not os.path.exists(model_path):
        print(f"Error: fastText model not found at {model_path}")
        print("Please run the download script (e.g., fasttext/init.py) first.")
        return None

    print(f"Loading fastText model for {lang} from {model_path}...")
    try:
        model = fasttext.load_model(model_path)
        loaded_models[lang] = model
        print(f"fastText model for {lang} loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading fastText model for {lang}: {e}")
        return None


def get_text_representation(record_data, fields=["bib1_title", "bib1_author"]):
    """
    レコードデータから、指定されたフィールドを結合して単一のテキスト文字列を生成します。
    """
    text_parts = []
    for field in fields:
        if field in record_data and record_data[field]:
            text_parts.append(str(record_data[field]).strip())
    return " ".join(text_parts)


def get_vector_for_record(record_id, records_map, fasttext_model, text_fields=["bib1_title", "bib1_author"]):
    """
    指定されたrecord_idの書誌レコードのベクトル表現を取得します。

    Args:
        record_id (str): ベクトルを取得するレコードのID。
        records_map (dict): 全レコードの情報を保持する辞書 (キー: record_id, バリュー: レコードデータ)。
        fasttext_model: ロード済みのfastTextモデル。
        text_fields (list): ベクトル化に使用するデータフィールドのリスト。

    Returns:
        numpy.ndarray: レコードのベクトル表現。見つからない場合はNone。
    """
    if record_id not in records_map:
        print(f"Warning: Record ID {record_id} not found in records_map.")
        return None
    if not fasttext_model:
        print("Error: fastText model is not loaded.")
        return None

    record_data = records_map[record_id].get("data", {})
    text_to_vectorize = get_text_representation(record_data, fields=text_fields)

    if not text_to_vectorize:
        # print(f"Warning: No text to vectorize for record ID {record_id} with fields {text_fields}.")
        # 全てが空の場合、ゼロベクトルを返すか、Noneを返すか選択。ここではNoneの代わりにゼロベクトルを返すことも検討。
        # Siamese Networkの入力として次元数を合わせるため、ここでは300次元のゼロベクトルを返す例も考えられる。
        # return np.zeros(fasttext_model.get_dimension())
        return None  # またはエラー処理

    return fasttext_model.get_sentence_vector(text_to_vectorize)


# --- メインのテスト処理 ---
if __name__ == "__main__":
    from .load_yaml_data import load_bibliographic_data  # テストのためにインポート

    # 1. YAMLデータから全レコード情報をロードし、records_map を作成
    yaml_path = "benchmark/bib_japan_20241024/1k/record.yml"
    print(f"Loading bibliographic data from: {yaml_path}")
    all_records_list = load_bibliographic_data(yaml_path)

    records_map = {}
    if all_records_list:
        for record in all_records_list:
            records_map[record["record_id"]] = record
        print(f"Created records_map with {len(records_map)} entries.")
    else:
        print("Could not load records to create records_map. Exiting test.")
        exit()

    # 2. fastTextモデルをロード (日本語モデルをデフォルトとする)
    ft_model_ja = load_fasttext_model(lang="ja")

    if not ft_model_ja:
        print("Failed to load Japanese fastText model. Exiting test.")
        exit()

    # 3. テスト用のレコードIDを指定してベクトルを取得してみる
    #    generate_pairs.py の出力などから実際のIDを持ってくると良い
    if all_records_list:
        # 最初のペアの最初のレコードID、または任意のIDを使用
        test_record_id_1 = all_records_list[0].get("record_id")
        # 異なるクラスタのレコードID（例として、適当にインデックスを指定）
        test_record_id_2 = None
        if len(all_records_list) > 15:  # 十分なレコードがあるかチェック
            # クラスタIDが異なる可能性が高い適当なインデックスのレコードを選ぶ
            first_cluster_id = all_records_list[0].get("cluster_id")
            for rec in all_records_list[15:]:
                if rec.get("cluster_id") != first_cluster_id:
                    test_record_id_2 = rec.get("record_id")
                    break
            if not test_record_id_2 and len(all_records_list) > 1:  # 見つからなければ2番目のレコード
                test_record_id_2 = all_records_list[1].get("record_id")
        elif len(all_records_list) > 1:
            test_record_id_2 = all_records_list[1].get("record_id")

        print(f"\n--- Testing get_vector_for_record for ID: {test_record_id_1} ---")
        vector1 = get_vector_for_record(test_record_id_1, records_map, ft_model_ja)
        if vector1 is not None:
            print(f"  Vector shape: {vector1.shape}, First 5 dims: {vector1[:5]}")
        else:
            print("  Failed to get vector.")

        if test_record_id_2:
            print(f"\n--- Testing get_vector_for_record for ID: {test_record_id_2} ---")
            vector2 = get_vector_for_record(test_record_id_2, records_map, ft_model_ja)
            if vector2 is not None:
                print(f"  Vector shape: {vector2.shape}, First 5 dims: {vector2[:5]}")
                # オプション: 2つのベクトル間のコサイン類似度を計算してみる (要 scipy)
                # from scipy.spatial.distance import cosine
                # if vector1 is not None and vector2 is not None:
                #     similarity = 1 - cosine(vector1, vector2)
                #     print(f"  Cosine similarity with vector1: {similarity:.4f}")
            else:
                print("  Failed to get vector.")

        # 存在しないIDでテスト
        print(f"\n--- Testing get_vector_for_record for a non-existent ID ---")
        vector_non_existent = get_vector_for_record("non-existent-id-123", records_map, ft_model_ja)
        if vector_non_existent is None:
            print("  Correctly returned None for non-existent ID.")
        else:
            print("  Error: Should have returned None for non-existent ID.")

    print("\nFeature extraction test finished.")
