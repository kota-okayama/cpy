import json
import os
import csv
import time
import sys

# プロジェクトルートをPythonパスに追加 (他モジュールをインポートするため)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- 設定値 ---
KNN_GRAPH_PATH = "siamese_model_pytorch/knn_graph"
KNN_GRAPH_FILENAME = "knn_graph_k10.json"  # build_knn_graph.py で K=10 としたファイル名
OUTPUT_PAIRS_PATH = "siamese_model_pytorch/llm_evaluation_pairs"
OUTPUT_PAIRS_FILENAME = "evaluation_candidate_pairs.csv"


def extract_unique_pairs_from_knn_graph(knn_graph_path):
    """
    K近傍グラフからユニークなレコードIDのペアを抽出する。

    Args:
        knn_graph_path (str): K近傍グラフのJSONファイルパス。

    Returns:
        set: ユニークなレコードIDのペアのセット。各ペアは (id1, id2) のタプルで、id1 < id2 となっている。
    """
    unique_pairs = set()
    try:
        with open(knn_graph_path, "r", encoding="utf-8") as f:
            knn_graph = json.load(f)
    except FileNotFoundError:
        print(f"エラー: K近傍グラフファイルが見つかりません: {knn_graph_path}")
        return unique_pairs
    except json.JSONDecodeError:
        print(f"エラー: K近傍グラフファイルのJSON形式が正しくありません: {knn_graph_path}")
        return unique_pairs

    for record_id, neighbors in knn_graph.items():
        for neighbor_id in neighbors:
            # 自分自身とのペアは除外
            if record_id == neighbor_id:
                continue
            # ペアを正規化 (id1, id2) where id1 < id2
            pair = tuple(sorted((record_id, neighbor_id)))
            unique_pairs.add(pair)

    return unique_pairs


def save_pairs_to_csv(pairs, output_csv_path):
    """
    レコードIDのペアをCSVファイルに保存する。

    Args:
        pairs (set): 保存するレコードIDのペアのセット。
        output_csv_path (str): 出力するCSVファイルのパス。
    """
    count = 0
    try:
        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["record_id_1", "record_id_2"])  # ヘッダー
            for pair in sorted(list(pairs)):  # 出力順序を安定させるためにソート
                writer.writerow(pair)
                count += 1
        print(f"{count} ペアを {output_csv_path} に保存しました。")
    except IOError:
        print(f"エラー: CSVファイルへの書き込みに失敗しました: {output_csv_path}")


if __name__ == "__main__":
    # --- 設定 ---
    # K近傍グラフのパス (build_knn_graph.pyの出力)
    KNN_GRAPH_DIR = "knn_graph"
    KNN_GRAPH_FILENAME = "knn_graph_k10.json"

    # 出力CSVファイルのパスとファイル名
    OUTPUT_CSV_DIR = "."  # スクリプトと同じディレクトリに保存
    OUTPUT_CSV_FILENAME = "evaluation_candidate_pairs.csv"
    # --- 設定ここまで ---

    # ファイルパスの構築
    # スクリプトのディレクトリを基準にする
    script_dir = os.path.dirname(os.path.abspath(__file__))
    knn_graph_path = os.path.join(script_dir, KNN_GRAPH_DIR, KNN_GRAPH_FILENAME)

    # 出力ディレクトリが存在しない場合は作成
    output_dir_full_path = os.path.join(script_dir, OUTPUT_CSV_DIR)
    if not os.path.exists(output_dir_full_path):
        os.makedirs(output_dir_full_path)
        print(f"出力ディレクトリを作成しました: {output_dir_full_path}")

    output_csv_path = os.path.join(output_dir_full_path, OUTPUT_CSV_FILENAME)

    print(f"K近傍グラフファイル: {knn_graph_path}")
    print(f"出力CSVファイル: {output_csv_path}")

    # ペアの抽出
    candidate_pairs = extract_unique_pairs_from_knn_graph(knn_graph_path)

    if candidate_pairs:
        # CSVに保存
        save_pairs_to_csv(candidate_pairs, output_csv_path)
    else:
        print("処理対象のペアが見つかりませんでした。")

    print("処理完了。")
