import itertools
from collections import defaultdict
import random  # For potential sampling later

# data_processingディレクトリ内のload_yaml_data.pyから関数をインポート
from .load_yaml_data import load_bibliographic_data


def generate_training_pairs(records_list):
    """
    書誌レコードのリストから、学習用の類似ペアと非類似ペアを生成します。

    Args:
        records_list (list): load_bibliographic_dataから返されるレコードのリスト。

    Returns:
        tuple: (positive_pairs, negative_pairs)
               各ペアは (record_id_1, record_id_2, label) のタプルです。
    """
    positive_pairs = []
    negative_pairs = []

    if not records_list:
        return positive_pairs, negative_pairs

    # cluster_idごとにレコードをグループ化
    clusters = defaultdict(list)
    for record in records_list:
        clusters[record["cluster_id"]].append(record)

    # 類似ペアの生成
    for cluster_id, records_in_cluster in clusters.items():
        if len(records_in_cluster) > 1:
            # 同一クラスタ内でペアを作成
            for r1, r2 in itertools.combinations(records_in_cluster, 2):
                positive_pairs.append((r1["record_id"], r2["record_id"], 1))  # ラベル1: 類似

    # 非類似ペアの生成
    # 全てのレコードのペアを調べ、クラスタIDが異なるものを非類似ペアとする
    # 注意: レコード数が多い場合、この処理は時間がかかり、非常に多くのペアを生成する可能性があります。
    for i in range(len(records_list)):
        for j in range(i + 1, len(records_list)):  # (i, j)のペアが重複しないように
            record1 = records_list[i]
            record2 = records_list[j]
            if record1["cluster_id"] != record2["cluster_id"]:
                negative_pairs.append((record1["record_id"], record2["record_id"], 0))  # ラベル0: 非類似

    # 非類似ペアのサンプリング (類似ペアと同数にする)
    if negative_pairs and len(negative_pairs) > len(positive_pairs):
        print(f"  Sampling negative pairs: from {len(negative_pairs)} down to {len(positive_pairs)}")
        negative_pairs = random.sample(negative_pairs, len(positive_pairs))

    return positive_pairs, negative_pairs


if __name__ == "__main__":
    # YAMLファイルのパス (ワークスペースルートからの相対パス)
    yaml_path = "benchmark/bib_japan_20241024/1k/record.yml"

    print(f"Loading bibliographic data from: {yaml_path}")
    bibliographic_records = load_bibliographic_data(yaml_path)

    if bibliographic_records:
        print(f"Successfully loaded {len(bibliographic_records)} records.")

        print("\nGenerating training pairs...")
        positive_pairs, negative_pairs = generate_training_pairs(bibliographic_records)

        num_positive = len(positive_pairs)
        num_negative = len(negative_pairs)

        print(f"\nGenerated {num_positive} positive pairs.")
        if positive_pairs:
            print("First 5 positive pairs:")
            for pair in positive_pairs[:5]:
                print(f"  {pair}")

        print(f"\nGenerated {num_negative} negative pairs.")
        if negative_pairs:
            print("First 5 negative pairs:")
            for pair in negative_pairs[:5]:
                print(f"  {pair}")

    else:
        print("Failed to load records. Cannot generate pairs.")
