import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys

# プロジェクトルートをPythonパスに追加 (data_processingをインポートするため)
# このファイルが siamese_model_pytorch ディレクトリにあると仮定
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from data_processing.load_yaml_data import load_bibliographic_data
from data_processing.generate_pairs import generate_training_pairs
from data_processing.feature_extraction import load_fasttext_model, get_vector_for_record, FASTTEXT_JA_MODEL_PATH


class BibliographicPairDataset(Dataset):
    """
    書誌レコードのペアとラベルを扱うPyTorch Dataset。
    """

    def __init__(self, pairs_list, records_map, fasttext_model, text_fields=["bib1_title", "bib1_author"]):
        """
        Args:
            pairs_list (list): (record_id1, record_id2, label) のタプルのリスト。
            records_map (dict): 全レコードの情報を保持する辞書 (キー: record_id)。
            fasttext_model: ロード済みのfastTextモデル。
            text_fields (list): ベクトル化に使用するデータフィールドのリスト。
        """
        self.pairs_list = pairs_list
        self.records_map = records_map
        self.fasttext_model = fasttext_model
        self.text_fields = text_fields

        # 事前にベクトル化してキャッシュすることも可能だが、メモリ使用量が増えるため
        # ここでは __getitem__ 内で都度ベクトル化する（状況に応じて変更可）

        # 有効なペアのみをフィルタリングする（ベクトル化できないペアを除外）
        self.valid_pairs = []
        print("Filtering valid pairs for dataset...")
        for i, (id1, id2, label) in enumerate(self.pairs_list):
            vec1_temp = get_vector_for_record(id1, self.records_map, self.fasttext_model, self.text_fields)
            vec2_temp = get_vector_for_record(id2, self.records_map, self.fasttext_model, self.text_fields)
            if vec1_temp is not None and vec2_temp is not None:
                self.valid_pairs.append((id1, id2, label))
            if (i + 1) % 5000 == 0:  # 進捗表示
                print(f"  Processed {i+1}/{len(self.pairs_list)} pairs for validation check...")
        print(f"Original pairs: {len(self.pairs_list)}, Valid pairs after vectorization check: {len(self.valid_pairs)}")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        record_id1, record_id2, label = self.valid_pairs[idx]

        # レコードIDからベクトルを取得
        vector1 = get_vector_for_record(record_id1, self.records_map, self.fasttext_model, self.text_fields)
        vector2 = get_vector_for_record(record_id2, self.records_map, self.fasttext_model, self.text_fields)

        # get_vector_for_record が None を返さない前提 (コンストラクタでフィルタリング済み)
        # もしNoneの可能性がある場合はここで再度チェックと対応が必要
        if vector1 is None or vector2 is None:
            # この状況はコンストラクタのフィルタリングにより基本的には発生しないはず
            # もし発生した場合のフォールバック処理 (例: ゼロベクトルやエラー)
            print(f"Error: Vector is None for pair at index {idx} ({record_id1}, {record_id2}). This shouldn't happen.")
            # 仮にゼロベクトルで埋める場合 (次元数はfastTextモデルから取得)
            dim = self.fasttext_model.get_dimension()
            vector1 = np.zeros(dim) if vector1 is None else vector1
            vector2 = np.zeros(dim) if vector2 is None else vector2

        # NumPy配列をPyTorchテンソルに変換
        tensor1 = torch.from_numpy(vector1.astype(np.float32))
        tensor2 = torch.from_numpy(vector2.astype(np.float32))
        label_tensor = torch.tensor(label, dtype=torch.float32)  # ContrastiveLossのlabelはfloatを期待することが多い

        return tensor1, tensor2, label_tensor


# --- メインのテスト処理 (動作確認用) ---
if __name__ == "__main__":
    print("--- Testing BibliographicPairDataset and DataLoader ---")

    # 1. データの準備 (data_processing モジュールから)
    yaml_path = "benchmark/bib_japan_20241024/1k/record.yml"
    print(f"Loading bibliographic data from: {yaml_path}")
    all_records_list = load_bibliographic_data(yaml_path)

    records_map = {}
    if all_records_list:
        for record in all_records_list:
            records_map[record["record_id"]] = record
        print(f"Created records_map with {len(records_map)} entries.")
    else:
        print("Could not load records to create records_map. Exiting.")
        exit()

    print("\nGenerating training pairs...")
    # サンプリングは generate_training_pairs 内で行われる想定
    positive_pairs, negative_pairs = generate_training_pairs(all_records_list)
    all_pairs = positive_pairs + negative_pairs
    # テストのため、ペア数を制限（例: 最初の100ペア）
    # all_pairs = all_pairs[:100]
    # print(f"Using a subset of {len(all_pairs)} pairs for testing the dataset.")
    if not all_pairs:
        print("No pairs generated. Exiting.")
        exit()
    print(f"Total pairs generated (before dataset filtering): {len(all_pairs)}")

    # 2. fastTextモデルのロード
    #    feature_extraction.py の load_fasttext_model を使用
    #    事前に fasttext_models/cc.ja.300.bin がダウンロードされている必要あり
    print("\nLoading fastText model...")
    ft_model = load_fasttext_model(lang="ja")
    if not ft_model:
        print("Failed to load fastText model. Exiting.")
        exit()

    # 3. BibliographicPairDatasetのインスタンス化
    print("\nInstantiating BibliographicPairDataset...")
    # text_fields は get_vector_for_record のデフォルト値を使用
    dataset = BibliographicPairDataset(all_pairs, records_map, ft_model)

    if len(dataset) == 0:
        print("Dataset is empty after filtering. Check vectorization or pair generation. Exiting.")
        exit()
    print(f"Dataset size: {len(dataset)} (after filtering valid pairs)")

    # 4. DataLoaderのテスト
    batch_size = 4  # 小さなバッチサイズでテスト
    # shuffle=True は通常訓練時に使用するが、テストではFalseでも可
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"\nTesting DataLoader with batch_size = {batch_size}...")
    try:
        # 最初の1バッチを取得して形状などを確認
        for i, (vec1_batch, vec2_batch, label_batch) in enumerate(dataloader):
            print(f"Batch {i+1}:")
            print(f"  Vector1 batch shape: {vec1_batch.shape}")  # 期待: (batch_size, embedding_dim)
            print(f"  Vector2 batch shape: {vec2_batch.shape}")  # 期待: (batch_size, embedding_dim)
            print(f"  Label batch shape: {label_batch.shape}")  # 期待: (batch_size)
            print(f"  Labels in batch: {label_batch}")
            if i == 0:  # 最初のバッチのみ表示して終了
                break
        print("DataLoader test passed (at least one batch processed).")
    except Exception as e:
        print(f"Error during DataLoader iteration: {e}")
        import traceback

        traceback.print_exc()

    print("\nDataset and DataLoader test finished.")
