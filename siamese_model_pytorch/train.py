import torch
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import os
import sys
import time

# プロジェクトルートをPythonパスに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from data_processing.load_yaml_data import load_bibliographic_data
from data_processing.generate_pairs import generate_training_pairs
from data_processing.feature_extraction import load_fasttext_model  # records_mapはここで使わない

from .network import BaseNetwork, SiameseNetwork, ContrastiveLoss
from .dataset import BibliographicPairDataset

# --- ハイパーパラメータ ---
LEARNING_RATE = 0.001
BATCH_SIZE = 64  # メモリに応じて調整
NUM_EPOCHS = 10  # まずは少ないエポック数でテスト
EMBEDDING_DIM = 128  # BaseNetworkの出力次元
DROPOUT_RATE = 0.3
CONTRASTIVE_MARGIN = 1.0
FASTTEXT_LANG = "ja"
# YAML_PATH = 'benchmark/bib_japan_20241024/1k/record.yml' # メイン関数内で定義
MODEL_SAVE_PATH = "siamese_model_pytorch/saved_models"


def train():
    print("Starting training process...")
    start_time = time.time()

    # --- デバイス設定 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- データの準備 ---
    print("\nStep 1: Loading and preparing data...")
    yaml_path = "benchmark/bib_japan_20241024/1k/record.yml"
    all_records_list = load_bibliographic_data(yaml_path)
    if not all_records_list:
        print("Failed to load bibliographic records. Exiting.")
        return

    records_map = {record["record_id"]: record for record in all_records_list}

    positive_pairs, negative_pairs = generate_training_pairs(all_records_list)
    all_pairs = positive_pairs + negative_pairs
    if not all_pairs:
        print("No training pairs generated. Exiting.")
        return
    print(f"Total pairs for dataset: {len(all_pairs)}")

    ft_model = load_fasttext_model(lang=FASTTEXT_LANG)
    if not ft_model:
        print("Failed to load fastText model. Exiting.")
        return

    full_dataset = BibliographicPairDataset(all_pairs, records_map, ft_model)
    if len(full_dataset) == 0:
        print("Dataset is empty after filtering. Cannot train. Exiting.")
        return
    print(f"Full dataset size (after filtering): {len(full_dataset)}")

    # TODO: 訓練データと検証データに分割 (今回は全データで訓練)
    # train_size = int(0.8 * len(full_dataset))
    # val_size = len(full_dataset) - train_size
    # indices = list(range(len(full_dataset)))
    # np.random.shuffle(indices)
    # train_indices, val_indices = indices[:train_size], indices[train_size:]
    # train_sampler = SubsetRandomSampler(train_indices)
    # val_sampler = SubsetRandomSampler(val_indices)
    # train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    # val_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

    # 今回は全データで訓練
    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"DataLoader created with batch size {BATCH_SIZE}.")

    # --- モデル、損失関数、オプティマイザの初期化 ---
    print("\nStep 2: Initializing model, loss function, and optimizer...")
    base_network = BaseNetwork(
        input_dim=ft_model.get_dimension(), embedding_dim=EMBEDDING_DIM, dropout_rate=DROPOUT_RATE
    ).to(device)
    siamese_model = SiameseNetwork(base_network).to(device)
    criterion = ContrastiveLoss(margin=CONTRASTIVE_MARGIN).to(device)
    optimizer = optim.Adam(siamese_model.parameters(), lr=LEARNING_RATE)

    print("Model, loss, and optimizer initialized.")

    # --- 学習ループ ---
    print("\nStep 3: Starting training loop...")
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        siamese_model.train()  # モデルを訓練モードに設定
        running_loss = 0.0
        processed_batches = 0

        for i, (input1, input2, label) in enumerate(train_loader):
            input1, input2, label = input1.to(device), input2.to(device), label.to(device)

            optimizer.zero_grad()

            output1, output2 = siamese_model(input1, input2)
            loss = criterion(output1, output2, label)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            processed_batches += 1

            if (i + 1) % 20 == 0:  # 20バッチごとに進捗を表示
                avg_batch_loss = loss.item()  # 現在のバッチのロス
                print(
                    f"  Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{i+1}/{len(train_loader)}], Current Batch Loss: {avg_batch_loss:.4f}"
                )

        epoch_loss = running_loss / processed_batches if processed_batches > 0 else 0
        epoch_end_time = time.time()
        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] completed. Average Training Loss: {epoch_loss:.4f}. Time: {epoch_end_time - epoch_start_time:.2f}s"
        )

        # TODO: ここで検証ループを実行し、検証ロスや精度を計算する

    # --- モデルの保存 ---
    print("\nStep 4: Saving the trained model...")
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    # BaseNetworkの重みのみを保存することが多い (SiameseNetworkはラッパーなので)
    model_filename = f"base_network_emb{EMBEDDING_DIM}_epoch{NUM_EPOCHS}.pth"
    save_path = os.path.join(MODEL_SAVE_PATH, model_filename)
    torch.save(base_network.state_dict(), save_path)
    print(f"Trained BaseNetwork model saved to {save_path}")

    total_time = time.time() - start_time
    print(f"\nTraining finished in {total_time // 60:.0f}m {total_time % 60:.0f}s.")


if __name__ == "__main__":
    train()
