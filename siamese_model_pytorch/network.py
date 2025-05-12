import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseNetwork(nn.Module):
    """
    Siamese Networkの基本的な構成要素となるネットワーク。
    入力ベクトルを受け取り、埋め込みベクトルを出力します。
    """

    def __init__(self, input_dim=300, embedding_dim=128, dropout_rate=0.3):
        super(BaseNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, embedding_dim)
        # 最後の層の後に活性化関数を挟むか、あるいは正規化 (L2 norm) を行うかは設計によります。
        # ここではシンプルに線形層の出力とします。

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        # オプション: 出力ベクトルをL2正規化することもあります。
        # x = F.normalize(x, p=2, dim=1)
        return x


class SiameseNetwork(nn.Module):
    """
    Siamese Network 本体。
    2つの入力を受け取り、それぞれをBaseNetworkに通して埋め込みベクトルペアを出力します。
    """

    def __init__(self, base_network):
        super(SiameseNetwork, self).__init__()
        self.base_network = base_network

    def forward(self, input1, input2):
        output1 = self.base_network(input1)
        output2 = self.base_network(input2)
        return output1, output2


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss 関数。
    L = label * distance^2 + (1 - label) * max(0, margin - distance)^2
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9  # ゼロ除算や不安定な勾配を防ぐための微小値

    def forward(self, output1, output2, label):
        # output1 と output2 は (batch_size, embedding_dim) の形状を想定
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=False, eps=self.eps)

        # label が 1 (類似) の場合: loss = distance^2
        # label が 0 (非類似) の場合: loss = max(0, margin - distance)^2
        loss_contrastive = torch.mean(
            (label.float() * torch.pow(euclidean_distance, 2))
            + ((1 - label).float() * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        )
        return loss_contrastive


# --- メインのテスト処理 (動作確認用) ---
if __name__ == "__main__":
    # 1. BaseNetworkのテスト
    print("--- Testing BaseNetwork ---")
    base_net = BaseNetwork(input_dim=300, embedding_dim=64)
    dummy_input = torch.randn(10, 300)  # バッチサイズ10, 入力次元300
    embedding_output = base_net(dummy_input)
    print(f"BaseNetwork input shape: {dummy_input.shape}")
    print(f"BaseNetwork output (embedding) shape: {embedding_output.shape}")  # 期待: (10, 64)
    assert embedding_output.shape == (10, 64)
    print("BaseNetwork test passed.")

    # 2. SiameseNetworkのテスト
    print("\n--- Testing SiameseNetwork ---")
    siamese_net = SiameseNetwork(base_net)  # BaseNetworkを共有
    dummy_input1 = torch.randn(10, 300)
    dummy_input2 = torch.randn(10, 300)
    output1, output2 = siamese_net(dummy_input1, dummy_input2)
    print(f"SiameseNetwork input1 shape: {dummy_input1.shape}")
    print(f"SiameseNetwork input2 shape: {dummy_input2.shape}")
    print(f"SiameseNetwork output1 shape: {output1.shape}")  # 期待: (10, 64)
    print(f"SiameseNetwork output2 shape: {output2.shape}")  # 期待: (10, 64)
    assert output1.shape == (10, 64)
    assert output2.shape == (10, 64)
    # 重みが共有されているか簡易的に確認 (勾配計算前なので出力は異なるが、片方の重みを変えれば両方影響するはず)
    # より厳密には、学習後にパラメータを比較するなど
    print("SiameseNetwork test passed.")

    # 3. ContrastiveLossのテスト
    print("\n--- Testing ContrastiveLoss ---")
    loss_fn = ContrastiveLoss(margin=1.0)
    # 仮の埋め込みベクトル (バッチサイズ10, 埋め込み次元64)
    emb1_sim = torch.randn(5, 64)  # 類似ペアの前半5つ
    emb2_sim = emb1_sim + torch.randn(5, 64) * 0.1  # 類似ペアなので近いベクトル
    emb1_dis = torch.randn(5, 64)  # 非類似ペアの後半5つ
    emb2_dis = torch.randn(5, 64) * 5  # 非類似ペアなので遠いベクトル (原点から)

    test_output1 = torch.cat([emb1_sim, emb1_dis], dim=0)
    test_output2 = torch.cat([emb2_sim, emb2_dis], dim=0)
    # ラベル: 最初の5つは類似(1), 次の5つは非類似(0)
    labels = torch.tensor([1] * 5 + [0] * 5, dtype=torch.int)

    loss = loss_fn(test_output1, test_output2, labels)
    print(f"Test embeddings output1 shape: {test_output1.shape}")
    print(f"Test embeddings output2 shape: {test_output2.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Calculated Contrastive Loss: {loss.item()}")
    assert loss.item() >= 0
    print("ContrastiveLoss test passed.")

    print("\nAll PyTorch network component tests finished.")
