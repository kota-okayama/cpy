#!/usr/bin/env python3
"""
書誌レコードマッチングのエンベディング類似度閾値分析ツール

このスクリプトは、エンベディング類似度の分布を分析し、
様々な閾値設定でのLLM評価削減効果をシミュレートします。
"""

import os
import sys
import yaml
import json
import argparse
import numpy as np
import pickle
import time
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Set, Union
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt

# エンベディング類似度を計算する関数
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """2つのベクトル間のコサイン類似度を計算"""
    # 異なる次元を処理（ベクトルは同じサイズであるべき）
    if len(vec1) != len(vec2):
        min_dim = min(len(vec1), len(vec2))
        vec1 = vec1[:min_dim]
        vec2 = vec2[:min_dim]
    
    # コサイン類似度の計算
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    
    if norm_a == 0 or norm_b == 0:
        return 0
    
    return dot_product / (norm_a * norm_b)

def load_yaml(filepath: str) -> Dict:
    """YAMLファイルからデータを読み込む"""
    print(f"ファイル読み込み: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data
    except Exception as e:
        print(f"エラー: ファイル読み込みに失敗しました: {e}")
        sys.exit(1)

def load_embedding_cache(cache_file: str) -> Dict[str, np.ndarray]:
    """エンベディングキャッシュを読み込む"""
    if os.path.exists(cache_file):
        try:
            print(f"エンベディングキャッシュを読み込み中: {cache_file}")
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
            print(f"{len(cache)}件のエンベディングをキャッシュから読み込みました")
            return cache
        except Exception as e:
            print(f"キャッシュ読み込みエラー: {e}")
            return {}
    return {}

def load_candidate_pairs(pairs_file: str) -> List[Tuple[str, str]]:
    """候補ペアをファイルから読み込む"""
    try:
        print(f"候補ペアを読み込み中: {pairs_file}")
        with open(pairs_file, 'r', encoding='utf-8') as f:
            pairs = []
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    pairs.append((parts[0], parts[1]))
        print(f"{len(pairs)}個の候補ペアを読み込みました")
        return pairs
    except Exception as e:
        print(f"ペア読み込みエラー: {e}")
        return []

def extract_records_from_test_books(data: Dict) -> List[Dict]:
    """test_books.yml形式からレコードを抽出する"""
    records = []
    
    if "records" in data and isinstance(data["records"], dict):
        for cluster_id, cluster_records in data["records"].items():
            if isinstance(cluster_records, list):
                for record in cluster_records:
                    if isinstance(record, dict) and "id" in record and "data" in record:
                        records.append(record)
    
    print(f"{len(records)}件のレコードを抽出しました")
    return records

def analyze_embedding_similarity_distribution(records, record_embeddings, candidate_pairs):
    """
    候補ペアのエンベディング類似度の分布を分析する
    
    Args:
        records: レコードのリスト
        record_embeddings: レコードIDからエンベディングへのマッピング
        candidate_pairs: 候補ペアのリスト
        
    Returns:
        similarity_scores: 類似度スコアのリスト
    """
    # レコードIDからレコードへのマッピングを作成
    id_to_record = {record.get("id", ""): record for record in records}
    
    # エンベディングキーとレコードIDの対応を表示
    print(f"レコードID数: {len(id_to_record)}")
    print(f"エンベディングキー数: {len(record_embeddings)}")
    
    # エンベディングキーの例を表示
    if record_embeddings:
        print("エンベディングキーの例:")
        for i, key in enumerate(list(record_embeddings.keys())[:5]):
            print(f"  {i+1}. {key}")
    
    # レコードIDの例を表示
    if id_to_record:
        print("レコードIDの例:")
        for i, key in enumerate(list(id_to_record.keys())[:5]):
            print(f"  {i+1}. {key}")
    
    similarity_scores = []
    valid_pairs = 0
    invalid_pairs = 0
    
    print(f"候補ペアの類似度分布を分析中...")
    
    for pair in tqdm(candidate_pairs, desc="類似度計算中"):
        # タプルから要素を抽出
        if isinstance(pair, tuple) and len(pair) == 2:
            record_id1, record_id2 = pair
        elif isinstance(pair, tuple) and len(pair) == 3:
            record_id1, record_id2, _ = pair
        else:
            invalid_pairs += 1
            continue
        
        # レコードとエンベディングが存在するか確認
        if record_id1 not in id_to_record or record_id2 not in id_to_record:
            invalid_pairs += 1
            continue
        
        if record_id1 not in record_embeddings or record_id2 not in record_embeddings:
            invalid_pairs += 1
            continue
        
        # エンベディング類似度を計算
        emb1 = record_embeddings[record_id1]
        emb2 = record_embeddings[record_id2]
        similarity = cosine_similarity(emb1, emb2)
        similarity_scores.append((record_id1, record_id2, similarity))
        valid_pairs += 1
    
    print(f"有効なペア: {valid_pairs}, 無効なペア: {invalid_pairs}")
    
    # 類似度でソート
    similarity_scores.sort(key=lambda x: x[2])
    
    return similarity_scores

def simulate_llm_thresholds(similarity_scores):
    """
    様々な閾値でのLLM評価数をシミュレートする
    
    Args:
        similarity_scores: (record_id1, record_id2, similarity)のタプルのリスト
    """
    total_pairs = len(similarity_scores)
    
    if total_pairs == 0:
        print("警告: 有効なペアがありません。閾値シミュレーションを実行できません。")
        return
    
    print(f"\n異なる閾値でのLLM評価数シミュレーション (全{total_pairs}ペア):")
    print("="*80)
    print("| 閾値範囲          | LLM評価ペア数 | 削減率   | 評価が必要なペアの類似度範囲           |")
    print("|-------------------|--------------|----------|--------------------------------------|")
    
    # 様々な閾値の組み合わせをテスト
    thresholds = [
        (0.0, 1.0),  # すべてのペアを評価
        (0.6, 0.95), # デフォルト設定
        (0.65, 0.9), 
        (0.7, 0.9),
        (0.7, 0.85),
        (0.75, 0.85),
        (0.8, 0.9),
        (0.7, 0.8),
        (0.75, 0.8),
        (0.8, 0.85),
        (0.85, 0.9),
    ]
    
    best_threshold = None
    best_reduction = -1
    
    for lower, upper in thresholds:
        # この閾値範囲内のペアをカウント
        llm_pairs = [s for s in similarity_scores if lower <= s[2] < upper]
        llm_count = len(llm_pairs)
        reduction = 1.0 - (llm_count / total_pairs)
        
        # より良い削減率を記録
        if reduction > best_reduction and not (lower == 0.0 and upper == 1.0):
            best_reduction = reduction
            best_threshold = (lower, upper)
        
        print(f"| {lower:.2f} - {upper:.2f}     | {llm_count:4d}/{total_pairs:4d}   | {reduction:6.2%} |                                      |")
    
    print("="*80)
    
    # 類似度分布の詳細
    print("\n類似度分布の詳細:")
    print("="*80)
    print("| 類似度範囲       | ペア数      | 割合     |")
    print("|------------------|------------|----------|")
    
    bins = [(i/10, (i+1)/10) for i in range(0, 10)]
    
    for lower, upper in bins:
        count = len([s for s in similarity_scores if lower <= s[2] < upper])
        percent = count / total_pairs if total_pairs > 0 else 0
        print(f"| {lower:.1f} - {upper:.1f}       | {count:4d}/{total_pairs:4d} | {percent:6.2%} |")
    
    print("="*80)
    
    # 最適な閾値を表示
    if best_threshold:
        lower, upper = best_threshold
        count = len([s for s in similarity_scores if lower <= s[2] < upper])
        print(f"\n最も削減効果が高い閾値: {lower:.2f} - {upper:.2f}")
        print(f"この閾値では {count} ペアを評価 (全体の {count/total_pairs:.2%})、{best_reduction:.2%} の削減効果")

def plot_similarity_distribution(similarity_scores, output_file=None):
    """
    類似度分布をヒストグラムとしてプロット
    
    Args:
        similarity_scores: 類似度スコアのリスト
        output_file: 出力ファイルパス（None の場合は表示のみ）
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("警告: matplotlib がインストールされていないため、グラフを生成できません")
        return
    
    if not similarity_scores:
        print("警告: 有効なペアがないため、グラフを生成できません")
        return
    
    # 類似度値のみを抽出
    similarities = [s[2] for s in similarity_scores]
    
    plt.figure(figsize=(12, 6))
    
    # ヒストグラム
    plt.subplot(1, 2, 1)
    bins = [i/20 for i in range(0, 21)]  # 0.0から1.0まで0.05刻み
    plt.hist(similarities, bins=bins, edgecolor='black')
    plt.title('エンベディング類似度の分布')
    plt.xlabel('類似度')
    plt.ylabel('ペア数')
    plt.grid(True, alpha=0.3)
    
    # 累積分布
    plt.subplot(1, 2, 2)
    plt.hist(similarities, bins=bins, edgecolor='black', cumulative=True, density=True)
    plt.title('類似度の累積分布')
    plt.xlabel('類似度')
    plt.ylabel('累積割合')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"類似度分布グラフを {output_file} に保存しました")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="エンベディング類似度閾値分析ツール")
    parser.add_argument("--input", "-i", type=str, required=True, help="入力YAMLファイルパス")
    parser.add_argument("--cache-dir", "-c", type=str, default=".cache", help="キャッシュディレクトリ")
    parser.add_argument("--pairs", "-p", type=str, help="候補ペアファイル（指定しない場合は全ペアを使用）")
    parser.add_argument("--output", "-o", type=str, help="分析結果の出力CSVファイル")
    parser.add_argument("--plot", "-P", type=str, help="類似度分布グラフの出力ファイル (例: distribution.png)")
    
    args = parser.parse_args()
    
    # 処理タイマーの開始
    start_time = datetime.now()
    print(f"分析開始時刻: {start_time}")
    
    # 入力データの読み込み
    data = load_yaml(args.input)
    
    # YAMLの構造を表示
    print("\nYAMLの構造:")
    if isinstance(data, dict):
        for key in data.keys():
            print(f"トップレベルキー: {key}")
            if key == "records" and isinstance(data[key], dict):
                print(f"  records内のクラスター数: {len(data[key])}")
                if data[key]:
                    first_key = next(iter(data[key]), None)
                    if first_key:
                        print(f"  最初のクラスターID: {first_key}")
                        print(f"  最初のクラスター内のレコード数: {len(data[key][first_key])}")
    
    # レコードの抽出
    records = extract_records_from_test_books(data)
    
    # エンベディングキャッシュの読み込み
    cache_file = os.path.join(args.cache_dir, "embeddings_cache.pkl")
    record_embeddings = load_embedding_cache(cache_file)
    
    # 候補ペアの準備
    if args.pairs:
        # ファイルから候補ペアを読み込む
        candidate_pairs = load_candidate_pairs(args.pairs)
    else:
        # すべてのペアを生成
        candidate_pairs = []
        record_ids = [record.get("id", "") for record in records]
        for i in range(len(record_ids)):
            for j in range(i + 1, len(record_ids)):
                candidate_pairs.append((record_ids[i], record_ids[j]))
        print(f"すべての組み合わせで {len(candidate_pairs)} 個の候補ペアを生成しました")
    
    # 類似度分布の分析
    similarity_scores = analyze_embedding_similarity_distribution(records, record_embeddings, candidate_pairs)
    
    # 閾値シミュレーション
    simulate_llm_thresholds(similarity_scores)
    
    # グラフ生成
    if args.plot:
        plot_similarity_distribution(similarity_scores, args.plot)
    
    # 終了時刻とサマリー
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    print(f"分析完了時刻: {end_time}")
    print(f"合計処理時間: {elapsed:.2f}秒")

if __name__ == "__main__":
    main()