#!/usr/bin/env python3
"""
類似度キャッシュ分析ツール

このスクリプトは書誌レコードマッチングで生成された類似度キャッシュを分析し、
キャッシュされた類似度情報をCSVファイルに出力します。
"""

import os
import sys
import pickle
import csv
import argparse
import hashlib
from typing import Dict, List, Tuple

def load_similarity_cache(cache_file: str) -> Dict[str, float]:
    """類似度キャッシュを読み込む"""
    if os.path.exists(cache_file):
        try:
            print(f"類似度キャッシュを読み込み中: {cache_file}")
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
            print(f"{len(cache)}件の類似度をキャッシュから読み込みました")
            return cache
        except Exception as e:
            print(f"キャッシュ読み込みエラー: {e}")
            return {}
    print(f"キャッシュファイルが見つかりません: {cache_file}")
    return {}

def decode_cache_key(cache_key: str) -> Tuple[str, str]:
    """キャッシュキーからレコードテキストを復元（部分的な復元のみ可能）"""
    # 注意: 完全な復元は不可能ですが、キーの形式からいくつかの情報を推測
    # 実際にはハッシュ関数は一方向なので完全な復元はできません
    return ("レコード1（ハッシュ値から復元不可）", "レコード2（ハッシュ値から復元不可）")

def extract_title_author(text: str) -> Tuple[str, str]:
    """テキストからタイトルと著者を抽出"""
    title = ""
    author = ""
    
    lines = text.split("\n")
    for line in lines:
        if line.startswith("タイトル:"):
            title = line.replace("タイトル:", "").strip()
        elif line.startswith("著者:"):
            author = line.replace("著者:", "").strip()
    
    return title, author

def export_similarity_cache(cache: Dict[str, float], output_file: str):
    """類似度キャッシュをCSVに出力"""
    # 出力データの準備
    output_data = []
    
    for i, (cache_key, similarity) in enumerate(cache.items()):
        # 類似度情報を収集
        record1, record2 = decode_cache_key(cache_key)
        
        info = {
            "index": i + 1,
            "cache_key": cache_key,
            "similarity": similarity,
            "record1": record1,
            "record2": record2
        }
        
        output_data.append(info)
    
    # 類似度で降順ソート
    output_data.sort(key=lambda x: x["similarity"], reverse=True)
    
    # CSVにエクスポート
    fieldnames = ["index", "cache_key", "similarity", "record1", "record2"]
    
    print(f"{len(output_data)}件の類似度情報をCSVに出力します: {output_file}")
    
    try:
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for info in output_data:
                writer.writerow(info)
        print(f"類似度情報を正常に保存しました: {output_file}")
    except Exception as e:
        print(f"エラー: 類似度情報の保存に失敗しました: {e}")

def analyze_similarity_distribution(cache: Dict[str, float], output_file: str):
    """類似度の分布を分析"""
    # 類似度の分布を計算
    ranges = {
        "0.0-0.1": 0,
        "0.1-0.2": 0,
        "0.2-0.3": 0,
        "0.3-0.4": 0,
        "0.4-0.5": 0,
        "0.5-0.6": 0,
        "0.6-0.7": 0,
        "0.7-0.8": 0,
        "0.8-0.9": 0,
        "0.9-1.0": 0
    }
    
    total_pairs = len(cache)
    sum_similarity = sum(cache.values())
    
    # 分布の計算
    for similarity in cache.values():
        range_key = f"{int(similarity * 10) / 10:.1f}-{int(similarity * 10 + 1) / 10:.1f}"
        if range_key in ranges:
            ranges[range_key] += 1
    
    # CSVにエクスポート
    output_data = []
    for range_key, count in ranges.items():
        percentage = count / total_pairs * 100 if total_pairs > 0 else 0
        info = {
            "similarity_range": range_key,
            "count": count,
            "percentage": f"{percentage:.2f}%"
        }
        output_data.append(info)
    
    # 統計情報を追加
    avg_similarity = sum_similarity / total_pairs if total_pairs > 0 else 0
    
    # CSV出力
    fieldnames = ["similarity_range", "count", "percentage"]
    
    print(f"類似度分布を分析しCSVに出力します: {output_file}")
    
    try:
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for info in output_data:
                writer.writerow(info)
            
            # 空行を挿入
            writer.writerow({"similarity_range": "", "count": "", "percentage": ""})
            
            # 統計情報を追加
            writer.writerow({"similarity_range": "総ペア数", "count": total_pairs, "percentage": "100.00%"})
            writer.writerow({"similarity_range": "平均類似度", "count": f"{avg_similarity:.4f}", "percentage": ""})
        
        print(f"類似度分布を正常に保存しました: {output_file}")
    except Exception as e:
        print(f"エラー: 類似度分布の保存に失敗しました: {e}")

def main():
    parser = argparse.ArgumentParser(description="類似度キャッシュの分析")
    parser.add_argument("--cache-file", "-c", type=str, default=".cache/similarity_cache.pkl", 
                       help="類似度キャッシュファイルのパス")
    parser.add_argument("--output-dir", "-o", type=str, default="./results", 
                       help="出力ディレクトリ")
    
    args = parser.parse_args()
    
    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # キャッシュの読み込み
    cache = load_similarity_cache(args.cache_file)
    
    # キャッシュが空の場合は終了
    if not cache:
        print("有効なキャッシュデータがありません。処理を終了します。")
        sys.exit(1)
    
    # 出力ファイルパス
    similarities_file = os.path.join(args.output_dir, "similarities.csv")
    distribution_file = os.path.join(args.output_dir, "similarity_distribution.csv")
    
    # キャッシュデータのエクスポート
    export_similarity_cache(cache, similarities_file)
    analyze_similarity_distribution(cache, distribution_file)
    
    print(f"\n全ての分析結果を {args.output_dir} ディレクトリに出力しました")

if __name__ == "__main__":
    main()
