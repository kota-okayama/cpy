#!/usr/bin/env python3
"""
ブロッキングコンポーネントを独立して実行するスクリプト
"""

import os
import sys
import yaml
import argparse
import numpy as np
import pickle
from datetime import datetime
import json
from typing import Dict, List

# ブロッキングコンポーネントをインポート
from blocking import BlockingComponent

# 定数
CACHE_DIR = ".cache"

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

def save_yaml(data: Dict, filepath: str):
    """データをYAMLファイルに保存"""
    print(f"ファイル保存: {filepath}")
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
        print(f"正常に保存されました: {filepath}")
    except Exception as e:
        print(f"エラー: ファイル保存に失敗しました: {e}")
        sys.exit(1)

def extract_records(data: Dict) -> List[Dict]:
    """入力データからレコードを抽出"""
    records = []
    
    # データ構造からレコードを抽出
    if "group" in data and isinstance(data["group"], list):
        for group_idx, group in enumerate(data["group"]):
            if "records" in group and isinstance(group["records"], list):
                for record_idx, record in enumerate(group["records"]):
                    records.append(record)
    elif "records" in data:
        if isinstance(data["records"], list):
            records = data["records"]
        elif isinstance(data["records"], dict):
            for cluster_id, cluster_records in data["records"].items():
                if isinstance(cluster_records, list):
                    records.extend(cluster_records)
                elif isinstance(cluster_records, dict):
                    records.append(cluster_records)
    
    print(f"{len(records)}件のレコードを抽出しました")
    return records

def analyze_record_fields(records: List[Dict]) -> Dict[str, str]:
    """レコードフィールドを分析してその型を決定する"""
    field_types = {}
    
    for record in records:
        if "data" in record and isinstance(record["data"], dict):
            for field, value in record["data"].items():
                if field not in field_types:
                    # フィールド名と内容に基づいて型を決定
                    if "title" in field.lower():
                        field_types[field] = "TEXT"
                    elif "author" in field.lower():
                        field_types[field] = "COMPLEMENT_JA"
                    elif "pubdate" in field.lower() or "date" in field.lower():
                        field_types[field] = "DATE"
                    elif "publisher" in field.lower():
                        field_types[field] = "TEXT"
                    else:
                        field_types[field] = "TEXT"
    
    return field_types

def load_embeddings(embeddings_file: str) -> Dict[str, np.ndarray]:
    """エンベディングファイルを読み込む"""
    print(f"エンベディングを読み込み中: {embeddings_file}")
    try:
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"{len(embeddings)}件のエンベディングを読み込みました")
        return embeddings
    except Exception as e:
        print(f"エンベディング読み込みエラー: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(description="ブロッキングコンポーネントスタンドアロン実行")
    parser.add_argument("--input", "-i", type=str, required=True, help="入力YAMLファイルパス")
    parser.add_argument("--output", "-o", type=str, default="candidate_pairs.json", help="出力ファイルパス")
    parser.add_argument("--embeddings", "-e", type=str, required=True, help="エンベディングファイルパス")
    parser.add_argument("--method", "-m", type=str, default="embedding", 
                       choices=["embedding", "lsh", "minihash"], help="ブロッキング方法")
    parser.add_argument("--threshold", "-t", type=float, default=0.75, help="類似度しきい値")
    parser.add_argument("--cache-dir", "-c", type=str, default=CACHE_DIR, help="キャッシュディレクトリ")
    
    args = parser.parse_args()
    
    # 処理タイマーの開始
    start_time = datetime.now()
    print(f"処理開始時刻: {start_time}")
    
    # 入力データの読み込み
    data = load_yaml(args.input)
    
    # レコードの抽出
    records = extract_records(data)
    
    # レコードフィールドの分析
    field_types = analyze_record_fields(records)
    print(f"{len(field_types)}種類のフィールドを分析しました:")
    for field, field_type in field_types.items():
        print(f"  {field}: {field_type}")
    
    # エンベディングの読み込み
    record_embeddings = load_embeddings(args.embeddings)
    
    # ブロッキングコンポーネントの初期化
    blocker = BlockingComponent(method=args.method, cache_dir=args.cache_dir)
    
    # インデックスの構築
    blocker.build_index(record_embeddings, records, field_types)
    
    # 候補ペアの生成
    candidate_pairs = blocker.find_candidate_pairs(threshold=args.threshold)
    
    # 統計情報の表示
    print(f"ブロッキングにより {len(candidate_pairs)} 個の候補ペアが生成されました")
    print(f"全ペア比較の場合: {len(records) * (len(records) - 1) // 2} ペア")
    print(f"削減率: {1.0 - len(candidate_pairs) / (len(records) * (len(records) - 1) // 2):.2%}")
    
    # 結果を保存
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "method": args.method,
        "threshold": args.threshold,
        "total_records": len(records),
        "total_possible_pairs": len(records) * (len(records) - 1) // 2,
        "candidate_pairs": candidate_pairs
    }
    
    # JSON形式で保存
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"候補ペアを保存しました: {args.output}")
    
    # 終了時刻とサマリー
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    print(f"処理完了時刻: {end_time}")
    print(f"合計処理時間: {elapsed:.2f}秒")

if __name__ == "__main__":
    main()