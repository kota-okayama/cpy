#!/usr/bin/env python3
"""
レコード分析ツール
YAMLファイル内のレコードを読み込み、クラスター構造と内容を分析・表示する
"""

import sys
import yaml
import argparse
from typing import Dict, List
from difflib import SequenceMatcher

def load_yaml(filepath: str):
    """YAMLファイルを読み込む"""
    print(f"ファイル読み込み: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data
    except Exception as e:
        print(f"エラー: ファイル読み込みに失敗しました: {e}")
        sys.exit(1)

def analyze_record_structure(data: Dict):
    """YAMLファイルの基本構造を分析して表示"""
    print("=== ファイル構造分析 ===")
    
    if not isinstance(data, dict):
        print(f"データ型: {type(data)}")
        print("YAMLファイルはdictionary形式ではありません。")
        return
    
    # トップレベルのキーを表示
    print(f"トップレベルのキー: {', '.join(data.keys())}")
    
    # inf_attr（属性フィールド）を表示
    if "inf_attr" in data:
        print("\n=== 属性フィールド ===")
        for field, field_type in data["inf_attr"].items():
            print(f"{field}: {field_type}")
    
    # レコード構造を確認
    if "records" in data:
        records_data = data["records"]
        
        if isinstance(records_data, dict):
            print("\n=== クラスター構造 ===")
            cluster_count = len(records_data)
            print(f"クラスター数: {cluster_count}")
            
            # 各クラスターのレコード数
            total_records = 0
            record_counts = []
            
            for cluster_id, records in records_data.items():
                if isinstance(records, list):
                    record_count = len(records)
                    total_records += record_count
                    record_counts.append(record_count)
                elif isinstance(records, dict):
                    total_records += 1
                    record_counts.append(1)
            
            print(f"総レコード数: {total_records}")
            
            if record_counts:
                print(f"1クラスターあたりの平均レコード数: {sum(record_counts) / len(record_counts):.2f}")
                print(f"最小レコード数: {min(record_counts)}")
                print(f"最大レコード数: {max(record_counts)}")
        
        elif isinstance(records_data, list):
            print("\n=== レコードリスト ===")
            print(f"レコード数: {len(records_data)}")
        
        else:
            print("\n=== 未知のレコード構造 ===")
            print(f"records フィールドの型: {type(records_data)}")

def show_cluster(data: Dict, cluster_id: str, show_details: bool = True):
    """指定されたクラスターの詳細を表示"""
    if "records" not in data or not isinstance(data["records"], dict):
        print("クラスター構造が見つかりません。")
        return
    
    if cluster_id not in data["records"]:
        print(f"クラスターID '{cluster_id}' が見つかりません。")
        return
    
    records = data["records"][cluster_id]
    
    print(f"\n=== クラスター '{cluster_id}' の詳細 ===")
    
    if not isinstance(records, list):
        print("クラスター内のレコードがリスト形式ではありません。")
        print(f"型: {type(records)}")
        return
    
    print(f"レコード数: {len(records)}")
    
    if show_details:
        for i, record in enumerate(records):
            print(f"\n--- レコード {i+1} ---")
            print(f"ID: {record.get('id', '不明')}")
            
            if "data" in record and isinstance(record["data"], dict):
                for field, value in record["data"].items():
                    print(f"{field}: {value}")
            else:
                print("データフィールドが見つかりません。")
                print(record)

def compare_records(data: Dict, record_id1: str, record_id2: str):
    """2つのレコードを比較して表示"""
    if "records" not in data:
        print("レコードが見つかりません。")
        return
    
    # レコードを検索
    record1 = None
    record2 = None
    
    # クラスター構造の場合
    if isinstance(data["records"], dict):
        for cluster_records in data["records"].values():
            if isinstance(cluster_records, list):
                for record in cluster_records:
                    if record.get("id") == record_id1:
                        record1 = record
                    if record.get("id") == record_id2:
                        record2 = record
    
    # リスト構造の場合
    elif isinstance(data["records"], list):
        for record in data["records"]:
            if record.get("id") == record_id1:
                record1 = record
            if record.get("id") == record_id2:
                record2 = record
    
    # レコードが見つからない場合
    if record1 is None:
        print(f"レコードID '{record_id1}' が見つかりません。")
        return
    if record2 is None:
        print(f"レコードID '{record_id2}' が見つかりません。")
        return
    
    print(f"\n=== レコード比較: '{record_id1}' vs '{record_id2}' ===")
    
    # データフィールドを取得
    data1 = record1.get("data", {})
    data2 = record2.get("data", {})
    
    # フィールドの集合を取得
    all_fields = set(data1.keys()) | set(data2.keys())
    
    # 各フィールドの比較を表示
    for field in sorted(all_fields):
        value1 = data1.get(field, "存在しません")
        value2 = data2.get(field, "存在しません")
        
        print(f"\n{field}:")
        print(f"  レコード1: {value1}")
        print(f"  レコード2: {value2}")
        
        # 両方のフィールドが存在する場合は類似度を計算
        if field in data1 and field in data2:
            similarity = SequenceMatcher(None, str(value1), str(value2)).ratio()
            print(f"  類似度: {similarity:.4f}")

def list_clusters(data: Dict, limit: int = None):
    """すべてのクラスターとサンプルレコードを一覧表示"""
    if "records" not in data or not isinstance(data["records"], dict):
        print("クラスター構造が見つかりません。")
        return
    
    clusters = data["records"]
    cluster_ids = list(clusters.keys())
    
    if limit is not None and limit < len(cluster_ids):
        print(f"最初の{limit}クラスターを表示します（全{len(cluster_ids)}クラスター中）")
        cluster_ids = cluster_ids[:limit]
    else:
        print(f"全{len(cluster_ids)}クラスターを表示します")
    
    for i, cluster_id in enumerate(cluster_ids):
        records = clusters[cluster_id]
        
        if not isinstance(records, list):
            record_count = 1
            sample = "リスト形式ではありません"
        else:
            record_count = len(records)
            
            # サンプルレコードのタイトルと著者を取得
            sample = "サンプルなし"
            if record_count > 0:
                sample_record = records[0]
                if "data" in sample_record and isinstance(sample_record["data"], dict):
                    data_fields = sample_record["data"]
                    
                    # タイトルと著者のフィールドを探す
                    title_field = None
                    author_field = None
                    
                    for field in data_fields.keys():
                        if "title" in field.lower():
                            title_field = field
                        elif "author" in field.lower():
                            author_field = field
                    
                    sample_parts = []
                    if title_field:
                        sample_parts.append(f"タイトル: {data_fields[title_field]}")
                    if author_field:
                        sample_parts.append(f"著者: {data_fields[author_field]}")
                    
                    if sample_parts:
                        sample = ", ".join(sample_parts)
        
        print(f"\nクラスター {i+1}: {cluster_id}")
        print(f"  レコード数: {record_count}")
        print(f"  サンプル: {sample}")

def search_records(data: Dict, query: str, fields: List[str] = None):
    """レコードを検索して表示"""
    if "records" not in data:
        print("レコードが見つかりません。")
        return
    
    query = query.lower()
    matches = []
    
    # 検索対象のフィールドが指定されていない場合、すべてのフィールドを検索
    if fields is None:
        fields = []
    
    # クラスター構造の場合
    if isinstance(data["records"], dict):
        for cluster_id, cluster_records in data["records"].items():
            if not isinstance(cluster_records, list):
                continue
                
            for record in cluster_records:
                if search_in_record(record, query, fields, cluster_id):
                    matches.append((cluster_id, record))
    
    # リスト構造の場合
    elif isinstance(data["records"], list):
        for record in data["records"]:
            if search_in_record(record, query, fields, None):
                matches.append((None, record))
    
    print(f"\n=== 検索結果: '{query}' ===")
    print(f"一致するレコード数: {len(matches)}")
    
    for i, (cluster_id, record) in enumerate(matches[:10]):  # 最初の10件のみ表示
        print(f"\n--- 一致 {i+1} ---")
        print(f"ID: {record.get('id', '不明')}")
        if cluster_id:
            print(f"クラスター: {cluster_id}")
        
        if "data" in record and isinstance(record["data"], dict):
            for field, value in record["data"].items():
                if not fields or field in fields:
                    print(f"{field}: {value}")
    
    if len(matches) > 10:
        print(f"\n... 他に{len(matches) - 10}件の一致があります。")

def search_in_record(record: Dict, query: str, fields: List[str], cluster_id: str = None) -> bool:
    """レコード内でクエリを検索"""
    if "data" not in record or not isinstance(record["data"], dict):
        return False
    
    for field, value in record["data"].items():
        if fields and field not in fields:
            continue
        
        if query in str(value).lower():
            return True
    
    return False

def main():
    parser = argparse.ArgumentParser(description="レコード分析ツール")
    parser.add_argument("--input", "-i", type=str, required=True, help="入力YAMLファイルパス")
    parser.add_argument("--analyze", "-a", action="store_true", help="ファイル構造を分析")
    parser.add_argument("--list-clusters", "-l", action="store_true", help="クラスター一覧を表示")
    parser.add_argument("--limit", type=int, help="表示するクラスター数の上限")
    parser.add_argument("--cluster", "-c", type=str, help="表示するクラスターID")
    parser.add_argument("--compare", "-p", type=str, nargs=2, help="比較する2つのレコードID")
    parser.add_argument("--search", "-s", type=str, help="レコード内を検索するクエリ")
    parser.add_argument("--fields", "-f", type=str, nargs="+", help="検索対象のフィールド")
    
    args = parser.parse_args()
    
    # データ読み込み
    data = load_yaml(args.input)
    
    if args.analyze:
        analyze_record_structure(data)
    
    if args.list_clusters:
        list_clusters(data, args.limit)
    
    if args.cluster:
        show_cluster(data, args.cluster)
    
    if args.compare:
        compare_records(data, args.compare[0], args.compare[1])
    
    if args.search:
        search_records(data, args.search, args.fields)
    
    # コマンドが指定されていない場合は構造分析を実行
    if not any([args.analyze, args.list_clusters, args.cluster, args.compare, args.search]):
        analyze_record_structure(data)

if __name__ == "__main__":
    main()