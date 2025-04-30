#!/usr/bin/env python3
"""
書誌レコードマッチング結果エクスポーター

このスクリプトは書誌レコードマッチングの結果を詳細に分析し、
マッチしたレコードペアやクラスター情報を詳細なCSVファイルに出力します。
"""

import os
import sys
import yaml
import csv
import argparse
from typing import Dict, List, Any

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

def extract_record_data(record: Dict) -> Dict:
    """レコードから重要な情報を抽出"""
    result = {
        "id": record.get("id", ""),
        "cluster_id": record.get("cluster_id", ""),
        "original_cluster_id": record.get("original_cluster_id", "")
    }
    
    # データフィールドから情報を抽出
    if "data" in record and isinstance(record["data"], dict):
        data = record["data"]
        
        # タイトルフィールドを探す
        for field, value in data.items():
            if "title" in field.lower():
                result["title"] = value
            elif "author" in field.lower():
                result["author"] = value
            elif "publisher" in field.lower():
                result["publisher"] = value
            elif "date" in field.lower() or "pubdate" in field.lower():
                result["pubdate"] = value
    
    return result

def export_matched_pairs(result_data: Dict, output_file: str):
    """マッチしたレコードペアをCSVに出力"""
    # 結果ファイルから全レコードを抽出
    all_records = []
    for group in result_data.get("group", []):
        all_records.extend(group.get("records", []))
    
    # レコードIDからレコードへのマッピング
    id_to_record = {}
    for record in all_records:
        if "id" in record:
            id_to_record[record["id"]] = record
    
    # マッチしたペアを収集
    matched_pairs = []
    # クラスターIDからレコードのリストへのマッピング
    cluster_to_records = {}
    
    for record in all_records:
        if "cluster_id" in record and "id" in record:
            cluster_id = record["cluster_id"]
            if cluster_id not in cluster_to_records:
                cluster_to_records[cluster_id] = []
            cluster_to_records[cluster_id].append(record["id"])
    
    # 各クラスター内のすべてのペアを生成
    for cluster_id, record_ids in cluster_to_records.items():
        # クラスターサイズが2以上の場合のみペア処理
        if len(record_ids) >= 2:
            for i in range(len(record_ids)):
                for j in range(i + 1, len(record_ids)):
                    record_id1 = record_ids[i]
                    record_id2 = record_ids[j]
                    
                    # レコード情報の取得
                    record1 = id_to_record[record_id1]
                    record2 = id_to_record[record_id2]
                    
                    # レコードデータの抽出
                    record_data1 = extract_record_data(record1)
                    record_data2 = extract_record_data(record2)
                    
                    # 元のクラスターが一致しているかどうか
                    is_correct_match = record_data1.get("original_cluster_id") == record_data2.get("original_cluster_id")
                    
                    # ペア情報を追加
                    pair_info = {
                        "cluster_id": cluster_id,
                        "record_id1": record_id1,
                        "record_id2": record_id2,
                        "title1": record_data1.get("title", ""),
                        "title2": record_data2.get("title", ""),
                        "author1": record_data1.get("author", ""),
                        "author2": record_data2.get("author", ""),
                        "original_cluster_id1": record_data1.get("original_cluster_id", ""),
                        "original_cluster_id2": record_data2.get("original_cluster_id", ""),
                        "is_correct_match": "正解" if is_correct_match else "不正解"
                    }
                    
                    matched_pairs.append(pair_info)
    
    # CSVにエクスポート
    fieldnames = [
        "cluster_id", "record_id1", "record_id2", 
        "title1", "title2", "author1", "author2",
        "original_cluster_id1", "original_cluster_id2", "is_correct_match"
    ]
    
    print(f"{len(matched_pairs)}件のマッチペアをCSVに出力します: {output_file}")
    
    try:
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for pair in matched_pairs:
                writer.writerow(pair)
        print(f"マッチペア情報を正常に保存しました: {output_file}")
    except Exception as e:
        print(f"エラー: マッチペア情報の保存に失敗しました: {e}")

def export_cluster_info(result_data: Dict, output_file: str):
    """クラスター情報をCSVに出力"""
    # クラスター情報を収集
    cluster_info = []
    
    for i, group in enumerate(result_data.get("group", [])):
        records = group.get("records", [])
        
        # クラスターサイズ
        cluster_size = len(records)
        
        # 元のクラスターIDの分布を計算
        original_clusters = {}
        for record in records:
            orig_id = record.get("original_cluster_id", "不明")
            if orig_id not in original_clusters:
                original_clusters[orig_id] = 0
            original_clusters[orig_id] += 1
        
        # 最も多い元クラスターを特定
        max_orig_id = ""
        max_count = 0
        for orig_id, count in original_clusters.items():
            if count > max_count:
                max_count = count
                max_orig_id = orig_id
        
        # クラスターの純度 (最も多い元クラスターの割合)
        purity = max_count / cluster_size if cluster_size > 0 else 0
        
        # レコードのタイトルとIDをリスト化
        record_titles = []
        record_ids = []
        for record in records:
            record_id = record.get("id", "")
            title = ""
            if "data" in record:
                for field, value in record["data"].items():
                    if "title" in field.lower():
                        title = value
                        break
            record_titles.append(title)
            record_ids.append(record_id)
        
        # クラスター情報を追加
        info = {
            "cluster_index": i + 1,
            "cluster_id": records[0].get("cluster_id", "") if records else "",
            "size": cluster_size,
            "purity": f"{purity:.2f}",
            "main_original_cluster": max_orig_id,
            "original_clusters_distribution": str(original_clusters),
            "record_ids": ", ".join(record_ids),
            "record_titles": ", ".join(record_titles)
        }
        
        cluster_info.append(info)
    
    # CSVにエクスポート
    fieldnames = [
        "cluster_index", "cluster_id", "size", "purity", 
        "main_original_cluster", "original_clusters_distribution",
        "record_ids", "record_titles"
    ]
    
    print(f"{len(cluster_info)}個のクラスター情報をCSVに出力します: {output_file}")
    
    try:
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for info in cluster_info:
                writer.writerow(info)
        print(f"クラスター情報を正常に保存しました: {output_file}")
    except Exception as e:
        print(f"エラー: クラスター情報の保存に失敗しました: {e}")

def export_record_info(result_data: Dict, output_file: str):
    """全レコード情報をCSVに出力"""
    # 全レコード情報を収集
    all_records_info = []
    
    for group in result_data.get("group", []):
        records = group.get("records", [])
        
        for record in records:
            record_data = extract_record_data(record)
            
            # レコード情報を追加
            info = {
                "record_id": record.get("id", ""),
                "cluster_id": record.get("cluster_id", ""),
                "original_cluster_id": record.get("original_cluster_id", ""),
                "title": record_data.get("title", ""),
                "author": record_data.get("author", ""),
                "publisher": record_data.get("publisher", ""),
                "pubdate": record_data.get("pubdate", ""),
                "is_correctly_clustered": record.get("cluster_id", "") == record.get("original_cluster_id", "")
            }
            
            all_records_info.append(info)
    
    # CSVにエクスポート
    fieldnames = [
        "record_id", "cluster_id", "original_cluster_id", 
        "title", "author", "publisher", "pubdate", 
        "is_correctly_clustered"
    ]
    
    print(f"{len(all_records_info)}件のレコード情報をCSVに出力します: {output_file}")
    
    try:
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for info in all_records_info:
                writer.writerow(info)
        print(f"レコード情報を正常に保存しました: {output_file}")
    except Exception as e:
        print(f"エラー: レコード情報の保存に失敗しました: {e}")

def main():
    parser = argparse.ArgumentParser(description="書誌レコードマッチング結果のエクスポート")
    parser.add_argument("--input", "-i", type=str, required=True, help="マッチング結果のYAMLファイルパス")
    parser.add_argument("--output-dir", "-o", type=str, default="./results", help="出力ディレクトリ")
    
    args = parser.parse_args()
    
    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 結果ファイルの読み込み
    result_data = load_yaml(args.input)
    
    # 出力ファイルパス
    pairs_file = os.path.join(args.output_dir, "matched_pairs.csv")
    clusters_file = os.path.join(args.output_dir, "clusters.csv")
    records_file = os.path.join(args.output_dir, "all_records.csv")
    
    # 各種情報をエクスポート
    export_matched_pairs(result_data, pairs_file)
    export_cluster_info(result_data, clusters_file)
    export_record_info(result_data, records_file)
    
    print(f"\n全ての情報を {args.output_dir} ディレクトリに出力しました")

if __name__ == "__main__":
    main()
