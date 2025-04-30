#!/usr/bin/env python3
"""
最適化版エンティティマッチングスクリプト
- リクエスト数を削減
- クラスター間のマッチングに特化
- 閾値の調整機能
"""

import os
import sys
import yaml
import json
import argparse
import numpy as np
import requests
import time
import random
from typing import Dict, List, Tuple, Any, Set
from difflib import SequenceMatcher

# OpenAI APIの設定
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_API_ENDPOINT = "https://api.openai.com/v1/embeddings"

# マッチングの設定
SIMILARITY_THRESHOLD = 0.75  # 類似度のしきい値（デフォルト値を下げた）
MAX_RECORDS = 100  # 処理する最大レコード数
MAX_SAMPLE_PER_CLUSTER = 2  # 各クラスターから抽出する最大サンプル数
MAX_RETRIES = 3  # API呼び出しの最大リトライ回数
RETRY_DELAY = 2  # リトライ間の待機時間（秒）

# エンベディング次元数 (実行時に検出)
EMBEDDING_DIM = None

def load_yaml(filepath: str) -> Dict:
    """YAMLファイルを読み込む"""
    print(f"ファイル読み込み: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data
    except Exception as e:
        print(f"エラー: ファイル読み込みに失敗しました: {e}")
        sys.exit(1)

def extract_records_by_cluster(data: Dict) -> Dict[str, List[Dict]]:
    """クラスター構造からレコードを抽出し、クラスターIDごとに整理する"""
    clusters = {}
    
    if "records" in data and isinstance(data["records"], dict):
        for cluster_id, records in data["records"].items():
            if isinstance(records, list):
                # レコードがリスト形式の場合
                for record in records:
                    # cluster_idを追加
                    if "cluster_id" not in record:
                        record["cluster_id"] = cluster_id
                
                # クラスター辞書に追加
                clusters[cluster_id] = records
            elif isinstance(records, dict):
                # レコードが辞書形式の場合
                record = records.copy()
                if "cluster_id" not in record:
                    record["cluster_id"] = cluster_id
                
                # クラスター辞書に追加
                clusters[cluster_id] = [record]
    
    return clusters

def detect_embedding_dim(api_key: str) -> int:
    """OpenAI APIからエンベディングの次元数を検出"""
    global EMBEDDING_DIM
    
    print("エンベディングの次元数を検出中...")
    
    # テスト用のテキスト
    test_text = "次元数検出用テキスト"
    
    embedding = get_embedding(test_text, api_key)
    dim = len(embedding)
    
    print(f"検出された次元数: {dim}")
    EMBEDDING_DIM = dim
    
    return dim

def get_embedding(text: str, api_key: str) -> np.ndarray:
    """テキストのエンベディングをOpenAI APIから取得（リトライ機能付き）"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": "text-embedding-3-large",
        "input": text
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                OPENAI_API_ENDPOINT,
                headers=headers,
                data=json.dumps(data),
                timeout=30  # タイムアウト設定
            )
            
            if response.status_code == 200:
                return np.array(response.json()["data"][0]["embedding"])
            elif response.status_code == 429:  # レート制限
                wait_time = (attempt + 1) * RETRY_DELAY
                print(f"レート制限に達しました。{wait_time}秒待機します...")
                time.sleep(wait_time)
            else:
                print(f"API呼び出しエラー ({attempt+1}/{MAX_RETRIES}): {response.text}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"エンベディング取得エラー ({attempt+1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    
    # 全リトライが失敗した場合はゼロベクトルを返す
    print(f"エンベディング取得に失敗しました。ゼロベクトルを使用します。")
    # 事前に検出した次元数でゼロベクトルを作成
    return np.zeros(EMBEDDING_DIM or 1536)

def get_record_text(record: Dict, attr_fields: Dict = None) -> str:
    """レコードからテキストを構築"""
    text = ""
    
    if "data" in record and isinstance(record["data"], dict):
        data = record["data"]
        
        # 属性フィールドが指定されている場合は、それに従って重みを設定
        if attr_fields:
            for field, field_type in attr_fields.items():
                if field in data:
                    value = data[field]
                    # タイトルと著者は重要度が高いので重みを大きくする
                    if field_type == "TEXT" or "title" in field.lower():
                        text += f"{field}: {value} {value} {value} "
                    elif "author" in field.lower() or field_type == "COMPLEMENT_JA":
                        text += f"{field}: {value} {value} "
                    else:
                        text += f"{field}: {value} "
        else:
            # 属性フィールドが指定されていない場合は、すべてのフィールドを使用
            for field, value in data.items():
                if "title" in field.lower():
                    text += f"{field}: {value} {value} {value} "
                elif "author" in field.lower():
                    text += f"{field}: {value} {value} "
                else:
                    text += f"{field}: {value} "
    
    # テキストが空の場合は、レコード全体を文字列化
    if not text:
        text = str(record)
    
    return text.strip()

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """コサイン類似度を計算（次元数の異なるベクトルにも対応）"""
    # 次元数が異なる場合は正規化
    if len(vec1) != len(vec2):
        min_dim = min(len(vec1), len(vec2))
        vec1 = vec1[:min_dim]
        vec2 = vec2[:min_dim]
    
    # 類似度計算
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    
    if norm_a == 0 or norm_b == 0:
        return 0
    
    return dot_product / (norm_a * norm_b)

def string_similarity(str1: str, str2: str) -> float:
    """文字列の類似度をSequenceMatcherで計算"""
    if not isinstance(str1, str):
        str1 = str(str1)
    if not isinstance(str2, str):
        str2 = str(str2)
    return SequenceMatcher(None, str1, str2).ratio()

def get_record_attribute(record: Dict, attr: str, default="") -> str:
    """レコードから属性値を取得"""
    if "data" in record and isinstance(record["data"], dict):
        return record["data"].get(attr, default)
    return record.get(attr, default)

def sample_records_from_clusters(clusters: Dict[str, List[Dict]], 
                               max_per_cluster: int = MAX_SAMPLE_PER_CLUSTER,
                               max_total: int = MAX_RECORDS) -> List[Dict]:
    """各クラスターから指定数のレコードをサンプリングする"""
    sampled_records = []
    cluster_ids = list(clusters.keys())
    
    # クラスター数が少ない場合は全てのクラスターを使用
    if len(cluster_ids) * max_per_cluster <= max_total:
        for cluster_id in cluster_ids:
            records = clusters[cluster_id]
            # 各クラスターから最大数サンプリング
            if len(records) <= max_per_cluster:
                sampled_records.extend(records)
            else:
                sampled_records.extend(random.sample(records, max_per_cluster))
    else:
        # クラスター数が多い場合はランダムに選択
        selected_clusters = random.sample(cluster_ids, max_total // max_per_cluster)
        for cluster_id in selected_clusters:
            records = clusters[cluster_id]
            # 各クラスターから最大数サンプリング
            if len(records) <= max_per_cluster:
                sampled_records.extend(records)
            else:
                sampled_records.extend(random.sample(records, max_per_cluster))
    
    return sampled_records

def find_matching_clusters(clusters: Dict[str, List[Dict]], api_key: str, attr_fields: Dict = None,
                         max_per_cluster: int = MAX_SAMPLE_PER_CLUSTER,
                         threshold: float = SIMILARITY_THRESHOLD) -> List[Tuple]:
    """クラスター間のマッチングを検出する"""
    # エンベディングの次元数を検出
    detect_embedding_dim(api_key)
    
    # クラスターごとの代表エンベディングを計算
    print(f"クラスター間の比較を実行します（全{len(clusters)}クラスター）...")
    
    cluster_embeddings = {}
    processed_clusters = 0
    
    for cluster_id, records in clusters.items():
        processed_clusters += 1
        if processed_clusters % 10 == 0:
            print(f"  {processed_clusters}/{len(clusters)}クラスター処理中...")
        
        # クラスターからレコードをサンプリング
        sample_size = min(max_per_cluster, len(records))
        if sample_size < len(records):
            sampled_records = random.sample(records, sample_size)
        else:
            sampled_records = records
        
        # 各レコードのテキストを結合
        combined_text = ""
        for record in sampled_records:
            combined_text += get_record_text(record, attr_fields) + " "
        
        # クラスター代表エンベディングを計算
        cluster_embeddings[cluster_id] = get_embedding(combined_text.strip(), api_key)
    
    # クラスター間の類似度を比較
    matches = []
    cluster_ids = list(cluster_embeddings.keys())
    total_comparisons = len(cluster_ids) * (len(cluster_ids) - 1) // 2
    processed = 0
    
    print(f"クラスター間の類似度を計算中（全{total_comparisons}ペア）...")
    
    for i in range(len(cluster_ids)):
        c1 = cluster_ids[i]
        emb1 = cluster_embeddings[c1]
        
        for j in range(i + 1, len(cluster_ids)):
            c2 = cluster_ids[j]
            emb2 = cluster_embeddings[c2]
            processed += 1
            
            if processed % max(1, total_comparisons // 10) == 0:
                print(f"  {processed}/{total_comparisons}ペア処理中... ({int(processed/total_comparisons*100)}%)")
            
            # 類似度計算
            similarity = cosine_similarity(emb1, emb2)
            
            # しきい値以上の類似度の場合、レコードレベルの詳細比較を行う
            if similarity >= threshold:
                # 各クラスターからサンプリング
                records1 = clusters[c1][:max_per_cluster] if len(clusters[c1]) > max_per_cluster else clusters[c1]
                records2 = clusters[c2][:max_per_cluster] if len(clusters[c2]) > max_per_cluster else clusters[c2]
                
                # サンプルレコード間の類似度を比較
                best_record_similarity = 0
                best_pair = None
                
                for r1 in records1:
                    for r2 in records2:
                        # タイトルと著者の文字列類似度
                        title_field = next((field for field, ftype in attr_fields.items() 
                                          if ftype == "TEXT" or "title" in field.lower()), None)
                        author_field = next((field for field, ftype in attr_fields.items() 
                                           if "author" in field.lower()), None)
                        
                        title_similarity = 0
                        author_similarity = 0
                        
                        if title_field:
                            title_similarity = string_similarity(
                                get_record_attribute(r1, title_field), 
                                get_record_attribute(r2, title_field)
                            )
                        
                        if author_field:
                            author_similarity = string_similarity(
                                get_record_attribute(r1, author_field), 
                                get_record_attribute(r2, author_field)
                            )
                        
                        # 総合類似度
                        record_similarity = similarity * 0.5 + title_similarity * 0.3 + author_similarity * 0.2
                        
                        if record_similarity > best_record_similarity:
                            best_record_similarity = record_similarity
                            best_pair = (r1, r2)
                
                # 最良ペアの情報を記録
                if best_pair and best_record_similarity >= threshold:
                    id1 = str(best_pair[0].get("id", f"c1_{c1}"))
                    id2 = str(best_pair[1].get("id", f"c2_{c2}"))
                    
                    # タイトルと著者の情報を取得
                    title_field = next((field for field, ftype in attr_fields.items() 
                                      if ftype == "TEXT" or "title" in field.lower()), None)
                    author_field = next((field for field, ftype in attr_fields.items() 
                                       if "author" in field.lower()), None)
                    
                    title1 = get_record_attribute(best_pair[0], title_field) if title_field else ""
                    title2 = get_record_attribute(best_pair[1], title_field) if title_field else ""
                    author1 = get_record_attribute(best_pair[0], author_field) if author_field else ""
                    author2 = get_record_attribute(best_pair[1], author_field) if author_field else ""
                    
                    matches.append((id1, id2, float(best_record_similarity), {
                        "cluster_similarity": float(similarity),
                        "title1": title1,
                        "title2": title2,
                        "author1": author1,
                        "author2": author2,
                        "cluster_id1": c1,
                        "cluster_id2": c2
                    }))
    
    # 類似度でソート（降順）
    matches.sort(key=lambda x: x[2], reverse=True)
    
    return matches

def main():
    parser = argparse.ArgumentParser(description="最適化版エンティティマッチングスクリプト")
    parser.add_argument("--input", "-i", type=str, required=True, help="入力YAMLファイルパス")
    parser.add_argument("--output", "-o", type=str, default="matches.yaml", help="出力YAMLファイルパス")
    parser.add_argument("--api-key", "-k", type=str, help="OpenAI APIキー")
    parser.add_argument("--threshold", "-t", type=float, default=SIMILARITY_THRESHOLD, help="類似度のしきい値")
    parser.add_argument("--max-samples", "-m", type=int, default=MAX_SAMPLE_PER_CLUSTER, help="クラスターごとの最大サンプル数")
    
    args = parser.parse_args()
    
    # APIキーの設定
    api_key = args.api_key or OPENAI_API_KEY
    if not api_key:
        print("エラー: OpenAI APIキーが指定されていません。--api-keyオプションまたは環境変数OPENAI_API_KEYで指定してください。")
        sys.exit(1)
    
    # データ読み込み
    data = load_yaml(args.input)
    
    # 属性フィールドの取得
    attr_fields = data.get("inf_attr", {})
    print(f"属性フィールド: {attr_fields}")
    
    # クラスター構造からレコードを抽出
    clusters = extract_records_by_cluster(data)
    if not clusters:
        print("エラー: クラスターが見つかりません。")
        sys.exit(1)
    
    print(f"クラスター数: {len(clusters)}")
    total_records = sum(len(records) for records in clusters.values())
    print(f"レコード総数: {total_records}")
    
    # クラスター間のマッチングを検出
    matches = find_matching_clusters(
        clusters, 
        api_key, 
        attr_fields,
        args.max_samples, 
        args.threshold
    )
    
    print(f"マッチング結果: {len(matches)}件のマッチを検出")
    
    # 上位10件を表示
    for i, (id1, id2, score, details) in enumerate(matches[:10]):
        print(f"マッチ {i+1}: {id1} <-> {id2} (類似度: {score:.4f})")
        if "title1" in details and "title2" in details:
            print(f"  タイトル1: {details['title1']}")
            print(f"  タイトル2: {details['title2']}")
        if "cluster_id1" in details and "cluster_id2" in details:
            print(f"  クラスター: {details['cluster_id1']} <-> {details['cluster_id2']}")
        print()
    
    # 結果を出力
    output_data = {
        "matches": [
            {
                "record1": id1,
                "record2": id2,
                "similarity": score,
                "details": details
            }
            for id1, id2, score, details in matches
        ]
    }
    
    # YAML形式で出力
    with open(args.output, "w", encoding="utf-8") as f:
        yaml.dump(output_data, f, default_flow_style=False, allow_unicode=True)
    
    print(f"結果を{args.output}に保存しました。")

if __name__ == "__main__":
    main()