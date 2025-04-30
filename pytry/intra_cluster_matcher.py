#!/usr/bin/env python3
"""
クラスター内比較対応マッチングスクリプト
クラスター内のレコード比較も行い、精度評価を可能にします
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
import hashlib
from typing import Dict, List, Tuple, Any, Set
from difflib import SequenceMatcher
from datetime import datetime

# OpenAI APIの設定
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_API_ENDPOINT = "https://api.openai.com/v1/embeddings"

# マッチングの設定
SIMILARITY_THRESHOLD = 0.65  # 類似度のしきい値
MAX_RETRIES = 5  # API呼び出しの最大リトライ回数
RETRY_DELAY = 5  # リトライ間の待機時間（秒）
RATE_LIMIT_DELAY = 60  # レート制限時の待機時間（秒）
REQUEST_INTERVAL = 0.5  # リクエスト間の待機時間（秒）

# クラスター内比較の設定
INTRA_CLUSTER_SAMPLE_SIZE = 10  # クラスター内で比較するレコードのサンプル数
INTRA_CLUSTER_MAX_PAIRS = 100  # クラスター内で比較する最大ペア数

# エンベディング次元数 (実行時に検出)
EMBEDDING_DIM = None

# デバッグフラグ
DEBUG_MODE = False
DEBUG_LOG_FILE = "debug_log.txt"

def debug_log(message, file=None):
    """デバッグログを出力する"""
    if DEBUG_MODE:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        if file:
            with open(file, 'a', encoding='utf-8') as f:
                f.write(log_message + "\n")

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

def get_embedding(text: str, api_key: str, debug_info: str = "") -> np.ndarray:
    """テキストのエンベディングをOpenAI APIから取得（リトライ機能付き）"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": "text-embedding-3-large",
        "input": text
    }
    
    # デバッグ情報を出力
    debug_log(f"エンベディング取得リクエスト: 長さ={len(text)}文字 ({debug_info})", DEBUG_LOG_FILE)
    
    for attempt in range(MAX_RETRIES):
        try:
            # リクエスト間隔を設定（レート制限対策）
            time.sleep(REQUEST_INTERVAL)
            
            response = requests.post(
                OPENAI_API_ENDPOINT,
                headers=headers,
                data=json.dumps(data),
                timeout=30  # タイムアウト設定
            )
            
            if response.status_code == 200:
                embedding = np.array(response.json()["data"][0]["embedding"])
                debug_log(f"エンベディング取得成功: 次元数={len(embedding)} ({debug_info})", DEBUG_LOG_FILE)
                return embedding
            elif response.status_code == 429:  # レート制限
                wait_time = RATE_LIMIT_DELAY * (attempt + 1)
                debug_log(f"レート制限に達しました。{wait_time}秒待機します... ({debug_info})", DEBUG_LOG_FILE)
                print(f"レート制限に達しました。{wait_time}秒待機します...")
                time.sleep(wait_time)
            else:
                debug_log(f"API呼び出しエラー ({attempt+1}/{MAX_RETRIES}): {response.text} ({debug_info})", DEBUG_LOG_FILE)
                print(f"API呼び出しエラー ({attempt+1}/{MAX_RETRIES}): {response.text}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
        except Exception as e:
            debug_log(f"エンベディング取得エラー ({attempt+1}/{MAX_RETRIES}): {e} ({debug_info})", DEBUG_LOG_FILE)
            print(f"エンベディング取得エラー ({attempt+1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    
    # 全リトライが失敗した場合はゼロベクトルを返す
    debug_log(f"エンベディング取得に失敗しました。ゼロベクトルを使用します。 ({debug_info})", DEBUG_LOG_FILE)
    print(f"エンベディング取得に失敗しました。ゼロベクトルを使用します。")
    # 事前に検出した次元数でゼロベクトルを作成
    return np.zeros(EMBEDDING_DIM or 1536)

def load_or_create_embedding_cache(cache_dir: str, identifier: str) -> Dict[str, np.ndarray]:
    """エンベディングキャッシュを読み込むか作成する"""
    # キャッシュディレクトリがなければ作成
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file = os.path.join(cache_dir, f"{identifier}_embedding_cache.npz")
    
    if os.path.exists(cache_file):
        print(f"エンベディングキャッシュを読み込んでいます: {cache_file}")
        try:
            data = np.load(cache_file, allow_pickle=True)
            cache = dict(data.items())
            print(f"キャッシュからエンベディングを{len(cache)}件読み込みました")
            return cache
        except Exception as e:
            print(f"キャッシュ読み込みエラー: {e}")
            return {}
    else:
        print(f"新しいエンベディングキャッシュを作成します: {cache_file}")
        return {}

def save_embedding_cache(cache: Dict[str, np.ndarray], cache_dir: str, identifier: str):
    """エンベディングキャッシュを保存する"""
    cache_file = os.path.join(cache_dir, f"{identifier}_embedding_cache.npz")
    
    try:
        np.savez(cache_file, **cache)
        print(f"エンベディングキャッシュを保存しました: {cache_file}")
    except Exception as e:
        print(f"キャッシュ保存エラー: {e}")

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

def normalize_japanese_text(text: str) -> str:
    """日本語テキストの正規化（カタカナとひらがなの違いなどを吸収）"""
    import unicodedata
    # 全角を半角に変換
    text = unicodedata.normalize('NFKC', text)
    # 記号を削除
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    return text.lower()

def get_record_attribute(record: Dict, attr: str, default="") -> str:
    """レコードから属性値を取得"""
    if "data" in record and isinstance(record["data"], dict):
        return record["data"].get(attr, default)
    return record.get(attr, default)

def compare_record_pair(r1: Dict, r2: Dict, similarity_base: float, attr_fields: Dict) -> Tuple[float, Dict]:
    """レコードペアを比較して類似度を計算"""
    # タイトルと著者の文字列類似度
    title_field = next((field for field, ftype in attr_fields.items() 
                      if ftype == "TEXT" or "title" in field.lower()), None)
    author_field = next((field for field, ftype in attr_fields.items() 
                       if "author" in field.lower()), None)
    
    title_similarity = 0
    author_similarity = 0
    
    if title_field:
        title1 = get_record_attribute(r1, title_field)
        title2 = get_record_attribute(r2, title_field)
        
        # 日本語の場合は正規化
        if any(ord(c) > 127 for c in title1 + title2):
            title1_norm = normalize_japanese_text(title1)
            title2_norm = normalize_japanese_text(title2)
            title_similarity = string_similarity(title1_norm, title2_norm)
        else:
            title_similarity = string_similarity(title1, title2)
    
    if author_field:
        author1 = get_record_attribute(r1, author_field)
        author2 = get_record_attribute(r2, author_field)
        
        # 日本語の場合は正規化
        if any(ord(c) > 127 for c in author1 + author2):
            author1_norm = normalize_japanese_text(author1)
            author2_norm = normalize_japanese_text(author2)
            author_similarity = string_similarity(author1_norm, author2_norm)
        else:
            author_similarity = string_similarity(author1, author2)
    
    # 総合類似度（重みを調整）
    record_similarity = similarity_base * 0.4 + title_similarity * 0.4 + author_similarity * 0.2
    
    # 詳細情報
    details = {
        "base_similarity": float(similarity_base),
        "title_similarity": float(title_similarity),
        "author_similarity": float(author_similarity),
        "title1": get_record_attribute(r1, title_field) if title_field else "",
        "title2": get_record_attribute(r2, title_field) if title_field else "",
        "author1": get_record_attribute(r1, author_field) if author_field else "",
        "author2": get_record_attribute(r2, author_field) if author_field else "",
        "cluster_id1": str(r1.get("cluster_id", "")),
        "cluster_id2": str(r2.get("cluster_id", ""))
    }
    
    return record_similarity, details

def process_intra_cluster(clusters: Dict[str, List[Dict]], api_key: str, attr_fields: Dict,
                        threshold: float = SIMILARITY_THRESHOLD, cache_dir: str = ".cache",
                        sample_size: int = INTRA_CLUSTER_SAMPLE_SIZE,
                        max_pairs: int = INTRA_CLUSTER_MAX_PAIRS) -> List[Tuple]:
    """クラスター内のレコードを比較してマッチングを検出"""
    print(f"クラスター内比較を実行中...")
    
    # エンベディングキャッシュを読み込む
    cache_identifier = hashlib.md5(json.dumps(attr_fields).encode()).hexdigest()[:8] + "_records"
    embedding_cache = load_or_create_embedding_cache(cache_dir, cache_identifier)
    
    matches = []
    
    for cluster_id, records in clusters.items():
        # レコード数が2未満のクラスターはスキップ
        if len(records) < 2:
            continue
        
        # サンプリング
        record_sample = records
        if len(records) > sample_size:
            record_sample = random.sample(records, sample_size)
        
        print(f"  クラスター {cluster_id}: {len(record_sample)}/{len(records)}件のレコードを処理中...")
        
        # 各レコードのエンベディングを取得
        record_embeddings = {}
        for record in record_sample:
            record_id = str(record.get("id", ""))
            
            # キャッシュを確認
            cache_key = f"record_{record_id}"
            if cache_key in embedding_cache:
                record_embeddings[record_id] = embedding_cache[cache_key]
            else:
                # テキストからエンベディングを計算
                text = get_record_text(record, attr_fields)
                embedding = get_embedding(text, api_key, f"レコード {record_id}")
                
                # キャッシュに保存
                embedding_cache[cache_key] = embedding
                record_embeddings[record_id] = embedding
        
        # 定期的にキャッシュを保存
        save_embedding_cache(embedding_cache, cache_dir, cache_identifier)
        
        # レコードペアを比較
        cluster_matches = []
        processed_pairs = 0
        
        for i, r1 in enumerate(record_sample):
            id1 = str(r1.get("id", ""))
            
            for j in range(i + 1, len(record_sample)):
                r2 = record_sample[j]
                id2 = str(r2.get("id", ""))
                
                # 最大ペア数に達したら終了
                processed_pairs += 1
                if processed_pairs > max_pairs:
                    break
                
                # エンベディングの類似度を計算
                similarity = cosine_similarity(record_embeddings[id1], record_embeddings[id2])
                
                # レコードペアの詳細比較
                record_similarity, details = compare_record_pair(r1, r2, similarity, attr_fields)
                
                # しきい値以上の類似度があればマッチとして追加
                if record_similarity >= threshold:
                    cluster_matches.append((id1, id2, float(record_similarity), details))
            
            # 最大ペア数に達したら終了
            if processed_pairs > max_pairs:
                break
        
        # クラスター内マッチングの結果を追加
        matches.extend(cluster_matches)
        print(f"    クラスター {cluster_id}: {len(cluster_matches)}件のマッチを検出")
    
    # 最終的なキャッシュを保存
    save_embedding_cache(embedding_cache, cache_dir, cache_identifier)
    
    # 類似度でソート（降順）
    matches.sort(key=lambda x: x[2], reverse=True)
    
    return matches

def process_inter_cluster(clusters: Dict[str, List[Dict]], api_key: str, attr_fields: Dict,
                        threshold: float = SIMILARITY_THRESHOLD, cache_dir: str = ".cache") -> List[Tuple]:
    """クラスター間の比較を実行してマッチングを検出"""
    print(f"クラスター間比較を実行中...")
    
    # エンベディングの次元数を検出
    detect_embedding_dim(api_key)
    
    # キャッシュの読み込み
    cache_identifier = hashlib.md5(json.dumps(attr_fields).encode()).hexdigest()[:8] + "_clusters"
    embedding_cache = load_or_create_embedding_cache(cache_dir, cache_identifier)
    
    # クラスターごとの代表エンベディングを計算
    cluster_embeddings = {}
    processed_clusters = 0
    
    for cluster_id, records in clusters.items():
        processed_clusters += 1
        
        # 定期的に進捗を表示
        if processed_clusters % 10 == 0 or processed_clusters == len(clusters):
            print(f"  クラスター {processed_clusters}/{len(clusters)} 処理中...")
        
        # レコードのテキストを結合してエンベディングを計算
        combined_text = ""
        for record in records:
            record_text = get_record_text(record, attr_fields)
            combined_text += record_text + " "
        
        # テキストが長すぎる場合は切り詰める（APIの制限に対応）
        if len(combined_text) > 8000:
            combined_text = combined_text[:8000]
        
        # キャッシュキーを生成
        cache_key = f"cluster_{cluster_id}"
        
        # キャッシュを確認
        if cache_key in embedding_cache:
            cluster_embeddings[cluster_id] = embedding_cache[cache_key]
        else:
            # エンベディングを計算してキャッシュに保存
            embedding = get_embedding(combined_text.strip(), api_key, f"クラスター {cluster_id}")
            embedding_cache[cache_key] = embedding
            cluster_embeddings[cluster_id] = embedding
    
    # 最終的なキャッシュを保存
    save_embedding_cache(embedding_cache, cache_dir, cache_identifier)
    
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
            
            # 定期的に進捗を表示
            if processed % max(1, total_comparisons // 10) == 0:
                print(f"  {processed}/{total_comparisons}ペア処理中... "
                      f"({int(processed/total_comparisons*100)}%)")
            
            # 類似度計算
            similarity = cosine_similarity(emb1, emb2)
            
            # しきい値以上の類似度の場合、レコードレベルの詳細比較を行う
            if similarity >= threshold:
                # 各クラスターからレコードを取得
                records1 = clusters[c1]
                records2 = clusters[c2]
                
                # レコード間の最良マッチを探す
                best_record_similarity = 0
                best_pair = None
                
                # サンプリングしてレコードを選択
                sample_size1 = min(len(records1), 3)
                sample_size2 = min(len(records2), 3)
                
                records1_sample = random.sample(records1, sample_size1) if sample_size1 < len(records1) else records1
                records2_sample = random.sample(records2, sample_size2) if sample_size2 < len(records2) else records2
                
                for r1 in records1_sample:
                    for r2 in records2_sample:
                        # レコードペアの詳細比較
                        record_similarity, details = compare_record_pair(r1, r2, similarity, attr_fields)
                        
                        if record_similarity > best_record_similarity:
                            best_record_similarity = record_similarity
                            best_pair = (r1, r2, record_similarity, details)
                
                # 最良ペアの情報を記録
                if best_pair and best_record_similarity >= threshold:
                    r1, r2, sim, details = best_pair
                    id1 = str(r1.get("id", f"c1_{c1}"))
                    id2 = str(r2.get("id", f"c2_{c2}"))
                    
                    matches.append((id1, id2, float(sim), details))
    
    # 類似度でソート（降順）
    matches.sort(key=lambda x: x[2], reverse=True)
    
    return matches

def calculate_metrics(matches: List[Tuple], clusters: Dict[str, List[Dict]]) -> Dict:
    """マッチング結果のメトリクスを計算"""
    metrics = {
        "total_matches": len(matches),
        "intra_cluster_matches": 0,
        "inter_cluster_matches": 0,
        "potential_matches": 0,
        "recall": 0.0,
        "clusters_with_matches": set(),
    }
    
    # クラスター内／クラスター間のマッチング数をカウント
    for _, _, _, details in matches:
        cluster_id1 = details.get("cluster_id1", "")
        cluster_id2 = details.get("cluster_id2", "")
        
        metrics["clusters_with_matches"].add(cluster_id1)
        metrics["clusters_with_matches"].add(cluster_id2)
        
        if cluster_id1 == cluster_id2:
            metrics["intra_cluster_matches"] += 1
        else:
            metrics["inter_cluster_matches"] += 1
    
    # 潜在的なマッチング数を計算
    potential_intra_matches = 0
    for cluster_id, records in clusters.items():
        if len(records) >= 2:
            # クラスター内の可能なペア数: nC2 = n(n-1)/2
            potential_intra_matches += (len(records) * (len(records) - 1)) // 2
    
    metrics["potential_intra_matches"] = potential_intra_matches
    
    # 再現率を計算（クラスター内マッチングのみ）
    if potential_intra_matches > 0:
        metrics["intra_cluster_recall"] = metrics["intra_cluster_matches"] / potential_intra_matches
    
    # クラスター統計
    metrics["total_clusters"] = len(clusters)
    metrics["clusters_with_matches"] = len(metrics["clusters_with_matches"])
    metrics["clusters_with_matches_ratio"] = metrics["clusters_with_matches"] / metrics["total_clusters"] if metrics["total_clusters"] > 0 else 0
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="クラスター内比較対応マッチングスクリプト")
    parser.add_argument("--input", "-i", type=str, required=True, help="入力YAMLファイルパス")
    parser.add_argument("--output", "-o", type=str, default="matches.yaml", help="出力YAMLファイルパス")
    parser.add_argument("--api-key", "-k", type=str, help="OpenAI APIキー")
    parser.add_argument("--threshold", "-t", type=float, default=SIMILARITY_THRESHOLD, help="類似度のしきい値")
    parser.add_argument("--cache-dir", "-c", type=str, default=".cache", help="キャッシュディレクトリ")
    parser.add_argument("--intra-only", action="store_true", help="クラスター内比較のみ実行")
    parser.add_argument("--inter-only", action="store_true", help="クラスター間比較のみ実行")
    parser.add_argument("--sample-size", "-s", type=int, default=INTRA_CLUSTER_SAMPLE_SIZE, help="クラスター内比較のサンプルサイズ")
    parser.add_argument("--max-pairs", "-m", type=int, default=INTRA_CLUSTER_MAX_PAIRS, help="クラスター内比較の最大ペア数")
    parser.add_argument("--debug", "-d", action="store_true", help="デバッグモードを有効にする")
    
    args = parser.parse_args()
    
    # デバッグモードの設定
    global DEBUG_MODE, DEBUG_LOG_FILE
    DEBUG_MODE = args.debug
    DEBUG_LOG_FILE = os.path.join("debug_logs", "debug_log.txt")
    
    if DEBUG_MODE:
        os.makedirs("debug_logs", exist_ok=True)
        with open(DEBUG_LOG_FILE, 'w', encoding='utf-8') as f:
            f.write(f"=== デバッグログ ===\n")
            f.write(f"開始時刻: {datetime.now()}\n\n")
    
    # APIキーの設定
    api_key = args.api_key or OPENAI_API_KEY
    if not api_key:
        print("エラー: OpenAI APIキーが指定されていません。--api-keyオプションまたは環境変数OPENAI_API_KEYで指定してください。")
        sys.exit(1)
    
    # キャッシュディレクトリの作成
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # 開始時間を記録
    start_time = datetime.now()
    print(f"処理開始: {start_time}")
    
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
    
    # マッチング結果
    matches = []
    
    # 比較モードの決定
    do_intra = not args.inter_only  # デフォルトはクラスター内比較を実行
    do_inter = not args.intra_only  # デフォルトはクラスター間比較を実行
    
    # クラスター内比較
    if do_intra:
        intra_matches = process_intra_cluster(
            clusters, 
            api_key, 
            attr_fields,
            args.threshold,
            args.cache_dir,
            args.sample_size,
            args.max_pairs
        )
        matches.extend(intra_matches)
        print(f"クラスター内マッチング: {len(intra_matches)}件のマッチを検出")
    
    # クラスター間比較
    if do_inter:
        inter_matches = process_inter_cluster(
            clusters, 
            api_key, 
            attr_fields,
            args.threshold,
            args.cache_dir
        )
        matches.extend(inter_matches)
        print(f"クラスター間マッチング: {len(inter_matches)}件のマッチを検出")
    
    # 結果を類似度順にソート
    matches.sort(key=lambda x: x[2], reverse=True)
    
    # メトリクスを計算
    metrics = calculate_metrics(matches, clusters)
    
    print(f"\n=== マッチング結果 ===")
    print(f"総マッチング数: {metrics['total_matches']}件")
    print(f"  クラスター内マッチング: {metrics['intra_cluster_matches']}件")
    print(f"  クラスター間マッチング: {metrics['inter_cluster_matches']}件")
    if 'intra_cluster_recall' in metrics:
        print(f"クラスター内マッチング再現率: {metrics['intra_cluster_recall']:.4f} ({metrics['intra_cluster_matches']}/{metrics['potential_intra_matches']})")
    print(f"マッチングのあるクラスター数: {metrics['clusters_with_matches']}/{metrics['total_clusters']} ({metrics['clusters_with_matches_ratio']:.2%})")
    
    # 上位10件を表示
    if len(matches) > 0:
        print(f"\n=== 上位マッチング ===")
        for i, (id1, id2, score, details) in enumerate(matches[:10]):
            print(f"マッチ {i+1}: {id1} <-> {id2} (類似度: {score:.4f})")
            if details.get("cluster_id1") == details.get("cluster_id2"):
                print(f"  [クラスター内] クラスター: {details.get('cluster_id1')}")
            else:
                print(f"  [クラスター間] クラスター: {details.get('cluster_id1')} <-> {details.get('cluster_id2')}")
            
            if "title1" in details and "title2" in details:
                print(f"  タイトル1: {details['title1']}")
                print(f"  タイトル2: {details['title2']}")
            if "author1" in details and "author2" in details:
                print(f"  著者1: {details['author1']}")
                print(f"  著者2: {details['author2']}")
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
        ],
        "metrics": metrics
    }
    
    # YAML形式で出力
    with open(args.output, "w", encoding='utf-8') as f:
        yaml.dump(output_data, f, default_flow_style=False, allow_unicode=True)
    
    print(f"結果を{args.output}に保存しました。")
    
    # 処理時間を表示
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    print(f"処理終了: {end_time}")
    print(f"処理時間: {elapsed:.1f}秒 ({elapsed/60:.1f}分)")

if __name__ == "__main__":
    import hashlib
    main()