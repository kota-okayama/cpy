#!/usr/bin/env python3
"""
デバッグ機能付きマッチングスクリプト
詳細なログとデバッグ情報を出力し、処理の各ステップを確認できます
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
SIMILARITY_THRESHOLD = 0.5  # 類似度のしきい値（より低く設定）
MAX_RETRIES = 5  # API呼び出しの最大リトライ回数
RETRY_DELAY = 5  # リトライ間の待機時間（秒）
RATE_LIMIT_DELAY = 60  # レート制限時の待機時間（秒）
REQUEST_INTERVAL = 0.5  # リクエスト間の待機時間（秒）

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
    debug_log(f"テキストサンプル: {text[:100]}...", DEBUG_LOG_FILE)
    
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

def dump_record_text(record: Dict, attr_fields: Dict = None) -> str:
    """デバッグ用：レコードのテキスト表現を詳細に出力"""
    result = []
    
    if "data" in record and isinstance(record["data"], dict):
        data = record["data"]
        
        # 属性フィールドが指定されている場合は、それに従って出力
        if attr_fields:
            for field, field_type in attr_fields.items():
                if field in data:
                    value = data[field]
                    weight = "重要" if field_type == "TEXT" or "title" in field.lower() or "author" in field.lower() else "通常"
                    result.append(f"{field} ({weight}): {value}")
        else:
            # 属性フィールドが指定されていない場合は、すべてのフィールドを出力
            for field, value in data.items():
                weight = "重要" if "title" in field.lower() or "author" in field.lower() else "通常"
                result.append(f"{field} ({weight}): {value}")
    
    return "\n".join(result)

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

def process_all_clusters(clusters: Dict[str, List[Dict]], api_key: str, attr_fields: Dict = None,
                        threshold: float = SIMILARITY_THRESHOLD, cache_dir: str = ".cache",
                        debug_dump_dir: str = "debug_dumps") -> List[Tuple]:
    """すべてのクラスターを処理してマッチングを検出する（デバッグ機能付き）"""
    # デバッグ情報の保存ディレクトリを作成
    if DEBUG_MODE:
        os.makedirs(debug_dump_dir, exist_ok=True)
    
    # エンベディングの次元数を検出
    detect_embedding_dim(api_key)
    
    # キャッシュの読み込み
    cache_identifier = hashlib.md5(json.dumps(attr_fields).encode()).hexdigest()[:8]
    embedding_cache = load_or_create_embedding_cache(cache_dir, cache_identifier)
    
    # クラスターごとの代表エンベディングを計算
    print(f"クラスター間の比較を実行します（全{len(clusters)}クラスター）...")
    
    cluster_embeddings = {}
    cluster_texts = {}  # デバッグ用：クラスターのテキスト表現
    processed_clusters = 0
    start_time = datetime.now()
    
    # デバッグ情報：クラスター情報ダンプ
    if DEBUG_MODE:
        with open(os.path.join(debug_dump_dir, "cluster_info.txt"), 'w', encoding='utf-8') as f:
            f.write(f"クラスター総数: {len(clusters)}\n")
            f.write(f"クラスターID一覧:\n")
            for cluster_id, records in clusters.items():
                f.write(f"  {cluster_id}: {len(records)}件のレコード\n")
    
    for cluster_id, records in clusters.items():
        processed_clusters += 1
        
        # 定期的に進捗を表示
        if processed_clusters % 10 == 0 or processed_clusters == len(clusters):
            elapsed = (datetime.now() - start_time).total_seconds()
            estimated_total = elapsed / processed_clusters * len(clusters)
            remaining = estimated_total - elapsed
            print(f"  クラスター {processed_clusters}/{len(clusters)} 処理中... "
                  f"経過: {int(elapsed)}秒, 残り: {int(remaining)}秒")
            
            # 定期的にキャッシュを保存
            if processed_clusters % 100 == 0:
                save_embedding_cache(embedding_cache, cache_dir, cache_identifier)
        
        # クラスターのレコードテキストを結合
        combined_text = ""
        record_texts = []  # 各レコードのテキスト
        
        for record in records:
            record_text = get_record_text(record, attr_fields)
            combined_text += record_text + " "
            record_texts.append(record_text)
        
        # デバッグ情報：クラスターとレコードの詳細ダンプ
        if DEBUG_MODE:
            cluster_dump_file = os.path.join(debug_dump_dir, f"cluster_{cluster_id}.txt")
            with open(cluster_dump_file, 'w', encoding='utf-8') as f:
                f.write(f"クラスターID: {cluster_id}\n")
                f.write(f"レコード数: {len(records)}\n\n")
                
                for i, record in enumerate(records):
                    f.write(f"=== レコード {i+1} ===\n")
                    f.write(f"ID: {record.get('id', '不明')}\n")
                    f.write(f"テキスト表現:\n{dump_record_text(record, attr_fields)}\n\n")
                    f.write(f"生成されたテキスト: {record_texts[i][:500]}...\n\n")
                
                f.write("\n=== 結合テキスト ===\n")
                f.write(combined_text[:1000] + "...\n")
        
        # テキストが長すぎる場合は切り詰める（APIの制限に対応）
        if len(combined_text) > 8000:
            debug_log(f"テキストが長すぎるため切り詰めます: {len(combined_text)} -> 8000文字 (クラスター {cluster_id})", DEBUG_LOG_FILE)
            combined_text = combined_text[:8000]
        
        # クラスターテキスト保存（デバッグ用）
        cluster_texts[cluster_id] = combined_text
        
        # キャッシュキーを生成
        cache_key = f"cluster_{cluster_id}"
        
        # キャッシュを確認
        if cache_key in embedding_cache:
            debug_log(f"キャッシュからエンベディングを取得: {cache_key}", DEBUG_LOG_FILE)
            cluster_embeddings[cluster_id] = embedding_cache[cache_key]
        else:
            # エンベディングを計算してキャッシュに保存
            debug_log(f"エンベディング計算: クラスター {cluster_id} (レコード数: {len(records)})", DEBUG_LOG_FILE)
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
    start_time = datetime.now()
    
    # デバッグ情報：類似度マトリックス
    if DEBUG_MODE:
        similarity_matrix_file = os.path.join(debug_dump_dir, "similarity_matrix.csv")
        with open(similarity_matrix_file, 'w', encoding='utf-8') as f:
            # ヘッダー行
            f.write("cluster_id_1,cluster_id_2,similarity,title_similarity,author_similarity,combined_score\n")
    
    high_similarity_pairs = []  # 高い類似度を持つペアを記録（デバッグ用）
    
    for i in range(len(cluster_ids)):
        c1 = cluster_ids[i]
        emb1 = cluster_embeddings[c1]
        
        for j in range(i + 1, len(cluster_ids)):
            c2 = cluster_ids[j]
            emb2 = cluster_embeddings[c2]
            processed += 1
            
            # 定期的に進捗を表示
            if processed % max(1, total_comparisons // 20) == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                estimated_total = elapsed / processed * total_comparisons
                remaining = estimated_total - elapsed
                print(f"  {processed}/{total_comparisons}ペア処理中... "
                      f"({int(processed/total_comparisons*100)}%) "
                      f"経過: {int(elapsed)}秒, 残り: {int(remaining)}秒")
            
            # 類似度計算
            similarity = cosine_similarity(emb1, emb2)
            
            # デバッグ用：類似度の詳細記録
            debug_log(f"クラスター類似度: {c1} <-> {c2} = {similarity:.4f}", DEBUG_LOG_FILE)
            
            # サンプルレコードのタイトルと著者の類似度を計算
            sample_record1 = clusters[c1][0] if clusters[c1] else None
            sample_record2 = clusters[c2][0] if clusters[c2] else None
            
            title_field = next((field for field, ftype in attr_fields.items() 
                              if ftype == "TEXT" or "title" in field.lower()), None)
            author_field = next((field for field, ftype in attr_fields.items() 
                               if "author" in field.lower()), None)
            
            title_similarity = 0
            author_similarity = 0
            
            if sample_record1 and sample_record2:
                if title_field:
                    title1 = get_record_attribute(sample_record1, title_field)
                    title2 = get_record_attribute(sample_record2, title_field)
                    title_similarity = string_similarity(title1, title2)
                
                if author_field:
                    author1 = get_record_attribute(sample_record1, author_field)
                    author2 = get_record_attribute(sample_record2, author_field)
                    author_similarity = string_similarity(author1, author2)
            
            # 総合スコア（仮）
            combined_score = similarity * 0.4 + title_similarity * 0.4 + author_similarity * 0.2
            
            # デバッグ情報：類似度マトリックスに追加
            if DEBUG_MODE:
                with open(similarity_matrix_file, 'a', encoding='utf-8') as f:
                    f.write(f"{c1},{c2},{similarity:.6f},{title_similarity:.6f},{author_similarity:.6f},{combined_score:.6f}\n")
            
            # 高い類似度のペアを記録（デバッグ用）
            if similarity >= 0.8 * threshold:
                high_similarity_pairs.append((c1, c2, similarity, title_similarity, author_similarity, combined_score))
            
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
                
                # デバッグ情報：レコードペアの詳細比較
                if DEBUG_MODE:
                    pair_comparison_file = os.path.join(debug_dump_dir, f"pair_{c1}_{c2}.txt")
                    with open(pair_comparison_file, 'w', encoding='utf-8') as f:
                        f.write(f"クラスター比較: {c1} <-> {c2}\n")
                        f.write(f"クラスター類似度: {similarity:.6f}\n")
                        f.write(f"サンプルタイトル類似度: {title_similarity:.6f}\n")
                        f.write(f"サンプル著者類似度: {author_similarity:.6f}\n")
                        f.write(f"総合類似度: {combined_score:.6f}\n\n")
                        
                        f.write("=== クラスター1のテキスト ===\n")
                        f.write(cluster_texts[c1][:500] + "...\n\n")
                        
                        f.write("=== クラスター2のテキスト ===\n")
                        f.write(cluster_texts[c2][:500] + "...\n\n")
                        
                        f.write("=== レコードペアの比較 ===\n")
                
                # レコードペアの比較
                for r1 in records1_sample:
                    for r2 in records2_sample:
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
                        record_similarity = similarity * 0.4 + title_similarity * 0.4 + author_similarity * 0.2
                        
                        # デバッグ情報：レコードペアの詳細比較
                        if DEBUG_MODE:
                            with open(pair_comparison_file, 'a', encoding='utf-8') as f:
                                f.write(f"\nレコード比較: {r1.get('id', '不明')} <-> {r2.get('id', '不明')}\n")
                                if title_field:
                                    f.write(f"タイトル1: {title1}\n")
                                    f.write(f"タイトル2: {title2}\n")
                                    f.write(f"タイトル類似度: {title_similarity:.6f}\n")
                                if author_field:
                                    f.write(f"著者1: {author1}\n")
                                    f.write(f"著者2: {author2}\n")
                                    f.write(f"著者類似度: {author_similarity:.6f}\n")
                                f.write(f"総合類似度: {record_similarity:.6f}\n")
                        
                        if record_similarity > best_record_similarity:
                            best_record_similarity = record_similarity
                            best_pair = (r1, r2)
                
                # 最良ペアの情報を記録
                if best_pair and best_record_similarity >= threshold:
                    id1 = str(best_pair[0].get("id", f"c1_{c1}"))
                    id2 = str(best_pair[1].get("id", f"c2_{c2}"))
                    
                    # レコード情報を取得
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
    
    # デバッグ情報：高い類似度のペアを出力
    if DEBUG_MODE:
        high_similarity_file = os.path.join(debug_dump_dir, "high_similarity_pairs.csv")
        with open(high_similarity_file, 'w', encoding='utf-8') as f:
            f.write("cluster_id_1,cluster_id_2,similarity,title_similarity,author_similarity,combined_score\n")
            for c1, c2, sim, title_sim, author_sim, combined in sorted(high_similarity_pairs, key=lambda x: x[5], reverse=True):
                f.write(f"{c1},{c2},{sim:.6f},{title_sim:.6f},{author_sim:.6f},{combined:.6f}\n")
    
    # 類似度でソート（降順）
    matches.sort(key=lambda x: x[2], reverse=True)
    
    return matches

def main():
    parser = argparse.ArgumentParser(description="デバッグ機能付きマッチングスクリプト")
    parser.add_argument("--input", "-i", type=str, required=True, help="入力YAMLファイルパス")
    parser.add_argument("--output", "-o", type=str, default="matches.yaml", help="出力YAMLファイルパス")
    parser.add_argument("--api-key", "-k", type=str, help="OpenAI APIキー")
    parser.add_argument("--threshold", "-t", type=float, default=SIMILARITY_THRESHOLD, help="類似度のしきい値")
    parser.add_argument("--cache-dir", "-c", type=str, default=".cache", help="キャッシュディレクトリ")
    parser.add_argument("--debug", "-d", action="store_true", help="デバッグモードを有効にする")
    parser.add_argument("--debug-dir", type=str, default="debug_dumps", help="デバッグ情報の保存先ディレクトリ")
    
    args = parser.parse_args()
    
    # デバッグモードの設定
    global DEBUG_MODE, DEBUG_LOG_FILE
    DEBUG_MODE = args.debug
    DEBUG_LOG_FILE = os.path.join(args.debug_dir, "debug_log.txt")
    
    if DEBUG_MODE:
        # デバッグディレクトリを作成
        os.makedirs(args.debug_dir, exist_ok=True)
        
        # デバッグログファイルを初期化
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
    
    # 全クラスターの処理（メモリ効率化とキャッシング）
    matches = process_all_clusters(
        clusters, 
        api_key, 
        attr_fields,
        args.threshold,
        args.cache_dir,
        args.debug_dir
    )
    
    print(f"マッチング結果: {len(matches)}件のマッチを検出")
    
    # 上位10件を表示
    for i, (id1, id2, score, details) in enumerate(matches[:10]):
        print(f"マッチ {i+1}: {id1} <-> {id2} (類似度: {score:.4f})")
        if "title1" in details and "title2" in details:
            print(f"  タイトル1: {details['title1']}")
            print(f"  タイトル2: {details['title2']}")
        if "author1" in details and "author2" in details:
            print(f"  著者1: {details['author1']}")
            print(f"  著者2: {details['author2']}")
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
    with open(args.output, "w", encoding='utf-8') as f:
        yaml.dump(output_data, f, default_flow_style=False, allow_unicode=True)
    
    print(f"結果を{args.output}に保存しました。")
    
    # 処理時間を表示
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    print(f"処理終了: {end_time}")
    print(f"処理時間: {elapsed:.1f}秒 ({elapsed/60:.1f}分)")
    
    # デバッグモードの場合、追加情報を表示
    if DEBUG_MODE:
        print(f"\nデバッグ情報:")
        print(f"  デバッグログ: {DEBUG_LOG_FILE}")
        print(f"  デバッグダンプ: {args.debug_dir}/")
        print(f"  高類似度ペア: {args.debug_dir}/high_similarity_pairs.csv")
        print(f"  類似度マトリックス: {args.debug_dir}/similarity_matrix.csv")
        print(f"  クラスター情報: {args.debug_dir}/cluster_info.txt")

if __name__ == "__main__":
    import hashlib
    main()