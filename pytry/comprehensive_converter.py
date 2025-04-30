#!/usr/bin/env python3
"""
修正版：エンベディングベースの書誌レコードクラスタリング

このスクリプトは書誌レコードのクラスタリングを行うための包括的なソリューションです。
元のクラスターIDは使用せず、純粋にテキストの意味的類似性に基づいてクラスタリングを実行します。

特徴:
- 元のクラスターIDに依存しない純粋なエンベディングベースのクラスタリング
- 日本語テキストの正規化による高精度なマッチング
- 詳細なメトリクス計算
- 例と一致する出力形式
"""

import os
import sys
import yaml
import json
import argparse
import numpy as np
import time
import uuid
import unicodedata
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Set, Union
from difflib import SequenceMatcher

# 依存ライブラリのインポート試行
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("警告: 'requests'ライブラリがインストールされていません。エンベディングはシミュレーションされます。")

# 設定
SIMILARITY_THRESHOLD = 0.75  # デフォルトの類似度しきい値
MAX_RETRIES = 10  # APIコールの最大リトライ回数
RETRY_DELAY = 1  # リトライ間の待機時間（秒）
EMBEDDING_DIM = 1536  # デフォルトのOpenAIエンベディング次元
DEBUG_MODE = False  # デバッグモード（詳細な出力）

def debug_log(message):
    """DEBUG_MODEが有効の場合、デバッグログメッセージを出力"""
    if DEBUG_MODE:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"[DEBUG] [{timestamp}] {message}")

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
    """
    入力データからレコードを抽出する。
    元のクラスターIDはoriginal_cluster_idとして保存するが、
    新しいクラスタリングには使用しない。
    """
    records = []
    
    # データ構造からレコードを抽出
    if "group" in data and isinstance(data["group"], list):
        for group_idx, group in enumerate(data["group"]):
            if "records" in group and isinstance(group["records"], list):
                for record_idx, record in enumerate(group["records"]):
                    # 新しいレコードを作成（ディープコピー）
                    new_record = {}
                    
                    # IDの処理
                    if "id" in record:
                        new_record["id"] = record["id"]
                    else:
                        new_record["id"] = f"record_{group_idx}_{record_idx}"
                    
                    # インデックスの処理
                    if "idx" in record:
                        new_record["idx"] = record["idx"]
                    else:
                        new_record["idx"] = record_idx
                    
                    # 元のクラスターIDを保存
                    if "cluster_id" in record:
                        new_record["original_cluster_id"] = record["cluster_id"]
                    else:
                        new_record["original_cluster_id"] = f"group_{group_idx}"
                    
                    # データの処理
                    if "data" in record and isinstance(record["data"], dict):
                        new_record["data"] = record["data"].copy()
                    else:
                        # データフィールドが無い場合、他のフィールドからデータを構築
                        new_record["data"] = {}
                        for key, value in record.items():
                            if key not in ["id", "idx", "cluster_id", "original_cluster_id"]:
                                new_record["data"][key] = value
                    
                    records.append(new_record)
    
    # その他の構造の処理
    elif "records" in data:
        if isinstance(data["records"], list):
            # レコードがリストの場合
            for idx, record in enumerate(data["records"]):
                new_record = {}
                
                if "id" in record:
                    new_record["id"] = record["id"]
                else:
                    new_record["id"] = f"record_{idx}"
                
                if "idx" in record:
                    new_record["idx"] = record["idx"]
                else:
                    new_record["idx"] = idx
                
                if "cluster_id" in record:
                    new_record["original_cluster_id"] = record["cluster_id"]
                
                if "data" in record and isinstance(record["data"], dict):
                    new_record["data"] = record["data"].copy()
                else:
                    new_record["data"] = {}
                    for key, value in record.items():
                        if key not in ["id", "idx", "cluster_id", "original_cluster_id"]:
                            new_record["data"][key] = value
                
                records.append(new_record)
        elif isinstance(data["records"], dict):
            # レコードがクラスターIDでグループ化されている場合
            for cluster_id, cluster_records in data["records"].items():
                if isinstance(cluster_records, list):
                    for idx, record in enumerate(cluster_records):
                        new_record = {}
                        
                        if "id" in record:
                            new_record["id"] = record["id"]
                        else:
                            new_record["id"] = f"record_{cluster_id}_{idx}"
                        
                        if "idx" in record:
                            new_record["idx"] = record["idx"]
                        else:
                            new_record["idx"] = idx
                        
                        new_record["original_cluster_id"] = cluster_id
                        
                        if "data" in record and isinstance(record["data"], dict):
                            new_record["data"] = record["data"].copy()
                        else:
                            new_record["data"] = {}
                            for key, value in record.items():
                                if key not in ["id", "idx", "cluster_id", "original_cluster_id"]:
                                    new_record["data"][key] = value
                        
                        records.append(new_record)
                elif isinstance(cluster_records, dict):
                    # 単一レコードの場合
                    record = cluster_records
                    new_record = {}
                    
                    if "id" in record:
                        new_record["id"] = record["id"]
                    else:
                        new_record["id"] = f"record_{cluster_id}_0"
                    
                    if "idx" in record:
                        new_record["idx"] = record["idx"]
                    else:
                        new_record["idx"] = 0
                    
                    new_record["original_cluster_id"] = cluster_id
                    
                    if "data" in record and isinstance(record["data"], dict):
                        new_record["data"] = record["data"].copy()
                    else:
                        new_record["data"] = {}
                        for key, value in record.items():
                            if key not in ["id", "idx", "cluster_id", "original_cluster_id"]:
                                new_record["data"][key] = value
                    
                    records.append(new_record)
    
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

def normalize_japanese_text(text: str) -> str:
    """より良い比較のために日本語テキストを正規化する"""
    if not text:
        return ""
    
    # 文字列でない場合は文字列に変換
    if not isinstance(text, str):
        text = str(text)
    
    # Unicodeの正規化（NFKC: 互換分解後に正準合成）
    text = unicodedata.normalize('NFKC', text)
    
    # 句読点や特殊文字を削除
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    
    # 小文字に変換
    text = text.lower()
    
    return text

def get_record_text(record: Dict, field_types: Dict[str, str] = None) -> str:
    """エンベディング計算のためにレコードから重み付きテキストを抽出"""
    text_parts = []
    
    if "data" in record and isinstance(record["data"], dict):
        data = record["data"]
        
        # フィールド型が提供されている場合はそれを使用、それ以外はフィールド名で優先順位付け
        if field_types:
            for field, field_type in field_types.items():
                if field in data:
                    value = data[field]
                    if field_type == "TEXT" or "title" in field.lower():
                        # タイトルを3回繰り返して重みを高く
                        text_parts.extend([f"{field}: {value}"] * 3)
                    elif field_type == "COMPLEMENT_JA" or "author" in field.lower():
                        # 著者を2回繰り返して中程度の重み
                        text_parts.extend([f"{field}: {value}"] * 2)
                    else:
                        text_parts.append(f"{field}: {value}")
        else:
            # フィールド型が提供されていない場合、フィールド名で重みを決定
            for field, value in data.items():
                if "title" in field.lower():
                    text_parts.extend([f"{field}: {value}"] * 3)
                elif "author" in field.lower():
                    text_parts.extend([f"{field}: {value}"] * 2)
                else:
                    text_parts.append(f"{field}: {value}")
    
    # テキストが抽出されなかった場合は文字列表現を使用
    if not text_parts:
        return str(record)
    
    return " ".join(text_parts)

def simulate_embedding(text: str) -> np.ndarray:
    """
    テキスト内容に基づいてエンベディングベクトルをシミュレートする。
    APIアクセスが利用できない場合のフォールバック。
    """
    # テキストから決定論的なシードを作成
    text_seed = sum(ord(c) for c in text)
    np.random.seed(text_seed)
    
    # ランダムなエンベディングベクトルを生成
    embedding = np.random.randn(EMBEDDING_DIM)
    
    # 単位長さに正規化（コサイン類似度計算の要件）
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding

def get_embedding(text: str, api_key: str = None) -> np.ndarray:
    """
    OpenAI APIまたはシミュレーションを使用してテキストのエンベディングを取得。
    
    requestsライブラリが利用可能でAPIキーが提供されている場合はAPIを呼び出す。
    それ以外の場合はシミュレーションにフォールバック。
    """
    # APIアクセスが利用できない場合、エンベディングをシミュレート
    if not REQUESTS_AVAILABLE or not api_key:
        debug_log(f"長さ{len(text)}のテキストのエンベディングをシミュレート")
        return simulate_embedding(text)
    
    # APIが利用可能、呼び出しを行う
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # テキスト長をAPI制限に制限
    input_text = text if len(text) <= 8000 else text[:8000]
    
    data = {
        "model": "text-embedding-ada-002",
        "input": input_text
    }
    
    debug_log(f"長さ{len(input_text)}のテキストのエンベディングをリクエスト")
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                data=json.dumps(data),
                timeout=30
            )
            
            if response.status_code == 200:
                embedding = np.array(response.json()["data"][0]["embedding"])
                return embedding
            elif response.status_code == 429:  # レート制限
                wait_time = RETRY_DELAY * (attempt + 1)
                print(f"レート制限に達しました。{wait_time}秒待機します...")
                time.sleep(wait_time)
            else:
                print(f"APIエラー ({attempt+1}/{MAX_RETRIES}): {response.text}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"リクエストエラー ({attempt+1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    
    # すべての試行が失敗した場合、エンベディングをシミュレート
    print("API呼び出しが失敗しました。シミュレートされたエンベディングにフォールバックします。")
    return simulate_embedding(text)

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

def string_similarity(str1: str, str2: str) -> float:
    """SequenceMatcherを使用して文字列の類似度比率を計算"""
    if not isinstance(str1, str):
        str1 = str(str1)
    if not isinstance(str2, str):
        str2 = str(str2)
    return SequenceMatcher(None, str1, str2).ratio()

def calculate_record_embeddings(records: List[Dict], field_types: Dict[str, str],
                               api_key: str = None) -> Dict[str, np.ndarray]:
    """すべてのレコードのエンベディングを計算"""
    record_embeddings = {}
    
    print(f"{len(records)}件のレコードのエンベディングを計算中...")
    start_time = datetime.now()
    
    for i, record in enumerate(records):
        # レコードIDの取得
        record_id = record.get("id", f"record_{i}")
        
        # テキスト表現の取得
        text = get_record_text(record, field_types)
        
        # エンベディングの計算
        embedding = get_embedding(text, api_key)
        record_embeddings[record_id] = embedding
        
        # 進捗状況を定期的に表示
        if (i + 1) % 50 == 0 or i == len(records) - 1:
            progress = (i + 1) / len(records) * 100
            elapsed = (datetime.now() - start_time).total_seconds()
            remaining = elapsed / (i + 1) * (len(records) - i - 1)
            print(f"  進捗: {i+1}/{len(records)}件 ({progress:.1f}%) - "
                  f"経過: {elapsed:.1f}秒, 残り: {remaining:.1f}秒")
    
    return record_embeddings

def create_new_clusters(records: List[Dict], record_embeddings: Dict[str, np.ndarray], 
                      field_types: Dict[str, str],
                      threshold: float = SIMILARITY_THRESHOLD) -> Dict[str, List[Dict]]:
    """エンベディングに基づいて完全に新しいクラスターを作成"""
    print(f"エンベディングに基づいて新しいクラスターを作成中...")
    
    # レコードIDから完全なレコードへのマッピングを作成
    id_to_record = {record.get("id", f"record_{i}"): record for i, record in enumerate(records)}
    
    # 新しいクラスターを初期化
    clusters = {}
    processed_record_ids = set()
    
    # 各レコードについて
    for i, record in enumerate(records):
        record_id = record.get("id", f"record_{i}")
        
        # すでに処理済みの場合はスキップ
        if record_id in processed_record_ids:
            continue
        
        # エンベディングがない場合はスキップ
        if record_id not in record_embeddings:
            continue
        
        # 新しいクラスターを作成
        new_cluster_id = f"cluster_{uuid.uuid4().hex[:8]}"
        cluster_records = [record]
        processed_record_ids.add(record_id)
        
        # 現在のクラスターに対する残りのレコードの類似度をチェック
        for j, other_record in enumerate(records):
            other_id = other_record.get("id", f"record_{j}")
            
            # すでに処理済みまたは同じレコードの場合はスキップ
            if other_id in processed_record_ids or other_id == record_id:
                continue
            
            # エンベディングがない場合はスキップ
            if other_id not in record_embeddings:
                continue
            
            # 類似度の計算
            record_emb = record_embeddings[record_id]
            other_emb = record_embeddings[other_id]
            emb_similarity = cosine_similarity(record_emb, other_emb)
            
            # しきい値レベルの類似度をチェック
            if emb_similarity >= threshold * 0.8:  # 最初のパスでしきい値を下げる
                # タイトルと著者の類似度を計算
                title_similarity = 0
                author_similarity = 0
                title_field = None
                author_field = None
                
                # タイトルと著者のフィールドを探す
                for field, field_type in field_types.items():
                    if field_type == "TEXT" and "title" in field.lower():
                        title_field = field
                    elif field_type == "COMPLEMENT_JA" and "author" in field.lower():
                        author_field = field
                
                # タイトルの類似度を計算
                if title_field and "data" in record and "data" in other_record:
                    title1 = record["data"].get(title_field, "")
                    title2 = other_record["data"].get(title_field, "")
                    
                    # タイトルを正規化（日本語テキストに特に重要）
                    title1_norm = normalize_japanese_text(title1)
                    title2_norm = normalize_japanese_text(title2)
                    
                    title_similarity = string_similarity(title1_norm, title2_norm)
                
                # 著者の類似度を計算
                if author_field and "data" in record and "data" in other_record:
                    author1 = record["data"].get(author_field, "")
                    author2 = other_record["data"].get(author_field, "")
                    
                    # 著者を正規化
                    author1_norm = normalize_japanese_text(author1)
                    author2_norm = normalize_japanese_text(author2)
                    
                    author_similarity = string_similarity(author1_norm, author2_norm)
                
                # 重み付き類似度スコアの計算
                weighted_similarity = (
                    emb_similarity * 0.3 +
                    title_similarity * 0.5 +
                    author_similarity * 0.2
                )
                
                # しきい値を超える類似性があれば、このレコードをクラスターに追加
                if weighted_similarity >= threshold:
                    cluster_records.append(other_record)
                    processed_record_ids.add(other_id)
                    
                    # クラスター内の各レコードに新しいクラスターIDを設定
                    for rec in cluster_records:
                        rec["cluster_id"] = new_cluster_id
        
        # クラスターが空でなければ追加
        if cluster_records:
            clusters[new_cluster_id] = cluster_records
    
    # 処理されていないレコードがあれば、それぞれを独自のクラスターに
    for record in records:
        record_id = record.get("id", "")
        if record_id not in processed_record_ids:
            new_cluster_id = f"cluster_{uuid.uuid4().hex[:8]}"
            record["cluster_id"] = new_cluster_id
            clusters[new_cluster_id] = [record]
            processed_record_ids.add(record_id)
    
    print(f"{len(clusters)}個のクラスターを作成しました")
    return clusters

def find_similar_clusters(clusters: Dict[str, List[Dict]], record_embeddings: Dict[str, np.ndarray],
                        field_types: Dict[str, str], 
                        threshold: float = SIMILARITY_THRESHOLD) -> List[Dict]:
    """エンベディングと類似度に基づいて、マージすべき類似クラスターを探す"""
    cluster_similarities = []
    
    print(f"{len(clusters)}個のクラスター間の類似性を計算中...")
    
    # クラスターIDのリストを取得
    cluster_ids = list(clusters.keys())
    total_comparisons = len(cluster_ids) * (len(cluster_ids) - 1) // 2
    
    if total_comparisons == 0:
        print("比較するクラスターがありません")
        return []
    
    processed = 0
    start_time = datetime.now()
    
    # 各クラスターペアを比較
    for i in range(len(cluster_ids)):
        cluster_id1 = cluster_ids[i]
        cluster1 = clusters[cluster_id1]
        
        for j in range(i + 1, len(cluster_ids)):
            cluster_id2 = cluster_ids[j]
            cluster2 = clusters[cluster_id2]
            
            processed += 1
            if processed % max(1, total_comparisons // 20) == 0:
                progress = processed / total_comparisons * 100
                elapsed = (datetime.now() - start_time).total_seconds()
                remaining = elapsed / processed * (total_comparisons - processed)
                print(f"  進捗: {processed}/{total_comparisons}回 ({progress:.1f}%) - "
                      f"経過: {elapsed:.1f}秒, 残り: {remaining:.1f}秒")
            
            # クラスター間の類似度を計算
            total_similarity = 0
            comparison_count = 0
            
            # 各クラスターから代表的なレコードを選択（最大5件）
            records1 = cluster1[:min(5, len(cluster1))]
            records2 = cluster2[:min(5, len(cluster2))]
            
            for record1 in records1:
                record_id1 = record1.get("id", "")
                if record_id1 not in record_embeddings:
                    continue
                    
                for record2 in records2:
                    record_id2 = record2.get("id", "")
                    if record_id2 not in record_embeddings:
                        continue
                    
                    # エンベディング類似度の計算
                    emb_similarity = cosine_similarity(record_embeddings[record_id1], record_embeddings[record_id2])
                    
                    # タイトルと著者の類似度を計算
                    title_similarity = 0
                    author_similarity = 0
                    
                    # タイトルと著者のフィールドを探す
                    title_field = None
                    author_field = None
                    
                    for field, field_type in field_types.items():
                        if field_type == "TEXT" and "title" in field.lower():
                            title_field = field
                        elif field_type == "COMPLEMENT_JA" and "author" in field.lower():
                            author_field = field
                    
                    # タイトルの類似度を計算
                    if title_field and "data" in record1 and "data" in record2:
                        title1 = record1["data"].get(title_field, "")
                        title2 = record2["data"].get(title_field, "")
                        
                        # タイトルを正規化
                        title1_norm = normalize_japanese_text(title1)
                        title2_norm = normalize_japanese_text(title2)
                        
                        title_similarity = string_similarity(title1_norm, title2_norm)
                    
                    # 著者の類似度を計算
                    if author_field and "data" in record1 and "data" in record2:
                        author1 = record1["data"].get(author_field, "")
                        author2 = record2["data"].get(author_field, "")
                        
                        # 著者を正規化
                        author1_norm = normalize_japanese_text(author1)
                        author2_norm = normalize_japanese_text(author2)
                        
                        author_similarity = string_similarity(author1_norm, author2_norm)
                    
                    # 重み付き類似度スコアの計算
                    weighted_similarity = (
                        emb_similarity * 0.3 +
                        title_similarity * 0.5 +
                        author_similarity * 0.2
                    )
                    
                    total_similarity += weighted_similarity
                    comparison_count += 1
            
            # 平均類似度の計算
            if comparison_count > 0:
                avg_similarity = total_similarity / comparison_count
                
                # しきい値を超える場合、マージ候補として追加
                if avg_similarity >= threshold:
                    # サンプルレコードを取得（代表として）
                    sample_record1 = cluster1[0] if cluster1 else None
                    sample_record2 = cluster2[0] if cluster2 else None
                    
                    # サンプルのタイトルと著者を取得
                    title1 = ""
                    title2 = ""
                    author1 = ""
                    author2 = ""
                    
                    if title_field and sample_record1 and "data" in sample_record1:
                        title1 = sample_record1["data"].get(title_field, "")
                    if title_field and sample_record2 and "data" in sample_record2:
                        title2 = sample_record2["data"].get(title_field, "")
                    
                    if author_field and sample_record1 and "data" in sample_record1:
                        author1 = sample_record1["data"].get(author_field, "")
                    if author_field and sample_record2 and "data" in sample_record2:
                        author2 = sample_record2["data"].get(author_field, "")
                    
                    # クラスター類似度情報を追加
                    cluster_similarities.append({
                        "cluster_id1": cluster_id1,
                        "cluster_id2": cluster_id2,
                        "similarity": float(avg_similarity),
                        "title1": title1,
                        "title2": title2,
                        "author1": author1,
                        "author2": author2,
                        "records1": len(cluster1),
                        "records2": len(cluster2)
                    })
    
    # 類似度でソート（降順）
    cluster_similarities.sort(key=lambda x: x["similarity"], reverse=True)
    
    return cluster_similarities

def merge_similar_clusters(clusters: Dict[str, List[Dict]], similarities: List[Dict]) -> Dict[str, List[Dict]]:
    """類似度に基づいて類似クラスターをマージする"""
    if not similarities:
        return clusters
    
    print(f"{len(similarities)}個の類似クラスターペアを見つけました")
    
    # マージ前のクラスター数
    original_cluster_count = len(clusters)
    
    # マージに使用するクラスターセット（実際に存在するクラスターのみ）
    valid_clusters = set(clusters.keys())
    
    # マージした数をカウント
    merged_count = 0
    
    # 類似度順にクラスターをマージ
    for sim_info in similarities:
        cluster_id1 = sim_info["cluster_id1"]
        cluster_id2 = sim_info["cluster_id2"]
        
        # 両方のクラスターがまだ存在するか確認
        if cluster_id1 not in valid_clusters or cluster_id2 not in valid_clusters:
            continue
        
        # クラスター1とクラスター2のレコードを取得
        records1 = clusters[cluster_id1]
        records2 = clusters[cluster_id2]
        
        # レコードをマージ
        for record in records2:
            # クラスターIDを更新
            record["cluster_id"] = cluster_id1
        
        # クラスター1にクラスター2のレコードを追加
        clusters[cluster_id1].extend(records2)
        
        # クラスター2を削除
        del clusters[cluster_id2]
        valid_clusters.remove(cluster_id2)
        
        merged_count += 1
    
    print(f"{merged_count}個のクラスターをマージしました")
    print(f"元のクラスター数: {original_cluster_count}, 最終クラスター数: {len(clusters)}")
    
    return clusters

def format_output_groups(clusters: Dict[str, List[Dict]]) -> List[Dict]:
    """クラスターを出力グループ形式にフォーマット"""
    output_groups = []
    
    for cluster_id, records in clusters.items():
        # indexでソート
        sorted_records = sorted(records, key=lambda r: r.get("idx", 0))
        
        # correctフィールドの作成
        correct_indices = list(range(len(sorted_records)))
        
        group = {
            "perfect_match": False,  # デフォルト値
            "records": sorted_records,
            "correct": [correct_indices]  # すべてのレコードが同じグループに属すると仮定
        }
        
        output_groups.append(group)
    
    return output_groups

def calculate_output_metrics(groups: List[Dict], all_records: List[Dict]) -> Dict:
    """出力サマリーのメトリクスを計算する"""
    total_records = sum(len(group.get("records", [])) for group in groups)
    
    # 最終クラスターの数
    num_of_groups_inference = len(groups)
    
    # 元のクラスターの数を計算
    original_cluster_ids = set()
    for record in all_records:
        if "original_cluster_id" in record:
            original_cluster_ids.add(record["original_cluster_id"])
    
    num_of_groups_correct = len(original_cluster_ids)
    
    # 元のクラスタリングと新しいクラスタリングを比較
    original_cluster_to_records = defaultdict(list)
    for record in all_records:
        if "id" in record and "original_cluster_id" in record:
            original_cluster_to_records[record["original_cluster_id"]].append(record["id"])
    
    new_cluster_to_records = defaultdict(list)
    for group in groups:
        for record in group.get("records", []):
            if "id" in record and "cluster_id" in record:
                new_cluster_to_records[record["cluster_id"]].append(record["id"])
    
    # ペアレベルの計算
    original_pairs = set()
    for cluster_id, record_ids in original_cluster_to_records.items():
        for i in range(len(record_ids)):
            for j in range(i+1, len(record_ids)):
                pair = tuple(sorted([record_ids[i], record_ids[j]]))
                original_pairs.add(pair)
    
    new_pairs = set()
    for cluster_id, record_ids in new_cluster_to_records.items():
        for i in range(len(record_ids)):
            for j in range(i+1, len(record_ids)):
                pair = tuple(sorted([record_ids[i], record_ids[j]]))
                new_pairs.add(pair)
    
    # 正しく予測されたペア
    correct_pairs = original_pairs.intersection(new_pairs)
    
    # ペアレベルの精度と再現率
    if len(new_pairs) > 0:
        precision_pair = len(correct_pairs) / len(new_pairs)
    else:
        precision_pair = 0
    
    if len(original_pairs) > 0:
        recall_pair = len(correct_pairs) / len(original_pairs)
    else:
        recall_pair = 0
    
    # F1スコアの計算
    if precision_pair + recall_pair > 0:
        f1_pair = 2 * precision_pair * recall_pair / (precision_pair + recall_pair)
    else:
        f1_pair = 0
    
    # グループレベルの精度と再現率
    precision_group_count = 0
    recall_group_count = 0
    total_new_records = 0
    total_original_records = 0
    
    # 各新しいクラスターについて、最も重複する元のクラスターとの類似度を計算
    for cluster_id, record_ids in new_cluster_to_records.items():
        max_overlap = 0
        for original_id, original_ids in original_cluster_to_records.items():
            overlap = len(set(record_ids).intersection(set(original_ids)))
            max_overlap = max(max_overlap, overlap)
        
        precision_group_count += max_overlap
        total_new_records += len(record_ids)
    
    # 各元のクラスターについて、最も重複する新しいクラスターとの類似度を計算
    for original_id, original_ids in original_cluster_to_records.items():
        max_overlap = 0
        for cluster_id, record_ids in new_cluster_to_records.items():
            overlap = len(set(original_ids).intersection(set(record_ids)))
            max_overlap = max(max_overlap, overlap)
        
        recall_group_count += max_overlap
        total_original_records += len(original_ids)
    
    # グループレベルの精度と再現率
    if total_new_records > 0:
        precision_group = precision_group_count / total_new_records
    else:
        precision_group = 0
    
    if total_original_records > 0:
        recall_group = recall_group_count / total_original_records
    else:
        recall_group = 0
    
    # 完全一致グループの計算
    complete_match_count = 0
    complete_matched_clusters = []
    
    for original_id, original_ids in original_cluster_to_records.items():
        original_set = set(original_ids)
        for cluster_id, record_ids in new_cluster_to_records.items():
            new_set = set(record_ids)
            if original_set == new_set:
                complete_match_count += 1
                complete_matched_clusters.append(original_id)
                break
    
    if len(original_cluster_to_records) > 0:
        complete_group = complete_match_count / len(original_cluster_to_records)
    else:
        complete_group = 0
    
    # メトリクスの作成
    metrics = {
        "type": "RESULT",
        "num_of_record": total_records,
        "num_of_groups(correct)": num_of_groups_correct,
        "num_of_groups(inference)": num_of_groups_inference,
        "config_match": None,
        "config_mismatch": None,
        "crowdsourcing_count": len(all_records),
        "f1(pair)": f"{f1_pair:.5f}",
        "precision(pair)": f"{precision_pair:.5f}",
        "recall(pair)": f"{recall_pair:.5f}",
        "complete(group)": complete_group,
        "precision(group)": precision_group,
        "recall(group)": recall_group,
        "complete_group": str(complete_matched_clusters)
    }
    
    return metrics

def format_final_output(groups: List[Dict], metrics: Dict) -> Dict:
    """出力を例と完全に一致する形式でフォーマット"""
    output = {
        "version": "3.1",
        "type": "RESULT",
        "id": str(uuid.uuid4()),
        "summary": metrics,
        "group": groups
    }
    
    return output

def main():
    parser = argparse.ArgumentParser(description="エンベディングベースの書誌レコードクラスタリング")
    parser.add_argument("--input", "-i", type=str, required=True, help="入力YAMLファイルパス")
    parser.add_argument("--output", "-o", type=str, default="result.yaml", help="出力YAMLファイルパス")
    parser.add_argument("--api-key", "-k", type=str, help="OpenAI APIキー（オプション）")
    parser.add_argument("--threshold", "-t", type=float, default=SIMILARITY_THRESHOLD, help="類似度しきい値")
    parser.add_argument("--debug", "-d", action="store_true", help="デバッグモードを有効化")
    
    args = parser.parse_args()
    
    # デバッグモードの設定
    global DEBUG_MODE
    DEBUG_MODE = args.debug
    
    # 処理タイマーの開始
    start_time = datetime.now()
    print(f"処理開始時刻: {start_time}")
    
    # 入力データの読み込み
    data = load_yaml(args.input)
    
    # レコードの抽出
    records = extract_records(data)
    print(f"{len(records)}件のレコードを抽出しました")
    
    # レコードフィールドの分析
    field_types = analyze_record_fields(records)
    print(f"{len(field_types)}種類のフィールドを分析しました:")
    for field, field_type in field_types.items():
        print(f"  {field}: {field_type}")
    
    # レコードエンベディングの計算
    record_embeddings = calculate_record_embeddings(records, field_types, args.api_key)
    print(f"{len(record_embeddings)}件のレコードのエンベディングを計算しました")
    
    # 新しいクラスターの作成
    clusters = create_new_clusters(records, record_embeddings, field_types, args.threshold)
    
    # 類似クラスターの検索
    cluster_similarities = find_similar_clusters(clusters, record_embeddings, field_types, args.threshold)
    
    # クラスター統合
    merged_clusters = merge_similar_clusters(clusters, cluster_similarities)
    
    # 出力グループのフォーマット
    output_groups = format_output_groups(merged_clusters)
    
    # 出力メトリクスの計算
    metrics = calculate_output_metrics(output_groups, records)
    
    # 最終出力のフォーマット
    output_data = format_final_output(output_groups, metrics)
    
    # 出力の保存
    save_yaml(output_data, args.output)
    
    # 終了時刻とサマリー
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    print(f"処理完了時刻: {end_time}")
    print(f"合計処理時間: {elapsed:.2f}秒")
    print(f"入力レコード数: {len(records)}")
    print(f"出力クラスター数: {len(merged_clusters)}")
    print(f"出力ファイル: {args.output}")
    
    # メトリクスの表示
    print("\n評価メトリクス:")
    print(f"  F1スコア (ペア): {metrics['f1(pair)']}")
    print(f"  精度 (ペア): {metrics['precision(pair)']}")
    print(f"  再現率 (ペア): {metrics['recall(pair)']}")
    print(f"  完全一致グループ率: {metrics['complete(group)']}")
    print(f"  精度 (グループ): {metrics['precision(group)']}")
    print(f"  再現率 (グループ): {metrics['recall(group)']}")

if __name__ == "__main__":
    main()