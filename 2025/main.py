#!/usr/bin/env python3
"""
ハイブリッド書誌レコードマッチングシステム

このスクリプトは、書誌レコードのクラスタリングを行うための包括的なソリューションです。
エンベディング、LLM、一貫性チェックを組み合わせて高精度なマッチングを実現します。

特徴:
- ブロッキングによる効率的な候補ペア生成
- エンベディング + LLMのハイブリッドマッチング
- 推移律に基づく一貫性チェック
- 詳細なメトリクス計算
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
import pickle
import hashlib
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Set, Union
from difflib import SequenceMatcher
import requests
from tqdm import tqdm

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("警告: networkx がインストールされていません。一貫性チェックに制限があります。")

try:
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("警告: scikit-learn がインストールされていません。ブロッキングに制限があります。")

# 定数
SIMILARITY_THRESHOLD = 0.8  # デフォルトの類似度しきい値
MAX_RETRIES = 3  # APIコールの最大リトライ回数
RETRY_DELAY = 2  # リトライ間の待機時間（秒）
EMBEDDING_DIM = 1536  # OpenAIエンベディング次元
DEBUG_MODE = False  # デバッグモード（詳細な出力）
CACHE_DIR = ".cache"  # キャッシュディレクトリ

# ユーティリティ関数
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
    """入力データからレコードを抽出する"""
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

def format_record_for_llm(record: Dict, field_types: Dict[str, str] = None) -> str:
    """LLMに渡すためのレコードテキスト形式を作成"""
    text_parts = []
    
    if "data" in record and isinstance(record["data"], dict):
        data = record["data"]
        
        # フィールド型が提供されている場合はそれを使用、それ以外はフィールド名で優先順位付け
        important_fields = []
        other_fields = []
        
        if field_types:
            for field, field_type in field_types.items():
                if field in data:
                    if field_type == "TEXT" and "title" in field.lower():
                        important_fields.append(f"タイトル: {data[field]}")
                    elif field_type == "COMPLEMENT_JA" and "author" in field.lower():
                        important_fields.append(f"著者: {data[field]}")
                    elif field_type == "TEXT" and "publisher" in field.lower():
                        important_fields.append(f"出版社: {data[field]}")
                    elif field_type == "DATE" and "date" in field.lower():
                        important_fields.append(f"出版日: {data[field]}")
                    else:
                        other_fields.append(f"{field}: {data[field]}")
        else:
            # フィールド型が提供されていない場合
            for field, value in data.items():
                if "title" in field.lower():
                    important_fields.append(f"タイトル: {value}")
                elif "author" in field.lower():
                    important_fields.append(f"著者: {value}")
                elif "publisher" in field.lower():
                    important_fields.append(f"出版社: {value}")
                elif "date" in field.lower() or "pubdate" in field.lower():
                    important_fields.append(f"出版日: {value}")
                else:
                    other_fields.append(f"{field}: {value}")
        
        # 重要なフィールドを先に、その他のフィールドを後に
        text_parts.extend(important_fields)
        text_parts.extend(other_fields)
    
    # テキストが抽出されなかった場合は文字列表現を使用
    if not text_parts:
        return str(record)
    
    return "\n".join(text_parts)

# エンベディング関連
def get_cache_key(text: str) -> str:
    """テキストからキャッシュキーを生成する"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

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

def save_embedding_cache(cache: Dict[str, np.ndarray], cache_file: str):
    """エンベディングキャッシュを保存する"""
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
        print(f"{len(cache)}件のエンベディングをキャッシュに保存しました: {cache_file}")
    except Exception as e:
        print(f"キャッシュ保存エラー: {e}")

def get_embedding_with_api(text: str, api_key: str) -> np.ndarray:
    """OpenAI APIを使用してエンベディングを取得する"""
    if not api_key:
        print("エラー: APIキーが提供されていません。")
        sys.exit(1)
    
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
    
    debug_log(f"OpenAI APIでエンベディングをリクエスト（テキスト長: {len(input_text)}）")
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                debug_log("APIリクエスト成功")
                embedding = np.array(response.json()["data"][0]["embedding"])
                return embedding
            elif response.status_code == 429:  # レート制限
                wait_time = RETRY_DELAY * (attempt + 1)
                print(f"レート制限に達しました。{wait_time}秒待機します...")
                time.sleep(wait_time)
            else:
                print(f"APIエラー ({attempt+1}/{MAX_RETRIES}): {response.status_code} - {response.text}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"リクエストエラー ({attempt+1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    
    print("すべてのAPIリクエスト試行が失敗しました。プログラムを終了します。")
    sys.exit(1)

def calculate_record_embeddings(records: List[Dict], field_types: Dict[str, str],
                               api_key: str, cache_dir: str = CACHE_DIR) -> Dict[str, np.ndarray]:
    """すべてのレコードのエンベディングを計算（キャッシュ機能付き）"""
    # キャッシュディレクトリの作成
    os.makedirs(cache_dir, exist_ok=True)
    
    # キャッシュファイル名
    cache_file = os.path.join(cache_dir, f"embeddings_cache.pkl")
    
    # キャッシュの読み込み
    embedding_cache = load_embedding_cache(cache_file)
    
    record_embeddings = {}
    cache_hits = 0
    cache_misses = 0
    
    print(f"{len(records)}件のレコードのエンベディングを計算中...")
    start_time = datetime.now()
    
    for i, record in enumerate(tqdm(records, desc="エンベディング計算中")):
        # レコードIDの取得
        record_id = record.get("id", f"record_{i}")
        
        # テキスト表現の取得
        text = get_record_text(record, field_types)
        
        # キャッシュキーの生成
        cache_key = get_cache_key(text)
        
        # キャッシュを確認
        if cache_key in embedding_cache:
            record_embeddings[record_id] = embedding_cache[cache_key]
            cache_hits += 1
        else:
            # APIでエンベディングを計算
            embedding = get_embedding_with_api(text, api_key)
            
            # キャッシュに保存
            embedding_cache[cache_key] = embedding
            record_embeddings[record_id] = embedding
            cache_misses += 1
        
        # 定期的にキャッシュを保存
        if cache_misses > 0 and cache_misses % 10 == 0:
            save_embedding_cache(embedding_cache, cache_file)
    
    # 最終的なキャッシュを保存
    save_embedding_cache(embedding_cache, cache_file)
    
    # 完了メッセージ
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"エンベディング計算完了: {len(record_embeddings)}件, 経過時間: {elapsed:.1f}秒")
    print(f"キャッシュヒット: {cache_hits}件, 新規計算: {cache_misses}件")
    
    return record_embeddings

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

# ブロッキングコンポーネント
class BlockingComponent:
    """効率的なブロッキング機能を提供するコンポーネント"""
    
    def __init__(self, method: str = "embedding", cache_dir: str = CACHE_DIR):
        """
        ブロッキングコンポーネントの初期化
        
        Args:
            method: ブロッキング方法 ("embedding", "lsh", "minihash")
            cache_dir: キャッシュディレクトリ
        """
        self.method = method
        self.cache_dir = cache_dir
        self.index = None
        self.record_ids = []
        self.record_minhashes = {}
        self.blocking_cache_file = os.path.join(cache_dir, f"blocking_index_{method}.pkl")

    def build_index(self, record_embeddings: Dict[str, np.ndarray], records: List[Dict], field_types: Dict[str, str] = None):
        """
        検索インデックスの構築
        
        Args:
            record_embeddings: レコードIDからエンベディングへのマップ
            records: レコードのリスト
            field_types: フィールドタイプの辞書
        """
        # キャッシュディレクトリの作成
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # キャッシュがあれば読み込み
        if os.path.exists(self.blocking_cache_file):
            try:
                print(f"ブロッキングインデックスをキャッシュから読み込み中: {self.blocking_cache_file}")
                with open(self.blocking_cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.index = cache_data.get("index")
                    self.record_ids = cache_data.get("record_ids", [])
                    self.record_minhashes = cache_data.get("record_minhashes", {})
                
                # キャッシュが有効かチェック
                if self.index is not None and len(self.record_ids) == len(record_embeddings):
                    embedding_hashes = [hashlib.md5(emb.tobytes()).hexdigest() for emb in record_embeddings.values()]
                    cache_hashes = cache_data.get("embedding_hashes", [])
                    
                    if embedding_hashes == cache_hashes:
                        print(f"有効なブロッキングインデックスがキャッシュから読み込まれました")
                        return
                    else:
                        print(f"キャッシュが最新ではないため、インデックスを再構築します")
            except Exception as e:
                print(f"キャッシュ読み込みエラー: {e}")
        
        print(f"ブロッキングインデックスを構築中...")
        start_time = datetime.now()
        
        # 選択したメソッドでインデックス構築
        if self.method == "embedding":
            self._build_embedding_index(record_embeddings)
        elif self.method == "lsh":
            self._build_lsh_index(record_embeddings, records, field_types)
        elif self.method == "minihash":
            self._build_minihash_index(record_embeddings, records, field_types)
        else:
            raise ValueError(f"未対応のブロッキング方法: {self.method}")
        
        # キャッシュに保存
        try:
            embedding_hashes = [hashlib.md5(record_embeddings[rid].tobytes()).hexdigest() 
                             for rid in self.record_ids if rid in record_embeddings]
            
            cache_data = {
                "index": self.index,
                "record_ids": self.record_ids,
                "record_minhashes": self.record_minhashes,
                "embedding_hashes": embedding_hashes,
                "method": self.method,
                "created_at": datetime.now().isoformat()
            }
            
            with open(self.blocking_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            print(f"ブロッキングインデックスをキャッシュに保存しました: {self.blocking_cache_file}")
        except Exception as e:
            print(f"キャッシュ保存エラー: {e}")
        
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"インデックス構築完了: {elapsed:.2f}秒")
    
    def _build_embedding_index(self, record_embeddings: Dict[str, np.ndarray]):
        """エンベディングベースのKNN検索インデックスを構築"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn がインストールされていません。pip install scikit-learn でインストールしてください。")
        
        self.record_ids = list(record_embeddings.keys())
        
        # エンベディング配列を作成
        if not self.record_ids:
            print("警告: レコードが見つかりません。空のインデックスを作成します。")
            self.index = None
            return
            
        embeddings_array = np.array([record_embeddings[rid] for rid in self.record_ids])
        
        # n_neighborsは少なくとも1以上
        n_neighbors = max(1, min(50, len(self.record_ids)))
        
        # NearestNeighborsインデックスを構築
        self.index = NearestNeighbors(n_neighbors=n_neighbors, 
                                     algorithm='auto', 
                                     metric='cosine')
        self.index.fit(embeddings_array)
    
    def _build_lsh_index(self, record_embeddings: Dict[str, np.ndarray], records: List[Dict], field_types: Dict[str, str]):
        """LSH (Locality Sensitive Hashing) インデックスを構築"""
        try:
            from datasketch import MinHashLSH, MinHash
        except ImportError:
            print("警告: 'datasketch' ライブラリがインストールされていません。以下のコマンドでインストールしてください:")
            print("pip install datasketch")
            raise ImportError("datasketch がインストールされていません")
        
        self.record_ids = list(record_embeddings.keys())
        
        # LSHインデックスを初期化
        self.index = MinHashLSH(threshold=0.7, num_perm=128)
        
        # レコードIDからレコードへのマッピングを作成
        id_to_record = {record.get("id", f"record_{i}"): record for i, record in enumerate(records)}
        
        # 各レコードのMinHashを計算
        self.record_minhashes = {}
        
        for record_id in tqdm(self.record_ids, desc="MinHash作成中"):
            if record_id not in id_to_record:
                continue
                
            record = id_to_record[record_id]
            
            # テキスト表現を取得
            text = get_record_text(record, field_types)
            
            # 日本語テキストを正規化
            normalized_text = normalize_japanese_text(text)
            
            # シャイングル（N-gram）を作成
            shingles = set()
            for i in range(len(normalized_text) - 2):
                shingles.add(normalized_text[i:i+3])
            
            # MinHashを計算
            m = MinHash(num_perm=128)
            for s in shingles:
                m.update(s.encode('utf-8'))
            
            # インデックスに追加
            self.index.insert(record_id, m)
            self.record_minhashes[record_id] = m
    
    def _build_minihash_index(self, record_embeddings: Dict[str, np.ndarray], records: List[Dict], field_types: Dict[str, str]):
        """MinHash + バケットベースのインデックスを構築（カスタム実装）"""
        self.record_ids = list(record_embeddings.keys())
        
        # 簡易的なバケットシステム
        # エンベディングの各次元を量子化してバケットに割り当て
        num_buckets = 20  # バケット数
        bucket_map = defaultdict(set)
        
        # レコードIDからレコードへのマッピングを作成
        id_to_record = {record.get("id", f"record_{i}"): record for i, record in enumerate(records)}
        
        # 重要な次元を選択（分散が大きい次元）
        all_embeddings = np.array([emb for emb in record_embeddings.values()])
        variances = np.var(all_embeddings, axis=0)
        important_dims = np.argsort(-variances)[:20]  # 上位20次元
        
        # 各レコードをバケットに割り当て
        for record_id in tqdm(self.record_ids, desc="バケット割り当て中"):
            if record_id not in record_embeddings:
                continue
                
            emb = record_embeddings[record_id]
            
            # 重要な次元のみ使用
            for dim_idx in important_dims:
                # 次元の値を量子化
                bucket_idx = int(np.floor(emb[dim_idx] * num_buckets))
                # バケットにレコードIDを追加
                bucket_map[(dim_idx, bucket_idx)].add(record_id)
        
        self.index = bucket_map
    
    def find_candidate_pairs(self, threshold: float = 0.8, max_candidates: int = 100) -> List[Tuple[str, str]]:
        """
        候補ペアを見つける
        
        Args:
            threshold: 類似度しきい値
            max_candidates: レコードあたりの最大候補数
            
        Returns:
            候補ペアのリスト（record_id1, record_id2のタプル）
        """
        if self.index is None:
            raise ValueError("インデックスが構築されていません。先にbuild_index()を呼び出してください。")
        
        candidate_pairs = set()
        
        if self.method == "embedding":
            candidate_pairs = self._find_candidates_embedding(threshold, max_candidates)
        elif self.method == "lsh":
            candidate_pairs = self._find_candidates_lsh(threshold)
        elif self.method == "minihash":
            candidate_pairs = self._find_candidates_minihash(threshold)
        
        print(f"{len(candidate_pairs)}個の候補ペアを見つけました")
        return list(candidate_pairs)
    
    def _find_candidates_embedding(self, threshold: float, max_candidates: int) -> Set[Tuple[str, str]]:
        """エンベディングベースのKNN検索で候補ペアを見つける"""
        # コサイン類似度からコサイン距離に変換（1 - 類似度）
        distance_threshold = 1.0 - threshold
        
        candidate_pairs = set()
        
        if not self.index or len(self.record_ids) == 0:
            print("警告: インデックスが空です。候補ペアは生成されません。")
            return candidate_pairs
        
        # 各レコードの近傍を検索
        for i, record_id1 in enumerate(tqdm(self.record_ids, desc="候補ペア検索中")):
            # 単一レコードの近傍を検索
            record_index = i
            try:
                distances, indices = self.index.kneighbors([self.index._fit_X[record_index]], 
                                                           n_neighbors=min(self.index.n_neighbors, len(self.record_ids)))
                
                # 1次元配列に変換
                distances = distances[0]
                indices = indices[0]
                
                # しきい値以下の距離を持つ近傍のみを追加
                for j, dist in enumerate(distances):
                    if dist <= distance_threshold and j > 0:  # j=0はレコード自身
                        neigh_idx = indices[j]
                        record_id2 = self.record_ids[neigh_idx]
                        
                        # 重複を避けるため、IDを辞書順にソート
                        pair = tuple(sorted([record_id1, record_id2]))
                        candidate_pairs.add(pair)
            except Exception as e:
                print(f"警告: レコード {record_id1} の近傍検索中にエラーが発生しました: {e}")
                continue
        
        return candidate_pairs
    
    
    
    def _find_candidates_lsh(self, threshold: float) -> Set[Tuple[str, str]]:
        """LSHで候補ペアを見つける"""
        candidate_pairs = set()
        
        # 各レコードについて、LSHで類似するレコードを検索
        for record_id in tqdm(self.record_ids, desc="LSH検索中"):
            if record_id not in self.record_minhashes:
                continue
                
            # LSHで近接レコードを検索
            minhash = self.record_minhashes[record_id]
            similar_ids = self.index.query(minhash)
            
            # 自分自身を除外し、ペアを作成
            for similar_id in similar_ids:
                if similar_id != record_id:
                    # 重複を避けるため、IDを辞書順にソート
                    pair = tuple(sorted([record_id, similar_id]))
                    candidate_pairs.add(pair)
        
        return candidate_pairs
    
    def _find_candidates_minihash(self, threshold: float) -> Set[Tuple[str, str]]:
        """カスタムバケットベース検索で候補ペアを見つける"""
        candidate_pairs = set()
        bucket_map = self.index
        
        # 候補ペアのカウントを保持する辞書
        pair_counts = defaultdict(int)
        
        # 各バケットについて、含まれるレコード間のペアを作成
        for bucket, record_ids in tqdm(bucket_map.items(), desc="バケット走査中"):
            record_ids_list = list(record_ids)
            
            # バケット内の全ペアを生成
            for i in range(len(record_ids_list)):
                for j in range(i+1, len(record_ids_list)):
                    rid1 = record_ids_list[i]
                    rid2 = record_ids_list[j]
                    
                    # 重複を避けるため、IDを辞書順にソート
                    pair = tuple(sorted([rid1, rid2]))
                    
                    # このペアが共有するバケット数をカウント
                    pair_counts[pair] += 1
        
        # 一定数以上のバケットを共有するペアを候補とする
        min_bucket_overlap = 3  # 最低限共有すべきバケット数
        for pair, count in pair_counts.items():
            if count >= min_bucket_overlap:
                candidate_pairs.add(pair)
        
        return candidate_pairs

# LLMマッチングモデル
class LLMMatchingModel:
    """LLMを使用してレコード間のマッチング確率を計算するモデル"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", cache_dir: str = CACHE_DIR):
        self.api_key = api_key
        self.model = model
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, f"llm_matching_cache_{model.replace('-', '_')}.pkl")
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, float]:
        """キャッシュを読み込む"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                print(f"{len(cache)}件のLLMマッチング結果をキャッシュから読み込みました")
                return cache
            except Exception as e:
                print(f"キャッシュ読み込みエラー: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """キャッシュを保存する"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            print(f"{len(self.cache)}件のLLMマッチング結果をキャッシュに保存しました")
        except Exception as e:
            print(f"キャッシュ保存エラー: {e}")
    
    def _get_cache_key(self, text1: str, text2: str) -> str:
        """キャッシュキーを生成"""
        combined = "\n===\n".join(sorted([text1, text2]))
        return hashlib.md5(combined.encode('utf-8')).hexdigest()
    
    def get_matching_probability(self, record1: Dict, record2: Dict, field_types: Dict[str, str] = None) -> float:
        """2つのレコード間のマッチング確率を計算"""
        # レコードテキストを作成
        text1 = format_record_for_llm(record1, field_types)
        text2 = format_record_for_llm(record2, field_types)
        
        # キャッシュキーを生成
        cache_key = self._get_cache_key(text1, text2)
        
        # キャッシュをチェック
        if cache_key in self.cache:
            debug_log(f"キャッシュヒット: {record1.get('id', '')} vs {record2.get('id', '')}")
            return self.cache[cache_key]
        
        # LLMで確率を計算
        probability = self._call_llm_api(text1, text2)
        
        # キャッシュに保存
        self.cache[cache_key] = probability
        
        # 定期的にキャッシュを保存
        if len(self.cache) % 10 == 0:
            self._save_cache()
        
        return probability
    
    def _call_llm_api(self, text1: str, text2: str) -> float:
        """LLM APIを呼び出して確率を計算"""
        if not self.api_key:
            print("エラー: APIキーが設定されていません")
            sys.exit(1)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # プロンプトの設計
        system_prompt = """
        あなたは書誌レコードの専門家です。2つの書誌レコードを比較して、それらが同じ作品を指しているかどうかの確率を判断してください。
        
        確率を0から1の間の小数で返してください（例: 0.95）。
        
        考慮すべき点:
        1. タイトルの類似性（最も重要）
        2. 著者の一致（非常に重要）
        3. 出版社と出版日（補足的な情報）
        4. 表記の違い（全角/半角、かな/カナなど）は同一とみなす
        
        返答は確率値のみを返してください。説明は不要です。
        """
        
        user_prompt = f"""
        以下の2つの書誌レコードが同じ作品を指している確率（0〜1）を判断してください。

        書誌レコード1:
        {text1}

        書誌レコード2:
        {text2}
        
        確率（0〜1の数値のみ）:
        """
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0
        }
        
        debug_log(f"LLM APIリクエスト送信")
        
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    # テキスト応答から数値を抽出
                    probability_text = result["choices"][0]["message"]["content"].strip()
                    
                    try:
                        # 数値を抽出
                        import re
                        probability_matches = re.findall(r'0\.\d+|\d+\.0|[01]', probability_text)
                        if probability_matches:
                            probability = float(probability_matches[0])
                            # 範囲を0〜1に制限
                            probability = max(0.0, min(1.0, probability))
                            return probability
                        else:
                            print(f"警告: LLMからの応答から確率を抽出できませんでした: {probability_text}")
                            return 0.5  # デフォルト値
                    except Exception as e:
                        print(f"警告: 確率の解析エラー: {e}")
                        return 0.5  # デフォルト値
                
                elif response.status_code == 429:  # レート制限
                    wait_time = RETRY_DELAY * (attempt + 1)
                    print(f"レート制限に達しました。{wait_time}秒待機します...")
                    time.sleep(wait_time)
                else:
                    print(f"APIエラー ({attempt+1}/{MAX_RETRIES}): {response.status_code} - {response.text}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
            except Exception as e:
                print(f"リクエストエラー ({attempt+1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
        
        print("すべてのAPIリクエスト試行が失敗しました。デフォルト値を使用します。")
        return 0.5  # デフォルト値
    
    def save_final_cache(self):
        """最終キャッシュを保存"""
        self._save_cache()

# 一貫性チェッカー
class ConsistencyChecker:
    """マッチング結果の一貫性をチェックするコンポーネント"""
    
    def __init__(self, threshold: float = 0.8):
        """
        一貫性チェッカーの初期化
        
        Args:
            threshold: 一貫性判定のしきい値
        """
        if not NETWORKX_AVAILABLE:
            raise ImportError("networkx がインストールされていません。pip install networkx でインストールしてください。")
            
        self.threshold = threshold
        self.similarity_graph = None
    
    def build_similarity_graph(self, record_pairs: List[Tuple[str, str, float]]):
        """
        類似度グラフを構築
        
        Args:
            record_pairs: (record_id1, record_id2, similarity)の形式のタプルのリスト
        """
        # NetworkXグラフを初期化
        self.similarity_graph = nx.Graph()
        
        # エッジを追加（各レコードペアの類似度を重みとして）
        for record_id1, record_id2, similarity in record_pairs:
            if similarity >= self.threshold:
                self.similarity_graph.add_edge(record_id1, record_id2, weight=similarity)
    
    def save_similarity_graph(self, filepath: str):
        """
        類似度グラフをファイルに保存
        
        Args:
            filepath: 保存先ファイルパス
        """
        if self.similarity_graph is None:
            raise ValueError("グラフが空なため保存できません")
        
        # グラフをGMLまたはGraphML形式で保存
        try:
            # ディレクトリが存在しない場合は作成
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            # 拡張子に基づいて保存形式を選択
            if filepath.endswith('.gml'):
                nx.write_gml(self.similarity_graph, filepath)
            elif filepath.endswith('.graphml'):
                nx.write_graphml(self.similarity_graph, filepath)
            elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
                with open(filepath, 'wb') as f:
                    pickle.dump(self.similarity_graph, f)
            else:
                # デフォルトはGML形式
                nx.write_gml(self.similarity_graph, filepath)
            
            print(f"類似度グラフを保存しました: {filepath}")
            return True
        except Exception as e:
            print(f"グラフ保存エラー: {e}")
            return False

    def load_similarity_graph(self, filepath: str):
        """
        類似度グラフをファイルから読み込み
        
        Args:
            filepath: 読み込み元ファイルパス
        
        Returns:
            読み込み成功したかどうか
        """
        try:
            if not os.path.exists(filepath):
                print(f"ファイルが存在しません: {filepath}")
                return False
            
            # 拡張子に基づいて読み込み形式を選択
            if filepath.endswith('.gml'):
                self.similarity_graph = nx.read_gml(filepath)
            elif filepath.endswith('.graphml'):
                self.similarity_graph = nx.read_graphml(filepath)
            elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
                with open(filepath, 'rb') as f:
                    self.similarity_graph = pickle.load(f)
            else:
                # デフォルトはGML形式と仮定
                self.similarity_graph = nx.read_gml(filepath)
            
            print(f"類似度グラフを読み込みました: {filepath}")
            return True
        except Exception as e:
            print(f"グラフ読み込みエラー: {e}")
            return False
    
    def find_inconsistencies(self) -> List[Dict]:
        """
        グラフ内の矛盾を検出
        
        Returns:
            矛盾のリスト（各矛盾は辞書形式でdetail情報を含む）
        """
        if self.similarity_graph is None:
            raise ValueError("類似度グラフが構築されていません。先にbuild_similarity_graph()を呼び出してください。")
        
        inconsistencies = []
        
        # すべてのノード（レコードID）を列挙
        all_nodes = list(self.similarity_graph.nodes())
        
        # 各ノードに対して検証
        for i, node_a in enumerate(tqdm(all_nodes, desc="矛盾検出中")):
            # 直接接続されたノード（類似レコード）
            neighbors_a = set(self.similarity_graph.neighbors(node_a))
            
            # 2ホップ先のノード（友達の友達）
            two_hop_neighbors = set()
            for neighbor in neighbors_a:
                two_hop_neighbors.update(self.similarity_graph.neighbors(neighbor))
            
            # 自分自身と直接の隣接ノードを除外
            two_hop_neighbors -= {node_a}
            two_hop_neighbors -= neighbors_a
            
            # 推移律に基づく一貫性をチェック
            for node_c in two_hop_neighbors:
                # ノードAとノードCが直接接続されていない場合、潜在的な矛盾
                if not self.similarity_graph.has_edge(node_a, node_c):
                    # 共通の隣接ノード（ノードB）を見つける
                    common_neighbors = set(self.similarity_graph.neighbors(node_a)) & set(self.similarity_graph.neighbors(node_c))
                    
                    for node_b in common_neighbors:
                        # 推移的な関係の強さ（ノードA→ノードBとノードB→ノードCの類似度の最小値）
                        ab_similarity = self.similarity_graph.edges[node_a, node_b]["weight"]
                        bc_similarity = self.similarity_graph.edges[node_b, node_c]["weight"]
                        transitive_similarity = min(ab_similarity, bc_similarity)
                        
                        # 推移的な類似度がしきい値より高い場合、矛盾として報告
                        if transitive_similarity > self.threshold + 0.05:  # より高いしきい値で確実な矛盾のみ検出
                            inconsistency = {
                                "type": "missing_edge",
                                "node_a": node_a,
                                "node_b": node_b,
                                "node_c": node_c,
                                "ab_similarity": ab_similarity,
                                "bc_similarity": bc_similarity,
                                "transitive_similarity": transitive_similarity,
                                "confidence": (transitive_similarity - self.threshold) / (1 - self.threshold)  # 矛盾の確信度
                            }
                            inconsistencies.append(inconsistency)
        
        # 確信度でソート
        inconsistencies.sort(key=lambda x: x["confidence"], reverse=True)
        
        return inconsistencies
    
    def resolve_inconsistencies(self, inconsistencies: List[Dict] = None):
        """
        検出された矛盾を解決
        
        Args:
            inconsistencies: 矛盾のリスト（Noneの場合は検出から実行）
        
        Returns:
            解決されたエッジのリスト（record_id1, record_id2, similarity）
        """
        if self.similarity_graph is None:
            raise ValueError("類似度グラフが構築されていません。先にbuild_similarity_graph()を呼び出してください。")
        
        if inconsistencies is None:
            inconsistencies = self.find_inconsistencies()
        
        resolved_edges = []
        
        # 各矛盾を解決
        for inconsistency in tqdm(inconsistencies, desc="矛盾解決中"):
            if inconsistency["type"] == "missing_edge":
                node_a = inconsistency["node_a"]
                node_c = inconsistency["node_c"]
                transitive_similarity = inconsistency["transitive_similarity"]
                
                # 新しいエッジを追加（推移的類似度を重みとして）
                self.similarity_graph.add_edge(node_a, node_c, weight=transitive_similarity)
                
                # 解決したエッジとして記録
                resolved_edges.append((node_a, node_c, transitive_similarity))
        
        return resolved_edges

# クラスタリングとメトリクス
def create_clusters_from_candidates(records: List[Dict], candidate_pairs: List[Tuple[str, str]], 
                                    record_embeddings: Dict[str, np.ndarray],
                                    field_types: Dict[str, str],
                                    threshold: float = 0.8,
                                    use_llm: bool = False,
                                    api_key: str = None) -> Dict[str, List[Dict]]:
    """候補ペアからクラスターを作成"""
    # レコードIDからレコードへのマッピングを作成
    id_to_record = {record.get("id", f"record_{i}"): record for i, record in enumerate(records)}
    
    # LLMモデルを初期化（ハイブリッドモードの場合）
    llm_model = None
    if use_llm and api_key:
        llm_model = LLMMatchingModel(api_key=api_key)
    
    # Union-Find データ構造（素集合データ構造）
    parent = {}
    rank = {}
    
    # 初期化
    for record in records:
        record_id = record.get("id", "")
        parent[record_id] = record_id
        rank[record_id] = 0
    
    # ルート要素を見つける関数
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])  # パス圧縮
        return parent[x]
    
    # 2つの集合を結合する関数
    def union(x, y):
        x_root = find(x)
        y_root = find(y)
        
        if x_root == y_root:
            return
        
        # ランクによる結合
        if rank[x_root] < rank[y_root]:
            parent[x_root] = y_root
        elif rank[x_root] > rank[y_root]:
            parent[y_root] = x_root
        else:
            parent[y_root] = x_root
            rank[x_root] += 1
    
    # 候補ペアの類似度を評価し、しきい値を超えるものをマージ
    for pair in tqdm(candidate_pairs, desc="候補ペア評価中"):
        # タプルから要素を抽出
        if isinstance(pair, tuple) and len(pair) == 2:
            record_id1, record_id2 = pair
            similarity = None  # 後で計算
        elif isinstance(pair, tuple) and len(pair) == 3:
            record_id1, record_id2, similarity = pair
        else:
            continue
            
        # レコードがマッピングに存在するか確認
        if record_id1 not in id_to_record or record_id2 not in id_to_record:
            continue
            
        record1 = id_to_record[record_id1]
        record2 = id_to_record[record_id2]
        
        # 類似度の計算（与えられていない場合）
        if similarity is None:
            if llm_model:
                # LLMで詳細な類似度評価
                similarity = llm_model.get_matching_probability(record1, record2, field_types)
                debug_log(f"LLM類似度: {record_id1} - {record_id2}: {similarity}")
            elif record_id1 in record_embeddings and record_id2 in record_embeddings:
                # エンベディング類似度と詳細評価の組み合わせ
                emb1 = record_embeddings[record_id1]
                emb2 = record_embeddings[record_id2]
                emb_similarity = cosine_similarity(emb1, emb2)
                
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
                if title_field and "data" in record1 and "data" in record2:
                    title1 = record1["data"].get(title_field, "")
                    title2 = record2["data"].get(title_field, "")
                    
                    # タイトルを正規化（日本語テキストに特に重要）
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
                similarity = (
                    emb_similarity * 0.3 +
                    title_similarity * 0.5 +
                    author_similarity * 0.2
                )
            else:
                # エンベディングが利用できない場合はデフォルト値
                similarity = 0.5
        
        # しきい値を超える場合、クラスターをマージ
        if similarity >= threshold:
            union(record_id1, record_id2)
    
    # LLMキャッシュの保存
    if llm_model:
        llm_model.save_final_cache()
    
    # クラスターの作成
    clusters = defaultdict(list)
    
    for record in records:
        record_id = record.get("id", "")
        
        if record_id in parent:
            root = find(record_id)
            
            # クラスターIDを設定
            record["cluster_id"] = root
            
            # クラスターにレコードを追加
            clusters[root].append(record)
    
    print(f"{len(clusters)}個のクラスターを作成しました")
    
    return clusters
def create_clusters_from_candidates_selective_llm(records: List[Dict], candidate_pairs: List[Tuple[str, str]], 
                                record_embeddings: Dict[str, np.ndarray],
                                field_types: Dict[str, str],
                                threshold: float = 0.8,
                                api_key: str = None,
                                llm_lower_threshold: float = 0.6,
                                llm_upper_threshold: float = 0.95) -> Dict[str, List[Dict]]:
        """候補ペアからクラスターを作成（あいまいなペアのみLLMを使用）"""
        # レコードIDからレコードへのマッピングを作成
        id_to_record = {record.get("id", f"record_{i}"): record for i, record in enumerate(records)}
        
        # LLMモデルを初期化
        llm_model = None
        if api_key:
            llm_model = LLMMatchingModel(api_key=api_key)
        
        # LLM評価の統計
        llm_eval_count = 0
        total_pairs = 0
        
        # Union-Find データ構造（素集合データ構造）
        parent = {}
        rank = {}
        
        # 初期化
        for record in records:
            record_id = record.get("id", "")
            parent[record_id] = record_id
            rank[record_id] = 0
        
        # ルート要素を見つける関数
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # パス圧縮
            return parent[x]
        
        # 2つの集合を結合する関数
        def union(x, y):
            x_root = find(x)
            y_root = find(y)
            
            if x_root == y_root:
                return
            
            # ランクによる結合
            if rank[x_root] < rank[y_root]:
                parent[x_root] = y_root
            elif rank[x_root] > rank[y_root]:
                parent[y_root] = x_root
            else:
                parent[y_root] = x_root
                rank[x_root] += 1
        
        # 候補ペアの類似度を評価し、しきい値を超えるものをマージ
        for pair in tqdm(candidate_pairs, desc="候補ペア評価中"):
            # タプルから要素を抽出
            if isinstance(pair, tuple) and len(pair) == 2:
                record_id1, record_id2 = pair
                similarity = None  # 後で計算
            elif isinstance(pair, tuple) and len(pair) == 3:
                record_id1, record_id2, similarity = pair
            else:
                continue
            
            total_pairs += 1
                
            # レコードがマッピングに存在するか確認
            if record_id1 not in id_to_record or record_id2 not in id_to_record:
                continue
                
            record1 = id_to_record[record_id1]
            record2 = id_to_record[record_id2]
            
            # 類似度の計算（与えられていない場合）
            if similarity is None:
                # まずエンベディング類似度を計算
                if record_id1 in record_embeddings and record_id2 in record_embeddings:
                    emb1 = record_embeddings[record_id1]
                    emb2 = record_embeddings[record_id2]
                    emb_similarity = cosine_similarity(emb1, emb2)
                    
                    # エンベディング類似度に基づいて判断
                    if emb_similarity >= llm_upper_threshold:
                        # 非常に高い類似度 - LLMなしで同一と判断
                        similarity = emb_similarity
                        debug_log(f"高類似度: {record_id1} - {record_id2}: {similarity} (LLMなし)")
                    elif emb_similarity <= llm_lower_threshold:
                        # 十分に低い類似度 - LLMなしで非同一と判断
                        similarity = emb_similarity
                        debug_log(f"低類似度: {record_id1} - {record_id2}: {similarity} (LLMなし)")
                    elif llm_model:
                        # あいまいな類似度範囲 - LLMで詳細評価
                        llm_eval_count += 1
                        similarity = llm_model.get_matching_probability(record1, record2, field_types)
                        debug_log(f"LLM類似度: {record_id1} - {record_id2}: {similarity}")
                    else:
                        # LLMが利用できない場合はルールベースの詳細評価
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
                        similarity = (
                            emb_similarity * 0.3 +
                            title_similarity * 0.5 +
                            author_similarity * 0.2
                        )
                else:
                    # エンベディングが利用できない場合はデフォルト値
                    similarity = 0.5
            
            # しきい値を超える場合、クラスターをマージ
            if similarity >= threshold:
                union(record_id1, record_id2)
        
        # LLMキャッシュの保存
        if llm_model:
            llm_model.save_final_cache()
            print(f"\nLLM評価統計: {llm_eval_count}/{total_pairs} ペア ({llm_eval_count/total_pairs:.2%})")
        
        # クラスターの作成
        clusters = defaultdict(list)
        
        for record in records:
            record_id = record.get("id", "")
            
            if record_id in parent:
                root = find(record_id)
                
                # クラスターIDを設定
                record["cluster_id"] = root
                
                # クラスターにレコードを追加
                clusters[root].append(record)
        
        print(f"{len(clusters)}個のクラスターを作成しました")
        
        return clusters

def apply_consistency_checking(record_pairs: List[Tuple[str, str, float]], threshold: float = 0.8,
                             graph_cache: str = None) -> List[Tuple[str, str, float]]:
    """一貫性チェックを適用して新しいエッジを生成"""
    # 一貫性チェッカーを初期化
    checker = ConsistencyChecker(threshold=threshold)
    
    # グラフキャッシュの読み込みを試行
    graph_loaded = False
    if graph_cache and os.path.exists(graph_cache):
        print(f"グラフキャッシュから読み込みを試行: {graph_cache}")
        graph_loaded = checker.load_similarity_graph(graph_cache)
    
    # グラフの構築（キャッシュから読み込めなかった場合）
    if not graph_loaded:
        checker.build_similarity_graph(record_pairs)
    
    # 矛盾の検出
    inconsistencies = checker.find_inconsistencies()
    print(f"{len(inconsistencies)}個の矛盾を検出しました")
    
    # 矛盾の解決
    resolved_edges = checker.resolve_inconsistencies(inconsistencies)
    print(f"{len(resolved_edges)}個のエッジを追加して矛盾を解決しました")
    
    # グラフをキャッシュに保存（指定されている場合）
    if graph_cache:
        checker.save_similarity_graph(graph_cache)
    
    # 新しいエッジを返す
    return resolved_edges

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
    parser = argparse.ArgumentParser(description="ハイブリッド書誌レコードマッチングシステム")
    parser.add_argument("--input", "-i", type=str, required=True, help="入力YAMLファイルパス")
    parser.add_argument("--output", "-o", type=str, default="result.yaml", help="出力YAMLファイルパス")
    parser.add_argument("--api-key", "-k", type=str, help="OpenAI APIキー")
    parser.add_argument("--threshold", "-t", type=float, default=SIMILARITY_THRESHOLD, help="類似度しきい値")
    parser.add_argument("--cache-dir", "-c", type=str, default=CACHE_DIR, help="キャッシュディレクトリ")
    parser.add_argument("--debug", "-d", action="store_true", help="デバッグモードを有効化")
    parser.add_argument("--blocking", "-b", type=str, default="embedding", 
                       choices=["embedding", "lsh", "minihash", "none"], help="ブロッキング方法")
    parser.add_argument("--llm-lower", type=float, default=0.6, help="LLM評価を行う最小類似度")
    parser.add_argument("--llm-upper", type=float, default=0.95, help="LLM評価を行う最大類似度")
    parser.add_argument("--hybrid", "-H", action="store_true", help="従来のLLMハイブリッドモードを有効化")
    parser.add_argument("--consistency", "-C", action="store_true", help="一貫性チェックを有効化")
    parser.add_argument("--graph-cache", "-g", type=str, help="グラフキャッシュファイル")
    parser.add_argument("--selective-llm", "-S", action="store_true", 
                       help="あいまいなペアのみにLLMを適用するモードを有効化")
    
    args = parser.parse_args()
    
    # デバッグモードの設定
    global DEBUG_MODE
    DEBUG_MODE = args.debug
    
    # APIキーの取得
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("エラー: APIキーが提供されていません。--api-key オプションまたは環境変数 OPENAI_API_KEY を設定してください。")
        sys.exit(1)
    
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
    
    # レコードエンベディングの計算（APIを使用、キャッシュ付き）
    record_embeddings = calculate_record_embeddings(records, field_types, api_key, args.cache_dir)
    
    # ブロッキングを使用して候補ペアを生成
    if args.blocking != "none":
        print(f"ブロッキング手法 '{args.blocking}' を使用します")
        blocker = BlockingComponent(method=args.blocking, cache_dir=args.cache_dir)
        blocker.build_index(record_embeddings, records, field_types)
        candidate_pairs = blocker.find_candidate_pairs(threshold=args.threshold * 0.8)
        
        print(f"ブロッキングにより {len(candidate_pairs)} 個の候補ペアが生成されました")
        print(f"全ペア比較の場合: {len(records) * (len(records) - 1) // 2} ペア")
        print(f"削減率: {1.0 - len(candidate_pairs) / (len(records) * (len(records) - 1) // 2):.2%}")
        
        # 候補ペアからクラスターを作成
        if args.selective_llm:
            # あいまいなペアのみにLLMを適用するモード
            print("あいまいなペアのみにLLMを適用するモードを使用します")
            clusters = create_clusters_from_candidates_selective_llm(
                records, 
                candidate_pairs, 
                record_embeddings, 
                field_types, 
                args.threshold,
                api_key=api_key,
                llm_lower_threshold=args.llm_lower,
                llm_upper_threshold=args.llm_upper
            )
        else:
            # 従来のモード
            clusters = create_clusters_from_candidates(
                records, 
                candidate_pairs, 
                record_embeddings, 
                field_types, 
                args.threshold,
                use_llm=args.hybrid,
                api_key=api_key if args.hybrid else None
            )
    else:
        # 従来の方法でクラスターを作成（全ペア比較）
        print("ブロッキングなしで処理を実行します")
        all_pairs = []
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                record_id1 = records[i].get("id", f"record_{i}")
                record_id2 = records[j].get("id", f"record_{j}")
                all_pairs.append((record_id1, record_id2))
        
        # 全ペアを使用してクラスタリング
        if args.selective_llm:
            # あいまいなペアのみにLLMを適用するモード
            print("あいまいなペアのみにLLMを適用するモードを使用します")
            clusters = create_clusters_from_candidates_selective_llm(
                records, 
                all_pairs, 
                record_embeddings, 
                field_types, 
                args.threshold,
                api_key=api_key,
                llm_lower_threshold=args.llm_lower,
                llm_upper_threshold=args.llm_upper
            )
        else:
            # 従来のモード
            clusters = create_clusters_from_candidates(
                records, 
                all_pairs, 
                record_embeddings, 
                field_types, 
                args.threshold,
                use_llm=args.hybrid,
                api_key=api_key if args.hybrid else None
            )
    
    # 一貫性チェックを適用
    if args.consistency:
        print("一貫性チェックを適用します")
        
        # クラスタリング結果から類似度ペアを抽出
        similarity_pairs = []
        for cluster_id, cluster_records in clusters.items():
            record_ids = [record.get("id", "") for record in cluster_records if "id" in record]
            for i in range(len(record_ids)):
                for j in range(i+1, len(record_ids)):
                    # デフォルトでは高い類似度を割り当て
                    similarity_pairs.append((record_ids[i], record_ids[j], 0.95))
        
        # 一貫性チェックで新しいエッジを生成
        resolved_edges = apply_consistency_checking(
            similarity_pairs, 
            threshold=args.threshold,
            graph_cache=args.graph_cache
        )
        
        # 解決されたエッジを追加してクラスターを更新
        if resolved_edges:
            print(f"一貫性チェックで追加された {len(resolved_edges)} 個のエッジに基づいてクラスターを更新")
            
            # クラスターをマージ
            for edge in resolved_edges:
                record_id1, record_id2, _ = edge
                
                # レコードIDからクラスターIDを取得
                cluster_id1 = None
                cluster_id2 = None
                
                for cluster_id, cluster_records in clusters.items():
                    record_ids = [record.get("id", "") for record in cluster_records]
                    if record_id1 in record_ids:
                        cluster_id1 = cluster_id
                    if record_id2 in record_ids:
                        cluster_id2 = cluster_id
                
                # 異なるクラスターに属している場合はマージ
                if cluster_id1 and cluster_id2 and cluster_id1 != cluster_id2:
                    # クラスター2のレコードをクラスター1に移動
                    for record in clusters[cluster_id2]:
                        record["cluster_id"] = cluster_id1
                        clusters[cluster_id1].append(record)
                    
                    # クラスター2を削除
                    del clusters[cluster_id2]
    
    # 出力グループのフォーマット
    output_groups = format_output_groups(clusters)
    
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
    print(f"出力クラスター数: {len(clusters)}")
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