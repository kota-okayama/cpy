#!/usr/bin/env python3
"""
書誌レコードマッチング用ブロッキングコンポーネント

このモジュールは、書誌レコードのマッチングプロセスを高速化するためのブロッキング機能を提供します。
エンベディングベースのKNN、LSH、カスタムMinHashなど複数のブロッキング方法をサポートしています。
"""

import os
import sys
import numpy as np
import pickle
import hashlib
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Union
import unicodedata
from difflib import SequenceMatcher

try:
    from tqdm import tqdm
except ImportError:
    # tqdmがない場合のフォールバック
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# 定数
CACHE_DIR = ".cache"
DEBUG_MODE = False

class BlockingComponent:
    """
    効率的なブロッキング機能を提供するコンポーネント
    """
    
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
        embeddings_array = np.array([record_embeddings[rid] for rid in self.record_ids])
        
        # NearestNeighborsインデックスを構築
        self.index = NearestNeighbors(n_neighbors=min(50, len(self.record_ids)), 
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
            sys.exit(1)
        
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
            text = self.get_record_text(record, field_types)
            
            # 日本語テキストを正規化
            normalized_text = self.normalize_japanese_text(text)
            
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
    
    def find_candidate_pairs(self, threshold: float = 0.75, max_candidates: int = 100) -> List[Tuple[str, str]]:
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
        
        # エンベディング配列を再作成（インデックス構築後に変更があった場合のため）
        # ここではグローバル変数record_embeddingsを参照しないように注意
        candidate_pairs = set()
        
        # 各レコードの近傍を検索
        for i, record_id1 in enumerate(tqdm(self.record_ids, desc="候補ペア検索中")):
            # 単一レコードの近傍を検索
            record_index = i
            distances, indices = self.index.kneighbors([self.index._fit_X[record_index]], n_neighbors=min(50, len(self.record_ids)))
            
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
    
    # ユーティリティメソッド
    @staticmethod
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
    
    @staticmethod
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