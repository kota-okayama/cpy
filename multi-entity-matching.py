import yaml
import re
import json
import unicodedata
import asyncio
import aiohttp
import os
import uuid
import time
import math
import copy
import argparse
import random
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
from jellyfish import jaro_winkler_similarity

# UnionFindクラスの実装
class UnionFind:
    """Union-Findデータ構造の実装"""
    
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        """xの根を見つける（経路圧縮）"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """xとyを結合する（ランクによる最適化）"""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return
        
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        else:
            self.parent[root_y] = root_x
            if self.rank[root_x] == self.rank[root_y]:
                self.rank[root_x] += 1

# 設定クラス
# 設定クラス
class Config:
    def __init__(self):
        self.similarity_threshold = 0.7
        self.jw_upper_threshold = 0.9
        self.jw_lower_threshold = 0.3
        self.cache_dir = ".cache"
        self.debug = False
        self.api_key = ""
        
        # キャッシュディレクトリの確認
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"キャッシュディレクトリ {self.cache_dir} を作成しました")
    
    def load_from_args(self, args):
        """コマンドライン引数から設定を読み込む"""
        if hasattr(args, 'threshold'):
            self.similarity_threshold = args.threshold
        if hasattr(args, 'jw_upper'):
            self.jw_upper_threshold = args.jw_upper
        if hasattr(args, 'jw_lower'):
            self.jw_lower_threshold = args.jw_lower
        if hasattr(args, 'cache_dir'):
            self.cache_dir = args.cache_dir
        if hasattr(args, 'debug'):
            self.debug = args.debug
        if hasattr(args, 'api_key') and args.api_key:
            self.api_key = args.api_key

# テキスト正規化関数
def normalize_text(text: str) -> str:
    """テキストを正規化し、特殊文字を削除して統一形式にする"""
    if not isinstance(text, str):
        return ""
    # Unicodeを正規化
    text = unicodedata.normalize('NFKC', text)
    # 括弧や特殊文字を削除
    text = re.sub(r'[\[\]\(\)\{\}【】〔〕]', ' ', text)
    # 連続した空白を単一の空白に置換
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 出版年の抽出
def extract_year(pubdate: str) -> Optional[str]:
    """出版日から年を抽出"""
    if not pubdate:
        return None
    # 4桁の数字パターンを検索
    year_match = re.search(r'(19|20)\d{2}', pubdate)
    if year_match:
        return year_match.group(0)
    return None

# クラスターの代表レコードを選択
def get_cluster_representative(records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """最も情報が充実しているレコードをクラスターの代表として選択"""
    if not records:
        return None
    
    try:
        best_record = None
        max_fields = -1
        
        for record in records:
            # データフィールドがない場合は作成
            if 'data' not in record:
                if all(k in record for k in ['bib1_title', 'bib1_author', 'bib1_publisher', 'bib1_pubdate']):
                    # データフィールドが直接含まれている場合
                    record_data = {
                        'bib1_title': record.get('bib1_title', ''),
                        'bib1_author': record.get('bib1_author', ''),
                        'bib1_publisher': record.get('bib1_publisher', ''),
                        'bib1_pubdate': record.get('bib1_pubdate', '')
                    }
                    record['data'] = record_data
            
            data = record.get('data', {})
            
            # 空でないフィールドの数をカウント
            non_empty_fields = sum(1 for v in data.values() if v and isinstance(v, str) and len(v.strip()) > 0)
            
            # タイトルと著者が存在する場合に優先度を上げる
            has_title = bool(data.get('bib1_title', '').strip())
            has_author = bool(data.get('bib1_author', '').strip())
            priority_score = non_empty_fields + (2 if has_title else 0) + (1 if has_author else 0)
            
            if priority_score > max_fields:
                max_fields = priority_score
                best_record = record
        
        return best_record
    
    except Exception as e:
        print(f"代表レコード選択中にエラーが発生: {e}")
        return records[0] if records else None  # エラーが発生した場合は最初のレコードを返す

def debug_yaml_structure(yaml_data: str) -> None:
    """YAMLの構造を詳細に分析してデバッグ出力する"""
    try:
        data = yaml.safe_load(yaml_data)
        print("===== YAMLデータ構造のデバッグ情報 =====")
        print(f"データ型: {type(data)}")
        
        if isinstance(data, dict):
            print(f"トップレベルのキー: {list(data.keys())}")
            
            if 'group' in data:
                print(f"group要素数: {len(data['group'])}")
                if len(data['group']) > 0:
                    first_group = data['group'][0]
                    print(f"最初のグループの型: {type(first_group)}")
                    if isinstance(first_group, dict):
                        print(f"最初のグループのキー: {list(first_group.keys())}")
                        
                        if 'records' in first_group:
                            print(f"最初のグループのレコード数: {len(first_group['records'])}")
                            if len(first_group['records']) > 0:
                                first_record = first_group['records'][0]
                                print(f"最初のレコードの型: {type(first_record)}")
                                if isinstance(first_record, dict):
                                    print(f"最初のレコードのキー: {list(first_record.keys())}")
                                    
                                    if 'data' in first_record:
                                        print(f"最初のレコードのデータキー: {list(first_record['data'].keys())}")
        
        print("======================================")
    except Exception as e:
        print(f"YAMLデータ構造の分析中にエラーが発生: {e}")

def extract_representatives(yaml_data: str) -> List[Dict[str, Any]]:
    """YAMLから各クラスターの代表レコードを抽出"""
    try:
        # まずYAMLの構造を分析
        debug_yaml_structure(yaml_data)
        
        data = yaml.safe_load(yaml_data)
        representatives = []
        
        # 辞書形式の処理
        if isinstance(data, dict):
            if 'group' in data and isinstance(data['group'], list):
                print(f"グループリストが見つかりました（{len(data['group'])}グループ）")
                
                for group_idx, group in enumerate(data['group']):
                    if isinstance(group, dict) and 'records' in group:
                        if group['records']:  # レコードが存在する場合のみ処理
                            print(f"グループ{group_idx}のレコード数: {len(group['records'])}")
                            rep = get_cluster_representative(group['records'])
                            if rep:
                                rep_data = rep.get('data', {})
                                representatives.append({
                                    "cluster_id": group_idx,
                                    "title": rep_data.get('bib1_title', ''),
                                    "author": rep_data.get('bib1_author', ''),
                                    "publisher": rep_data.get('bib1_publisher', ''),
                                    "pubdate": rep_data.get('bib1_pubdate', ''),
                                    "original_id": rep.get('id', ''),
                                    "original_idx": rep.get('idx', -1),
                                    "original_cluster_id": group_idx,  # グループインデックスをクラスターIDとして使用
                                    "all_records": group['records']  # クラスター内の全レコードを保存
                                })
                                print(f"グループ{group_idx}の代表レコードを抽出しました: {rep_data.get('bib1_title', '未設定')}")
        
        # リスト形式の処理も保持
        elif isinstance(data, list):
            for group_idx, group in enumerate(data):
                if isinstance(group, dict) and 'records' in group:
                    rep = get_cluster_representative(group['records'])
                    if rep:
                        rep_data = rep.get('data', {})
                        representatives.append({
                            "cluster_id": group_idx,
                            "title": rep_data.get('bib1_title', ''),
                            "author": rep_data.get('bib1_author', ''),
                            "publisher": rep_data.get('bib1_publisher', ''),
                            "pubdate": rep_data.get('bib1_pubdate', ''),
                            "original_id": rep.get('id', ''),
                            "original_idx": rep.get('idx', -1),
                            "original_cluster_id": group_idx,
                            "all_records": group['records']
                        })
        
        print(f"合計{len(representatives)}個の代表レコードを抽出しました")
        return representatives
    
    except Exception as e:
        print(f"代表レコードの抽出中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return []

def select_cluster_representatives(clusters: List[Dict[str, Any]], max_representatives: int = 5) -> Dict[str, List[Dict[str, Any]]]:
    """
    各クラスターから複数の代表レコードを選択する
    
    Args:
        clusters: クラスターリスト
        max_representatives: クラスターごとの最大代表レコード数
        
    Returns:
        クラスターIDをキー、代表レコードリストを値とする辞書
    """
    cluster_representatives = {}
    
    for cluster in clusters:
        cluster_id = cluster.get('cluster_id', '')
        all_records = cluster.get('all_records', [])
        
        if not all_records or not cluster_id:
            continue
            
        # クラスターサイズに基づいて代表レコード数を計算
        cluster_size = len(all_records)
        num_representatives = min(max(2, int(math.sqrt(cluster_size))), max_representatives)
        
        selected_records = []
        
        # メインの代表レコードを含める（存在する場合）
        if 'representative' in cluster and cluster['representative']:
            rep_record = cluster['representative']
            if isinstance(rep_record, dict) and 'id' in rep_record:
                selected_records.append(rep_record)
        
        # 多様性を確保するためにさらにレコードを追加
        remaining_records = [r for r in all_records if r not in selected_records]
        
        # 多様性を高めるための基準でソート
        # 例：タイトルの長さを簡単な指標として使用
        if remaining_records:
            # ソート用にタイトルの長さを取得（エラー処理付き）
            def get_title_length(record):
                try:
                    if isinstance(record, dict):
                        if 'data' in record and isinstance(record['data'], dict):
                            title = record['data'].get('bib1_title', '')
                        else:
                            title = record.get('bib1_title', '')
                        return len(title) if title else 0
                    return 0
                except Exception:
                    return 0
            
            # タイトル長でソートして多様性を確保
            remaining_records.sort(key=get_title_length)
            
            # より良い多様性のために一定間隔でレコードを選択
            if len(remaining_records) <= num_representatives - len(selected_records):
                selected_records.extend(remaining_records)
            else:
                step = len(remaining_records) / (num_representatives - len(selected_records))
                indices = [int(i * step) for i in range(num_representatives - len(selected_records))]
                for idx in indices:
                    if 0 <= idx < len(remaining_records):
                        selected_records.append(remaining_records[idx])
        
        cluster_representatives[cluster_id] = selected_records
    
    return cluster_representatives

def create_cross_cluster_comparison_pairs(cluster_representatives: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    異なるクラスターの代表レコード間のすべての組み合わせペアを生成
    
    Args:
        cluster_representatives: クラスターごとの代表レコード辞書
        
    Returns:
        比較ペアのリスト（レコード情報付き）
    """
    comparison_pairs = []
    
    # すべてのクラスターIDを取得
    cluster_ids = list(cluster_representatives.keys())
    
    # 各クラスターペア間で代表レコードを比較
    for i, cluster_id1 in enumerate(cluster_ids):
        for cluster_id2 in cluster_ids[i+1:]:  # まだ比較していないクラスターとのみ比較
            reps1 = cluster_representatives[cluster_id1]
            reps2 = cluster_representatives[cluster_id2]
            
            # 2つのクラスターの代表レコード間のすべての可能な組み合わせを作成
            for rep1 in reps1:
                for rep2 in reps2:
                    # エラー処理付きでレコード情報を抽出
                    try:
                        rep1_id = rep1.get('id', '')
                        rep2_id = rep2.get('id', '')
                        
                        # IDが欠けている場合はスキップ
                        if not rep1_id or not rep2_id:
                            continue
                        
                        # データフィールドを取得
                        rep1_data = rep1.get('data', {}) if isinstance(rep1, dict) else {}
                        rep2_data = rep2.get('data', {}) if isinstance(rep2, dict) else {}
                        
                        # データが欠けている場合は直接フィールドから抽出
                        if not rep1_data:
                            rep1_data = {k: v for k, v in rep1.items() if k.startswith('bib1_')}
                        if not rep2_data:
                            rep2_data = {k: v for k, v in rep2.items() if k.startswith('bib1_')}
                        
                        # 比較ペアを作成
                        pair_info = {
                            "cluster_pair": [cluster_id1, cluster_id2],
                            "record_pair": [rep1_id, rep2_id],
                            "titles": [
                                rep1_data.get('bib1_title', ''),
                                rep2_data.get('bib1_title', '')
                            ],
                            "authors": [
                                rep1_data.get('bib1_author', ''),
                                rep2_data.get('bib1_author', '')
                            ],
                            "publishers": [
                                rep1_data.get('bib1_publisher', ''),
                                rep2_data.get('bib1_publisher', '')
                            ],
                            "pubdates": [
                                rep1_data.get('bib1_pubdate', ''),
                                rep2_data.get('bib1_pubdate', '')
                            ]
                        }
                        comparison_pairs.append(pair_info)
                    except Exception as e:
                        print(f"比較ペア作成中にエラー発生: {e}")
    
    return comparison_pairs

def extract_records(data: Any) -> List[Dict[str, Any]]:
    """入力データからすべてのレコードを抽出"""
    records = []
    
    try:
        if isinstance(data, dict):
            if 'group' in data and isinstance(data['group'], list):
                for group_idx, group in enumerate(data['group']):
                    if isinstance(group, dict) and 'records' in group and group['records']:
                        for record in group['records']:
                            # レコードにクラスターIDを追加
                            record_copy = copy.deepcopy(record)  # 元のレコードを変更しないようにコピー
                            if 'original_cluster_id' not in record_copy:
                                record_copy['original_cluster_id'] = str(group_idx)
                            records.append(record_copy)
        
        elif isinstance(data, list):
            for group_idx, group in enumerate(data):
                if isinstance(group, dict) and 'records' in group:
                    for record in group['records']:
                        record_copy = copy.deepcopy(record)
                        if 'original_cluster_id' not in record_copy:
                            record_copy['original_cluster_id'] = str(group_idx)
                        records.append(record_copy)
        
        print(f"合計{len(records)}個のレコードを抽出しました")
        return records
    
    except Exception as e:
        print(f"レコード抽出中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return []

# 特徴量の事前計算
def precompute_features(representatives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """代表レコードから特徴量を抽出して事前計算"""
    features = []
    
    for rep in representatives:
        # テキストの正規化
        title = normalize_text(rep['title'])
        author = normalize_text(rep['author'])
        publisher = normalize_text(rep['publisher'])
        
        # タイトルの主要部分を抽出
        main_title_match = re.match(r'^([^\(\[\{【〔]+)', title)
        main_title = main_title_match.group(1).strip() if main_title_match else title
        
        # シリーズ番号や巻数を抽出
        volume_numbers = re.findall(r'\d+', title)
        
        # 出版年を抽出
        year = extract_year(rep['pubdate'])
        
        features.append({
            "cluster_id": rep['cluster_id'],
            "main_title": main_title,
            "full_title": title,
            "author": author,
            "publisher": publisher,
            "year": year,
            "volume_info": volume_numbers,
            "original_data": rep,
            "original_cluster_id": rep.get('original_cluster_id', '')
        })
    
    return features

def batch_comparison_pairs(pairs: List[Dict[str, Any]], batch_size: int = 300) -> List[List[Dict[str, Any]]]:
    """
    比較ペアをAPI処理用のバッチに分割
    
    Args:
        pairs: すべての比較ペアのリスト
        batch_size: バッチあたりの最大ペア数
        
    Returns:
        比較ペアを含むバッチのリスト
    """
    return [pairs[i:i+batch_size] for i in range(0, len(pairs), batch_size)]

# 効率的なAPIリクエストの作成
def create_efficient_api_request(features_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """GPT APIに送信する最適化されたリクエストデータを作成"""
    request_data = {
        "records": []
    }
    
    for feature in features_batch:
        record_data = {
            "id": feature['cluster_id'],
            "main_title": feature['main_title'],
            "full_title": feature['full_title'],
            "author": feature['author'],
            "publisher": feature['publisher'],
            "year": feature['year'],
            "volume_info": feature['volume_info'],
            "original_cluster_id": feature.get('original_cluster_id', '')
        }
        request_data["records"].append(record_data)
    
    return request_data

def create_pair_api_request(pairs_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    レコードペアの類似度計算用APIリクエストを作成
    
    Args:
        pairs_batch: 比較するレコードペアのバッチ
        
    Returns:
        APIリクエストデータ構造
    """
    request_data = {
        "record_pairs": []
    }
    
    for pair in pairs_batch:
        record_pair = {
            "pair_id": f"{pair['record_pair'][0]}_{pair['record_pair'][1]}",
            "cluster_pair": pair["cluster_pair"],
            "titles": pair["titles"],
            "authors": pair["authors"],
            "publishers": pair.get("publishers", ["", ""]),
            "pubdates": pair.get("pubdates", ["", ""])
        }
        request_data["record_pairs"].append(record_pair)
    
    return request_data

# GPT APIリクエストのプロンプト作成
def create_gpt_prompt(batch_data: Dict[str, Any]) -> str:
    """GPT APIに送信するプロンプトを作成"""
    records = batch_data["records"]
    
    prompt = """
あなたは書誌レコードの類似度を分析する専門家です。以下の書籍レコードの各ペア間の類似度を計算し、類似度スコア（0〜1）を返してください。

類似度の計算では以下の要素を考慮してください：
- タイトルの類似性（最も重要、重み付け: 70%）
- 著者名の類似性（重み付け: 20%）
- 出版社の類似性（重み付け: 5%）
- 出版年の類似性（重み付け: 5%）

重要：
- 同じシリーズの異なる巻は高い類似度を持ちますが、完全に一致ではありません（例: 0.8-0.9）
- タイトルと著者が完全に一致する場合は非常に高い類似度（0.95-1.0）
- 日本語のタイトルには様々な表記ゆれがあるため、正規化して比較してください

書籍レコード：
"""
    
    # レコード情報の追加
    for i, record in enumerate(records):
        prompt += f"\n記録{i} (ID: {record['id']}):\n"
        prompt += f"  メインタイトル: {record['main_title']}\n"
        prompt += f"  完全タイトル: {record['full_title']}\n"
        prompt += f"  著者: {record['author']}\n"
        prompt += f"  出版社: {record['publisher']}\n"
        prompt += f"  出版年: {record['year'] if record['year'] else '不明'}\n"
        prompt += f"  巻数情報: {', '.join(record['volume_info']) if record['volume_info'] else '不明'}\n"
    
    prompt += """
指示：
1. すべての可能なレコードペア間の類似度を計算してください
2. 類似度が0.5以上のペアのみを返してください（より多くの詳細な分析情報を取得するため）
3. 結果は以下のJSON形式で返してください：

{
  "similarity_pairs": [
    {"pair": [0, 1], "score": 0.95, "reason": "タイトルと著者が完全に一致、同じシリーズの可能性が高い"},
    {"pair": [2, 4], "score": 0.85, "reason": "同じシリーズの異なる巻と思われる"},
    ...
  ]
}

JSONデータのみを返してください。説明や追加のテキストは不要です。
"""
    return prompt

def create_gpt_prompt_for_pairs(batch_data: Dict[str, Any]) -> str:
    """
    レコードペア間の類似度計算用のGPT APIプロンプトを作成
    
    Args:
        batch_data: レコードペアデータのバッチ
        
    Returns:
        GPT API用のプロンプト文字列
    """
    record_pairs = batch_data["record_pairs"]
    
    prompt = """
あなたは書誌レコードの類似度を分析する専門家です。以下の書籍レコードのペア間の類似度を計算し、類似度スコア（0〜1）を返してください。

類似度の計算では以下の要素を考慮してください：
- タイトルの類似性（最も重要、重み付け: 70%）
- 著者名の類似性（重み付け: 20%）
- 出版社の類似性（重み付け: 5%）
- 出版年の類似性（重み付け: 5%）

重要：
- 同じシリーズの異なる巻は高い類似度を持ちますが、完全に一致ではありません（例: 0.8-0.9）
- タイトルと著者が完全に一致する場合は非常に高い類似度（0.95-1.0）
- 日本語のタイトルには様々な表記ゆれがあるため、正規化して比較してください

書籍レコードペア：
"""
    
    # レコードペア情報の追加
    for i, pair in enumerate(record_pairs):
        prompt += f"\nペア{i} (ID: {pair['pair_id']}):\n"
        prompt += f"  レコード1:\n"
        prompt += f"    タイトル: {pair['titles'][0]}\n"
        prompt += f"    著者: {pair['authors'][0]}\n"
        prompt += f"    出版社: {pair['publishers'][0]}\n"
        prompt += f"    出版年: {pair['pubdates'][0] if pair['pubdates'][0] else '不明'}\n"
        
        prompt += f"  レコード2:\n"
        prompt += f"    タイトル: {pair['titles'][1]}\n"
        prompt += f"    著者: {pair['authors'][1]}\n"
        prompt += f"    出版社: {pair['publishers'][1]}\n"
        prompt += f"    出版年: {pair['pubdates'][1] if pair['pubdates'][1] else '不明'}\n"
    
    prompt += """
指示：
1. 各レコードペアの類似度を計算してください
2. 結果は以下のJSON形式で返してください：

{
  "similarity_results": [
    {"pair_id": "id1_id2", "score": 0.95, "reason": "タイトルと著者が完全に一致、同じシリーズの可能性が高い"},
    {"pair_id": "id3_id4", "score": 0.85, "reason": "同じシリーズの異なる巻と思われる"},
    ...
  ]
}

JSONデータのみを返してください。説明や追加のテキストは不要です。
"""
    return prompt


# GPT APIの並列呼び出し
async def call_gpt_api_async(prompt: str, api_key: str) -> Dict[str, Any]:
    """GPT APIを非同期で呼び出す"""
    async with aiohttp.ClientSession() as session:
        payload = {
            "model": "gpt-4o-mini",  # 使用するモデルを適宜変更
            "messages": [
                {"role": "system", "content": "あなたは日本語の書誌レコードの類似度を計算する専門家です。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,  # 決定論的な結果を得るために低温設定
            "max_tokens": 2000,
            "response_format": {"type": "json_object"}  # JSON形式の応答を要求
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        async with session.post(
            "https://api.openai.com/v1/chat/completions", 
            json=payload, 
            headers=headers
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"API呼び出しエラー: {response.status} - {error_text}")
            
            result = await response.json()
            return result

# バッチの並列処理
async def process_requests_parallel(batch_requests: List[Dict[str, Any]], api_key: str) -> List[Dict[str, Any]]:
    """複数のバッチリクエストを並列処理"""
    tasks = []
    for batch_data in batch_requests:
        prompt = create_gpt_prompt(batch_data)
        tasks.append(call_gpt_api_async(prompt, api_key))
    
    # 並列実行（レート制限を考慮して少し遅延を入れる）
    results = []
    for i, task in enumerate(tasks):
        try:
            result = await task
            results.append({
                "batch_index": i,
                "batch_data": batch_requests[i],
                "api_response": result
            })
            # レート制限対策で少し待機
            if i < len(tasks) - 1:
                await asyncio.sleep(0.5)
        except Exception as e:
            print(f"バッチ {i} の処理中にエラーが発生: {e}")
            results.append({
                "batch_index": i,
                "batch_data": batch_requests[i],
                "error": str(e)
            })
    
    return results

# 比較ペア用のAPIリクエストを作成と処理関数を修正
async def process_pair_requests_parallel(batch_requests: List[Dict[str, Any]], api_key: str) -> List[Dict[str, Any]]:
    """ペア比較用のバッチリクエストを並列処理"""
    tasks = []
    for batch_data in batch_requests:
        # ここでペア用のプロンプト生成関数を使用
        prompt = create_gpt_prompt_for_pairs(batch_data)
        tasks.append(call_gpt_api_async(prompt, api_key))
    
    # 以下は元の関数と同じ
    results = []
    for i, task in enumerate(tasks):
        try:
            result = await task
            results.append({
                "batch_index": i,
                "batch_data": batch_requests[i],
                "api_response": result
            })
            if i < len(tasks) - 1:
                await asyncio.sleep(0.5)
        except Exception as e:
            print(f"バッチ {i} の処理中にエラーが発生: {e}")
            results.append({
                "batch_index": i,
                "batch_data": batch_requests[i],
                "error": str(e)
            })
    
    return results

async def process_with_multi_representatives(yaml_data: str, api_key: str, batch_size: int = 300, threshold: float = 0.7) -> Dict[str, Any]:
    """
    クラスターごとに複数の代表レコードを使用してYAMLデータを処理
    
    Args:
        yaml_data: 入力YAMLデータ
        api_key: OpenAI APIキー
        batch_size: APIリクエストのバッチサイズ
        threshold: クラスタリングの類似度閾値
        
    Returns:
        処理結果
    """
    print("クラスターごとの複数代表レコードによる処理を開始...")
    
    # 代表レコードを抽出して初期クラスターを作成
    representatives = extract_representatives(yaml_data)
    features = precompute_features(representatives)
    
    # 基本的な類似度を取得するための初期API呼び出し
    feature_batches = [features[i:i+batch_size] for i in range(0, len(features), batch_size)]
    api_requests = [create_efficient_api_request(batch) for batch in feature_batches]
    api_results = await process_requests_parallel(api_requests, api_key)
    similarities = extract_similarity_results(api_results, representatives)
    
    # 初期クラスターを作成
    initial_clusters, merge_decisions = create_clusters_from_matches(similarities, representatives, threshold)
    
    # 各クラスターから複数の代表レコードを選択
    max_representatives_per_cluster = 5  # 必要に応じて調整
    cluster_representatives = select_cluster_representatives(initial_clusters, max_representatives_per_cluster)
    
    print(f"{len(cluster_representatives)}個のクラスターから代表レコードを選択しました")
    
    # 異なるクラスターの代表レコード間の比較ペアを作成
    comparison_pairs = create_cross_cluster_comparison_pairs(cluster_representatives)
    
    print(f"クラスター代表間で{len(comparison_pairs)}個の比較ペアを作成しました")
    
    # 比較ペアがない場合はスキップ
    if not comparison_pairs:
        print("比較ペアが作成されませんでした。初期クラスタリングを使用します")
        all_records = extract_records(yaml.safe_load(yaml_data))
        output_groups = format_output_groups(initial_clusters)
        metrics = calculate_output_metrics(output_groups, all_records)
        return format_final_output(output_groups, metrics)
    
    # 比較ペアをバッチに分割
    max_pairs_per_batch = 200  # API制限に基づいて調整
    comparison_batches = batch_comparison_pairs(comparison_pairs, max_pairs_per_batch)
    
    # 比較ペア用のAPIリクエストを作成
    pair_api_requests = []
    for batch in comparison_batches:
        request_data = create_pair_api_request(batch)
        pair_api_requests.append(request_data)

    # 比較ペアを処理
    pair_api_results = await process_pair_requests_parallel(pair_api_requests, api_key)
    pair_similarities = extract_pair_similarity_results(pair_api_results, comparison_pairs)
    
    print(f"{len(pair_similarities)}個の代表ペアの類似度を計算しました")
    
    # 初期の類似度と新しいペアの類似度を組み合わせる
    # ペアの類似度をcreate_clusters_from_matches関数が期待する形式に変換
    additional_similarities = []
    for pair_sim in pair_similarities:
        if pair_sim["similarity_score"] >= threshold:
            similarity_entry = {
                "cluster_pair": pair_sim["cluster_pair"],
                "similarity_score": pair_sim["similarity_score"],
                "reason": pair_sim["reason"],
                "records": [
                    {"cluster_id": pair_sim["cluster_pair"][0], "title": pair_sim["titles"][0], 
                     "author": pair_sim["authors"][0]},
                    {"cluster_id": pair_sim["cluster_pair"][1], "title": pair_sim["titles"][1], 
                     "author": pair_sim["authors"][1]}
                ]
            }
            additional_similarities.append(similarity_entry)
    
    # 類似度を結合
    all_similarities = similarities + additional_similarities
    
    # 結合した類似度で最終クラスターを作成
    final_clusters, final_merge_decisions = create_clusters_from_matches(all_similarities, representatives, threshold)
    
    print(f"{len(final_clusters)}個の最終クラスターを作成しました")
    
    # 評価用にすべてのレコードを抽出
    all_records = extract_records(yaml.safe_load(yaml_data))
    
    # 出力をフォーマットしてメトリクスを計算
    output_groups = format_output_groups(final_clusters)
    metrics = calculate_output_metrics(output_groups, all_records)
    
    # 最終出力をフォーマット
    results = format_final_output(output_groups, metrics)
    results["multi_representative_info"] = {
        "initial_clusters": len(initial_clusters),
        "final_clusters": len(final_clusters),
        "representatives_selected": sum(len(reps) for reps in cluster_representatives.values()),
        "comparison_pairs": len(comparison_pairs),
        "additional_similarities_found": len(additional_similarities)
    }
    
    return results

# API応答から類似度結果を抽出
def extract_similarity_results(api_results: List[Dict[str, Any]], representatives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """API応答から類似度結果を抽出して統合"""
    all_similarities = []
    
    for result in api_results:
        if "error" in result:
            print(f"エラーのためバッチ {result['batch_index']} の結果をスキップ: {result['error']}")
            continue
        
        try:
            # APIレスポンスから類似度ペアを抽出
            content = result["api_response"]["choices"][0]["message"]["content"]
            
            # JSONデータをサニタイズ
            sanitized_content = sanitize_json_response(content)
            
            # サニタイズされたJSONをパース
            response_data = json.loads(sanitized_content)
            
            if "similarity_pairs" not in response_data:
                print(f"バッチ {result['batch_index']} のレスポンスに 'similarity_pairs' がありません")
                continue
                        
            # バッチ内のレコードIDマッピング
            batch_records = result["batch_data"]["records"]
            id_mapping = {i: record["id"] for i, record in enumerate(batch_records)}
            
            # 類似度ペアを変換して追加
            for pair_data in response_data["similarity_pairs"]:
                local_ids = pair_data["pair"]
                global_ids = [id_mapping[local_id] for local_id in local_ids]
                
                # 元のレコード情報を取得
                record_info = []
                for cluster_id in global_ids:
                    for rep in representatives:
                        if rep["cluster_id"] == cluster_id:
                            record_info.append({
                                "cluster_id": cluster_id,
                                "title": rep["title"],
                                "author": rep["author"],
                                "publisher": rep["publisher"],
                                "pubdate": rep["pubdate"],
                                "original_cluster_id": rep.get("original_cluster_id", "")
                            })
                            break
                
                # 類似度情報を追加
                all_similarities.append({
                    "cluster_pair": global_ids,
                    "similarity_score": pair_data["score"],
                    "reason": pair_data.get("reason", ""),
                    "records": record_info
                })
        except Exception as e:
            print(f"バッチ {result['batch_index']} の結果処理中にエラーが発生: {e}")
    
    # 類似度スコアで降順ソート
    all_similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    return all_similarities


def extract_pair_similarity_results(api_results: List[Dict[str, Any]], all_comparison_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    APIレスポンスから類似度結果を抽出してフォーマット
    
    Args:
        api_results: APIレスポンス結果のリスト
        all_comparison_pairs: 元の比較ペア
        
    Returns:
        フォーマットされた類似度結果
    """
    # ペアIDから元のペアへのマッピングを作成
    pair_id_to_original = {}
    for pair in all_comparison_pairs:
        pair_id = f"{pair['record_pair'][0]}_{pair['record_pair'][1]}"
        pair_id_to_original[pair_id] = pair
    
    # 類似度結果を抽出
    all_similarities = []
    
    for result in api_results:
        if "error" in result:
            print(f"バッチ {result['batch_index']} でエラー: {result['error']}")
            continue
        
        try:
            # APIレスポンスから抽出
            content = result["api_response"]["choices"][0]["message"]["content"]
            
            # JSONデータをサニタイズ
            sanitized_content = sanitize_json_response(content)
            
            # サニタイズされたJSONをパース
            response_data = json.loads(sanitized_content)
            
            if "similarity_results" not in response_data:
                print(f"バッチ {result['batch_index']} に 'similarity_results' がありません")
                continue
            
            # 各類似度結果を処理
            for sim_result in response_data["similarity_results"]:
                pair_id = sim_result["pair_id"]
                
                if pair_id not in pair_id_to_original:
                    print(f"不明なペアID: {pair_id}")
                    continue
                
                original_pair = pair_id_to_original[pair_id]
                
                # 類似度エントリを作成
                similarity_entry = {
                    "cluster_pair": original_pair["cluster_pair"],
                    "record_pair": original_pair["record_pair"],
                    "similarity_score": sim_result["score"],
                    "reason": sim_result.get("reason", ""),
                    "titles": original_pair["titles"],
                    "authors": original_pair["authors"]
                }
                
                all_similarities.append(similarity_entry)
                
        except Exception as e:
            print(f"バッチ {result.get('batch_index', '不明')} から結果抽出中にエラー: {e}")
    
    # 類似度スコアで降順ソート（最高値が先頭）
    all_similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    return all_similarities


# 類似度結果の詳細なレポートを作成する関数
def generate_similarity_report(similarities: List[Dict[str, Any]], 
                              representatives: List[Dict[str, Any]], 
                              threshold: float = 0.7,
                              output_dir: str = None,
                              output_file: str = None) -> Dict[str, Any]:
    """類似度結果の詳細なレポートを作成する"""
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_pairs": len(similarities),
        "similarity_threshold": threshold,
        "cluster_pairs": []
    }
    
    # 類似度ペアを整形
    for sim in similarities:
        pair = sim["cluster_pair"]
        score = sim["similarity_score"]
        reason = sim.get("reason", "")
        
        # ペアの詳細情報を作成
        pair_info = {
            "cluster_ids": pair,
            "similarity_score": score,
            "reason": reason,
            "titles": [
                next((rep["title"] for rep in representatives if rep["cluster_id"] == pair[0]), "不明"),
                next((rep["title"] for rep in representatives if rep["cluster_id"] == pair[1]), "不明")
            ],
            "authors": [
                next((rep["author"] for rep in representatives if rep["cluster_id"] == pair[0]), "不明"),
                next((rep["author"] for rep in representatives if rep["cluster_id"] == pair[1]), "不明")
            ]
        }
        
        report["cluster_pairs"].append(pair_info)
    
    # 類似度スコアの統計情報を追加
    scores = [sim["similarity_score"] for sim in similarities]
    if scores:
        report["statistics"] = {
            "min_score": min(scores),
            "max_score": max(scores),
            "avg_score": sum(scores) / len(scores),
            "pairs_above_threshold": sum(1 for s in scores if s >= threshold),
            "score_distribution": {
                "0.9-1.0": sum(1 for s in scores if 0.9 <= s <= 1.0),
                "0.8-0.9": sum(1 for s in scores if 0.8 <= s < 0.9),
                "0.7-0.8": sum(1 for s in scores if 0.7 <= s < 0.8),
                "0.6-0.7": sum(1 for s in scores if 0.6 <= s < 0.7),
                "0.5-0.6": sum(1 for s in scores if 0.5 <= s < 0.6),
                "0.0-0.5": sum(1 for s in scores if 0.0 <= s < 0.5)
            }
        }
    
    if output_dir and output_file:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        full_path = os.path.join(output_dir, output_file)
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"類似度レポートを {full_path} に保存しました")
        
    return report

def sanitize_json_response(json_string: str) -> str:
    """
    APIから返されたJSONレスポンスを修正してパース可能な状態にする
    
    Args:
        json_string: APIから返された生のJSON文字列
        
    Returns:
        サニタイズされたJSON文字列
    """
    if not json_string:
        return "{}"
    
    # 1. 文字列中のエスケープされていない引用符を処理
    # ダブルクォート内のダブルクォートをエスケープ
    cleaned = ""
    in_string = False
    escape_next = False
    
    for i, char in enumerate(json_string):
        if char == '"' and not escape_next:
            in_string = not in_string
            cleaned += char
        elif char == '\\':
            escape_next = True
            cleaned += char
        elif char == '"' and escape_next:
            escape_next = False
            cleaned += char
        elif in_string and char == '\n':
            # 文字列内の改行を削除
            cleaned += " "
            escape_next = False
        else:
            cleaned += char
            escape_next = False
    
    # 2. 文字列が正しく閉じられているか確認
    # 引用符の数をカウント
    quote_count = cleaned.count('"')
    if quote_count % 2 != 0:
        # 引用符の数が奇数の場合、最後に引用符を追加
        cleaned += '"'
    
    # 3. カンマの修正 - 余分なカンマの削除と不足しているカンマの追加
    lines = cleaned.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        line = line.rstrip()
        
        # 最後の行でない場合の処理
        if i < len(lines) - 1:
            # オブジェクトや配列の閉じ括弧の前のカンマを削除
            if line.rstrip().endswith(',') and lines[i+1].strip() in [']}', ']', '}']:
                line = line.rstrip(',')
            # オブジェクトや配列の要素の後にカンマがない場合追加
            elif not line.rstrip().endswith(',') and not line.rstrip().endswith('{') and \
                 not line.rstrip().endswith('[') and not line.rstrip().endswith('}') and \
                 not line.rstrip().endswith(']') and lines[i+1].strip() not in [']}', ']', '}']:
                line += ','
        
        fixed_lines.append(line)
    
    cleaned = '\n'.join(fixed_lines)
    
    # 4. プロパティ名が引用符で囲まれているか確認
    # 正規表現で簡易的にチェック
    import re
    property_pattern = r'([a-zA-Z0-9_]+):'
    cleaned = re.sub(property_pattern, r'"\1":', cleaned)
    
    # 5. 日本語の処理
    # すでにエスケープされている場合は処理しない
    
    try:
        # 一度パースしてみて、エラーがなければそのまま返す
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError as e:
        # エラーが継続する場合、回復不能としてシンプルな構造を返す
        print(f"JSONサニタイズ後もエラーが継続: {e}")
        error_info = {"error": str(e), "original_text_sample": json_string[:100] + "..."}
        
        # GPT APIの応答フォーマットに合わせた構造を作成
        if "similarity_results" in json_string:
            return json.dumps({"similarity_results": []})
        elif "similarity_pairs" in json_string:
            return json.dumps({"similarity_pairs": []})
        else:
            return json.dumps({"error": "Parse failed", "results": []})

# 類似度結果を分析する
def analyze_similarity_results(similarities: List[Dict[str, Any]], threshold: float = 0.7) -> Dict[str, Any]:
    """類似度結果を分析する"""
    # 類似度スコアの統計
    scores = [sim["similarity_score"] for sim in similarities]
    
    stats = {
        "count": len(scores),
        "min_score": min(scores) if scores else 0,
        "max_score": max(scores) if scores else 0,
        "avg_score": sum(scores) / len(scores) if scores else 0,
        "threshold": threshold,
        "pairs_above_threshold": sum(1 for s in scores if s >= threshold),
        "score_distribution": {
            "0.9-1.0": sum(1 for s in scores if 0.9 <= s <= 1.0),
            "0.8-0.9": sum(1 for s in scores if 0.8 <= s < 0.9),
            "0.7-0.8": sum(1 for s in scores if 0.7 <= s < 0.8),
            "0.6-0.7": sum(1 for s in scores if 0.6 <= s < 0.7),
            "0.5-0.6": sum(1 for s in scores if 0.5 <= s < 0.6),
            "0.0-0.5": sum(1 for s in scores if 0.0 <= s < 0.5)
        }
    }
    
    return stats

def create_clusters_from_matches(similarities: List[Dict[str, Any]], representatives: List[Dict[str, Any]], threshold: float = 0.7) -> tuple:
    """類似度結果からクラスターを作成し、代表レコードが類似の場合はクラスター全体を統合"""
    
    # Union-Findデータ構造を初期化
    uf = UnionFind(len(representatives))
    
    # クラスターIDからインデックスへのマッピング
    cluster_id_to_index = {rep["cluster_id"]: i for i, rep in enumerate(representatives)}
    
    # クラスター結合の記録
    merge_decisions = []
    
    # 類似度に基づいて結合
    for similarity in similarities:
        if similarity["similarity_score"] >= threshold:  # 類似度閾値
            pair = similarity["cluster_pair"]
            if pair[0] in cluster_id_to_index and pair[1] in cluster_id_to_index:
                idx1 = cluster_id_to_index[pair[0]]
                idx2 = cluster_id_to_index[pair[1]]
                
                # クラスター統合を記録
                merge_info = {
                    "pair": [pair[0], pair[1]],
                    "titles": [representatives[idx1]['title'], representatives[idx2]['title']],
                    "similarity_score": similarity["similarity_score"],
                    "reason": similarity.get("reason", ""),
                    "merged": True
                }
                merge_decisions.append(merge_info)
                
                print(f"クラスター統合: {representatives[idx1]['title']} と {representatives[idx2]['title']} (類似度: {similarity['similarity_score']})")
                uf.union(idx1, idx2)
                
        # しきい値以下の類似度ペアも記録（統合なし）
        elif similarity["similarity_score"] >= 0.5:  # 記録する下限閾値
            pair = similarity["cluster_pair"]
            if pair[0] in cluster_id_to_index and pair[1] in cluster_id_to_index:
                idx1 = cluster_id_to_index[pair[0]]
                idx2 = cluster_id_to_index[pair[1]]
                
                # 統合しなかった類似ペアも記録
                not_merged_info = {
                    "pair": [pair[0], pair[1]],
                    "titles": [representatives[idx1]['title'], representatives[idx2]['title']],
                    "similarity_score": similarity["similarity_score"],
                    "reason": similarity.get("reason", ""),
                    "merged": False
                }
                merge_decisions.append(not_merged_info)
    
    # クラスターの構築（統合されたクラスターの全レコードを含む）
    merged_clusters = defaultdict(list)
    for i, rep in enumerate(representatives):
        root = uf.find(i)
        # クラスターの代表レコードを追加
        merged_clusters[root].append(rep)
    
    # 結果のフォーマット
    result_clusters = []
    for cluster_idx, members in merged_clusters.items():
        # 代表レコードを選択（便宜上、最初のメンバーを使用）
        representative = members[0] if members else None
        
        # 統合されたクラスター内の全レコードを収集
        all_merged_records = []
        for member in members:
            if "all_records" in member and member["all_records"]:
                all_merged_records.extend(member["all_records"])
        
        result_clusters.append({
            "cluster_id": str(uuid.uuid4()),
            "members": members,
            "representative": representative,
            "all_records": all_merged_records,  # 統合された全レコード
            "member_count": len(members),
            "record_count": len(all_merged_records)
        })
        
        print(f"統合クラスター作成: {len(members)}個のクラスターが統合され、{len(all_merged_records)}個のレコードを含む")
    
    # クラスター統合の決定情報も返す
    return result_clusters, merge_decisions

# 出力グループのフォーマット
def format_output_groups(clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """クラスターを出力形式にフォーマット"""
    output_groups = []
    
    print(f"出力グループをフォーマット中... {len(clusters)}個のクラスターを処理")
    
    for cluster_idx, cluster in enumerate(clusters):
        records = []
        # デバッグ情報を追加
        print(f"クラスター{cluster_idx}を処理中:")
        print(f"  メンバー数: {len(cluster.get('members', []))}")
        print(f"  代表レコード: {cluster.get('representative', {}).get('title', 'なし')}")
        
        # all_recordsフィールドを確認
        if 'all_records' in cluster and cluster['all_records']:
            all_records = cluster['all_records']
            print(f"  全レコード数: {len(all_records)}")
            
            for record in all_records:
                # レコードにdataフィールドがあるか確認
                if isinstance(record, dict):
                    record_id = record.get('id', '')
                    
                    # dataフィールドがある場合は直接使用
                    if 'data' in record and isinstance(record['data'], dict):
                        data_fields = record['data']
                    else:
                        # 必要なフィールドを探す
                        data_fields = {}
                        for key, value in record.items():
                            if key.startswith('bib1_'):
                                data_fields[key] = value
                    
                    # レコードを出力形式に追加
                    records.append({
                        "id": record_id,
                        "cluster_id": cluster.get("cluster_id", str(uuid.uuid4())),
                        "original_cluster_id": record.get("original_cluster_id", ""),
                        "data": {
                            "bib1_title": data_fields.get("bib1_title", ""),
                            "bib1_author": data_fields.get("bib1_author", ""),
                            "bib1_publisher": data_fields.get("bib1_publisher", ""),
                            "bib1_pubdate": data_fields.get("bib1_pubdate", "")
                        }
                    })
        
        # 完全一致フラグ（すべてのレコードが同じ元のクラスターIDを持つか）
        perfect_match = False
        if records:
            first_original_id = records[0].get("original_cluster_id", "")
            perfect_match = all(r.get("original_cluster_id", "") == first_original_id for r in records)
            print(f"  出力レコード数: {len(records)}, 完全一致: {perfect_match}")
        else:
            print("  出力レコード数: 0")
        
        group = {
            "correct": [[i] for i in range(len(records))],
            "perfect_match": perfect_match,
            "records": records
        }
        
        output_groups.append(group)
    
    print(f"合計{len(output_groups)}個の出力グループを作成しました")
    return output_groups

# 出力メトリクスの計算
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

# human_in_the_loop_process関数内で評価履歴をCSVファイルとして出力する機能を追加
def write_iteration_results_csv(evaluation_history, output_prefix, strategy, human_accuracy, output_dir="results"):
    """反復ごとの評価結果をCSVファイルに出力"""
    # 出力ディレクトリの作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"出力ディレクトリ {output_dir} を作成しました")
    
    import csv
    # ファイル名の作成
    output_csv = f"results/{output_prefix}_{strategy}_{int(human_accuracy*100)}_iterations.csv"
    
    # CSVファイルに出力
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # ヘッダー行
        writer.writerow([
            "iteration", "f1_pair", "precision_pair", "recall_pair", 
            "complete_group", "num_groups"
        ])
        
        # 各反復の結果
        for entry in evaluation_history:
            writer.writerow([
                entry.get("iteration", 0),
                entry.get("f1_pair", 0),
                entry.get("precision_pair", 0),
                entry.get("recall_pair", 0),
                entry.get("complete_group", 0),
                entry.get("num_groups", 0)
            ])
    
    print(f"反復結果を {output_csv} に保存しました")
    return output_csv

# 最終出力のフォーマット
def format_final_output(groups: List[Dict], metrics: Dict) -> Dict:
    """出力を最終形式でフォーマット"""
    output = {
        "version": "3.1",
        "type": "RESULT",
        "id": str(uuid.uuid4()),
        "summary": metrics,
        "group": groups
    }
    
    return output

# 結果をファイルに保存
# 結果をファイルに保存
def save_result_files(results: Dict[str, Any], output_yaml: str, output_json: str = None) -> None:
    """結果をYAMLとJSONファイルに保存"""
    # 出力ディレクトリを作成
    output_dir = os.path.dirname(output_yaml)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"出力ディレクトリ {output_dir} を作成しました")
    
    # YAMLファイルに保存
    with open(output_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(results, f, allow_unicode=True, sort_keys=False)
    print(f"結果を {output_yaml} に保存しました")
    
    # JSONファイルにも保存（指定があれば）
    if output_json:
        if os.path.dirname(output_json) and not os.path.exists(os.path.dirname(output_json)):
            os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"結果を {output_json} にも保存しました")

# 矛盾を検知して修正するための関数
def detect_transitivity_violations(similarity_pairs: List[Dict[str, Any]], threshold: float = 0.7) -> List[tuple]:
    """
    推移律の矛盾を検出する
    A=B, B=C, A≠C の形のパターンを検出
    """
    # グラフ構造を作成
    match_graph = defaultdict(set)
    unmatch_graph = defaultdict(set)
    
    # 類似度がしきい値以上のものをマッチと見なし、グラフに追加
    for pair in similarity_pairs:
        pair_ids = pair["cluster_pair"]
        score = pair["similarity_score"]
        
        if score >= threshold:
            match_graph[pair_ids[0]].add(pair_ids[1])
            match_graph[pair_ids[1]].add(pair_ids[0])
        else:
            # しきい値未満は不一致
            unmatch_graph[pair_ids[0]].add(pair_ids[1])
            unmatch_graph[pair_ids[1]].add(pair_ids[0])
    
    # 推移律に矛盾するトリプルを発見
    inconsistent_triplets = []
    
    for entity_a in match_graph:
        for entity_b in match_graph[entity_a]:
            for entity_c in match_graph[entity_b]:
                # A=B, B=C, A≠C という矛盾パターンを検出
                if entity_c in unmatch_graph.get(entity_a, set()) or entity_a in unmatch_graph.get(entity_c, set()):
                    inconsistent_triplets.append((entity_a, entity_b, entity_c))
    
    return inconsistent_triplets

# 矛盾検出とインコンシステンシースコア計算
def calculate_inconsistency_score(entity_a: str, entity_b: str, entity_c: str, 
                                 similarity_pairs: List[Dict[str, Any]]) -> float:
    """
    3つのエンティティ間の非一貫性スコアを計算（論文の式4.1に基づく）
    """
    # ペアのマッピングを作成
    pair_to_similarity = {}
    for sim in similarity_pairs:
        pair_ids = tuple(sorted(sim["cluster_pair"]))
        pair_to_similarity[pair_ids] = sim["similarity_score"]
    
    # a-b, b-c, c-aの類似度を取得
    p_a_b = pair_to_similarity.get(tuple(sorted([entity_a, entity_b])), 0.5)
    p_b_c = pair_to_similarity.get(tuple(sorted([entity_b, entity_c])), 0.5)
    p_c_a = 1.0 - pair_to_similarity.get(tuple(sorted([entity_c, entity_a])), 0.5)  # 不一致確率
    
    # インコンシステンシースコアを計算
    inconsistency = p_a_b * p_b_c * p_c_a
    
    return inconsistency

# 最も矛盾が大きいトリプルを検出する拡張関数
def detect_inconsistent_triplets(similarity_pairs: List[Dict[str, Any]], 
                                representatives: List[Dict[str, Any]], 
                                threshold: float = 0.7) -> List[Dict[str, Any]]:
    """
    推移律に矛盾するトリプルを検出し、インコンシステンシースコアでランク付け
    """
    # すべてのエンティティIDのリスト
    entity_ids = [rep["cluster_id"] for rep in representatives]
    
    # マッチングと非マッチングのグラフを構築
    match_graph = defaultdict(set)
    
    for sim in similarity_pairs:
        pair_ids = sim["cluster_pair"]
        score = sim["similarity_score"]
        
        if score >= threshold:
            match_graph[pair_ids[0]].add(pair_ids[1])
            match_graph[pair_ids[1]].add(pair_ids[0])
    
    # 矛盾するトリプルを検出
    inconsistent_triplets = []
    
    # すべての可能な3つ組みを検討
    for i, entity_a in enumerate(entity_ids):
        for j, entity_b in enumerate(entity_ids[i+1:], i+1):
            if entity_b not in match_graph[entity_a]:
                continue  # A-Bがマッチしない場合はスキップ
                
            for k, entity_c in enumerate(entity_ids[j+1:], j+1):
                if entity_c not in match_graph[entity_b]:
                    continue  # B-Cがマッチしない場合はスキップ
                
                # a-cがマッチしない場合、矛盾を検出
                if entity_c not in match_graph[entity_a]:
                    # インコンシステンシースコアを計算
                    score = calculate_inconsistency_score(entity_a, entity_b, entity_c, similarity_pairs)
                    
                    inconsistent_triplets.append({
                        "triplet": (entity_a, entity_b, entity_c),
                        "inconsistency_score": score
                    })
    
    # インコンシステンシースコアで降順ソート
    inconsistent_triplets.sort(key=lambda x: x["inconsistency_score"], reverse=True)
    
    return inconsistent_triplets

# 核心となるペアのみを抽出する戦略（論文5.5.4節）
def extract_core_inconsistent_pairs(inconsistent_triplets: List[Dict[str, Any]], 
                                   similarity_pairs: List[Dict[str, Any]],
                                   max_samples: int = 300) -> List[Dict[str, Any]]:
    """
    矛盾の核心と考えられるペアのみを抽出する戦略
    """
    core_pairs = []
    pair_set = set()  # 重複を避けるため
    
    # ペアの類似度マッピング
    pair_to_similarity = {}
    for sim in similarity_pairs:
        pair = tuple(sorted(sim["cluster_pair"]))
        pair_to_similarity[pair] = sim["similarity_score"]
    
    for triplet_info in inconsistent_triplets:
        triplet = triplet_info["triplet"]
        entity_a, entity_b, entity_c = triplet
        
        # 3つのペアの不確実性（0.5からの距離）を計算
        pairs = [
            ((entity_a, entity_b), abs(pair_to_similarity.get(tuple(sorted([entity_a, entity_b])), 0.5) - 0.5)),
            ((entity_b, entity_c), abs(pair_to_similarity.get(tuple(sorted([entity_b, entity_c])), 0.5) - 0.5)),
            ((entity_a, entity_c), abs(pair_to_similarity.get(tuple(sorted([entity_a, entity_c])), 0.5) - 0.5))
        ]
        
        # 最も不確実な（0.5に最も近い）ペアを選択
        pairs.sort(key=lambda x: x[1])
        most_uncertain_pair = pairs[0][0]
        uncertainty = pairs[0][1]
        
        # 重複チェック
        sorted_pair = tuple(sorted(most_uncertain_pair))
        if sorted_pair not in pair_set:
            pair_set.add(sorted_pair)
            
            core_pairs.append({
                "pair": most_uncertain_pair,
                "triplet": triplet,
                "uncertainty": uncertainty,
                "inconsistency_score": triplet_info["inconsistency_score"]
            })
    
    # 不確実性でソートして上位のサンプルを返す
    core_pairs.sort(key=lambda x: x["uncertainty"])
    return core_pairs[:max_samples]

# 人間のフィードバックをシミュレートする関数
async def simulate_human_feedback(samples: List[Dict[str, Any]], 
                                representatives: List[Dict[str, Any]], 
                                correct_labels: Dict[Tuple[str, str], bool] = None,
                                human_accuracy: float = 1.0) -> List[Dict[str, Any]]:
    """
    矛盾検知に基づく人間フィードバックのシミュレーション
    """
    corrected_pairs = []
    
    # 代表エンティティのID -> インデックスマッピング
    cluster_id_to_index = {rep["cluster_id"]: i for i, rep in enumerate(representatives)}
    
    for sample in samples:
        pair = sample["pair"]
        triplet = sample.get("triplet", None)
        
        # 実際の正解ラベルを取得（シミュレーションのため）
        actual_match = False
        if correct_labels and tuple(sorted(pair)) in correct_labels:
            actual_match = correct_labels[tuple(sorted(pair))]
        else:
            # 正解ラベルが提供されていない場合、実際のデータから推測
            # 例：同じoriginal_cluster_idを持つなら一致
            try:
                idx1 = cluster_id_to_index.get(pair[0], -1)
                idx2 = cluster_id_to_index.get(pair[1], -1)
                if idx1 >= 0 and idx2 >= 0:
                    rep1 = representatives[idx1]
                    rep2 = representatives[idx2]
                    if rep1.get("original_cluster_id", "") == rep2.get("original_cluster_id", "") and rep1.get("original_cluster_id", ""):
                        actual_match = True
            except Exception as e:
                print(f"正解ラベル推測中にエラー: {e}")
        
        # 人間の正解率に基づいて回答をシミュレート
        human_answer = actual_match
        if random.random() > human_accuracy:
            # エラーの場合は逆の回答
            human_answer = not actual_match
        
        # 修正情報を記録
        correction = {
            "pair": pair,
            "triplet": triplet,
            "label": "match" if human_answer else "unmatch",
            "actual_match": actual_match,  # デバッグ用
            "correct_answer": human_answer == actual_match  # デバッグ用
        }
        
        corrected_pairs.append(correction)
    
    return corrected_pairs

# 人間のフィードバックに基づいて類似度結果を更新
def update_similarity_results(similarity_pairs: List[Dict[str, Any]], 
                             corrected_pairs: List[Dict[str, Any]]) -> None:
    """
    人間のフィードバックに基づいて類似度結果を更新
    """
    # ペアのマッピングを作成
    pair_to_index = {}
    for i, sim in enumerate(similarity_pairs):
        pair = tuple(sorted(sim["cluster_pair"]))
        pair_to_index[pair] = i
    
    # 修正を適用
    for correction in corrected_pairs:
        pair = tuple(sorted(correction["pair"]))
        if pair in pair_to_index:
            idx = pair_to_index[pair]
            
            # 類似度を更新
            if correction["label"] == "match":
                similarity_pairs[idx]["similarity_score"] = 1.0  # 一致の場合は最高スコア
                similarity_pairs[idx]["reason"] = "Human feedback: match"
            else:
                similarity_pairs[idx]["similarity_score"] = 0.0  # 不一致の場合は最低スコア
                similarity_pairs[idx]["reason"] = "Human feedback: unmatch"
            
            print(f"ペア {pair} の類似度を {similarity_pairs[idx]['similarity_score']} に更新しました")

# Human-in-the-loopプロセスのメイン関数
async def human_in_the_loop_process(yaml_data: str, 
                                   api_key: str, 
                                   batch_size: int = 300, 
                                   threshold: float = 0.7, 
                                   human_accuracy: float = 1.0,
                                   max_iterations: int = 10,
                                   strategy: str = 'core_inconsistency',
                                   log_iterations: bool = False,
                                   output_prefix: str = "result",
                                   output_dir: str = "results") -> Dict[str, Any]:
    """
    Human-in-the-loopエンティティマッチングの完全なプロセス
    反復ごとの結果をログに記録する機能を追加
    """
    print(f"=== Human-in-the-loop処理を開始 (戦略: {strategy}, 人間精度: {human_accuracy}) ===")
    
    # 1. 代表レコードの抽出
    representatives = extract_representatives(yaml_data)
    print(f"{len(representatives)}個のクラスター代表を抽出しました")
    
    # 2. 全レコードの抽出
    data = yaml.safe_load(yaml_data)
    all_records = extract_records(data)
    print(f"合計{len(all_records)}個のレコードを処理します")
    
    # 3. 特徴量の事前計算と初期類似度計算
    features = precompute_features(representatives)
    
    # バッチに分割
    feature_batches = [features[i:i+batch_size] for i in range(0, len(features), batch_size)]
    
    # APIリクエストを準備
    api_requests = []
    for batch in feature_batches:
        request_data = create_efficient_api_request(batch)
        api_requests.append(request_data)
    
    # 初期のAPI呼び出し
    api_results = await process_requests_parallel(api_requests, api_key)
    similarity_pairs = extract_similarity_results(api_results, representatives)
    
    # 正解ラベルを抽出（シミュレーション用）
    correct_labels = {}
    for i, rep1 in enumerate(representatives):
        for j, rep2 in enumerate(representatives):
            if i < j:  # 重複を避ける
                # 同じ元クラスターIDなら一致と見なす
                is_match = rep1.get("original_cluster_id", "") == rep2.get("original_cluster_id", "") and rep1.get("original_cluster_id", "")
                correct_labels[tuple(sorted([rep1["cluster_id"], rep2["cluster_id"]]))] = is_match
    
    # 4. 初期クラスタリング
    clusters, _ = create_clusters_from_matches(similarity_pairs, representatives, threshold)
    
    # 評価結果の追跡
    evaluation_history = []
    
    # 5. 反復プロセス
    for iteration in range(max_iterations):
        print(f"\n----- 反復 {iteration+1}/{max_iterations} -----")
        
        # 現在の結果を評価
        output_groups = format_output_groups(clusters)
        metrics = calculate_output_metrics(output_groups, all_records)
        
        # 反復結果を記録
        iteration_result = {
            "iteration": iteration,
            "f1_pair": float(metrics.get("f1(pair)", "0")),
            "precision_pair": float(metrics.get("precision(pair)", "0")),
            "recall_pair": float(metrics.get("recall(pair)", "0")),
            "complete_group": float(metrics.get("complete(group)", "0")),
            "num_groups": metrics.get("num_of_groups(inference)", 0)
        }
        evaluation_history.append(iteration_result)
        
        print(f"現在のF1スコア: {metrics.get('f1(pair)', 'N/A')}")
        
        # 反復の詳細をログに記録（フラグが有効な場合）
        if log_iterations:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                print(f"出力ディレクトリ {output_dir} を作成しました")
        
            csv_file = f"{output_dir}/{output_prefix}_{strategy}_{int(human_accuracy*100)}_iterations.csv"
            iteration_log_file = f"{output_dir}/{output_prefix}_{strategy}_{int(human_accuracy*100)}_iter{iteration}.json"
            with open(iteration_log_file, 'w', encoding='utf-8') as f:
                iteration_data = {
                    "iteration": iteration,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "metrics": metrics,
                    "clusters": {
                        "count": len(clusters),
                        "sizes": [len(cluster.get("all_records", [])) for cluster in clusters]
                    }
                }
                json.dump(iteration_data, f, ensure_ascii=False, indent=2)
            print(f"反復{iteration}の情報を {iteration_log_file} に保存しました")
        
        # 5.1 矛盾するトリプルを検出
        # 以下、既存の反復処理コード（変更なし）...
        
        # 5.3 人間からのフィードバックをシミュレーション
        corrected_pairs = await simulate_human_feedback(
            samples, representatives, correct_labels, human_accuracy)
        
        # 修正情報をログに記録（フラグが有効な場合）
        if log_iterations:
            corrections_log_file = f"{output_dir}/{output_prefix}_{strategy}_{int(human_accuracy*100)}_iter{iteration}_corrections.json"
            with open(corrections_log_file, 'w', encoding='utf-8') as f:
                json.dump(corrected_pairs, f, ensure_ascii=False, indent=2)
            print(f"反復{iteration}の修正情報を {corrections_log_file} に保存しました")
        
        # 5.4 類似度結果を更新（既存コード）...
        
    # 最終評価
    output_groups = format_output_groups(clusters)
    final_metrics = calculate_output_metrics(output_groups, all_records)
    
    # 最終結果を評価履歴に追加
    evaluation_history.append({
        "iteration": max_iterations,
        "f1_pair": float(final_metrics.get("f1(pair)", "0")),
        "precision_pair": float(final_metrics.get("precision(pair)", "0")),
        "recall_pair": float(final_metrics.get("recall(pair)", "0")),
        "complete_group": float(final_metrics.get("complete(group)", "0")),
        "num_groups": final_metrics.get("num_of_groups(inference)", 0)
    })
    
    # 反復結果をCSV形式で保存
    if log_iterations:
        csv_file = f"{output_dir}/{output_prefix}_{strategy}_{int(human_accuracy*100)}_iterations.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            import csv
            writer = csv.writer(f)
            # ヘッダー行
            writer.writerow([
                "iteration", "f1_pair", "precision_pair", "recall_pair", 
                "complete_group", "num_groups"
            ])
            
            # 各反復の結果
            for entry in evaluation_history:
                writer.writerow([
                    entry.get("iteration", 0),
                    entry.get("f1_pair", 0),
                    entry.get("precision_pair", 0),
                    entry.get("recall_pair", 0),
                    entry.get("complete_group", 0),
                    entry.get("num_groups", 0)
                ])
        print(f"反復結果サマリーを {csv_file} に保存しました")
    
    print("\n=== Human-in-the-loop処理完了 ===")
    print(f"最終F1スコア: {final_metrics.get('f1(pair)', 'N/A')}")
    
    # 結果を返す
    final_results = format_final_output(output_groups, final_metrics)
    final_results["evaluation_history"] = evaluation_history
    final_results["human_accuracy"] = human_accuracy
    final_results["strategy"] = strategy
    
    return final_results

def log_iteration_details(iteration, corrected_pairs, similarity_changes, 
                         output_prefix, strategy, human_accuracy, output_dir="results"):
    """各反復の詳細情報をログファイルに記録"""
    # 出力ディレクトリを確認・作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"出力ディレクトリ {output_dir} を作成しました")
        
    # ログファイル名
    log_file = f"{output_dir}/{output_prefix}_{strategy}_{int(human_accuracy*100)}_iter{iteration}_details.json"
    
    # 詳細情報
    details = {
        "iteration": iteration,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "strategy": strategy,
        "human_accuracy": human_accuracy,
        "corrected_pairs": corrected_pairs,
        "similarity_changes": similarity_changes
    }
    
    # JSONファイルに出力
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    
    print(f"反復{iteration}の詳細を {log_file} に保存しました")

def generate_iteration_graphs(output_prefix, strategy, human_accuracy, output_dir="results"):
    """反復結果のCSVファイルからグラフを生成"""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        
        # 出力ディレクトリを確認・作成
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"出力ディレクトリ {output_dir} を作成しました")
        
        # CSVファイルを読み込み
        csv_file = f"results/{output_prefix}_{strategy}_{int(human_accuracy*100)}_iterations.csv"
        df = pd.read_csv(csv_file)
        
        # F1スコアの推移グラフ
        plt.figure(figsize=(10, 6))
        plt.plot(df["iteration"], df["f1_pair"], marker='o', linewidth=2, label="F1スコア")
        plt.plot(df["iteration"], df["precision_pair"], marker='s', linewidth=2, label="精度")
        plt.plot(df["iteration"], df["recall_pair"], marker='^', linewidth=2, label="再現率")
        
        plt.title(f"反復によるスコア推移 (戦略: {strategy}, 人間精度: {human_accuracy})")
        plt.xlabel("反復回数")
        plt.ylabel("スコア")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        # グラフを保存
        graph_file = f"results/{output_prefix}_{strategy}_{int(human_accuracy*100)}_scores.png"
        plt.savefig(graph_file)
        print(f"反復スコアグラフを {graph_file} に保存しました")
        
        # クラスター数の推移グラフ
        plt.figure(figsize=(10, 6))
        plt.plot(df["iteration"], df["num_groups"], marker='o', linewidth=2, color='green')
        
        plt.title(f"反復によるクラスター数の推移 (戦略: {strategy}, 人間精度: {human_accuracy})")
        plt.xlabel("反復回数")
        plt.ylabel("クラスター数")
        plt.grid(True)
        plt.tight_layout()
        
        # グラフを保存
        graph_file = f"results/{output_prefix}_{strategy}_{int(human_accuracy*100)}_clusters.png"
        plt.savefig(graph_file)
        print(f"クラスター数グラフを {graph_file} に保存しました")
        
    except ImportError:
        print("グラフ生成には matplotlib と pandas が必要です")
    except Exception as e:
        print(f"グラフ生成中にエラーが発生: {e}")
        
# 実験用の関数 - 様々な人間精度と戦略の組み合わせでテスト
async def run_experiments(yaml_file_path: str, 
                       output_prefix: str = None, 
                       batch_size: int = 8, 
                       threshold: float = 0.7,
                       api_key: str = None,
                       output_dir: str = "results"):
    """YAMLファイルを読み込み、複数の人間精度と戦略の組み合わせで実験"""
    try:
        # 出力ディレクトリを作成
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"出力ディレクトリ {output_dir} を作成しました")
            
        if output_prefix is None:
            input_basename = os.path.basename(yaml_file_path)
            input_name = os.path.splitext(input_basename)[0]
            output_prefix = f"{input_name}_experiment"
        
        os.makedirs("results", exist_ok=True)
        
        # YAMLファイルを読み込み
        with open(yaml_file_path, 'r', encoding='utf-8') as f:
            yaml_data = f.read()
        
        print(f"YAMLファイル '{yaml_file_path}' を読み込みました")
        
        # 実験設定
        human_accuracies = [1.0, 0.95, 0.9]
        strategies = ['core_inconsistency', 'inconsistency', 'uncertainty', 'hybrid']
        
        # 実験結果を格納する辞書
        all_results = {
            "experiment_setup": {
                "yaml_file": yaml_file_path,
                "threshold": threshold,
                "batch_size": batch_size,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "results": []
        }
        
        # 各実験を実行
        for accuracy in human_accuracies:
            for strategy in strategies:
                print(f"\n\n======= 実験: 精度={accuracy}, 戦略={strategy} =======\n")
                
                # 処理を実行
                start_time = time.time()
                results = await human_in_the_loop_process(
                    yaml_data, api_key, batch_size, threshold, 
                    human_accuracy=accuracy, strategy=strategy
                )
                end_time = time.time()
                
                # 処理時間を記録
                results["processing_time"] = end_time - start_time
                
                # 結果ファイル名
                output_file = f"results/{output_prefix}_{strategy}_{int(accuracy*100)}.yaml"
                output_json = f"results/{output_prefix}_{strategy}_{int(accuracy*100)}.json"
                
                # 結果を保存
                save_result_files(results, output_file, output_json)
                
                # サマリーを収集
                result_summary = {
                    "strategy": strategy,
                    "human_accuracy": accuracy,
                    "processing_time": results["processing_time"],
                    "final_f1": float(results.get("summary", {}).get("f1(pair)", "0")),
                    "final_precision": float(results.get("summary", {}).get("precision(pair)", "0")),
                    "final_recall": float(results.get("summary", {}).get("recall(pair)", "0")),
                    "final_complete_group": float(results.get("summary", {}).get("complete(group)", "0")),
                    "evaluation_history": results.get("evaluation_history", [])
                }
                
                all_results["results"].append(result_summary)
                
                print(f"実験完了: 精度={accuracy}, 戦略={strategy}")
                print(f"結果を {output_file} に保存しました")
        
        # 全実験結果のサマリーを保存
        summary_file = f"results/{output_prefix}_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n全実験結果のサマリーを {summary_file} に保存しました")
        
        # 結果をグラフで可視化
        visualize_experiment_results(all_results, output_prefix)
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


# クラスターごとの代表ペア数を調整する関数
def select_representative_pairs(clusters: List[Dict[str, Any]], max_representatives: int = 10) -> List[Dict[str, Any]]:
    """
    クラスターごとに代表ペアを選択する関数
    クラスターサイズに応じて代表数を調整する
    
    Args:
        clusters: クラスターのリスト
        max_representatives: 1クラスターあたりの最大代表数
        
    Returns:
        選択された代表ペアのリスト
    """
    representative_pairs = []
    
    for cluster in clusters:
        # クラスターサイズの取得
        cluster_size = len(cluster.get('all_records', []))
        
        # クラスターサイズに応じて代表数を計算
        # 小さいクラスターでは少なく、大きいクラスターでは多く選ぶ
        if cluster_size == 0:
            continue
            
        # 代表数をクラスターサイズの平方根に比例させる（最大値あり）
        num_representatives = min(max(1, int(math.sqrt(cluster_size))), max_representatives)
        
        # クラスター内のレコードからランダムに代表を選択
        all_records = cluster.get('all_records', [])
        if len(all_records) <= num_representatives:
            selected_records = all_records  # すべてのレコードを使用
        else:
            # 代表を選ぶ戦略：
            # 1. クラスターの代表レコードを必ず含める
            # 2. 残りはクラスター内でランダムに選択（ただし多様性を考慮）
            selected_records = []
            
            # 代表レコードを追加
            if 'representative' in cluster and cluster['representative']:
                rep_record = cluster['representative']
                selected_records.append(rep_record)
            
            # 残りのレコードをランダムに選択
            remaining_records = [r for r in all_records if r not in selected_records]
            
            # シンプルな多様性のための選択（より高度な選択ロジックも可能）
            # 例：タイトル長や出版年でソートして均等に選ぶなど
            remaining_needed = num_representatives - len(selected_records)
            if remaining_needed > 0 and remaining_records:
                # ランダムサンプルを取得
                random_sample = random.sample(remaining_records, 
                                            min(remaining_needed, len(remaining_records)))
                selected_records.extend(random_sample)
        
        # 選択したレコードから代表ペアを作成
        for i, rec1 in enumerate(selected_records):
            for j, rec2 in enumerate(selected_records[i+1:], i+1):
                # 同じクラスター内のペアの情報を作成
                pair_info = {
                    "cluster_id": cluster.get('cluster_id', ''),
                    "pair": [rec1.get('id', ''), rec2.get('id', '')],
                    "titles": [
                        rec1.get('data', {}).get('bib1_title', ''),
                        rec2.get('data', {}).get('bib1_title', '')
                    ],
                    "authors": [
                        rec1.get('data', {}).get('bib1_author', ''),
                        rec2.get('data', {}).get('bib1_author', '')
                    ],
                    "publishers": [
                        rec1.get('data', {}).get('bib1_publisher', ''),
                        rec2.get('data', {}).get('bib1_publisher', '')
                    ],
                    "is_same_cluster": True
                }
                representative_pairs.append(pair_info)
    
    return representative_pairs

# 実験結果を可視化する関数
def visualize_experiment_results(all_results, output_prefix, output_dir="results"):
    """
    実験結果をグラフで可視化
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"出力ディレクトリ {output_dir} を作成しました")
        # 結果をDataFrameに変換
        experiment_data = []
        
        for result in all_results["results"]:
            strategy = result["strategy"]
            accuracy = result["human_accuracy"]
            
            for entry in result.get("evaluation_history", []):
                experiment_data.append({
                    "strategy": strategy,
                    "human_accuracy": accuracy,
                    "iteration": entry.get("iteration", 0),
                    "f1": entry.get("f1_pair", 0),
                    "precision": entry.get("precision_pair", 0),
                    "recall": entry.get("recall_pair", 0),
                    "complete_group": entry.get("complete_group", 0),
                    "num_groups": entry.get("num_groups", 0)
                })
        
        df = pd.DataFrame(experiment_data)
        
        # 1. 精度別のF1スコア推移
        plt.figure(figsize=(12, 8))
        
        for accuracy in df["human_accuracy"].unique():
            for strategy in df["strategy"].unique():
                data = df[(df["human_accuracy"] == accuracy) & (df["strategy"] == strategy)]
                plt.plot(data["iteration"], data["f1"], 
                         marker='o', 
                         label=f"{strategy} (精度={accuracy})")
        
        plt.title("Human-in-the-loopプロセスにおけるF1スコアの推移")
        plt.xlabel("反復回数")
        plt.ylabel("F1スコア")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/{output_prefix}_f1_trend.png")
        
        # 2. 戦略ごとの最終F1スコア比較
        plt.figure(figsize=(12, 8))
        
        final_results = []
        for result in all_results["results"]:
            final_results.append({
                "strategy": result["strategy"],
                "human_accuracy": result["human_accuracy"],
                "final_f1": result["final_f1"]
            })
        
        final_df = pd.DataFrame(final_results)
        
        for accuracy in final_df["human_accuracy"].unique():
            data = final_df[final_df["human_accuracy"] == accuracy]
            plt.bar(
                [f"{strategy} ({accuracy})" for strategy in data["strategy"]], 
                data["final_f1"]
            )
        
        plt.title("戦略と精度ごとの最終F1スコア")
        plt.xlabel("戦略 (人間精度)")
        plt.ylabel("最終F1スコア")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"results/{output_prefix}_final_f1_comparison.png")
        
        # 3. 精度低下の影響分析
        plt.figure(figsize=(12, 8))
        
        for strategy in df["strategy"].unique():
            strategy_data = []
            for accuracy in sorted(df["human_accuracy"].unique(), reverse=True):
                data = df[(df["strategy"] == strategy) & (df["human_accuracy"] == accuracy)]
                if not data.empty:
                    last_iteration = data["iteration"].max()
                    final_f1 = data[data["iteration"] == last_iteration]["f1"].iloc[0]
                    strategy_data.append({
                        "accuracy": accuracy,
                        "f1": final_f1
                    })
            
            strategy_df = pd.DataFrame(strategy_data)
            plt.plot(strategy_df["accuracy"], strategy_df["f1"], 
                     marker='o', linewidth=2, label=strategy)
        
        plt.title("人間の精度低下によるF1スコアへの影響")
        plt.xlabel("人間の回答精度")
        plt.ylabel("最終F1スコア")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/{output_prefix}_accuracy_impact.png")
        
        print("グラフの可視化が完了しました")
        
    except ImportError:
        print("グラフ可視化には matplotlib と pandas が必要です")
    except Exception as e:
        print(f"可視化中にエラーが発生しました: {e}")

# YAMLファイルを読み込み、処理して結果を保存する関数
async def process_yaml_file(yaml_file_path: str, 
                         output_file: str = None, 
                         batch_size: int = 8, 
                         threshold: float = 0.7, 
                         api_key: str = None,
                         strategy: str = 'core_inconsistency',
                         human_accuracy: float = 1.0,
                         max_iterations: int = 10,
                         similarity_report_path: str = None,
                         multi_rep: bool = False,
                         reps_per_cluster: int = 5,
                         log_iterations: bool = False,
                         output_dir: str = "results") -> Dict[str, Any]:
    """
    YAMLファイルを読み込み、処理して結果を保存
    """
    try:
        # コマンドライン引数の処理
        if output_file is None:
            # 入力ファイル名から出力ファイル名を自動生成
            input_basename = os.path.basename(yaml_file_path)
            input_name = os.path.splitext(input_basename)[0]
            
            if multi_rep:
                # 複数代表レコードモードの場合
                output_file = f"{output_dir}/{input_name}_multi_rep_{reps_per_cluster}.yaml"
            elif human_accuracy < 1.0:
                # Human-in-the-loopモードの場合
                output_file = f"{output_dir}/{input_name}_{strategy}_{int(human_accuracy*100)}.yaml"
            else:
                # 通常処理の場合
                output_file = f"{output_dir}/{input_name}_result.yaml"
        elif not output_file.startswith(f"{output_dir}/"):
            output_file = f"{output_dir}/{output_file}"
        
        # JSON出力ファイル名も生成
        output_json = f"{os.path.splitext(output_file)[0]}.json"
        
        # YAMLファイルを読み込み
        with open(yaml_file_path, 'r', encoding='utf-8') as f:
            yaml_data = f.read()
        
        print(f"YAMLファイル '{yaml_file_path}' を読み込みました")
        
        # 処理開始時刻
        start_time = datetime.now()
        print(f"処理開始時刻: {start_time}")
        
        # 処理の種類に応じて実行
        if multi_rep:
            # 複数代表レコードモードで実行
            results = await process_with_multi_representatives(
                yaml_data,
                api_key,
                batch_size,
                threshold
            )
        elif human_accuracy < 1.0 or strategy != 'core_inconsistency':
            # Human-in-the-loopモードで実行（反復処理）
            input_prefix = os.path.splitext(os.path.basename(yaml_file_path))[0]
            
            results = await human_in_the_loop_process(
                yaml_data,
                api_key,
                batch_size,
                threshold,
                human_accuracy,
                max_iterations,
                strategy,
                log_iterations=log_iterations,
                output_prefix=input_prefix,
                output_dir=output_dir
            )        
        else:
            # 通常の処理を実行
            # 1. 代表レコードの抽出
            representatives = extract_representatives(yaml_data)
            
            # 2. 特徴量の事前計算
            features = precompute_features(representatives)
            
            # 3. バッチに分割
            feature_batches = [features[i:i+batch_size] for i in range(0, len(features), batch_size)]
            
            # 4. APIリクエストを準備
            api_requests = []
            for batch in feature_batches:
                request_data = create_efficient_api_request(batch)
                api_requests.append(request_data)
            
            # 5. 並列にAPIリクエストを実行
            api_results = await process_requests_parallel(api_requests, api_key)
            
            # 6. 結果を抽出して統合
            similarities = extract_similarity_results(api_results, representatives)
            
            # 7. クラスターを作成
            clusters, merge_decisions = create_clusters_from_matches(similarities, representatives, threshold)
            # クラスターごとの代表ペアを選択
            max_representatives_per_cluster = 10  # コマンドライン引数から設定可能にするとよい
            representative_pairs = select_representative_pairs(clusters, max_representatives_per_cluster)
            # 代表ペアの情報を結果に追加
            results["representative_pairs"] = representative_pairs
            
            # 8. 全レコードの抽出
            data = yaml.safe_load(yaml_data)
            all_records = extract_records(data)
            
            # 9. 出力グループのフォーマット
            output_groups = format_output_groups(clusters)
            
            # 10. 出力メトリクスの計算
            metrics = calculate_output_metrics(output_groups, all_records)
            
            # 11. 最終出力のフォーマット
            results = format_final_output(output_groups, metrics)
            
            # 類似度情報を出力に追加
            similarity_report = generate_similarity_report(similarities, representatives, threshold)
            similarity_analysis = analyze_similarity_results(similarities, threshold)
            
            results["similarity_report"] = similarity_report
            results["similarity_analysis"] = similarity_analysis
            results["merge_decisions"] = merge_decisions
            
        # 処理時間の追加
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        results["processing_time_seconds"] = processing_time
        
        # 結果を保存
        save_result_files(results, output_file, output_json)
        
        # 類似度レポートを別途保存（指定があれば）
        if similarity_report_path:
            if not similarity_report_path.startswith(f"{output_dir}/"):
                similarity_report_path = f"{output_dir}/{similarity_report_path}"
            
            with open(similarity_report_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "similarity_report": results.get("similarity_report", {}),
                    "similarity_analysis": results.get("similarity_analysis", {}),
                    "merge_decisions": results.get("merge_decisions", [])
                }, f, ensure_ascii=False, indent=2)
            print(f"類似度レポートを {similarity_report_path} に保存しました")
        
        # 終了時刻とサマリー
        print(f"処理完了時刻: {end_time}")
        print(f"合計処理時間: {processing_time:.2f}秒")
        
        # サマリー情報の表示
        if "summary" in results:
            metrics = results["summary"]
            print("\n評価メトリクス:")
            print(f"  F1スコア (ペア): {metrics.get('f1(pair)', 'N/A')}")
            print(f"  精度 (ペア): {metrics.get('precision(pair)', 'N/A')}")
            print(f"  再現率 (ペア): {metrics.get('recall(pair)', 'N/A')}")
            print(f"  完全一致グループ率: {metrics.get('complete(group)', 'N/A')}")
        
        return results
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
    
# メイン関数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='改善された複数代表レコードによるエンティティマッチング')
    
    # 基本的な引数
    parser.add_argument('--input', '-i', type=str, required=True, help='処理するYAMLファイルのパス')
    parser.add_argument('--output', '-o', type=str, help='結果を保存するYAMLファイルのパス')
    parser.add_argument('--api-key', '-k', type=str, help='OpenAI APIキー（環境変数が設定されていない場合）')
    parser.add_argument('--threshold', '-t', type=float, default=0.7, help='類似度しきい値')
    parser.add_argument('--batch-size', '-b', type=int, default=8, help='APIリクエストのバッチサイズ')
    parser.add_argument('--similarity-report', '-s', type=str, help='類似度レポートを別途保存するJSONファイルのパス')
    
    # Human-in-the-loop関連の引数
    parser.add_argument('--strategy', '-st', type=str, default='core_inconsistency', 
                     choices=['core_inconsistency', 'inconsistency', 'uncertainty', 'hybrid'],
                     help='サンプリング戦略の選択')
    parser.add_argument('--human-accuracy', '-ha', type=float, default=1.0,
                     help='人間の回答精度のシミュレーション（0.0-1.0）')
    parser.add_argument('--iterations', '-it', type=int, default=10, 
                     help='Human-in-the-loopの最大反復回数')
    
    # 複数代表モード用の引数
    parser.add_argument('--multi-rep', '-mr', action='store_true', 
                     help='より正確なマッチングのためにクラスターごとに複数の代表レコードを使用')
    parser.add_argument('--reps-per-cluster', '-rpc', type=int, default=5,
                     help='クラスターごとの最大代表レコード数（デフォルト: 5）')
    
    # 出力設定関連の引数（新規追加）
    parser.add_argument('--output-dir', '-od', type=str, default='results',
                     help='結果ファイルを保存するディレクトリ（デフォルト: results）')
    parser.add_argument('--log-iterations', '-li', action='store_true', 
                     help='各反復の詳細をログファイルに出力')
    parser.add_argument('--generate-graphs', '-gg', action='store_true',
                     help='反復結果のグラフを自動生成')
    
    # デバッグ関連の引数
    parser.add_argument('--debug', '-d', action='store_true', help='デバッグモードを有効化')
    
    args = parser.parse_args()
    
    # 出力ディレクトリの作成
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"出力ディレクトリ {args.output_dir} を作成しました")
    
    # APIキーの取得
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("警告: OpenAI APIキーが指定されていません。")
        exit(1)
    
    # ファイル処理の実行
    asyncio.run(process_yaml_file(
        args.input, 
        args.output, 
        args.batch_size, 
        args.threshold, 
        api_key,
        strategy=args.strategy,
        human_accuracy=args.human_accuracy,
        max_iterations=args.iterations,
        similarity_report_path=args.similarity_report,
        multi_rep=args.multi_rep,
        reps_per_cluster=args.reps_per_cluster,
        log_iterations=args.log_iterations,
        output_dir=args.output_dir
    ))
    
    