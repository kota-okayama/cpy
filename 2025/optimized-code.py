"""
LLMベースの書誌レコードマッチングシステム (最適化版)

このスクリプトはLLM（大規模言語モデル）を使用して書誌レコードの類似性を判定し、
クラスタリングを行います。OpenAI ChatCompletionを使用してレコード間の類似度を判断し、
結果をYAMLファイルで出力します。

API使用量を削減するための最適化:
- バッチ処理による効率的なAPI呼び出し
- 事前フィルタリングによる比較ペア数の削減
- キャッシュ機能による処理の高速化と費用の節約
- より効率的なサンプリング戦略
- 専用のスレッドプールによる並列処理
"""

import os
import sys
import yaml
import json
import argparse
import pickle
import hashlib
import uuid
import time
import threading
import queue
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Set, Union
import concurrent.futures

# 依存ライブラリのインポート試行
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("警告: 'requests'ライブラリがインストールされていません。以下のコマンドでインストールしてください:")
    print("pip install requests")
    sys.exit(1)

# 設定（デフォルト値）
DEBUG_MODE = False  # デバッグモード（詳細な出力）
SIMILARITY_THRESHOLD = 0.7  # LLMの類似度判定のしきい値（0-1）
MAX_RETRIES = 3  # APIコールの最大リトライ回数
RETRY_DELAY = 2  # リトライ間の待機時間（秒）
CACHE_DIR = ".cache"  # キャッシュディレクトリ
BATCH_SIZE_DEFAULT = 5  # 一度に処理するAPIリクエストの最大数（デフォルト値）
MAX_WORKERS_DEFAULT = 4  # 並列処理時のワーカー数（デフォルト値）
REQUEST_INTERVAL = 0.5  # APIリクエスト間の最小間隔（秒）
PRE_FILTER_THRESHOLD_DEFAULT = 0.5  # 事前フィルタリングのしきい値（デフォルト値）

# 実行時に設定される値（グローバル変数）
batch_size = BATCH_SIZE_DEFAULT
max_workers = MAX_WORKERS_DEFAULT
pre_filter_threshold = PRE_FILTER_THRESHOLD_DEFAULT

# APIリクエスト制御のためのセマフォとタイマー
api_semaphore = None
last_request_time = None
last_request_lock = None

def initialize_api_control():
    """APIリクエスト制御のための変数を初期化"""
    global api_semaphore, last_request_time, last_request_lock
    api_semaphore = threading.Semaphore(max_workers)
    last_request_time = datetime.now()
    last_request_lock = threading.Lock()

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
    field_counts = defaultdict(int)
    
    # まずフィールドの出現回数を集計
    for record in records:
        if "data" in record and isinstance(record["data"], dict):
            for field in record["data"].keys():
                field_counts[field] += 1
    
    # 出現回数の多いフィールドを優先
    sorted_fields = sorted(field_counts.items(), key=lambda x: x[1], reverse=True)
    
    for field, count in sorted_fields:
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

def get_cache_key(text1: str, text2: str) -> str:
    """2つのテキストからキャッシュキーを生成する"""
    # 2つのテキストの組み合わせハッシュ（順序に依存しないようにソート）
    combined = "\n===\n".join(sorted([text1, text2]))
    return hashlib.md5(combined.encode('utf-8')).hexdigest()

def load_similarity_cache(cache_file: str) -> Dict[str, float]:
    """類似度キャッシュを読み込む"""
    if os.path.exists(cache_file):
        try:
            print(f"類似度キャッシュを読み込み中: {cache_file}")
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
            print(f"{len(cache)}件の類似度をキャッシュから読み込みました")
            return cache
        except Exception as e:
            print(f"キャッシュ読み込みエラー: {e}")
            return {}
    return {}

def save_similarity_cache(cache: Dict[str, float], cache_file: str):
    """類似度キャッシュを保存する"""
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
        print(f"{len(cache)}件の類似度をキャッシュに保存しました: {cache_file}")
    except Exception as e:
        print(f"キャッシュ保存エラー: {e}")

def wait_for_rate_limit():
    """APIリクエストのレート制限に対応するための待機"""
    global last_request_time
    
    with last_request_lock:
        now = datetime.now()
        elapsed = (now - last_request_time).total_seconds()
        
        if elapsed < REQUEST_INTERVAL:
            wait_time = REQUEST_INTERVAL - elapsed
            time.sleep(wait_time)
        
        last_request_time = datetime.now()

def get_similarity_with_llm(text1: str, text2: str, api_key: str) -> float:
    """
    LLMを使用して2つのテキスト間の類似度を計算する。
    返り値は0〜1の数値で、1が完全一致、0が完全に異なることを表す。
    """
    batch_data = [{'text1': text1, 'text2': text2}]
    results = get_similarity_batch(batch_data, api_key)
    return results[0]['similarity']

def pre_filter_records(record1: Dict, record2: Dict, field_types: Dict[str, str]) -> float:
    """
    LLMを呼び出す前に、簡易な方法でレコードの類似度を事前評価する
    高速で低コストな方法で明らかに異なるレコードを除外する
    """
    similarity_score = 0.0
    total_weight = 0
    
    if "data" not in record1 or "data" not in record2:
        return 0.0
    
    # 重要なフィールドの重み付け
    field_weights = {
        "title": 5,    # タイトルが最も重要
        "author": 3,   # 著者も重要
        "publisher": 1, # 出版社はやや重要
        "pubdate": 1    # 出版日はやや重要
    }
    
    for field_name, weight in field_weights.items():
        # フィールド名を含むキーを探す
        matching_fields1 = [f for f in record1["data"].keys() if field_name in f.lower()]
        matching_fields2 = [f for f in record2["data"].keys() if field_name in f.lower()]
        
        if matching_fields1 and matching_fields2:
            field1 = matching_fields1[0]
            field2 = matching_fields2[0]
            
            value1 = str(record1["data"][field1]).lower()
            value2 = str(record2["data"][field2]).lower()
            
            # 簡易な類似度計算
            if field_name == "title" or field_name == "author":
                # より単純な正規化
                import re
                
                # 基本的な正規化
                value1 = re.sub(r'[^\w\s]', '', value1).strip()
                value2 = re.sub(r'[^\w\s]', '', value2).strip()
                
                # 完全一致の場合
                if value1 == value2:
                    similarity_score += weight
                # 部分一致の場合
                elif value1 in value2 or value2 in value1:
                    similarity_score += weight * 0.7
                # 単語レベルの類似性
                else:
                    words1 = set(value1.split())
                    words2 = set(value2.split())
                    
                    if words1 and words2:
                        common_words = words1.intersection(words2)
                        word_similarity = len(common_words) / max(len(words1), len(words2))
                        similarity_score += weight * word_similarity
            
            # 出版社や出版日は完全一致のみを考慮
            else:
                if value1 == value2:
                    similarity_score += weight
            
            total_weight += weight
    
    # 類似度スコアを0-1の範囲に正規化
    if total_weight > 0:
        normalized_score = similarity_score / total_weight
        return normalized_score
    
    return 0.0

def get_similarity_batch(batch_data: List[Dict], api_key: str) -> List[Dict]:
    """
    LLMを使用して複数のテキストペアの類似度を一括で計算する
    """
    if not api_key:
        print("エラー: APIキーが提供されていません。--api-key オプションまたは環境変数 OPENAI_API_KEY を設定してください。")
        sys.exit(1)
    
    # レート制限のための待機
    wait_for_rate_limit()
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # システムプロンプトは1回だけ送信
    system_prompt = """
    あなたは書誌レコードの類似性を判断する専門家です。複数の書誌レコードペアを評価していただきます。
    各ペアについて、それらが同じ作品を参照しているかどうかを判断してください。
    タイトル、著者、出版社、出版日などの情報を考慮して、類似度を0〜1の数値で評価してください。
    1は完全に同じ作品、0は完全に異なる作品を表します。
    
    以下の基準で評価してください：
    - タイトルが非常に似ている場合は高い類似度
    - 著者が一致する場合も高い類似度
    - 出版社と出版日も考慮するが、タイトルと著者ほど重要ではない
    - 表記の違い（全角/半角、かな/カナ、句読点の有無など）は同じものとみなす
    
    各ペアに対する回答は必ず0〜1の数値のみを返してください。例：0.95
    """
    
    user_prompt = "以下の書誌レコードペアを評価してください。各ペアの類似度を0～1の数値で回答してください。\n\n"
    
    # 各ペアのプロンプトを作成
    for i, item in enumerate(batch_data):
        pair_prompt = f"""ペア{i+1}:
書誌レコード1:
{item['text1']}

書誌レコード2:
{item['text2']}

類似度（0〜1の数値のみ）:
"""
        user_prompt += pair_prompt + "\n"
    
    # APIリクエストデータ
    data = {
        "model": "gpt-4",  # または "gpt-3.5-turbo" などの適切なモデル
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0  # 決定論的な回答を得るために低い温度を設定
    }
    
    debug_log(f"{len(batch_data)}件のペアに対して類似度判定をリクエスト")
    
    for attempt in range(MAX_RETRIES):
        try:
            with api_semaphore:
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=60  # バッチ処理のための長めのタイムアウト
                )
            
            if response.status_code == 200:
                result = response.json()
                # テキスト応答から数値を抽出
                similarity_text = result["choices"][0]["message"]["content"].strip()
                
                try:
                    # バッチ応答を解析
                    import re
                    similarity_values = []
                    
                    # 数値をすべて抽出
                    matches = re.findall(r'(0\.\d+|\d+\.0|[01])', similarity_text)
                    
                    # 各ペアに対する結果を割り当て
                    for i, item in enumerate(batch_data):
                        if i < len(matches):
                            similarity = float(matches[i])
                            # 範囲を0〜1に制限
                            similarity = max(0.0, min(1.0, similarity))
                            item['similarity'] = similarity
                        else:
                            print(f"警告: ペア{i+1}の類似度を抽出できませんでした")
                            item['similarity'] = 0.5  # デフォルト値
                    
                    return batch_data
                
                except Exception as e:
                    print(f"警告: 類似度の解析エラー: {e}")
                    # 各ペアにデフォルト値を設定
                    for item in batch_data:
                        item['similarity'] = 0.5
                    return batch_data
            
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
    # 各ペアにデフォルト値を設定
    for item in batch_data:
        item['similarity'] = 0.5
    return batch_data

def batch_worker(batch_queue, results_queue, api_key, similarity_cache, cache_file, cache_lock):
    """バッチ処理ワーカー（スレッド/プロセス用）"""
    while True:
        try:
            batch = batch_queue.get(block=False)
            if batch is None:
                batch_queue.task_done()
                break
                
            # キャッシュをチェックし、未キャッシュのペアのみをAPIに送信
            uncached_batch = []
            cached_results = []
            
            for item in batch:
                cache_key = get_cache_key(item['text1'], item['text2'])
                
                with cache_lock:
                    if cache_key in similarity_cache:
                        # キャッシュヒット
                        item['similarity'] = similarity_cache[cache_key]
                        cached_results.append(item)
                    else:
                        # キャッシュミス
                        uncached_batch.append(item)
            
            # キャッシュされていないペアがあればAPI呼び出し
            if uncached_batch:
                api_results = get_similarity_batch(uncached_batch, api_key)
                
                # 結果をキャッシュに保存
                with cache_lock:
                    for item in api_results:
                        cache_key = get_cache_key(item['text1'], item['text2'])
                        similarity_cache[cache_key] = item['similarity']
                    
                    # 定期的にキャッシュを保存
                    if len(uncached_batch) >= 5:
                        save_similarity_cache(similarity_cache, cache_file)
                
                # すべての結果を結果キューに追加
                for item in api_results:
                    results_queue.put(item)
            
            # キャッシュされた結果も結果キューに追加
            for item in cached_results:
                results_queue.put(item)
                
            batch_queue.task_done()
            
        except queue.Empty:
            break
        except Exception as e:
            print(f"バッチワーカーエラー: {e}")
            batch_queue.task_done()

def calculate_record_similarities(records: List[Dict], field_types: Dict[str, str],
                                api_key: str, cache_dir: str = CACHE_DIR,
                                sample_size: int = 20) -> List[Tuple]:
    """
    レコード間の類似度を計算する（バッチ処理と事前フィルタリングを使用）
    
    流れ:
    1. レコードをサンプリング
    2. 事前フィルタリングで比較するペア数を削減
    3. ユーザーに確認を求める
    4. バッチで複数ペアを一度に処理
    5. 類似度が高いペアをマッチとしてリスト化
    """
    # グローバル変数の参照
    global batch_size, max_workers, pre_filter_threshold

    # APIリクエスト制御の初期化
    initialize_api_control()
    
    # キャッシュディレクトリの作成
    os.makedirs(cache_dir, exist_ok=True)
    
    # キャッシュファイル名
    cache_file = os.path.join(cache_dir, f"similarity_cache.pkl")
    
    # キャッシュの読み込み
    similarity_cache = load_similarity_cache(cache_file)
    cache_lock = threading.Lock()
    
    # 結果を格納するリスト
    matches = []
    
    # サンプリング戦略
    # 1. 各クラスターから代表的なレコードを選択
    # 2. 残りのレコードをランダムサンプリング
    if len(records) > sample_size:
        import random
        from collections import defaultdict
        
        # 元のクラスターごとにレコードをグループ化
        clusters = defaultdict(list)
        for record in records:
            cluster_id = record.get("original_cluster_id", "unknown")
            clusters[cluster_id].append(record)
        
        # 各クラスターから1レコードずつ選択
        sampled_records = []
        for cluster_id, cluster_records in clusters.items():
            sampled_records.append(random.choice(cluster_records))
        
        # 残りのサンプルサイズをランダムに選択
        remaining_records = [r for r in records if r not in sampled_records]
        if remaining_records and len(sampled_records) < sample_size:
            additional_samples = random.sample(remaining_records, min(sample_size - len(sampled_records), len(remaining_records)))
            sampled_records.extend(additional_samples)
        
        print(f"{len(records)}件のレコードから{len(sampled_records)}件をサンプリングしました")
    else:
        sampled_records = records
        print(f"全{len(records)}件のレコードを使用します")
    
    # 比較ペアの準備（事前フィルタリング適用）
    comparison_pairs = []
    filtered_count = 0
    total_possible_pairs = len(sampled_records) * (len(sampled_records) - 1) // 2
    
    print(f"事前フィルタリング中...")
    
    for i in range(len(sampled_records)):
        record1 = sampled_records[i]
        
        for j in range(i + 1, len(sampled_records)):
            record2 = sampled_records[j]
            
            # 事前フィルタリング
            pre_filter_score = pre_filter_records(record1, record2, field_types)
            
            # しきい値を超える場合のみLLMで詳細評価
            if pre_filter_score >= pre_filter_threshold:
                # LLM向けにフォーマット
                text1 = format_record_for_llm(record1, field_types)
                text2 = format_record_for_llm(record2, field_types)
                
                # 比較ペアとして追加
                comparison_pairs.append({
                    'text1': text1,
                    'text2': text2,
                    'record_id1': record1.get("id", ""),
                    'record_id2': record2.get("id", ""),
                    'record1': record1,
                    'record2': record2
                })
            else:
                filtered_count += 1
    
    print(f"事前フィルタリング: {filtered_count}/{total_possible_pairs}ペアをスキップ ({filtered_count/total_possible_pairs*100:.1f}%)")
    
    # キャッシュヒット数を計算
    cache_hit_count = 0
    for pair in comparison_pairs:
        cache_key = get_cache_key(pair['text1'], pair['text2'])
        if cache_key in similarity_cache:
            cache_hit_count += 1
    
    # 実際に必要なAPI呼び出し回数を計算
    required_api_calls = len(comparison_pairs) - cache_hit_count
    batch_count = (required_api_calls + batch_size - 1) // batch_size if batch_size > 0 else required_api_calls
    
    # APIコストの概算（GPT-4の場合）
    tokens_per_pair = 800  # 平均トークン数の概算
    total_tokens = required_api_calls * tokens_per_pair
    cost_per_1k_tokens = 0.06  # ドル、GPT-4の場合
    estimated_cost = total_tokens * cost_per_1k_tokens / 1000  # ドル
    
    print(f"必要なAPI呼び出し情報:")
    print(f"  評価対象ペア数: {len(comparison_pairs)}ペア")
    print(f"  キャッシュヒット: {cache_hit_count}ペア")
    print(f"  API呼び出し必要数: {required_api_calls}ペア")
    print(f"  バッチ数: {batch_count}バッチ（バッチサイズ: {batch_size}）")
    print(f"  予想トークン数: 約{total_tokens}トークン")
    print(f"  概算コスト: 約${estimated_cost:.2f}")
    
    print(f"\n{len(comparison_pairs)}ペアの詳細評価を実行します")
    
    # バッチ処理のためのキュー設定
    batch_queue = queue.Queue()
    results_queue = queue.Queue()
    
    # バッチに分割
    current_batch = []
    for pair in comparison_pairs:
        current_batch.append(pair)
        
        if len(current_batch) >= batch_size:
            batch_queue.put(current_batch)
            current_batch = []
    
    # 残りのペアがあればバッチに追加
    if current_batch:
        batch_queue.put(current_batch)
    
    # ワーカースレッドの設定
    threads = []
    for _ in range(min(max_workers, batch_queue.qsize())):
        thread = threading.Thread(
            target=batch_worker,
            args=(batch_queue, results_queue, api_key, similarity_cache, cache_file, cache_lock)
        )
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    # 処理進捗の表示
    total_pairs = len(comparison_pairs)
    processed = 0
    start_time = datetime.now()
    
    print(f"レコード間の類似度を計算中...")
    
    # 結果の収集
    while processed < total_pairs:
        try:
            result = results_queue.get(timeout=1)
            processed += 1
            
            # 進捗表示
            if processed % 5 == 0 or processed == total_pairs:
                progress = processed / total_pairs * 100
                elapsed = (datetime.now() - start_time).total_seconds()
                remaining = elapsed / processed * (total_pairs - processed) if processed > 0 else 0
                print(f"  進捗: {processed}/{total_pairs}回 ({progress:.1f}%) - "
                      f"経過: {elapsed:.1f}秒, 残り: {remaining:.1f}秒")
            
            # しきい値を超える場合、マッチとして記録
            if result['similarity'] >= SIMILARITY_THRESHOLD:
                record1 = result['record1']
                record2 = result['record2']
                
                # 詳細情報を抽出
                title_field = next((field for field, ftype in field_types.items() 
                                if ftype == "TEXT" and "title" in field.lower()), None)
                author_field = next((field for field, ftype in field_types.items() 
                                if ftype == "COMPLEMENT_JA" and "author" in field.lower()), None)
                
                title1 = record1["data"].get(title_field, "") if title_field and "data" in record1 else ""
                title2 = record2["data"].get(title_field, "") if title_field and "data" in record2 else ""
                author1 = record1["data"].get(author_field, "") if author_field and "data" in record1 else ""
                author2 = record2["data"].get(author_field, "") if author_field and "data" in record2 else ""
                
                # マッチ情報を追加
                match_info = {
                    "record_id1": result['record_id1'],
                    "record_id2": result['record_id2'],
                    "similarity": result['similarity'],
                    "title1": title1,
                    "title2": title2,
                    "author1": author1,
                    "author2": author2,
                    "original_cluster_id1": record1.get("original_cluster_id", ""),
                    "original_cluster_id2": record2.get("original_cluster_id", "")
                }
                
                matches.append(match_info)
            
            results_queue.task_done()
            
        except queue.Empty:
            # キューが空の場合は少し待機
            time.sleep(0.1)
            
            # すべてのスレッドが終了しているか確認
            if all(not t.is_alive() for t in threads) and results_queue.empty():
                break
    
    # スレッドの終了を待機
    for t in threads:
        t.join()
    
    # 最終的なキャッシュを保存
    with cache_lock:
        save_similarity_cache(similarity_cache, cache_file)
    
    # 類似度でソート
    matches.sort(key=lambda x: x["similarity"], reverse=True)
    
    print(f"{len(matches)}件の類似レコードペアを見つけました")
    
    return matches
def main():
    parser = argparse.ArgumentParser(description="LLMベースの書誌レコードマッチング (最適化版)")
    parser.add_argument("--input", "-i", type=str, required=True, help="入力YAMLファイルパス")
    parser.add_argument("--output", "-o", type=str, default="result.yaml", help="出力YAMLファイルパス")
    parser.add_argument("--api-key", "-k", type=str, help="OpenAI APIキー")
    parser.add_argument("--threshold", "-t", type=float, default=SIMILARITY_THRESHOLD, help="類似度しきい値")
    parser.add_argument("--cache-dir", "-c", type=str, default=CACHE_DIR, help="キャッシュディレクトリ")
    parser.add_argument("--sample", "-s", type=int, default=20, help="サンプルサイズ（レコード数が多い場合）")
    parser.add_argument("--debug", "-d", action="store_true", help="デバッグモードを有効化")
    parser.add_argument("--batch-size", "-b", type=int, default=BATCH_SIZE_DEFAULT, help="バッチサイズ（一度に処理するペア数）")
    parser.add_argument("--max-workers", "-w", type=int, default=MAX_WORKERS_DEFAULT, help="最大ワーカー数")
    parser.add_argument("--pre-filter", "-p", type=float, default=PRE_FILTER_THRESHOLD_DEFAULT, help="事前フィルタリングのしきい値")
    parser.add_argument("--yes", "-y", action="store_true", help="すべての確認に自動的に「はい」と回答（確認をスキップ）")
    
    args = parser.parse_args()
    
    # グローバル変数の更新
    global DEBUG_MODE
    DEBUG_MODE = args.debug
    
    global batch_size
    batch_size = args.batch_size
    
    global max_workers
    max_workers = args.max_workers
    
    global pre_filter_threshold
    pre_filter_threshold = args.pre_filter
    
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
    
    # レコード間の類似度を計算（最適化版）
    matches = calculate_record_similarities(
        records, 
        field_types, 
        api_key, 
        args.cache_dir,
        args.sample
    )
    
    # ユーザーに確認（--yesフラグが指定されていない場合のみ）
    if args.yes:
        print("\n自動確認モード: APIリクエストを実行します")
    else:
        confirmation = input("\nAPIリクエストを実行しますか？ (y/n): ")
        if confirmation.lower() != 'y':
            print("処理を中止します。")
            sys.exit(0)
    
    # 類似度マッチからクラスターを作成
    clusters = create_clusters_from_matches(records, matches)
    
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
    
    # APIコスト削減の統計
    total_possible_comparisons = len(records) * (len(records) - 1) // 2
    filtered_by_sampling = total_possible_comparisons - (args.sample * (args.sample - 1) // 2) if len(records) > args.sample else 0
    filtered_by_blocking = int(args.sample * (args.sample - 1) / 2) - len(matches) if len(records) > args.sample else int(len(records) * (len(records) - 1) / 2) - len(matches)
    batching_reduction = len(matches) - (len(matches) // batch_size + (1 if len(matches) % batch_size > 0 else 0)) if batch_size > 1 else 0
    
    print("\nAPI使用量削減の効果:")
    print(f"  総比較可能ペア数: {total_possible_comparisons}回")
    if filtered_by_sampling > 0:
        print(f"  サンプリングによる削減: {filtered_by_sampling}回")
    print(f"  事前フィルタリングによる削減: {filtered_by_blocking}回")
    print(f"  バッチ処理による削減: {batching_reduction}回")
    print(f"  合計削減率: {((filtered_by_sampling + filtered_by_blocking + batching_reduction) / total_possible_comparisons * 100):.1f}%")
    
    # メトリクスの表示
    print("\n評価メトリクス:")
    print(f"  F1スコア (ペア): {metrics['f1(pair)']}")
    print(f"  精度 (ペア): {metrics['precision(pair)']}")
    print(f"  再現率 (ペア): {metrics['recall(pair)']}")
    print(f"  完全一致グループ率: {metrics['complete(group)']}")
    print(f"  精度 (グループ): {metrics['precision(group)']}")
    print(f"  再現率 (グループ): {metrics['recall(group)']}")

if __name__ == "__main__":
    main()#!/usr/bin/env python3