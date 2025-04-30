def main():
    parser = argparse.ArgumentParser(description="LLMベースの書誌レコードマッチング")
    parser.add_argument("--input", "-i", type=str, required=True, help="入力YAMLファイルパス")
    parser.add_argument("--output", "-o", type=str, default="result.yaml", help="出力YAMLファイルパス")
    parser.add_argument("--api-key", "-k", type=str, help="OpenAI APIキー")
    parser.add_argument("--threshold", "-t", type=float, default=0.7, help="類似度しきい値")
    parser.add_argument("--cache-dir", "-c", type=str, default=".cache", help="キャッシュディレクトリ")
    parser.add_argument("--sample", "-s", type=int, default=None, help="サンプルサイズ（指定しない場合は全レコードを使用）")
    parser.add_argument("--debug", "-d", action="store_true", help="デバッグモードを有効化")
    parser.add_argument("--similarities-output", "-so", type=str, help="すべての類似度情報を出力するCSVファイルパス")
    
    args = parser.parse_args()
    
    # デバッグモードの取得
    debug_mode = args.debug
    
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
    
    # レコード間の類似度を計算（LLMを使用）
    matches = calculate_record_similarities(
        records=records, 
        field_types=field_types, 
        api_key=api_key, 
        cache_dir=args.cache_dir,
        sample_size=args.sample,
        similarity_threshold=args.threshold,
        similarity_csv_path=args.similarities_output
    )
    
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
"""
LLMベースの書誌レコードマッチングシステム

このスクリプトはLLM（大規模言語モデル）を使用して書誌レコードの類似性を判定し、
クラスタリングを行います。OpenAI ChatCompletionを使用してレコード間の類似度を判断し、
結果をYAMLファイルで出力します。

特徴:
- LLMの高度な文脈理解を活用した類似性判定
- レコードペアごとの詳細な類似度分析
- キャッシュ機能による処理の高速化と費用の節約
- 元のフォーマットに一致する出力生成
- すべてのペアの類似度情報をCSVファイルに出力
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
import csv
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Set, Union

# 依存ライブラリのインポート試行
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("警告: 'requests'ライブラリがインストールされていません。以下のコマンドでインストールしてください:")
    print("pip install requests")
    sys.exit(1)

# 設定
SIMILARITY_THRESHOLD = 0.7  # LLMの類似度判定のしきい値（0-1）
MAX_RETRIES = 3  # APIコールの最大リトライ回数
RETRY_DELAY = 2  # リトライ間の待機時間（秒）
CACHE_DIR = ".cache"  # キャッシュディレクトリ
DEBUG_MODE = False  # デバッグモード（詳細な出力）

def debug_log(message, debug_mode=False):
    """デバッグモードが有効の場合、デバッグログメッセージを出力"""
    if debug_mode:
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

def get_similarity_with_llm(text1: str, text2: str, api_key: str, max_retries: int = 3, retry_delay: int = 2) -> float:
    """
    LLMを使用して2つのテキスト間の類似度を計算する。
    返り値は0〜1の数値で、1が完全一致、0が完全に異なることを表す。
    """
    if not api_key:
        print("エラー: APIキーが提供されていません。--api-key オプションまたは環境変数 OPENAI_API_KEY を設定してください。")
        sys.exit(1)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # LLMへのプロンプト
    system_prompt = """
    あなたは書誌レコードの類似性を判断する専門家です。2つの書誌レコードの情報を比較し、それらが同じ作品を参照しているかどうかを判断してください。
    タイトル、著者、出版社、出版日などの情報を考慮して、類似度を0〜1の数値で評価してください。
    1は完全に同じ作品、0は完全に異なる作品を表します。
    
    以下の基準で評価してください：
    - タイトルが非常に似ている場合は高い類似度
    - 著者が一致する場合も高い類似度
    - 出版社と出版日も考慮するが、タイトルと著者ほど重要ではない
    - 表記の違い（全角/半角、かな/カナ、句読点の有無など）は同じものとみなす
    
    回答は必ず0〜1の数値のみを返してください。例：0.95
    """
    
    user_prompt = f"""
    以下の2つの書誌レコードを比較し、類似度を0〜1の数値で回答してください。

    書誌レコード1:
    {text1}

    書誌レコード2:
    {text2}
    
    類似度（0〜1の数値のみ）:
    """
    
    # APIリクエストデータ
    data = {
        "model": "gpt-4",  # または "gpt-3.5-turbo" などの適切なモデル
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0  # 決定論的な回答を得るために低い温度を設定
    }
    
    if DEBUG_MODE:
        print(f"LLMに類似度判定をリクエスト")
    
    for attempt in range(max_retries):
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
                similarity_text = result["choices"][0]["message"]["content"].strip()
                
                try:
                    # 数値を抽出（LLMが説明を追加した場合に対応）
                    import re
                    similarity_matches = re.findall(r'0\.\d+|\d+\.0|[01]', similarity_text)
                    if similarity_matches:
                        similarity = float(similarity_matches[0])
                        # 範囲を0〜1に制限
                        similarity = max(0.0, min(1.0, similarity))
                        return similarity
                    else:
                        print(f"警告: LLMからの応答から類似度を抽出できませんでした: {similarity_text}")
                        return 0.5  # デフォルト値
                except Exception as e:
                    print(f"警告: 類似度の解析エラー: {e}")
                    return 0.5  # デフォルト値
            
            elif response.status_code == 429:  # レート制限
                wait_time = retry_delay * (attempt + 1)
                print(f"レート制限に達しました。{wait_time}秒待機します...")
                time.sleep(wait_time)
            else:
                print(f"APIエラー ({attempt+1}/{max_retries}): {response.status_code} - {response.text}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        except Exception as e:
            print(f"リクエストエラー ({attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    
    print("すべてのAPIリクエスト試行が失敗しました。デフォルト値を使用します。")
    return 0.5  # デフォルト値

def calculate_record_similarities(records: List[Dict], field_types: Dict[str, str],
                                api_key: str, cache_dir: str = ".cache",
                                sample_size: int = None, similarity_threshold: float = 0.7,
                                similarity_csv_path: str = None) -> List[Tuple]:
    """
    レコード間の類似度を計算する
    
    引数:
        records: レコードのリスト
        field_types: フィールドタイプの辞書
        api_key: OpenAI APIキー
        cache_dir: キャッシュディレクトリ
        sample_size: サンプルサイズ（Noneの場合は全レコードを使用）
        similarity_threshold: 類似度のしきい値
        similarity_csv_path: 類似度結果を出力するCSVファイルのパス
    
    戻り値:
        類似度がしきい値以上のレコードペア情報のリスト
    """
    # キャッシュディレクトリの作成
    os.makedirs(cache_dir, exist_ok=True)
    
    # キャッシュファイル名
    cache_file = os.path.join(cache_dir, f"similarity_cache.pkl")
    
    # キャッシュの読み込み
    similarity_cache = load_similarity_cache(cache_file)
    
    # 結果を格納するリスト
    matches = []
    all_similarities = []  # すべての類似度情報を格納
    
    # サンプリング（サンプルサイズが指定されている場合）
    if sample_size is not None and len(records) > sample_size:
        import random
        sampled_records = random.sample(records, min(sample_size, len(records)))
        print(f"{len(records)}件のレコードから{len(sampled_records)}件をサンプリングしました")
    else:
        sampled_records = records
        print(f"全{len(records)}件のレコードを使用します")
    
    total_comparisons = len(sampled_records) * (len(sampled_records) - 1) // 2
    processed = 0
    start_time = datetime.now()
    cache_hits = 0
    cache_misses = 0
    
    print(f"レコード間の類似度を計算中...")
    
    # CSV出力ファイルの準備（指定されている場合）
    csv_file = None
    csv_writer = None
    
    if similarity_csv_path:
        try:
            csv_file = open(similarity_csv_path, 'w', newline='', encoding='utf-8')
            csv_writer = csv.writer(csv_file)
            # CSVヘッダーの書き込み
            csv_writer.writerow([
                'record_id1', 'record_id2', 'similarity',
                'title1', 'title2', 'author1', 'author2',
                'original_cluster_id1', 'original_cluster_id2'
            ])
            print(f"類似度情報をCSVファイルに出力します: {similarity_csv_path}")
        except Exception as e:
            print(f"CSVファイルの作成に失敗しました: {e}")
            csv_file = None
            csv_writer = None
    
    # レコードペアごとに類似度を計算
    for i in range(len(sampled_records)):
        record1 = sampled_records[i]
        
        for j in range(i + 1, len(sampled_records)):
            record2 = sampled_records[j]
            
            processed += 1
            if processed % 5 == 0 or processed == total_comparisons:
                progress = processed / total_comparisons * 100
                elapsed = (datetime.now() - start_time).total_seconds()
                remaining = elapsed / processed * (total_comparisons - processed)
                print(f"  進捗: {processed}/{total_comparisons}回 ({progress:.1f}%) - "
                      f"経過: {elapsed:.1f}秒, 残り: {remaining:.1f}秒 - "
                      f"キャッシュヒット: {cache_hits}, ミス: {cache_misses}")
            
            # レコードテキストの作成
            text1 = format_record_for_llm(record1, field_types)
            text2 = format_record_for_llm(record2, field_types)
            
            # キャッシュキーの生成
            cache_key = get_cache_key(text1, text2)
            
            # キャッシュを確認
            if cache_key in similarity_cache:
                similarity = similarity_cache[cache_key]
                cache_hits += 1
            else:
                # LLMで類似度を計算
                similarity = get_similarity_with_llm(text1, text2, api_key)
                
                # キャッシュに保存
                similarity_cache[cache_key] = similarity
                cache_misses += 1
                
                # キャッシュを定期的に保存
                if cache_misses % 10 == 0:
                    save_similarity_cache(similarity_cache, cache_file)
            
            # 詳細情報を抽出
            title_field = next((field for field, ftype in field_types.items() 
                              if ftype == "TEXT" and "title" in field.lower()), None)
            author_field = next((field for field, ftype in field_types.items() 
                               if ftype == "COMPLEMENT_JA" and "author" in field.lower()), None)
            
            title1 = record1["data"].get(title_field, "") if title_field and "data" in record1 else ""
            title2 = record2["data"].get(title_field, "") if title_field and "data" in record2 else ""
            author1 = record1["data"].get(author_field, "") if author_field and "data" in record1 else ""
            author2 = record2["data"].get(author_field, "") if author_field and "data" in record2 else ""
            
            # ペア情報を作成
            pair_info = {
                "record_id1": record1.get("id", ""),
                "record_id2": record2.get("id", ""),
                "similarity": similarity,
                "title1": title1,
                "title2": title2,
                "author1": author1,
                "author2": author2,
                "original_cluster_id1": record1.get("original_cluster_id", ""),
                "original_cluster_id2": record2.get("original_cluster_id", "")
            }
            
            # すべての類似度情報を保存
            all_similarities.append(pair_info)
            
            # CSVファイルに書き込み
            if csv_writer:
                csv_writer.writerow([
                    pair_info["record_id1"],
                    pair_info["record_id2"],
                    pair_info["similarity"],
                    pair_info["title1"],
                    pair_info["title2"],
                    pair_info["author1"],
                    pair_info["author2"],
                    pair_info["original_cluster_id1"],
                    pair_info["original_cluster_id2"]
                ])
                
                # CSVファイルを定期的にフラッシュ
                if processed % 20 == 0:
                    csv_file.flush()
            
            # しきい値を超える場合、マッチとして記録
            if similarity >= similarity_threshold:
                matches.append(pair_info)
    
    # 最終的なキャッシュを保存
    save_similarity_cache(similarity_cache, cache_file)
    
    # CSVファイルを閉じる
    if csv_file:
        csv_file.close()
        print(f"類似度情報をCSVファイルに保存しました: {similarity_csv_path}")
    
    # 類似度でソート
    matches.sort(key=lambda x: x["similarity"], reverse=True)
    all_similarities.sort(key=lambda x: x["similarity"], reverse=True)
    
    print(f"{len(matches)}件の類似レコードペアを見つけました（しきい値{similarity_threshold}以上）")
    print(f"合計{len(all_similarities)}件のペア比較を実行しました")
    
    return matches

def save_all_similarities_to_csv(similarities: List[Dict], output_path: str):
    """すべての類似度情報をCSVファイルに保存する"""
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # ヘッダー行
            writer.writerow([
                'record_id1', 'record_id2', 'similarity',
                'title1', 'title2', 'author1', 'author2',
                'original_cluster_id1', 'original_cluster_id2'
            ])
            
            # データ行
            for pair in similarities:
                writer.writerow([
                    pair["record_id1"],
                    pair["record_id2"],
                    pair["similarity"],
                    pair["title1"],
                    pair["title2"],
                    pair["author1"],
                    pair["author2"],
                    pair["original_cluster_id1"],
                    pair["original_cluster_id2"]
                ])
        
        print(f"{len(similarities)}件の類似度情報をCSVファイルに保存しました: {output_path}")
    except Exception as e:
        print(f"CSVファイルの保存に失敗しました: {e}")

def create_clusters_from_matches(records: List[Dict], matches: List[Dict]) -> Dict[str, List[Dict]]:
    """マッチ情報からクラスターを作成"""
    # レコードIDからレコードへのマッピング
    id_to_record = {record.get("id", f"record_{i}"): record for i, record in enumerate(records)}
    
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
    
    # マッチ情報を使って集合を結合
    for match in matches:
        record_id1 = match["record_id1"]
        record_id2 = match["record_id2"]
        union(record_id1, record_id2)
    
    # クラスターを作成
    clusters = defaultdict(list)
    
    for record in records:
        record_id = record.get("id", "")
        
        if record_id in parent:
            root = find(record_id)
            new_cluster_id = f"cluster_{uuid.uuid4().hex[:8]}"
            
            # 既存のルートに対するクラスターIDをマッピング
            if root not in clusters:
                clusters[root] = []
            
            # レコードにクラスターIDを設定
            record["cluster_id"] = root
            
            # クラスターにレコードを追加
            clusters[root].append(record)
    
    print(f"{len(clusters)}個のクラスターを作成しました")
    
    return clusters

def format_output_groups(clusters: Dict[str, List[Dict]]) -> List[Dict]:
    """クラスターを出力グループ形式にフォーマット"""
    output_groups = []
    
    for cluster_id, records in clusters.items():
        # インデックスでソート
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
    parser = argparse.ArgumentParser(description="LLMベースの書誌レコードマッチング")
    parser.add_argument("--input", "-i", type=str, required=True, help="入力YAMLファイルパス")
    parser.add_argument("--output", "-o", type=str, default="result.yaml", help="出力YAMLファイルパス")
    parser.add_argument("--api-key", "-k", type=str, help="OpenAI APIキー")
    parser.add_argument("--threshold", "-t", type=float, default=SIMILARITY_THRESHOLD, help="類似度しきい値")
    parser.add_argument("--cache-dir", "-c", type=str, default=CACHE_DIR, help="キャッシュディレクトリ")
    parser.add_argument("--sample", "-s", type=int, default=None, help="サンプルサイズ（指定しない場合は全レコードを使用）")
    parser.add_argument("--debug", "-d", action="store_true", help="デバッグモードを有効化")
    parser.add_argument("--similarities-output", "-so", type=str, help="すべての類似度情報を出力するCSVファイルパス")
    
    args = parser.parse_args()
    
    # デバッグモードの設定
    global DEBUG_MODE
    DEBUG_MODE = args.debug
    
    # しきい値の設定
    global SIMILARITY_THRESHOLD
    if args.threshold is not None:
        SIMILARITY_THRESHOLD = args.threshold
    
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
    
    # レコード間の類似度を計算（LLMを使用）
    all_similarities = []  # すべての類似度情報を格納
    
    matches = calculate_record_similarities(
        records, 
        field_types, 
        api_key, 
        args.cache_dir,
        args.sample,
        args.similarities_output
    )
    
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