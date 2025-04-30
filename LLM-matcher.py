import yaml
import re
import json
import unicodedata
import asyncio
import aiohttp
import os
import uuid
import time
import copy
import argparse
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
from jellyfish import jaro_winkler_similarity as jw


# APIキーの設定（環境変数から取得）
API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not API_KEY:
    print("警告: OpenAI APIキーが環境変数に設定されていません。コマンドラインで指定してください。")

# 設定クラス
class Config:
    def __init__(self):
        self.similarity_threshold = 0.7
        self.jw_upper_threshold = 0.9
        self.jw_lower_threshold = 0.3
        self.cache_dir = ".cache"
        self.debug = False
        self.api_key = API_KEY
    
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

def create_clusters_from_matches(similarities: List[Dict[str, Any]], representatives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """類似度結果からクラスターを作成し、代表レコードが類似の場合はクラスター全体を統合"""
    
    # Union-Findデータ構造を初期化
    uf = UnionFind(len(representatives))
    
    # クラスターIDからインデックスへのマッピング
    cluster_id_to_index = {rep["cluster_id"]: i for i, rep in enumerate(representatives)}
    
    # 類似度に基づいて結合
    for similarity in similarities:
        if similarity["similarity_score"] >= 0.7:  # 類似度閾値
            pair = similarity["cluster_pair"]
            if pair[0] in cluster_id_to_index and pair[1] in cluster_id_to_index:
                idx1 = cluster_id_to_index[pair[0]]
                idx2 = cluster_id_to_index[pair[1]]
                uf.union(idx1, idx2)
    
    # クラスターの構築（統合されたクラスターの全レコードを含む）
    merged_clusters = defaultdict(list)
    for i, rep in enumerate(representatives):
        root = uf.find(i)
        # クラスターの代表レコードを追加
        merged_clusters[root].append(rep)
    
    # 結果のフォーマット
    result_clusters = []
    for cluster_idx, members in merged_clusters.items():
        # 統合されたクラスター内の全レコードを収集
        all_merged_records = []
        for member in members:
            if "all_records" in member:
                all_merged_records.extend(member["all_records"])
        
        result_clusters.append({
            "cluster_id": str(uuid.uuid4()),
            "members": members,
            "representative": members[0] if members else None,
            "all_records": all_merged_records  # 統合された全レコード
        })
    
    return result_clusters

# レコードフィールドの分析
def analyze_record_fields(records: List[Dict[str, Any]]) -> Dict[str, str]:
    """レコードのフィールドタイプを分析"""
    field_types = {}
    
    for record in records:
        data = record.get('data', {})
        for field, value in data.items():
            if field not in field_types:
                if isinstance(value, str):
                    if re.search(r'\d{4}', value) and len(value) <= 10:
                        field_types[field] = "date"
                    else:
                        field_types[field] = "text"
                elif isinstance(value, (int, float)):
                    field_types[field] = "number"
                else:
                    field_types[field] = "unknown"
    
    return field_types

# Jaro-Winklerの類似度計算
def jaro_winkler_similarity(s1: str, s2: str) -> float:
    """Jaro-Winkler類似度を計算"""
    from jellyfish import jaro_winkler_similarity as jw
    
    if not s1 or not s2:
        return 0.0
    
    # 文字列の正規化
    s1 = normalize_text(s1)
    s2 = normalize_text(s2)
    
    if not s1 or not s2:
        return 0.0
    
    return jw(s1, s2)

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
2. 類似度が0.6以上のペアのみを返してください
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
            response_data = json.loads(content)
            
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

# 類似度に基づいてクラスターを作成
def create_clusters_from_matches(similarities: List[Dict[str, Any]], representatives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """類似度結果からクラスターを作成し、代表レコードが類似の場合はクラスター全体を統合"""
    
    # Union-Findデータ構造を初期化
    uf = UnionFind(len(representatives))
    
    # クラスターIDからインデックスへのマッピング
    cluster_id_to_index = {rep["cluster_id"]: i for i, rep in enumerate(representatives)}
    
    # 類似度に基づいて結合
    for similarity in similarities:
        if similarity["similarity_score"] >= 0.7:  # 類似度閾値
            pair = similarity["cluster_pair"]
            if pair[0] in cluster_id_to_index and pair[1] in cluster_id_to_index:
                idx1 = cluster_id_to_index[pair[0]]
                idx2 = cluster_id_to_index[pair[1]]
                print(f"クラスター統合: {representatives[idx1]['title']} と {representatives[idx2]['title']} (類似度: {similarity['similarity_score']})")
                uf.union(idx1, idx2)
    
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
            "all_records": all_merged_records  # 統合された全レコード
        })
        
        print(f"統合クラスター作成: {len(members)}個のクラスターが統合され、{len(all_merged_records)}個のレコードを含む")
    
    return result_clusters

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
        # membersフィールドから直接レコードを取得
        elif 'members' in cluster and cluster['members']:
            members = cluster['members']
            for member in members:
                if 'all_records' in member and member['all_records']:
                    for record in member['all_records']:
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

# 結果をフォーマット
def format_results(similarities: List[Dict[str, Any]]) -> Dict[str, Any]:
    """最終結果を構造化されたデータとしてフォーマット"""
    return {
        "similarity_count": len(similarities),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "similarity_results": similarities
    }

# YAML読み込み
def load_yaml(file_path: str) -> Any:
    """YAMLファイルを読み込む"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# YAML保存
def save_yaml(data: Any, file_path: str) -> None:
    """データをYAMLファイルに保存"""
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)

# メイン処理関数
async def process_yaml_efficiently(yaml_data: str, api_key: str, batch_size: int = 8) -> Dict[str, Any]:
    """YAMLデータを効率的に処理するメイン関数"""
    try:
        # 処理開始時間
        start_time = time.time()
        
        # 1. 代表レコードの抽出
        print("代表レコードを抽出中...")
        representatives = extract_representatives(yaml_data)
        print(f"{len(representatives)}個のクラスター代表を抽出しました")
        
        # 2. 全レコードの抽出
        data = yaml.safe_load(yaml_data)
        all_records = extract_records(data)
        print(f"合計{len(all_records)}個のレコードを処理します")
        
        # 3. 特徴量の事前計算
        print("特徴量を計算中...")
        features = precompute_features(representatives)
        
        # 4. バッチに分割（GPTのコンテキスト制限に基づいて設定）
        batches = [features[i:i+batch_size] for i in range(0, len(features), batch_size)]
        print(f"{len(batches)}個のバッチに分割しました（1バッチあたり最大{batch_size}レコード）")
        
        # 5. 各バッチについて効率的なAPIリクエストを作成
        print("APIリクエストを準備中...")
        api_requests = []
        for batch in batches:
            request_data = create_efficient_api_request(batch)
            api_requests.append(request_data)
        
        # 6. 並列にAPIリクエストを実行
        print("GPT APIリクエストを実行中...")
        api_results = await process_requests_parallel(api_requests, api_key)
        
        # 7. 結果を抽出して統合
        print("結果を処理中...")
        similarities = extract_similarity_results(api_results, representatives)
        
        # 8. 類似度に基づいてクラスターを作成
        print("クラスターを作成中...")
        clusters = create_clusters_from_matches(similarities, representatives)
        
        # 9. 出力グループのフォーマット
        print("出力フォーマットを作成中...")
        output_groups = format_output_groups(clusters)
        
        # 10. 出力メトリクスの計算
        metrics = calculate_output_metrics(output_groups, all_records)
        
        # 11. 最終出力のフォーマット
        final_results = format_final_output(output_groups, metrics)
        
        # 処理時間の計算
        end_time = time.time()
        processing_time = end_time - start_time
        final_results["processing_time_seconds"] = processing_time
        
        print(f"処理完了：{len(clusters)}個のクラスターを作成しました（処理時間: {processing_time:.2f}秒）")
        
        return final_results
    
    except Exception as e:
        return {
            "error": f"処理中にエラーが発生しました: {str(e)}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

# 結果をJSONファイルに保存
def save_result_files(results: Dict[str, Any], output_yaml: str, output_json: str = None) -> None:
    """結果をYAMLとJSONファイルに保存"""
    # YAMLファイルに保存
    save_yaml(results, output_yaml)
    print(f"結果を {output_yaml} に保存しました")
    
    # JSONファイルにも保存（指定があれば）
    if output_json:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"結果を {output_json} にも保存しました")

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

# コマンドライン実行用のメイン関数
async def main(yaml_file_path: str, output_file: str = None, batch_size: int = 8) -> None:
    """YAMLファイルを読み込み、処理して結果を保存"""
    try:
        # コマンドライン引数の処理
        if output_file is None:
            # 入力ファイル名から出力ファイル名を自動生成
            input_basename = os.path.basename(yaml_file_path)
            input_name = os.path.splitext(input_basename)[0]
            output_file = f"{input_name}_result.yaml"
        
        # JSON出力ファイル名も生成
        output_json = f"{os.path.splitext(output_file)[0]}.json"
        
        # YAMLファイルを読み込み
        with open(yaml_file_path, 'r', encoding='utf-8') as f:
            yaml_data = f.read()
        
        print(f"YAMLファイル '{yaml_file_path}' を読み込みました")
        
        # 処理開始時刻
        start_time = datetime.now()
        print(f"処理開始時刻: {start_time}")
        
        # 処理実行
        results = await process_yaml_efficiently(yaml_data, API_KEY, batch_size)
        
        # 結果を保存
        save_result_files(results, output_file, output_json)
        
        # 終了時刻とサマリー
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        print(f"処理完了時刻: {end_time}")
        print(f"合計処理時間: {elapsed:.2f}秒")
        
        # サマリー情報の表示
        if "summary" in results:
            metrics = results["summary"]
            print("\n評価メトリクス:")
            print(f"  F1スコア (ペア): {metrics.get('f1(pair)', 'N/A')}")
            print(f"  精度 (ペア): {metrics.get('precision(pair)', 'N/A')}")
            print(f"  再現率 (ペア): {metrics.get('recall(pair)', 'N/A')}")
            print(f"  完全一致グループ率: {metrics.get('complete(group)', 'N/A')}")
            print(f"  精度 (グループ): {metrics.get('precision(group)', 'N/A')}")
            print(f"  再現率 (グループ): {metrics.get('recall(group)', 'N/A')}")
            print(f"  入力レコード数: {metrics.get('num_of_record', 'N/A')}")
            print(f"  元のグループ数: {metrics.get('num_of_groups(correct)', 'N/A')}")
            print(f"  推論されたグループ数: {metrics.get('num_of_groups(inference)', 'N/A')}")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

# スクリプト実行
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='書誌レコードクラスターの類似度を計算')
    parser.add_argument('--input', '-i', type=str, required=True, help='処理するYAMLファイルのパス')
    parser.add_argument('--output', '-o', type=str, help='結果を保存するYAMLファイルのパス')
    parser.add_argument('--api-key', '-k', type=str, help='OpenAI APIキー（環境変数が設定されていない場合）')
    parser.add_argument('--threshold', '-t', type=float, default=0.7, help='類似度しきい値')
    parser.add_argument('--jw-upper', '-ju', type=float, default=0.9, 
                    help='Jaro-Winkler上限しきい値 (この値以上は直接マッチ)')
    parser.add_argument('--jw-lower', '-jl', type=float, default=0.3, 
                    help='Jaro-Winkler下限しきい値 (この値以下は直接不一致)')
    parser.add_argument('--batch-size', '-b', type=int, default=8, help='APIリクエストのバッチサイズ')
    parser.add_argument('--cache-dir', '-c', type=str, default=".cache", help='キャッシュディレクトリ')
    parser.add_argument('--debug', '-d', action='store_true', help='デバッグモードを有効化')
    
    args = parser.parse_args()
    
    # APIキーの更新（コマンドラインから指定された場合）
    if args.api_key:
        API_KEY = args.api_key
    
    # 設定の初期化
    config = Config()
    config.load_from_args(args)
    
    # 非同期処理を実行
    asyncio.run(main(args.input, args.output, args.batch_size))