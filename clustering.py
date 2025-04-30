#!/usr/bin/env python3
"""
適応型しきい値調整クラスタリングツール

このスクリプトは書誌レコードクラスタリングのしきい値を自動的に調整し、
精度と再現率のバランスを最適化します。精度が1.0未満になるまでしきい値を
段階的に下げていくことで、より高い再現率を実現します。
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
import re
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Set, Union

try:
    import jellyfish
except ImportError:
    print("警告: 'jellyfish'ライブラリがインストールされていません。以下のコマンドでインストールしてください:")
    print("pip install jellyfish")
    sys.exit(1)

class Config:
    """設定値を管理するクラス"""
    def __init__(self):
        # デフォルト設定
        self.initial_threshold = 0.8   # 初期類似度しきい値
        self.min_threshold = 0.3       # 最小類似度しきい値
        self.threshold_step = 0.05     # しきい値の減少ステップ
        self.target_precision = 0.98   # 目標精度（この値以下になったら停止）
        self.cache_dir = ".cache"      # キャッシュディレクトリ
        self.debug = False             # デバッグモード（詳細な出力）
        self.normalize_fields = True   # フィールド正規化の有効化
        self.max_iterations = 10       # 最大反復回数
    
    def load_from_args(self, args):
        """コマンドライン引数から設定を読み込む"""
        if hasattr(args, 'initial_threshold') and args.initial_threshold is not None:
            self.initial_threshold = args.initial_threshold
        if hasattr(args, 'min_threshold') and args.min_threshold is not None:
            self.min_threshold = args.min_threshold
        if hasattr(args, 'threshold_step') and args.threshold_step is not None:
            self.threshold_step = args.threshold_step
        if hasattr(args, 'target_precision') and args.target_precision is not None:
            self.target_precision = args.target_precision
        if hasattr(args, 'cache_dir') and args.cache_dir is not None:
            self.cache_dir = args.cache_dir
        if hasattr(args, 'debug'):
            self.debug = args.debug
        if hasattr(args, 'normalize'):
            self.normalize_fields = args.normalize
        if hasattr(args, 'max_iterations') and args.max_iterations is not None:
            self.max_iterations = args.max_iterations
    
    def debug_log(self, message):
        """デバッグモードが有効の場合、デバッグログメッセージを出力"""
        if self.debug:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            print(f"[DEBUG] [{timestamp}] {message}")

class BibRecordNormalizer:
    """書誌レコードの正規化を行うクラス"""
    
    @staticmethod
    def normalize_title(title: str) -> str:
        """タイトルを正規化する"""
        if not title:
            return ""
        
        # シリーズ情報を一時的に分離
        series_info = ""
        main_title = title
        
        # 括弧内のシリーズ情報を抽出
        bracket_match = re.search(r'(\s*[\[(（].*?[\])）])\s*$', title)
        if bracket_match:
            series_info = bracket_match.group(1)
            main_title = title[:bracket_match.start()].strip()
        
        # スペースの正規化
        main_title = re.sub(r'\s+', ' ', main_title).strip()
        
        # 全角/半角の統一（半角に）
        main_title = main_title.replace('　', ' ')
        
        # 記号の除去（「」や『』などの除去）
        main_title = re.sub(r'[「」『』]', '', main_title)
        
        return main_title
    
    @staticmethod
    def normalize_author(author: str) -> str:
        """著者名を正規化する"""
        if not author:
            return ""
        
        # 著者役割表示の除去
        author = re.sub(r'[\\/∥‖][著作編訳監].*?$', '', author)
        
        # カッコ内の情報を除去
        author = re.sub(r'[\[(（].*?[\])）]', '', author)
        
        # 字体の統一
        author = author.replace('龍', '竜').replace('彦', '彦')
        
        # 区切り文字の除去
        author = re.sub(r'[\s/∥‖,，、]', '', author)
        
        return author.strip()
    
    @staticmethod
    def normalize_date(date: str) -> str:
        """出版日を正規化する"""
        if not date:
            return ""
        
        # 年のみを抽出
        year_match = re.search(r'(\d{4})', date)
        if year_match:
            return year_match.group(1)
        
        return date.strip()
    
    @staticmethod
    def normalize_publisher(publisher: str) -> str:
        """出版社を正規化する"""
        if not publisher:
            return ""
        
        # 出版地の除去
        publisher = re.sub(r'^.*?:', '', publisher)
        
        # カッコ内の情報を除去
        publisher = re.sub(r'[\[(（].*?[\])）]', '', publisher)
        
        # スペースと区切り文字の除去
        publisher = re.sub(r'[\s/∥‖,，、]', '', publisher)
        
        return publisher.strip()
    
    @staticmethod
    def normalize_record_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """レコードデータを正規化する"""
        normalized_data = {}
        
        for field, value in data.items():
            if not value:
                normalized_data[field] = value
                continue
                
            if "title" in field.lower():
                normalized_data[field] = BibRecordNormalizer.normalize_title(str(value))
            elif "author" in field.lower():
                normalized_data[field] = BibRecordNormalizer.normalize_author(str(value))
            elif "publisher" in field.lower():
                normalized_data[field] = BibRecordNormalizer.normalize_publisher(str(value))
            elif "date" in field.lower() or "pubdate" in field.lower():
                normalized_data[field] = BibRecordNormalizer.normalize_date(str(value))
            else:
                normalized_data[field] = value
        
        return normalized_data

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

def extract_clusters(data: Dict) -> List[Dict]:
    """YAMLから既存のクラスター情報を抽出する"""
    clusters = []
    
    for group_idx, group in enumerate(data.get("group", [])):
        cluster = {
            "index": group_idx,
            "cluster_id": f"cluster_{group_idx}",
            "records": group.get("records", []),
            "size": len(group.get("records", [])),
            "original_correct": group.get("correct", [])
        }
        clusters.append(cluster)
    
    print(f"{len(clusters)}個のクラスターを抽出しました")
    return clusters

def get_representative_record(cluster: Dict) -> Dict:
    """クラスターから代表的なレコードを選択する（最も情報量の多いレコード）"""
    if not cluster.get("records", []):
        return {}
    
    # レコードごとの情報量（フィールド値の長さの合計）を計算
    records_with_info = []
    for record in cluster["records"]:
        info_amount = 0
        if "data" in record and isinstance(record["data"], dict):
            for field, value in record["data"].items():
                if value:
                    info_amount += len(str(value))
        
        records_with_info.append((record, info_amount))
    
    # 情報量でソートし、最も情報量の多いレコードを返す
    records_with_info.sort(key=lambda x: x[1], reverse=True)
    return records_with_info[0][0]

def calculate_field_similarity(field1: str, field2: str) -> float:
    """2つのフィールド値の類似度を計算する"""
    if not field1 or not field2:
        return 0.0
    
    # Jaro-Winkler類似度を計算
    return jellyfish.jaro_winkler_similarity(str(field1), str(field2))

def extract_key_fields(record: Dict) -> Dict[str, str]:
    """レコードから重要なフィールドを抽出する"""
    result = {}
    
    if "data" in record and isinstance(record["data"], dict):
        data = record["data"]
        
        # フィールド名を確認して抽出
        for field, value in data.items():
            if "title" in field.lower():
                result["title"] = value
            elif "author" in field.lower():
                result["author"] = value
            elif "publisher" in field.lower():
                result["publisher"] = value
            elif "date" in field.lower() or "pubdate" in field.lower():
                result["date"] = value
    
    return result

def calculate_cluster_similarity(cluster1: Dict, cluster2: Dict, config: Config) -> float:
    """2つのクラスター間の類似度を計算する"""
    # 代表レコードを取得
    rep_record1 = get_representative_record(cluster1)
    rep_record2 = get_representative_record(cluster2)
    
    # レコードがない場合は類似度を0とする
    if not rep_record1 or not rep_record2:
        return 0.0
    
    # 重要フィールドの抽出
    fields1 = extract_key_fields(rep_record1)
    fields2 = extract_key_fields(rep_record2)
    
    # フィールドの正規化（オプション）
    if config.normalize_fields:
        normalized_fields1 = {}
        normalized_fields2 = {}
        
        for field, value in fields1.items():
            if field == "title":
                normalized_fields1[field] = BibRecordNormalizer.normalize_title(value)
            elif field == "author":
                normalized_fields1[field] = BibRecordNormalizer.normalize_author(value)
            elif field == "publisher":
                normalized_fields1[field] = BibRecordNormalizer.normalize_publisher(value)
            elif field == "date":
                normalized_fields1[field] = BibRecordNormalizer.normalize_date(value)
            else:
                normalized_fields1[field] = value
        
        for field, value in fields2.items():
            if field == "title":
                normalized_fields2[field] = BibRecordNormalizer.normalize_title(value)
            elif field == "author":
                normalized_fields2[field] = BibRecordNormalizer.normalize_author(value)
            elif field == "publisher":
                normalized_fields2[field] = BibRecordNormalizer.normalize_publisher(value)
            elif field == "date":
                normalized_fields2[field] = BibRecordNormalizer.normalize_date(value)
            else:
                normalized_fields2[field] = value
        
        fields1 = normalized_fields1
        fields2 = normalized_fields2
    
    # フィールドの重み付け
    weights = {
        "title": 0.6,     # タイトルの重み
        "author": 0.3,    # 著者の重み
        "publisher": 0.05, # 出版社の重み
        "date": 0.05      # 出版日の重み
    }
    
    total_similarity = 0.0
    total_weight = 0.0
    
    # 各フィールドの類似度を計算
    for field in weights:
        if field in fields1 and field in fields2 and fields1[field] and fields2[field]:
            field_weight = weights[field]
            # フィールド類似度の計算
            similarity = calculate_field_similarity(fields1[field], fields2[field])
            
            # デバッグ出力
            config.debug_log(f"フィールド '{field}' の類似度: {similarity:.4f}")
            config.debug_log(f"  値1: {fields1[field]}")
            config.debug_log(f"  値2: {fields2[field]}")
            
            total_similarity += similarity * field_weight
            total_weight += field_weight
    
    # 有効な比較がない場合は低い類似度を返す
    if total_weight == 0:
        return 0.0
    
    # 正規化された類似度
    normalized_similarity = total_similarity / total_weight
    
    # クラスターサイズに基づく補正（オプション）
    # 小さいクラスター同士の統合を優先するための調整
    size_factor = 1.0
    max_size = max(cluster1["size"], cluster2["size"])
    if max_size > 5:  # 大きすぎるクラスターへのペナルティ
        size_factor = 0.95
    
    return normalized_similarity * size_factor

def find_mergeable_clusters(clusters: List[Dict], similarity_threshold: float, config: Config) -> List[Tuple[int, int, float]]:
    """マージ可能なクラスターのペアを見つける"""
    mergeable_pairs = []
    
    print(f"クラスター間の類似度を計算中 (しきい値: {similarity_threshold})...")
    
    total_comparisons = len(clusters) * (len(clusters) - 1) // 2
    processed = 0
    
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            processed += 1
            if processed % 10 == 0 or processed == total_comparisons:
                progress = processed / total_comparisons * 100
                print(f"  クラスター類似度計算の進捗: {processed}/{total_comparisons} ({progress:.1f}%)")
            
            # クラスター間の類似度を計算
            similarity = calculate_cluster_similarity(clusters[i], clusters[j], config)
            
            # しきい値を超える場合、マージ候補として記録
            if similarity >= similarity_threshold:
                mergeable_pairs.append((i, j, similarity))
                config.debug_log(f"マージ候補: クラスター {i} と {j} (類似度: {similarity:.4f})")
    
    # 類似度の高い順にソート
    mergeable_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print(f"{len(mergeable_pairs)}個のマージ可能なクラスターペアを見つけました")
    
    return mergeable_pairs

def merge_clusters(clusters: List[Dict], merge_pairs: List[Tuple[int, int, float]]) -> List[Dict]:
    """指定されたペアに基づいてクラスターをマージする"""
    # マージ済みクラスターを追跡するセット
    merged = set()
    
    # 新しいクラスターのリスト
    new_clusters = []
    
    # 各マージペアを処理
    for i, j, similarity in merge_pairs:
        # すでにマージされているクラスターはスキップ
        if i in merged or j in merged:
            continue
        
        # クラスターiとjをマージ
        merged_cluster = {
            "index": len(new_clusters),
            "cluster_id": f"merged_cluster_{uuid.uuid4().hex[:8]}",
            "records": clusters[i]["records"] + clusters[j]["records"],
            "size": clusters[i]["size"] + clusters[j]["size"],
            "original_correct": clusters[i].get("original_correct", []) + clusters[j].get("original_correct", []),
            "merged_from": [clusters[i]["cluster_id"], clusters[j]["cluster_id"]],
            "merge_similarity": similarity
        }
        
        new_clusters.append(merged_cluster)
        merged.add(i)
        merged.add(j)
    
    # マージされなかったクラスターを追加
    for i, cluster in enumerate(clusters):
        if i not in merged:
            new_cluster = cluster.copy()
            new_cluster["index"] = len(new_clusters)
            new_clusters.append(new_cluster)
    
    print(f"{len(merge_pairs)}個のマージを実行し、{len(new_clusters)}個のクラスターになりました")
    
    return new_clusters

def format_output_groups(clusters: List[Dict]) -> List[Dict]:
    """クラスターを出力グループ形式にフォーマット"""
    output_groups = []
    
    for cluster in clusters:
        records = cluster["records"]
        
        # correctフィールドはオリジナルを保持
        correct = cluster.get("original_correct", [])
        if not correct:
            correct = [[i for i in range(len(records))]]
        
        group = {
            "perfect_match": False,  # デフォルト値
            "records": records,
            "correct": correct
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
    for group_idx, group in enumerate(groups):
        for record in group.get("records", []):
            if "id" in record:
                new_cluster_to_records[f"group_{group_idx}"].append(record["id"])
    
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
        "complete_group": str(complete_matched_clusters),
        "threshold": None,  # しきい値情報を追加
    }
    
    return metrics

def format_final_output(groups: List[Dict], metrics: Dict, threshold: float) -> Dict:
    """出力を例と完全に一致する形式でフォーマット"""
    metrics_copy = metrics.copy()
    metrics_copy["threshold"] = threshold  # 使用したしきい値を記録
    
    output = {
        "version": "3.1",
        "type": "RESULT",
        "id": str(uuid.uuid4()),
        "summary": metrics_copy,
        "group": groups
    }
    
    return output

def adaptive_clustering(data: Dict, config: Config) -> Dict:
    """適応型しきい値調整クラスタリングを実行する"""
    # 元のクラスターを抽出
    clusters = extract_clusters(data)
    
    # すべてのレコードのリストを作成（メトリクス計算用）
    all_records = []
    for cluster in clusters:
        all_records.extend(cluster["records"])
    
    # 初期しきい値から開始
    current_threshold = config.initial_threshold
    best_result = None
    best_metrics = None
    best_threshold = None
    
    print(f"\n===== 適応型しきい値調整クラスタリングを開始 =====")
    print(f"初期しきい値: {current_threshold}")
    print(f"目標精度: < {config.target_precision}")
    
    iteration = 0
    previous_cluster_count = len(clusters)
    previous_precision = 1.0
    
    # しきい値を段階的に下げながらクラスタリングを実行
    while current_threshold >= config.min_threshold and iteration < config.max_iterations:
        iteration += 1
        print(f"\n----- 反復 {iteration}/{config.max_iterations}: しきい値 {current_threshold:.2f} -----")
        
        # 現在のしきい値でマージ可能なクラスターを見つける
        merge_pairs = find_mergeable_clusters(clusters, current_threshold, config)
        
        # マージペアがない場合はしきい値を下げて続行
        if not merge_pairs:
            print(f"マージ可能なクラスターがありません。しきい値を下げます: {current_threshold} -> {current_threshold - config.threshold_step}")
            current_threshold -= config.threshold_step
            continue
        
        # クラスターをマージ
        merged_clusters = merge_clusters(clusters, merge_pairs)
        
        # 出力グループのフォーマット
        output_groups = format_output_groups(merged_clusters)
        
        # 出力メトリクスの計算
        metrics = calculate_output_metrics(output_groups, all_records)
        
        # 精度の取得
        precision_pair = float(metrics.get("precision(pair)", "1.0"))
        recall_pair = float(metrics.get("recall(pair)", "0.0"))
        f1_pair = float(metrics.get("f1(pair)", "0.0"))
        
        print(f"精度(ペア): {precision_pair:.5f}")
        print(f"再現率(ペア): {recall_pair:.5f}")
        print(f"F1スコア(ペア): {f1_pair:.5f}")
        print(f"クラスター数: {len(merged_clusters)} (元: {previous_cluster_count})")
        
        # 最適な結果を記録
        if best_result is None or f1_pair > float(best_metrics.get("f1(pair)", "0.0")):
            best_result = {
                "summary": metrics,
                "group": output_groups
            }
            best_metrics = metrics
            best_threshold = current_threshold
            print(f"※ 新しい最適解を更新: しきい値={current_threshold}, 精度={precision_pair:.5f}, 再現率={recall_pair:.5f}, F1={f1_pair:.5f}")
        
        # 目標精度を達成したら終了
        if precision_pair <= config.target_precision:
            print(f"目標精度を達成しました: {precision_pair:.5f} <= {config.target_precision}")
            if iteration >= 2:  # 少なくとも2回は実行
                break
        
        # 精度が急激に低下した場合は前の結果を採用
        if previous_precision - precision_pair > 0.2 and previous_precision > 0.8:
            print(f"精度が急激に低下しました ({previous_precision:.5f} -> {precision_pair:.5f})。探索を終了します。")
            break
        
        # 次の反復に備えて変数を更新
        previous_precision = precision_pair
        previous_cluster_count = len(merged_clusters)
        clusters = merged_clusters
        
        # しきい値を下げる
        current_threshold -= config.threshold_step
    
    # 最適な結果が見つからなかった場合は現在の結果を使用
    if best_result is None:
        output_groups = format_output_groups(clusters)
        metrics = calculate_output_metrics(output_groups, all_records)
        best_result = {
            "summary": metrics,
            "group": output_groups
        }
        best_threshold = current_threshold
    
    # 最終出力のフォーマット
    output_data = format_final_output(best_result["group"], best_result["summary"], best_threshold)
    
    print(f"\n===== 適応型クラスタリング結果 =====")
    print(f"最適しきい値: {best_threshold:.2f}")
    print(f"クラスター数: {len(best_result['group'])} (元: {len(data.get('group', []))})")
    print(f"精度(ペア): {best_metrics.get('precision(pair)', 'N/A')}")
    print(f"再現率(ペア): {best_metrics.get('recall(pair)', 'N/A')}")
    print(f"F1スコア(ペア): {best_metrics.get('f1(pair)', 'N/A')}")
    
    return output_data

def main():
    parser = argparse.ArgumentParser(description="適応型しきい値調整クラスタリング")
    parser.add_argument("--input", "-i", type=str, required=True, help="入力YAMLファイルパス")
    parser.add_argument("--output", "-o", type=str, default="adaptive_result.yaml", help="出力YAMLファイルパス")
    parser.add_argument("--initial-threshold", "-it", type=float, default=0.8, help="初期類似度しきい値")
    parser.add_argument("--min-threshold", "-mt", type=float, default=0.3, help="最小類似度しきい値")
    parser.add_argument("--threshold-step", "-ts", type=float, default=0.05, help="しきい値の減少ステップ")
    parser.add_argument("--target-precision", "-tp", type=float, default=0.98, help="目標精度（この値以下になったら停止）")
    parser.add_argument("--max-iterations", "-m", type=int, default=10, help="最大反復回数")
    parser.add_argument("--normalize", "-n", action="store_true", help="フィールド正規化を有効化")
    parser.add_argument("--debug", "-d", action="store_true", help="デバッグモードを有効化")
    
    args = parser.parse_args()
    
    # 設定の初期化
    config = Config()
    config.load_from_args(args)
    
    # 処理タイマーの開始
    start_time = datetime.now()
    print(f"処理開始時刻: {start_time}")
    print(f"設定: 初期しきい値={config.initial_threshold}, 最小しきい値={config.min_threshold}, "
          f"ステップ={config.threshold_step}, 目標精度={config.target_precision}")
    
    # 入力データの読み込み
    data = load_yaml(args.input)
    
    # 適応型クラスタリングの実行
    output_data = adaptive_clustering(data, config)
    
    # 出力の保存
    save_yaml(output_data, args.output)
    
    # 終了時刻とサマリー
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    print(f"\n処理完了時刻: {end_time}")
    print(f"合計処理時間: {elapsed:.2f}秒")
    print(f"入力クラスター数: {len(data.get('group', []))}")
    print(f"出力クラスター数: {len(output_data.get('group', []))}")
    
    # メトリクスの表示
    metrics = output_data.get("summary", {})
    print("\n最終評価メトリクス:")
    print(f"  F1スコア (ペア): {metrics.get('f1(pair)', 'N/A')}")
    print(f"  精度 (ペア): {metrics.get('precision(pair)', 'N/A')}")
    print(f"  再現率 (ペア): {metrics.get('recall(pair)', 'N/A')}")
    print(f"  完全一致グループ率: {metrics.get('complete(group)', 'N/A')}")
    print(f"  精度 (グループ): {metrics.get('precision(group)', 'N/A')}")
    print(f"  再現率 (グループ): {metrics.get('recall(group)', 'N/A')}")
    print(f"  使用したしきい値: {metrics.get('threshold', 'N/A')}")

if __name__ == "__main__":
    main()