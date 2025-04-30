#!/usr/bin/env python3
"""
書誌レコードマッチング用一貫性チェックメカニズム

このモジュールは、書誌レコードのマッチング結果における一貫性をチェックし、
推移律に基づいて矛盾を検出・修正する機能を提供します。
"""

import os
import sys
import numpy as np
import pickle
import json
import argparse
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Union
import networkx as nx
from tqdm import tqdm

class ConsistencyChecker:
    """
    マッチング結果の一貫性をチェックするコンポーネント
    """
    
    def __init__(self, threshold: float = 0.8):
        """
        一貫性チェッカーの初期化
        
        Args:
            threshold: 一貫性判定のしきい値
        """
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
    
    def update_clusters(self, existing_clusters: Dict[str, List[Dict]], resolved_edges: List[Tuple[str, str, float]]) -> Dict[str, List[Dict]]:
        """
        解決したエッジに基づいてクラスターを更新
        
        Args:
            existing_clusters: 既存のクラスター
            resolved_edges: 解決されたエッジのリスト
        
        Returns:
            更新されたクラスター
        """
        # レコードIDからクラスターIDへのマッピングを作成
        record_to_cluster = {}
        for cluster_id, records in existing_clusters.items():
            for record in records:
                record_id = record.get("id", "")
                if record_id:
                    record_to_cluster[record_id] = cluster_id
        
        # 解決されたエッジに基づいてクラスターをマージ
        updated_clusters = existing_clusters.copy()
        clusters_to_merge = []
        
        for record_id1, record_id2, _ in resolved_edges:
            if record_id1 in record_to_cluster and record_id2 in record_to_cluster:
                cluster_id1 = record_to_cluster[record_id1]
                cluster_id2 = record_to_cluster[record_id2]
                
                # 異なるクラスターに属している場合のみマージ
                if cluster_id1 != cluster_id2:
                    clusters_to_merge.append((cluster_id1, cluster_id2))
        
        # マージリストをユニークに
        clusters_to_merge = list(set(clusters_to_merge))
        
        # クラスターをマージ
        for cluster_id1, cluster_id2 in clusters_to_merge:
            if cluster_id1 in updated_clusters and cluster_id2 in updated_clusters:
                # クラスター2のレコードをクラスター1に移動
                for record in updated_clusters[cluster_id2]:
                    record["cluster_id"] = cluster_id1
                    updated_clusters[cluster_id1].append(record)
                
                # クラスター2を削除
                del updated_clusters[cluster_id2]
                
                # レコードIDのマッピングを更新
                for record_id, cluster_id in record_to_cluster.items():
                    if cluster_id == cluster_id2:
                        record_to_cluster[record_id] = cluster_id1
        
        return updated_clusters
    
    def evaluate_cluster_consistency(self, clusters: Dict[str, List[Dict]], record_similarities: Dict[Tuple[str, str], float]) -> Dict:
        """
        クラスタリング結果の一貫性を評価
        
        Args:
            clusters: クラスター
            record_similarities: レコードペアの類似度
        
        Returns:
            一貫性メトリクス
        """
        total_pairs = 0
        consistent_pairs = 0
        inconsistent_pairs = 0
        
        # 各クラスター内のすべてのペアをチェック
        for cluster_id, records in clusters.items():
            # クラスター内のレコードIDのリスト
            record_ids = [record.get("id", "") for record in records if "id" in record]
            
            # すべての可能なペアを生成
            for i in range(len(record_ids)):
                for j in range(i+1, len(record_ids)):
                    record_id1 = record_ids[i]
                    record_id2 = record_ids[j]
                    
                    # ペアのキー（順序に依存しない）
                    pair_key = tuple(sorted([record_id1, record_id2]))
                    
                    # 類似度が記録されているか確認
                    if pair_key in record_similarities:
                        similarity = record_similarities[pair_key]
                        total_pairs += 1
                        
                        # しきい値を超えるかどうかをチェック
                        if similarity >= self.threshold:
                            consistent_pairs += 1
                        else:
                            inconsistent_pairs += 1
        
        # メトリクスの計算
        consistency_ratio = consistent_pairs / total_pairs if total_pairs > 0 else 1.0
        
        metrics = {
            "total_pairs": total_pairs,
            "consistent_pairs": consistent_pairs,
            "inconsistent_pairs": inconsistent_pairs,
            "consistency_ratio": consistency_ratio
        }
        
        return metrics
    
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

def main():
    parser = argparse.ArgumentParser(description="書誌レコードマッチングの一貫性チェック")
    parser.add_argument("--input", "-i", type=str, required=True, help="入力類似度ペアファイル（JSON）")
    parser.add_argument("--output", "-o", type=str, default="resolved_edges.json", help="出力解決エッジファイル")
    parser.add_argument("--threshold", "-t", type=float, default=0.8, help="一貫性判定のしきい値")
    parser.add_argument("--graph-cache", "-g", type=str, help="グラフキャッシュファイル")
    parser.add_argument("--save-graph", "-s", action="store_true", help="解析後にグラフを保存")
    
    args = parser.parse_args()
    
    # 処理タイマーの開始
    start_time = datetime.now()
    print(f"処理開始時刻: {start_time}")
    
    # 一貫性チェッカーの初期化
    checker = ConsistencyChecker(threshold=args.threshold)
    
    # グラフキャッシュからの読み込みを試行
    graph_loaded = False
    if args.graph_cache and os.path.exists(args.graph_cache):
        print(f"グラフキャッシュからの読み込みを試みます: {args.graph_cache}")
        graph_loaded = checker.load_similarity_graph(args.graph_cache)
    
    # グラフ読み込みに失敗または指定されていない場合は新規構築
    if not graph_loaded:
        # 入力データの読み込み
        print(f"類似度ペアを読み込み中: {args.input}")
        with open(args.input, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        # 類似度ペアの抽出
        if "record_pairs" in input_data:
            record_pairs = [(pair[0], pair[1], pair[2]) for pair in input_data["record_pairs"]]
        elif "candidate_pairs" in input_data:
            # 候補ペアのみの場合、類似度を0.5（不明）として設定
            record_pairs = [(pair[0], pair[1], 0.5) for pair in input_data["candidate_pairs"]]
        else:
            print("エラー: 有効な類似度ペアまたは候補ペアが見つかりません")
            sys.exit(1)
        
        print(f"{len(record_pairs)}個のレコードペアを読み込みました")
        
        # 類似度グラフの構築
        checker.build_similarity_graph(record_pairs)
    
    # 矛盾の検出
    inconsistencies = checker.find_inconsistencies()
    print(f"{len(inconsistencies)}個の矛盾を検出しました")
    
    # 矛盾の解決
    resolved_edges = checker.resolve_inconsistencies(inconsistencies)
    print(f"{len(resolved_edges)}個のエッジを追加して矛盾を解決しました")
    
    # グラフの保存（オプション）
    if args.save_graph:
        graph_path = args.graph_cache if args.graph_cache else "similarity_graph.gml"
        checker.save_similarity_graph(graph_path)
    
    # 結果の保存
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "threshold": args.threshold,
        "inconsistencies_found": len(inconsistencies),
        "edges_resolved": len(resolved_edges),
        "resolved_edges": resolved_edges
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"解決されたエッジを保存しました: {args.output}")
    
    # 終了時刻とサマリー
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    print(f"処理完了時刻: {end_time}")
    print(f"合計処理時間: {elapsed:.2f}秒")

if __name__ == "__main__":
    main()