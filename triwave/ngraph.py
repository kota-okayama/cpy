"""Neighbor graph"""

# from dataclasses import dataclass
import os
import sys

import codecs
from datetime import datetime, timedelta
import json
from enum import IntEnum
from random import random, shuffle
import math

import numpy as np
import faiss
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
from scipy import stats

from .utils import path
from .inference import Inference, THRESHOLD
from .datatype.record import RecordMG
from .datatype.pairdata import Pairdata
from .datatype.result import ContractionResult
from .datatype.workflow import WorkflowConfig, WorkflowState, CacheMode, CSPlatform, CSAxiomStrategy, InferenceMode
from .datatype.fitting_params import (
    ParamsType,
    BetaParams,
    GammaParams,
    GaussianParams,
)
from .file_container import RecordContainer
from .logger import Logger, LoggerConfig


class NGraphConstructMode:
    """近傍グラフ構築モード"""

    KNN = "KNN"  # ランキングベースグラフ (k近傍法)
    DIST = "DIST"  # 距離ベースグラフ
    DIVERSITY = "DIVERSITY"  # 多様な k 近傍法
    DIVERSITY_COMPRESSED = "DIVERSITY_COMPRESSED"  # 多様な k 近傍法 (圧縮版)
    SYNERGY = "SYNERGY"  # シナジー


class NGraph:
    """近傍グラフ構築クラス"""

    def __init__(
        self,
        inference: Inference,
        inference_mode: InferenceMode = None,
        config: WorkflowConfig = None,
        log_filepath: str = None,
    ):
        """
        コンストラクタ

        - - -

        Params
        ------
        inference: Inference
            推論処理およびfasttextに関わるAPIを提供してくれるインスタンス
        inference_mode: InferenceMode
            推論モード (Bayesian/SUN) を切り替える
        config: WorkflowConfig, by default None
            ワークフローに渡されるconfig
        log_filepath: str, by default None
            ログファイルパス
        """
        self.inference = inference  # 推論処理を担うクラスを格納
        self.config: WorkflowConfig = config if config is not None else WorkflowConfig()

        if inference_mode is not None:
            self.config.inference_mode = inference_mode  # 推論処理のモードを格納

        # self.save_memory: bool = False  # メモリを節約する(SUN推論のみ)
        # self.num_in_edges: "dict[tuple[int, int], int]" = {}  # 2つのノードの共通の被リンク数を格納

        self.re_container: RecordContainer = RecordContainer()
        self.record: "list[RecordMG]" = []

        self.demmed_pairs: "list[tuple[int, int]]" = []  # TODO: config 等で管理する

        self.logger: Logger = Logger(
            __name__,
            logger_config=LoggerConfig(level=self.config.log_level),
            filepath=log_filepath,
        )

        # Parameter (TODO: 教師データから分析して適切な値を得られると良い)
        if self.config.inference_mode == InferenceMode.BAYESIAN:
            self.tau_same = self.tau_not = 0.1

        elif self.config.inference_mode == InferenceMode.SUN:
            self.tau_same = 0.3
            self.tau_not = 0.5

    def load_file(self, filepath: str):
        """
        書誌データが格納されたyamlファイルをre_containerに読み込む

        - - -

        Params
        ------
        filepath: str
            yamlファイルのパス
        """

        self.re_container.load_file(filepath)

    def output_distribution_of_mgraph(
        self,
        indices: np.array,
        distances: np.array,
        image_dirpath: str = None,
        image_title: str = None,
        k: int = 10,
        dist: float = 1.0,
    ):
        """
        [DEIM 2025] mgraph の分布を出力する

        - - -

        Params
        ------
        indices: np.array
            Faiss から出力されるインデックス
        dist: np.array
            Faiss から出力される距離
        """

        # 正解データから全ての一致ペア群を取得 (初期値は 10)
        all_match_pairs_index = self.re_container.get_all_match_pairs_index()
        all_match_pairs = dict.fromkeys(all_match_pairs_index, 10.0)

        # 距離学習機から取得した各距離を格納
        for i, (index, distance) in enumerate(zip(indices, distances)):
            for j, d in zip(index, distance):
                if tuple(sorted([int(i), int(j)])) in all_match_pairs.keys():
                    all_match_pairs[tuple(sorted([int(i), int(j)]))] = min(10.0, float(d))

        # mgraph から全てのペアを取得
        exist_pairs: "list[float]" = []
        for pair in WorkflowConfig.graph.get_all_edges():
            if tuple(pair) in all_match_pairs.keys():
                # ペアの距離を格納
                exist_pairs.append(all_match_pairs[tuple(pair)])
                # 辞書から削除
                all_match_pairs.pop(tuple(pair))

        # mgraph から全てのポテンシャルペアを取得
        potential_pairs: "list[float]" = []
        _, potential_edges, _, _ = WorkflowConfig.graph.get_potential_recall(all_match_pairs_index)
        for pair in potential_edges:
            if tuple(pair) in all_match_pairs.keys():
                # ペアの距離を格納
                potential_pairs.append(all_match_pairs[tuple(pair)])
                # 辞書から削除
                all_match_pairs.pop(tuple(pair))

        # 該当しなかったペアの取得
        not_exist_pairs: "list[float]" = list(all_match_pairs.values())
        del all_match_pairs

        # matplotlib によるヒストグラムの描画
        # 初期化
        plt.clf()
        sum_pairs = len(WorkflowConfig.graph.get_all_edges())
        image_title_suffix = f"{image_title} / " if image_title is not None else ""
        image_title_complete = (
            f"{image_title_suffix}Proposed method $(k={math.ceil(k)}, |E|={sum_pairs})$"
            if self.config.ngraph_construction == NGraphConstructMode.SYNERGY
            else (
                f"{image_title_suffix}Threshold based method $(d={dist}, |E|={sum_pairs})$"
                if self.config.ngraph_construction == NGraphConstructMode.DIST
                else f"{image_title_suffix}nearest-K method $(k={k}, |E|={sum_pairs})$"
            )
        )
        plt.figure(figsize=(6, 4))
        bins = np.arange(0, 10.1, 0.1)
        categories = [
            f"Exist ({len(exist_pairs)})",
            f"Potential ({len(potential_pairs)})",
            f"Not Exist ({len(not_exist_pairs)})",
        ]
        colors = ["#FF6060", "#00afe4", "#808080"]

        plt.hist(
            [exist_pairs, potential_pairs, not_exist_pairs],  # 実際のデータ
            bins=bins,
            label=categories,
            color=colors,
            stacked=True,
            rwidth=0.7,
        )
        plt.title(image_title_complete)
        plt.yscale("log")
        plt.xlim(0, 10)
        plt.xlabel("Metric")
        plt.ylabel("#Match pairs")

        plt.legend()

        # グリッドの表示（オプション）
        plt.grid(True, which="both", linestyle="--", alpha=0.3)

        # レイアウトの調整
        plt.tight_layout()

        # 保存先ディレクトリ生成
        os.makedirs(image_dirpath, exist_ok=True)

        # ファイル名生成
        filepath = None
        for i in range(1, 10000):
            tmpname = path.join(image_dirpath, "deim2025-{:0=4}.pdf".format(i))
            if not path.exists(tmpname):
                filepath = tmpname
                break

        # 既に規定数以上のファイルが保存されている場合はエラーを出す
        if filepath is None:
            raise FileExistsError("Cannot save file because there are already too many files (over 100000).")

        # 画像の保存
        plt.savefig(
            filepath,
            format="pdf",
            bbox_inches="tight",
            transparent=True,
            pad_inches=0.1,
        )

    def generate_mgraph_and_pairdata(
        self,
        k: int | None = None,
        dist: float | None = None,
        min_k: int = 10,
        reflect_mgraph: bool = True,
        reflect_pairdata: bool = True,
        max_cores: int = None,
        limited_k: bool = True,
        contract_edges: bool = False,
        image_dirpath: str = None,
        image_title: str = None,
        log_minutes: int = 5,
    ):
        """
        行列グラフとペアデータを生成する

        - - -

        Params
        ------
        [Deprecated] k: int | None, by default None
            k近傍グラフの深さ (KNN)
        [Deprecated] dist: float | None, by default None
            距離近傍で構築する際の距離 (DIST)
        min_k: int, by default 10
            k近傍における最低エッジ数
        reflect_mgraph: bool, by default True
            現在のMGraphに反映するか否か
        reflect_pairdata: bool, by default True
            現在のPairdataに反映するか否か
        max_cores: int, by default None
            使用するコア数
        limited_k: bool, by default True
            k 近傍によるエッジに対して制約をかける (DIST 戦略のみ)
        contract_edges: bool, by default False
            エッジを縮約するか否か (DIVERSITY 戦略のみ)
        image_dirpath: str, by default None
            画像の保存先ディレクトリ
        log_minutes: int, by default 5
            進捗をログに出力する間隔

        Return
        ------

        """
        # TODO: メソッドを整理する
        # NGraphConstructMode に応じて、受け付けるべきパラメータを変更する

        # パラメータの初期化
        k = self.config.ngraph_params.k if k is None else k
        dist = THRESHOLD if dist is None else dist
        min_k = 10 if min_k is None else min_k
        self.logger.info(f"K: {k}, Dist: {dist}")

        # recordからそれぞれをベクトルに変換し、行列化する (multiprocessによる並列化)
        self.logger.info("Start make_fasttext_dist_matrix.")
        record_array = self.inference.get_fasttext_vectors_with_mp(
            CacheMode.WRITE,
            self.re_container.filepath,
            self.record,
            max_cores,
            log_minutes,
        )
        self.logger.info("Finished make_fasttext_dist_matrix.")

        # 全ての書誌データに対して距離を換算しながら、隣接行列グラフの構築
        if reflect_mgraph:
            WorkflowConfig.graph.generate(len(self.re_container.records))

        # 距離学習機によりベクトルを変換する
        converted_matrix = self.inference.convert_vector_by_metric_learner(np.array(record_array))

        self.logger.info("Start initializing faiss index.")

        # faissにより距離を計算するためのインデックス構築
        # TODO: FlatL2(総当り)以外にIVF(ボロノイを利用した近似最近傍探索)もあるため、選択できるようにする
        # ただし、現状 Intel MKL FATAL ERROR: Cannot load libmkl_avx2.so or libmkl_def.so. エラーが出てる
        # quatizer = faiss.IndexFlatL2(len(converted_matrix[0]))
        # faiss_index = faiss.IndexIVFFlat(quatizer, len(converted_matrix[0]), 1)
        # faiss_index.train(converted_matrix)
        faiss_index = faiss.IndexFlatL2(len(converted_matrix[0]))

        # GPUが利用できる場合は、GPU利用のためのインデックス実装に切り替える
        if self.config.gpu_status.faiss:
            res = faiss.StandardGpuResources()
            self.logger.info(">> Done: faiss.StandardGpuResources")
            faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
            self.logger.info(">> Done: faiss.index_cpu_to_gpu")

        self.logger.info("Start adding faiss vectors.")

        # インデックスにベクトルを追加する
        faiss_index.add(converted_matrix)

        # ある頂点をキーとして、近傍に接続している頂点 (距離 < 1) を格納する (DIVERSITY のみ)
        near_vertices = {}

        # 探索
        if self.config.ngraph_construction == NGraphConstructMode.KNN:  # k近傍法
            distances, indices = faiss_index.search(converted_matrix, min(len(converted_matrix), 2048))
            # faiss の distances は、2乗のままなので sqrt をかける
            distances = np.sqrt(distances)

        elif self.config.ngraph_construction == NGraphConstructMode.DIST:  # 距離ベース
            # distances, indices = faiss_index.range_search(converted_matrix, dist)
            distances, indices = faiss_index.search(converted_matrix, min(len(converted_matrix), 2048))
            # faiss の distances は、2乗のままなので sqrt をかける
            distances = np.sqrt(distances)

        elif self.config.ngraph_construction == NGraphConstructMode.DIVERSITY:  # 多様な k 近傍法
            # distances, indices = faiss_index.range_search(converted_matrix, dist)
            distances, indices = faiss_index.search(converted_matrix, min(len(converted_matrix), 2048))
            # faiss の distances は、2乗のままなので sqrt をかける
            distances = np.sqrt(distances)
            # 距離の近い頂点インデックスを格納する
            for i in range(len(converted_matrix)):
                near_vertices[i] = indices[i][distances[i] <= dist]

        self.logger.info("Start generating neighbor graph.")

        target_indices = []

        # faissにより距離計算を行い、近傍グラフを構築する
        for i in range(len(converted_matrix)):
            if self.config.ngraph_construction == NGraphConstructMode.KNN:
                k1 = k2 = k  # k近傍法の場合はk1とk2は同じ値になる
                target_distances = distances[i][0:k]

            elif self.config.ngraph_construction == NGraphConstructMode.DIST:
                n = len(self.re_container.records)
                logn = int(math.log(n))

                if limited_k:
                    k1 = min(max(len((distances[i])[distances[i] <= dist]), logn), 2 * logn)  # MGraphに接続するエッジ数
                else:
                    k1 = len((distances[i])[distances[i] <= dist])

                target_distances = (distances[i])[distances[i] <= dist * 2]
                k2 = 2 * k1  # Pairdataで保持するエッジ数

            elif self.config.ngraph_construction == NGraphConstructMode.DIVERSITY:
                # 接続するエッジ数は、頂点数 n に対して log(n)
                logn = int(math.log(len(self.re_container.records), 2))
                target_indices = []

                __near_vertices = set(near_vertices[i])
                for j in range(len(indices[i])):

                    if not indices[i][j] in __near_vertices:
                        target_indices.append(j)
                        # インデックスの和集合
                        __near_vertices = __near_vertices | set(near_vertices[indices[i][j]])

                    if len(target_indices) >= logn:
                        break

            if reflect_mgraph:
                # MGraphに接続するエッジを張る
                if self.config.ngraph_construction != NGraphConstructMode.DIVERSITY:
                    WorkflowConfig.graph.connect_edges(i, indices[i][0:k1], False)

                elif self.config.ngraph_construction == NGraphConstructMode.DIVERSITY:
                    WorkflowConfig.graph.connect_edges(i, target_indices, False)

            if reflect_pairdata:
                # Pairdata に距離情報を追加する
                if self.config.ngraph_construction != NGraphConstructMode.DIVERSITY:
                    target_indices = indices[i][0 : len(target_distances)]
                    delete_distance = np.where(target_indices == i)[0]
                    if delete_distance.size > 0:
                        target_distances = np.delete(target_distances, delete_distance[0])
                    target_indices = target_indices[target_indices != i]
                    for t_i, t_d in zip(target_indices, target_distances):
                        self.config.pairdata[tuple(sorted([i, t_i]))] = Pairdata(inf_dist=float(t_d))

                else:
                    for t_i in target_indices:
                        self.config.pairdata[tuple(sorted([i, t_i]))] = Pairdata(inf_dist=float(distances[i][t_i]))

        if reflect_mgraph:
            # 縮約が必要であれば、縮約処理を行う
            if self.config.ngraph_construction == NGraphConstructMode.DIVERSITY and contract_edges:
                for i in range(len(near_vertices)):
                    for j in near_vertices[i]:
                        if i < j:
                            self.contraction(i, j)

            # 統計情報の更新
            WorkflowConfig.graph.update_statics()

        # みなし一致ペアのリストを返す (DIVERSITY のみ)
        if self.config.ngraph_construction == NGraphConstructMode.DIVERSITY:
            demmed_pairs = set([])
            for k, v in near_vertices.items():
                demmed_pairs = demmed_pairs | set([(tuple(sorted([k, __v]))) for __v in v])

            return list(demmed_pairs)

        # 分布グラフの描画
        if image_dirpath is not None:
            self.output_distribution_of_mgraph(indices, distances, image_dirpath, image_title, k, dist)

        return []

    def generate_mgraph_and_pairdata_in_diversity(
        self,
        reflect_mgraph: bool = True,
        reflect_pairdata: bool = True,
        max_cores: int = None,
        contract_edges: bool = False,
        log_minutes: int = 5,
    ):
        """行列グラフとペアデータを生成する

        - - -

        Params
        ------
        reflect_mgraph: bool, by default True
            現在のMGraphに反映するか否か
        reflect_pairdata: bool, by default True
            現在のPairdataに反映するか否か
        max_cores: int, by default None
            使用するコア数
        contract_edges: bool, by default False
            エッジを縮約するか否か (DIVERSITY 戦略のみ)
        log_minutes: int, by default 5
            進捗をログに出力する間隔

        Return
        ------
        "list[tuple[int, int]]", by default []
            みなし一致ペアのリスト
        """
        # TODO: メソッドを整理する
        # NGraphConstructMode に応じて、受け付けるべきパラメータを変更する

        # パラメータの初期化
        # dist = THRESHOLD
        dist = 1

        # recordからそれぞれをベクトルに変換し、行列化する (multiprocessによる並列化)
        self.logger.info("Start make_fasttext_dist_matrix.")
        record_array = self.inference.get_fasttext_vectors_with_mp(
            CacheMode.WRITE,
            self.re_container.filepath,
            self.record,
            max_cores,
            log_minutes,
        )
        self.logger.info("Finished make_fasttext_dist_matrix.")

        # 全ての書誌データに対して距離を換算しながら、隣接行列グラフの構築
        if reflect_mgraph:
            WorkflowConfig.graph.generate(len(self.re_container.records))
            self.logger.info(WorkflowConfig.graph.graph)

        # 距離学習機によりベクトルを変換する
        converted_matrix = self.inference.convert_vector_by_metric_learner(np.array(record_array))

        # ========== 1 周目 ==========
        self.logger.info("Start initializing faiss index (1st).")

        # faissにより距離を計算するためのインデックス構築
        # TODO: FlatL2(総当り)以外にIVF(ボロノイを利用した近似最近傍探索)もあるため、選択できるようにする
        # ただし、現状 Intel MKL FATAL ERROR: Cannot load libmkl_avx2.so or libmkl_def.so. エラーが出てる
        # quatizer = faiss.IndexFlatL2(len(converted_matrix[0]))
        # faiss_index = faiss.IndexIVFFlat(quatizer, len(converted_matrix[0]), 1)
        # faiss_index.train(converted_matrix)
        faiss_index = faiss.IndexFlatL2(len(converted_matrix[0]))

        # GPUが利用できる場合は、GPU利用のためのインデックス実装に切り替える
        if self.config.gpu_status.faiss:
            res = faiss.StandardGpuResources()
            self.logger.info(">>> Done: faiss.StandardGpuResources")
            faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
            self.logger.info(">>> Done: faiss.index_cpu_to_gpu")

        self.logger.info("Start adding faiss vectors.")

        # インデックスにベクトルを追加する
        faiss_index.add(converted_matrix)

        # ある頂点をキーとして、近傍に接続している頂点 (距離 < 1) を格納する
        near_vertices = {}

        # 探索
        distances, indices = faiss_index.search(converted_matrix, min(len(converted_matrix), 2048))
        # faiss の distances は、2乗のままなので sqrt をかける
        distances = np.sqrt(distances)
        # 近傍の頂点インデックスを格納する
        for i in range(len(converted_matrix)):
            near_vertices[i] = indices[i][distances[i] <= dist]

        # 近傍の頂点によるコレクションを生成
        collection = np.arange(len(converted_matrix), dtype=np.int32)
        for i, nv in near_vertices.items():

            # collection の i と同じ値を持っていない n を選出
            __nv = set(nv) - set(np.where(collection == collection[i])[0])

            for j in __nv:
                p, s = (i, j) if len(near_vertices[i]) > len(near_vertices[j]) else (j, i)
                collection[collection == collection[s]] = collection[p]

        # コレクションから代表ノード選出
        representative_nodes = np.unique(collection)

        # GPU 解放
        del faiss_index

        # ========== 2 周目 ==========
        self.logger.info("Start initializing faiss index (2nd).")

        # 代表ノードの matrix を生成
        representative_matrix = np.array([converted_matrix[i] for i in representative_nodes])

        faiss_index = faiss.IndexFlatL2(len(representative_matrix[0]))

        # GPUが利用できる場合は、GPU利用のためのインデックス実装に切り替える
        if self.config.gpu_status.faiss:
            res = faiss.StandardGpuResources()
            self.logger.info(">>> Done: faiss.StandardGpuResources")
            faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
            self.logger.info(">>> Done: faiss.index_cpu_to_gpu")

        self.logger.info("Start adding faiss vectors.")

        # インデックスにベクトルを追加する
        faiss_index.add(representative_matrix)

        # 探索
        distances, indices = faiss_index.search(representative_matrix, min(len(representative_matrix), 2048))

        self.logger.info("Start generating neighbor graph.")
        # faissにより距離計算を行い、近傍グラフを構築する
        for i, rn in enumerate(representative_nodes):
            # 接続するエッジ数は、頂点数 n に対して log(n)
            logn = int(math.log(len(self.re_container.records), 2))

            m = len(np.where(collection == collection[rn])[0])
            __vertices = list(map(lambda x: representative_nodes[x], indices[i][0 : (m * logn)]))

            if reflect_mgraph:
                # MGraphに接続するエッジを張る
                WorkflowConfig.graph.connect_edges(rn, __vertices, False)

            if reflect_pairdata:
                # Pairdata に距離情報を追加する
                for rn_v, t_i in zip(__vertices, distances[i][0 : (m * logn)]):
                    if rn == rn_v:
                        continue

                    self.config.pairdata[tuple(sorted([rn, rn_v]))] = Pairdata(inf_dist=float(distances[i][int(t_i)]))

        # =========================================

        if reflect_mgraph:
            # 縮約が必要であれば、縮約処理を行う
            if contract_edges:
                for i in range(len(near_vertices)):
                    for j in near_vertices[i]:
                        if i < j:
                            self.contraction(i, j)

            # 統計情報の更新
            WorkflowConfig.graph.update_statics()

        # みなし一致ペアのリストを返す
        demmed_pairs = set([])
        for k, v in near_vertices.items():
            demmed_pairs = demmed_pairs | set([(tuple(sorted([k, __v]))) for __v in v])

        return list(demmed_pairs)

    def generate_mgraph_and_pairdata_in_synergy(
        self,
        reflect_mgraph: bool = True,
        reflect_pairdata: bool = True,
        max_cores: int = None,
        contract_edges: bool = False,
        image_dirpath: str = None,
        image_title: str = None,
        log_minutes: int = 5,
    ):
        """行列グラフとペアデータを生成する

        - - -

        Params
        ------
        reflect_mgraph: bool, by default True
            現在のMGraphに反映するか否か
        reflect_pairdata: bool, by default True
            現在のPairdataに反映するか否か
        max_cores: int, by default None
            使用するコア数
        contract_edges: bool, by default False
            エッジを縮約するか否か (DIVERSITY 戦略のみ)
        log_minutes: int, by default 5
            進捗をログに出力する間隔

        Return
        ------
        "list[tuple[int, int]]", by default []
            みなし一致ペアのリスト
        """
        # TODO: メソッドを整理する
        # NGraphConstructMode に応じて、受け付けるべきパラメータを変更する

        # パラメータの初期化
        # dist = THRESHOLD
        dist = 1

        # recordからそれぞれをベクトルに変換し、行列化する (multiprocessによる並列化)
        self.logger.info("Start make_fasttext_dist_matrix.")
        record_array = self.inference.get_fasttext_vectors_with_mp(
            CacheMode.WRITE,
            self.re_container.filepath,
            self.record,
            max_cores,
            log_minutes,
        )
        self.logger.info("Finished make_fasttext_dist_matrix.")

        # 全ての書誌データに対して距離を換算しながら、隣接行列グラフの構築
        if reflect_mgraph:
            WorkflowConfig.graph.generate(len(self.re_container.records), True)

        # 距離学習機によりベクトルを変換する
        converted_matrix = self.inference.convert_vector_by_metric_learner(np.array(record_array))

        # ========== 1 周目 ==========
        self.logger.info("Start initializing faiss index (1st).")

        # faissにより距離を計算するためのインデックス構築
        # TODO: FlatL2(総当り)以外にIVF(ボロノイを利用した近似最近傍探索)もあるため、選択できるようにする
        # ただし、現状 Intel MKL FATAL ERROR: Cannot load libmkl_avx2.so or libmkl_def.so. エラーが出てる
        # quatizer = faiss.IndexFlatL2(len(converted_matrix[0]))
        # faiss_index = faiss.IndexIVFFlat(quatizer, len(converted_matrix[0]), 1)
        # faiss_index.train(converted_matrix)
        faiss_index = faiss.IndexFlatL2(len(converted_matrix[0]))

        # GPUが利用できる場合は、GPU利用のためのインデックス実装に切り替える
        if self.config.gpu_status.faiss:
            res = faiss.StandardGpuResources()
            self.logger.info(">>> Done: faiss.StandardGpuResources")
            faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
            self.logger.info(">>> Done: faiss.index_cpu_to_gpu")

        self.logger.info("Start adding faiss vectors.")

        # インデックスにベクトルを追加する
        faiss_index.add(converted_matrix)

        # ある頂点をキーとして、近傍に接続している頂点 (距離 < 1) を格納する
        near_vertices = {}

        # 探索
        distances, indices = faiss_index.search(converted_matrix, min(len(converted_matrix), 2048))
        # faiss の distances は、2乗のままなので sqrt をかける
        distances = np.sqrt(distances)
        # 近傍の頂点インデックスを格納する
        for i in range(len(converted_matrix)):
            near_vertices[i] = indices[i][distances[i] <= dist]

        # 近傍の頂点によるコレクションを生成
        collection = np.arange(len(converted_matrix), dtype=np.int32)
        for i, nv in near_vertices.items():

            # collection の i と同じ値を持っていない n を選出
            __nv = set(nv) - set(np.where(collection == collection[i])[0])

            for j in __nv:
                p, s = (i, j) if len(near_vertices[i]) > len(near_vertices[j]) else (j, i)
                collection[collection == collection[s]] = collection[p]

        # コレクションから代表ノード選出
        representative_nodes = list(map(lambda x: int(x), np.unique(collection)))

        # より多くのノードを持つ代表ノード順にソートする
        representative_nodes = sorted(
            representative_nodes, key=lambda x: len(np.where(collection == collection[x])[0]), reverse=True
        )

        calltime = datetime.now() + timedelta(minutes=log_minutes)
        relation: "set[tuple[int, int]]" = set([])
        global_counter = 0

        # 代表点でイテレーション
        for cnt, rn in enumerate(representative_nodes):
            counter = int(len(near_vertices[rn]) * math.log(len(self.re_container.records), 2))
            global_counter += counter

            if reflect_mgraph:
                __origin = np.array([])
                __indices = np.array([])
                __distances = np.array([])

                # 代表ノードから各近傍ノードにエッジを張る
                WorkflowConfig.graph.connect_edges(rn, (collection == rn).nonzero()[0], False)
                counter -= len(near_vertices[rn])

                for nv in near_vertices[rn]:
                    # __indices と __distances に追加
                    __validated = distances[nv] > dist
                    __origin = np.append(__origin, np.full(len(indices[nv][__validated]), nv))
                    __indices = np.append(__indices, indices[nv][__validated])
                    __distances = np.append(__distances, distances[nv][__validated])

                # __origin と __indices を __distances を元に距離の近い順にソート
                __origin = __origin[np.argsort(__distances)]
                __indices = __indices[np.argsort(__distances)]

                for e, i, o in zip(range(len(__indices)), __indices, __origin):
                    if counter <= 0:
                        break

                    # i が所属するクラスタを調べ、そのクラスタへのエッジがまだ存在しなければ、エッジを張る
                    # if rn != collection[int(i)] and tuple(sorted([rn, int(collection[int(i)])])) not in relation:
                    if rn != collection[int(i)] and tuple(sorted([rn, int(collection[int(i)])])) not in relation:
                        WorkflowConfig.graph.connect_edges(int(o), [int(i)], False)
                        relation.add(tuple(sorted([rn, int(collection[int(i)])])))

                        # pairdata に追加
                        if reflect_pairdata:
                            self.config.pairdata[tuple(sorted([int(o), int(i)]))] = Pairdata(
                                inf_dist=float(__distances[e])
                            )

                        counter -= 1

                    # if rn != collection[int(i)] and tuple(sorted([int(o), int(i)])) not in relation:
                    #     WorkflowConfig.graph.connect_edges(int(o), [int(i)], False)
                    #     relation.add(tuple(sorted([int(o), int(i)])))

                    #     # pairdata に追加
                    #     if reflect_pairdata:
                    #         self.config.pairdata[tuple(sorted([int(o), int(i)]))] = Pairdata(
                    #             inf_dist=float(__distances[e])
                    #         )

                    #     counter -= 1

                # 一定時間を超えた場合にログを出力する
                if datetime.now() > calltime:
                    self.logger.info(
                        "#representive_nodes: {} / {} ({}%)".format(
                            cnt,
                            len(representative_nodes),
                            int(cnt * 100 / len(representative_nodes)),
                        )
                    )
                    calltime += timedelta(minutes=log_minutes)

        if reflect_mgraph:
            # 縮約が必要であれば、縮約処理を行う
            if contract_edges:
                for i in range(len(near_vertices)):
                    for j in near_vertices[i]:
                        if i < j:
                            self.contraction(i, j)

            # 統計情報の更新
            WorkflowConfig.graph.update_statics()

        # 分布グラフの描画
        if image_dirpath is not None:
            self.output_distribution_of_mgraph(
                indices,
                distances,
                image_dirpath,
                image_title,
                k=math.log(len(self.re_container.records), 2),
                dist=None,
            )

        # エッジに接続するため、みなし一致ペアは存在しない

        return []

    def output_ngraph_summary(self, demmed_pairs: "list[tuple[int, int]]" = []):
        """
        生成された近傍グラフについて概要を出力する

        - - -

        Params
        ------
        demmed_pairs: "list[tuple[int, int]]", by default []
            みなし一致ペアのリスト
        """

        # 含まれていたエッジの数をカウント
        match_pairs = self.re_container.get_all_match_pairs_index()
        recall = WorkflowConfig.graph.get_recall(match_pairs, demmed_pairs)
        potential_recall = WorkflowConfig.graph.get_potential_recall(match_pairs, demmed_pairs)

        demmed = (
            ""
            if recall[3] is None
            else f"""
 Recall (Demmed): {len(recall[3])} / {len(match_pairs)} ( {recall[2]} )
PRecall (Demmed): {len(potential_recall[3])} / {len(match_pairs)} ( {potential_recall[2]} )
"""
        )

        result = f"""
[[ NGraph Summary ]]
        Vertices: {WorkflowConfig.graph.num_of_verts}
           Edges: {WorkflowConfig.graph.num_of_edges}
          Recall: {len(recall[1])} / {len(match_pairs)} ( {recall[0]} )
Potential Recall: {len(potential_recall[1])} / {len(match_pairs)} ( {potential_recall[0]} )
{demmed[1:-1]}
"""

        self.logger.info(result)

    def generate_ngraph(
        self,
        filepath: str,
        k: int = None,
        dist: float = None,
        min_k: int = None,
        reflect_crowdsourcing_result: bool = True,
        limited_k: bool = True,
        connect_edge_by_cs_result: bool = True,
        max_cores: int = None,
        image_dirpath: str = None,
        image_title: str = None,
        log_minutes: int = 5,
    ):
        """
        対象のjson/yamlファイルを読み込み、近傍グラフを構築したあと、結果をインスタンス変数に格納する

        - - -

        Params
        ------
        filepath: str
            書誌データを格納したjson/yamlのファイルパス
        graph_construct_mode: NGraphConstructMode
            k近傍で構築するか、距離近傍で構築するかを決定する
        [Deprecated] k: int, by default None
            k近傍グラフの深さ (KNN)
        [Deprecated] dist: float, by default None
            距離近傍で構築する際の距離 (DIST)
        reflect_crowdsourcing_result: bool, by default True
            クラウドソーシングの結果を反映するか否か
        limited_k: bool, by default True
            k 近傍によるエッジに対して制約をかける (DIST 戦略のみ)
        connect_edge_by_cs_result: bool, by default True
            クラウドソーシングの結果が持つエッジを張るかどうか (reflect_crowdsourcing_resultがTrueの時は実行しない)
        max_cores: int, by default None
            並列処理時に使用するコア数
        log_minutes: int, by default 10
            進捗をログに出力する間隔
        """

        # 距離を取得するためにfasttext用の行列を作成し、正解書誌データインデックスを作成する
        self.load_file(filepath)
        self.record = self.re_container.get_recordmg()

        # ペアの推論値を持つデータを初期化
        self.config.pairdata = {}

        # MGraphと推論データを生成する
        self.demmed_pairs = []

        if self.config.ngraph_construction == NGraphConstructMode.DIVERSITY_COMPRESSED:
            self.demmed_pairs = self.generate_mgraph_and_pairdata_in_diversity(
                reflect_mgraph=True,
                reflect_pairdata=True,
                max_cores=max_cores,
                contract_edges=True,
                log_minutes=log_minutes,
            )

        elif self.config.ngraph_construction == NGraphConstructMode.SYNERGY:
            self.demmed_pairs = self.generate_mgraph_and_pairdata_in_synergy(
                reflect_mgraph=True,
                reflect_pairdata=True,
                max_cores=max_cores,
                contract_edges=False,
                image_dirpath=image_dirpath,
                image_title=image_title,
                log_minutes=log_minutes,
            )

        else:
            self.demmed_pairs = self.generate_mgraph_and_pairdata(
                k=k,
                dist=dist,
                reflect_mgraph=True,
                reflect_pairdata=True,
                max_cores=max_cores,
                limited_k=limited_k,
                image_dirpath=image_dirpath,
                image_title=image_title,
                log_minutes=log_minutes,
            )

        self.logger.info("Finished generating neighbor graph.")

        # 作成された近傍グラフの概要の出力
        self.output_ngraph_summary(self.demmed_pairs)

        # グラフを出力する
        # draw_kgraph(candidate_k)

        # 必要があればクラウドソーシングの結果を反映する
        if reflect_crowdsourcing_result:
            start_edges = len(self.config.crowdsourcing_result)

            # クラウドソーシングの結果がない場合は、処理を終了する
            if start_edges == 0:
                return

            calltime = datetime.now() + timedelta(minutes=log_minutes)

            count = start_edges

            self.logger.info("Start neighbor graph contraction by crowdsourcing result.")
            self.logger.info(f"target edges: {start_edges}")

            for key, value in self.config.crowdsourcing_result.items():
                a, b = key

                if sum(value) / len(value) >= 0.5:  # クラウドソーシングで一致と判定 -> エッジを縮約
                    WorkflowConfig.graph.contraction_edge(a, b)

                else:  # クラウドソーシングで不一致と判定 -> エッジを削除
                    WorkflowConfig.graph.remove_edge(a, b)

                count -= 1

                # 一定時間を超えた場合にログを出力する
                if datetime.now() > calltime:
                    self.logger.info(
                        "Number of remaining edges: {} / {} ({}%)".format(
                            count,
                            start_edges,
                            int((start_edges - count) * 100 / start_edges),
                        )
                    )
                    calltime += timedelta(minutes=log_minutes)

            # 統計情報の更新
            WorkflowConfig.graph.update_statics()

            self.logger.info("Finished neighbor graph contraction by crowdsourcing result.")

        # 必要があれば、クラウドソーシングの結果を持つペアに対してエッジを張る
        elif connect_edge_by_cs_result:
            self.logger.info("Start neighbor graph connection by crowdsourcing result.")

            for a, b in self.config.crowdsourcing_result.keys():
                WorkflowConfig.graph.connect_edges(a, [b], False)

            # 統計情報の更新
            WorkflowConfig.graph.update_statics()

            self.logger.info("Finished neighbor graph connection by crowdsourcing result.")

    def contraction(self, a: int, b: int, force: bool = False):
        """
        [Deprecated] 縮約処理を行い、k近傍グラフを更新する

        - - -

        Params
        ------
        a: int
            縮約するノード(残存側)のインデックス
        b: int
            縮約するノード(削除側)のインデックス
        force: bool, by default False
            エッジの有無に関わらず強制的に縮約を行う
        """

        # 今後はこのメソッドを経由せず、mgraphに直接アクセスして縮約する
        WorkflowConfig.graph.contraction_edge(a, b)

    def remove_edge(self, a: int, b: int):
        """
        [Deprecated] エッジを削除し、k近傍グラフを更新する

        - - -

        Params
        ------
        a : int
            リンクされているノードのインデックス
        b : int
            リンクされているノードのインデックス
        """

        # 今後はこのメソッドを経由せず、mgraphに直接アクセスしてエッジを削除する
        WorkflowConfig.graph.remove_edge(a, b)

    def update_pairdata(self, a: int, b: int, use_sun_params: bool = False):
        """
        指定されたインデックスのインスタンス変数に保存されているself.config.pairdataの推論値を更新する

        - - -

        Params
        ------
        a: int
            self.recordに保存されている書誌データインデックス
        b: int
            self.recordに保存されている書誌データインデックス
        use_sun_params: bool, by default False
            自動更新されたパラメータを使う
        """
        a, b = sorted([a, b])

        # すでに推論済みであれば何もしない
        if (a, b) not in self.config.pairdata or self.config.pairdata[(a, b)].inf_same is None:

            if (a, b) not in self.config.pairdata:
                self.config.pairdata[(a, b)] = Pairdata()

            # ペア間の距離が未計算であれば、既存のクラスターを利用して距離を計算する
            if self.config.pairdata[(a, b)].inf_dist is None:
                __dist = []

                # クラスターが所属するインデックスを取得
                vertices1 = self.config.graph.get_vertices_in_cluster(a)
                vertices2 = self.config.graph.get_vertices_in_cluster(b)

                # ランダムにペアを抽出し、出現した距離を持っている5つのペアの距離の平均を利用する
                __pair = [tuple(sorted([v1, v2])) for v1 in vertices1 for v2 in vertices2]
                shuffle(__pair)
                for p in __pair:
                    if p in self.config.pairdata and self.config.pairdata[p].inf_dist is not None:
                        __dist.append(self.config.pairdata[p].inf_dist)
                        if len(__dist) >= 5:
                            break

                # TODO: クラスタ間で1つも距離を持たない場合があるらしいため、原因を究明して改修する
                if len(__dist) == 0:
                    __dist = [10]

                self.config.pairdata[(a, b)].inf_dist = sum(__dist) / len(__dist)

            # 推論により確率を算出
            if self.config.inference_mode == InferenceMode.BAYESIAN:
                (
                    self.config.pairdata[(a, b)].inf_same,
                    self.config.pairdata[(a, b)].inf_unknown,
                    self.config.pairdata[(a, b)].inf_not,
                    sim,
                    same,
                    not_same,
                    self.config.pairdata[(a, b)].inf_dist,
                ) = self.inference.greet_bayesian(
                    self.record[a],
                    self.record[b],
                    self.config.pairdata[(a, b)].inf_dist,
                )

            elif self.config.inference_mode == InferenceMode.SUN:
                (
                    self.config.pairdata[(a, b)].inf_same,
                    self.config.pairdata[(a, b)].inf_unknown,
                    self.config.pairdata[(a, b)].inf_not,
                    sim,
                    same,
                    not_same,
                    self.config.pairdata[(a, b)].inf_dist,
                ) = self.inference.greet_sun(
                    self.record[a],
                    self.record[b],
                    self.config.pairdata[(a, b)].inf_dist,
                    use_sun_params,
                )

            return same, not_same, sim

    def update_all_pairdata(self):
        """インスタンス変数に保存されているself.config.pairdataについてすべてのペアに対し推論を行いデータを更新する"""

        # TODO: データ圧縮のためにNot-sameと判定されたものについてはメモリ開放をしてもいいかも

        # すべてのペアに対して推論を開始する
        start_num_of_pairs = len(self.config.pairdata.keys())
        notice_edges = (start_num_of_pairs // 1000) * 1000 - 1
        self.logger.info("Start inference on all pairs")
        self.logger.info("number of pairs: {}".format(start_num_of_pairs))

        for i, k in enumerate(self.config.pairdata.keys()):
            self.update_pairdata(*k)
            if notice_edges >= start_num_of_pairs - i - 1:
                self.logger.info(
                    "Number of remaining pairs: {} / {}".format(start_num_of_pairs - i - 1, start_num_of_pairs)
                )
                notice_edges -= 1000

        self.logger.info("Finish inference on all pairs")

    def apply_axiomatic_system_for_current_graph(self):
        """[Deprecated] 現在のグラフに対し公理系を適用しグラフの更新を行うクラス"""

        class InfResult(IntEnum):
            """推論結果の内容を保持するクラス"""

            SAME = 1
            UNKNOWN = 10
            NOT_SAME = 100

        def return_sun_result(prob_same: float, prob_not: float, tau_same: float, tau_not: float) -> InfResult:
            """SUN推論の確率値から推論結果を返すメソッド"""

            if 1 - prob_same < tau_same:
                return InfResult.SAME

            elif 1 - prob_not < tau_not:
                return InfResult.NOT_SAME

            else:
                return InfResult.UNKNOWN

        inf_result: "dict[tuple[int, int], InfResult | float]" = (
            {}
        )  # 推論した書誌データペアをキーに、その推論結果を値に格納
        target_triangle: "list[tuple[int, int, int]]" = []
        query_for_crowd: "set[tuple[int, int]]" = set([])  # クラウドソーシングに投げて縮約を行うペアの組み合わせ

        # グラフのエッジに対して、公理系を適用して矛盾したグラフからクラウドソーシングする候補ペアを探す
        triangles = WorkflowConfig.graph.get_all_triangles()
        for vs in triangles:
            # 推論により確率を算出
            for i in range(len(vs) - 1):
                for j in range(i + 1, len(vs)):
                    if (i, j) not in self.config.pairdata or self.config.pairdata[(i, j)].inf_same is None:
                        self.update_pairdata(i, j)

                    if (i, j) not in inf_result:
                        if self.config.inference_mode == InferenceMode.BAYESIAN:
                            inf_result[(i, j)] = self.config.pairdata[(i, j)].inf_same
                        elif self.config.inference_mode == InferenceMode.SUN:
                            inf_result[(i, j)] = return_sun_result(
                                self.config.pairdata[(i, j)].inf_same,
                                self.config.pairdata[(i, j)].inf_not,
                                self.tau_same,
                                self.tau_not,
                            )

            if self.config.inference_mode == InferenceMode.BAYESIAN:  # bayesian
                # (1-a)bc + a(1-b)c + ab(1-c) の領域が大きい場合に矛盾として計算
                judgement = (
                    (1 - inf_result[(vs[0], vs[1])]) * inf_result[(vs[0], vs[2])] * inf_result[(vs[1], vs[2])]
                    + (1 - inf_result[(vs[0], vs[2])]) * inf_result[(vs[1], vs[2])] * inf_result[(vs[0], vs[1])]
                    + (1 - inf_result[(vs[1], vs[2])]) * inf_result[(vs[0], vs[1])] * inf_result[(vs[0], vs[2])]
                )

                if judgement >= 0.5:
                    target_triangle.append((vs[0], vs[1], vs[2]))
                    query_for_crowd.add((vs[0], vs[1]))
                    query_for_crowd.add((vs[0], vs[2]))
                    query_for_crowd.add((vs[1], vs[2]))

            elif self.config.inference_mode == InferenceMode.SUN:  # SUN
                # 条件にあてはまる書誌ペアをクラウドソーシングで問い合わせる書誌ペアとして追加 (条件: (S, S, N), (S, S, U), (S, N, U), (S, U, U), (N, U, U), (U, U, U))
                judgement = inf_result[(vs[0], vs[1])] + inf_result[(vs[0], vs[2])] + inf_result[(vs[1], vs[2])]

                if (
                    judgement == 102
                    or judgement == 12
                    or judgement == 111
                    or judgement == 21
                    or judgement == 120
                    or judgement == 30
                ):
                    target_triangle.append((vs[0], vs[1], vs[2]))
                    query_for_crowd.add((vs[0], vs[1]))
                    query_for_crowd.add((vs[0], vs[2]))
                    query_for_crowd.add((vs[1], vs[2]))

        # クラウドソーシングに問い合わせて縮約処理を行う (シミュレーション)
        human_count = 0

        for a, b in list(query_for_crowd):
            # クラウドソーシングで問い合わせる (シミュレーション)
            human_score = self.request_crowdsourcing((a, b))[(a, b)]
            human_count += 1

            # 縮約 (クラウドソーシング)
            if human_score >= 0.5:
                self.logger.debug(
                    "contraction(human/axiom): ({}, {}) ({}, {})".format(
                        a, b, WorkflowConfig.graph.collection[a], WorkflowConfig.graph.collection[b]
                    )
                )

                # 縮約処理
                self.contraction(a, b)
                inf_result[(a, b)] += InfResult.SAME * 1000

                # 操作結果をログに残す
                # correct_result = self.request_crowdsourcing(self.record[a].re.id, self.record[b].re.id)

            # エッジ削除 (クラウドソーシング)
            else:
                # エッジ削除処理
                self.remove_edge(a, b)
                inf_result[(a, b)] += InfResult.NOT_SAME * 1000

                # 操作結果をログに残す
                # correct_result = self.request_crowdsourcing(self.record[a].re.id, self.record[b].re.id)
                self.logger.debug("remove(human/axiom): ({}, {})".format(a, b))

        if self.config.inference_mode == InferenceMode.BAYESIAN:
            cat = {}
            for cor in [3, 201, 300]:
                cat[cor * 1000] = 0

            pair = {}
            for inf in [1, 10, 100]:
                for cor in [1, 100]:
                    pair[inf + cor * 1000] = 0

            # カテゴリごとに集計
            for a, b, c in target_triangle:
                target = int((inf_result[(a, b)] + inf_result[(a, c)] + inf_result[(b, c)]) // 1000 * 1000)
                cat[target] += 1

            # ペアに関して集計
            for a, b in list(query_for_crowd):
                pair[inf_result[(a, b)]] += 1

            result = "[[ Axiomatic System Result (Bayesian) ]]\n"
            result += "   Group Evaluation |     inference |\n"
            result += "                    | inconsistency |\n"
            result += "------------------------------------\n"
            result += " correct |  3 pairs |      {:>8d} |\n".format(cat[300000])
            result += "             1 pair |      {:>8d} |\n".format(cat[201000])
            result += "            no pair |      {:>8d} |\n".format(cat[3000])
            result += "\n"
            result += "    Pair Evaluation | inference\n"
            result += "                    |     Same |  Unknown | Not-same |\n"
            result += "-------------------------------------------------------\n"
            result += " correct |     Same | {:>8d} | {:>8d} | {:>8d} |\n".format(pair[1001], pair[1010], pair[1100])
            result += "           Not-same | {:>8d} | {:>8d} | {:>8d} |\n".format(
                pair[100001], pair[100010], pair[100100]
            )
            result += " human tasks ... {}\n".format(human_count)

        elif self.config.inference_mode == InferenceMode.SUN:
            cat = {}
            for inf in [102, 12, 111, 21, 120, 30]:
                for cor in [3, 201, 300]:
                    cat[inf + cor * 1000] = 0

            pair = {}
            for inf in [1, 10, 100]:
                for cor in [1, 100]:
                    pair[inf + cor * 1000] = 0

            # カテゴリごとに集計
            for a, b, c in target_triangle:
                cat[inf_result[(a, b)] + inf_result[(a, c)] + inf_result[(b, c)]] += 1

            # ペアに関して集計
            for a, b in list(query_for_crowd):
                pair[inf_result[(a, b)]] += 1

            result = "[[ Axiomatic System Result (SUN) ]]\n"
            result += "   Group Evaluation | inference\n"
            result += "                    |  S/S/N |  S/S/U |  S/N/U |  S/U/U |  N/U/U |  U/U/U |\n"
            result += "---------------------------------------------------------------------------\n"
            result += " correct |  3 pairs | {:>6d} | {:>6d} | {:>6d} | {:>6d} | {:>6d} | {:>6d} |\n".format(
                cat[300102], cat[300012], cat[300111], cat[300021], cat[300120], cat[300030]
            )
            result += "             1 pair | {:>6d} | {:>6d} | {:>6d} | {:>6d} | {:>6d} | {:>6d} |\n".format(
                cat[201102], cat[201012], cat[201111], cat[201021], cat[201120], cat[201030]
            )
            result += "            no pair | {:>6d} | {:>6d} | {:>6d} | {:>6d} | {:>6d} | {:>6d} |\n".format(
                cat[3102], cat[3012], cat[3111], cat[3021], cat[3120], cat[3030]
            )
            result += "\n"
            result += "    Pair Evaluation | inference\n"
            result += "                    |     Same |  Unknown | Not-same |\n"
            result += "-------------------------------------------------------\n"
            result += " correct |     Same | {:>8d} | {:>8d} | {:>8d} |\n".format(pair[1001], pair[1010], pair[1100])
            result += "           Not-same | {:>8d} | {:>8d} | {:>8d} |\n".format(
                pair[100001], pair[100010], pair[100100]
            )
            result += " human tasks ... {}\n".format(human_count)

        self.logger.info(result)

    def apply_axiomatic_system_for_retraining(
        self,
        cs_task_limit: int = None,
        cs_worker_accuracy: float = None,
        cs_result_priority: bool = True,
        cs_axiom_strategy: CSAxiomStrategy = CSAxiomStrategy.BINARY,
        priority_rule: "dict[str, int]" = {},
    ):
        """
        現在のグラフに対し公理系を適用し、矛盾からクラウドソーシングすべきペアを発見する

        - - -

        Params
        ------
        cs_task_limit: int, by default None
            クラウドソーシングするペアの上限数
        cs_worker_accuracy: float, by default None
            クラウドソーシングワーカーの回答精度
        cs_result_priority: bool, by default True
            クラウドソーシングの結果を優先して利用するか否か
        cs_axiom_strategy: CSAxiomStrategy, by default CSAxiomStrategy.BINARY
            クラウドソーシング問い合わせ時の戦略
        priority: dict[str, int]
            "SSN", "SSU", "SUN", "SUU", "UUN", "UUU" をキーにして、優先度を値に持つ辞書
        """

        if cs_task_limit is None:
            cs_task_limit = sys.maxsize

        if cs_worker_accuracy is None:
            cs_worker_accuracy = self.config.crowdsourcing_worker_accuracy

        class InfResult(IntEnum):
            """推論結果の内容を保持するクラス"""

            SAME = 1
            UNKNOWN = 10
            NOT_SAME = 100

        def return_sun_result(prob_same: float, prob_not: float, tau_same: float, tau_not: float) -> InfResult:
            """SUN推論の確率値から推論結果を返すメソッド"""

            if 1 - prob_same < tau_same:
                return InfResult.SAME

            elif 1 - prob_not < tau_not:
                return InfResult.NOT_SAME

            else:
                return InfResult.UNKNOWN

        # 推論した書誌データペアをキーに、その推論結果を値に格納
        inf_result: "dict[tuple[int, int], InfResult | float]" = {}

        # クラウドソーシングに投げて縮約を行うペアの組み合わせ
        query_for_crowd: "dict[list[int], float]" = {}

        cost = 1
        if cs_axiom_strategy == CSAxiomStrategy.TRINARY:
            cost = 3

        # ログに関する設定
        LOG_MINUTES = 10
        calltime = datetime.now() + timedelta(minutes=LOG_MINUTES)

        # 公理矛盾の種類によって優先順位を付与するための辞書型 (SUN推論)
        inf_result_priority = {
            102: priority_rule.get("SSN", 4),
            12: priority_rule.get("SSU", 3),
            111: priority_rule.get("SUN", 2),
            21: priority_rule.get("SUU", 1),
            120: priority_rule.get("UUN", 0),
            30: priority_rule.get("UUU", 0),
        }

        # グラフのエッジに対して、公理系を適用して矛盾したグラフからクラウドソーシングする候補ペアを探す
        triangles = WorkflowConfig.graph.get_all_triangles()
        count = 0

        for vs in triangles:
            # 推論により確率を算出
            for i in range(len(vs) - 1):
                for j in range(i + 1, len(vs)):
                    target = (vs[i], vs[j])
                    if target not in self.config.pairdata or self.config.pairdata[target].inf_same is None:
                        self.update_pairdata(vs[i], vs[j])

                    if target not in inf_result:
                        # クラウドソーシング結果を利用し、さらにそのペアの結果を保持しているならその結果を使う
                        if (
                            cs_result_priority
                            and (vs[i], vs[j]) in self.config.crowdsourcing_result
                            and len(self.config.crowdsourcing_result[(vs[i], vs[j])]) > 0
                        ):
                            cs_result = self.config.crowdsourcing_result[(vs[i], vs[j])]
                            inf_result[(vs[i], vs[j])] = sum(cs_result) / len(cs_result)

                        elif self.config.inference_mode == InferenceMode.BAYESIAN:
                            inf_result[(vs[i], vs[j])] = self.config.pairdata[(vs[i], vs[j])].inf_same

                        elif self.config.inference_mode == InferenceMode.SUN:
                            inf_result[(vs[i], vs[j])] = return_sun_result(
                                self.config.pairdata[(vs[i], vs[j])].inf_same,
                                self.config.pairdata[(vs[i], vs[j])].inf_not,
                                self.tau_same,
                                self.tau_not,
                            )

            if self.config.inference_mode == InferenceMode.BAYESIAN:  # bayesian
                # (1-a)bc + a(1-b)c + ab(1-c) の領域が大きい場合に矛盾として計算
                score = (
                    (1 - inf_result[(vs[0], vs[1])]) * inf_result[(vs[0], vs[2])] * inf_result[(vs[1], vs[2])]
                    + (1 - inf_result[(vs[0], vs[2])]) * inf_result[(vs[1], vs[2])] * inf_result[(vs[0], vs[1])]
                    + (1 - inf_result[(vs[1], vs[2])]) * inf_result[(vs[0], vs[1])] * inf_result[(vs[0], vs[2])]
                )

                if score >= 0:
                    if cs_axiom_strategy == CSAxiomStrategy.BINARY:
                        query_for_crowd[(vs[0], vs[1])] = max(query_for_crowd.get((vs[0], vs[1]), 0), score)
                        query_for_crowd[(vs[0], vs[2])] = max(query_for_crowd.get((vs[0], vs[2]), 0), score)
                        query_for_crowd[(vs[1], vs[2])] = max(query_for_crowd.get((vs[1], vs[2]), 0), score)

                    elif cs_axiom_strategy == CSAxiomStrategy.BINARY_UNCERTAINTY:
                        query_for_crowd[tuple(vs)] = score

                    else:
                        query_for_crowd[tuple(vs)] = score

            elif self.config.inference_mode == InferenceMode.SUN:  # SUN
                # 条件にあてはまる書誌ペアをクラウドソーシングで問い合わせる書誌ペアとして追加 (条件: (S, S, N), (S, S, U), (S, N, U), (S, U, U), (N, U, U), (U, U, U))
                judgement = inf_result[(vs[0], vs[1])] + inf_result[(vs[0], vs[2])] + inf_result[(vs[1], vs[2])]

                if (
                    judgement == 102
                    or judgement == 12
                    or judgement == 111
                    or judgement == 21
                    or judgement == 120
                    or judgement == 30
                ):
                    if cs_axiom_strategy == CSAxiomStrategy.BINARY:
                        score = (
                            self.config.pairdata[(vs[0], vs[1])].inf_not
                            * self.config.pairdata[(vs[0], vs[2])].inf_same
                            * self.config.pairdata[(vs[1], vs[2])].inf_same
                            + self.config.pairdata[(vs[0], vs[1])].inf_same
                            * self.config.pairdata[(vs[0], vs[2])].inf_not
                            * self.config.pairdata[(vs[1], vs[2])].inf_same
                            + self.config.pairdata[(vs[0], vs[1])].inf_same
                            * self.config.pairdata[(vs[0], vs[2])].inf_same
                            * self.config.pairdata[(vs[1], vs[2])].inf_not
                            + inf_result_priority[judgement]
                        )
                        query_for_crowd[(vs[0], vs[1])] = max(query_for_crowd.get((vs[0], vs[1]), 0), score)
                        query_for_crowd[(vs[0], vs[2])] = max(query_for_crowd.get((vs[0], vs[2]), 0), score)
                        query_for_crowd[(vs[1], vs[2])] = max(query_for_crowd.get((vs[1], vs[2]), 0), score)

                    elif cs_axiom_strategy == CSAxiomStrategy.BINARY_UNCERTAINTY:
                        query_for_crowd[tuple(vs)] = score

                    else:
                        query_for_crowd[tuple(vs)] = score

            # 定期ログを発行する
            count += 1
            if datetime.now() > calltime:
                all_ts = len(triangles)
                self.logger.info(
                    "Number of remaining triangles: {} / {} ({}%)".format(count, all_ts, int(count * 100 / all_ts))
                )
                calltime += timedelta(minutes=LOG_MINUTES)

        # スコアの高い順に並び替える
        q_f_c: "dict[float, set]" = {}
        for key, value in query_for_crowd.items():
            q_f_c[value] = q_f_c.get(value, set([]))
            q_f_c[value].add(key)

        # クラウドソーシングで聞くべきペアを選出する
        query_for_crowd_list: "list[tuple[int, int]]" = []
        if cs_axiom_strategy == CSAxiomStrategy.BINARY:
            for key in sorted(q_f_c.keys(), reverse=True):
                query_for_crowd_list += list(q_f_c[key])

        elif cs_axiom_strategy == CSAxiomStrategy.BINARY_UNCERTAINTY:
            for key in sorted(q_f_c.keys(), reverse=True):
                for vs in q_f_c[key]:

                    # 一番不確実なもの (0.5 に近いもの) を取得し、それだけを対象にする
                    # 同値の場合は、クラウドソーシング結果がないものを優先して候補にする
                    target = sorted(
                        list(
                            zip(
                                [
                                    abs(inf_result[(vs[0], vs[1])] - 0.5),
                                    abs(inf_result[(vs[0], vs[2])] - 0.5),
                                    abs(inf_result[(vs[1], vs[2])] - 0.5),
                                ],
                                [
                                    len(self.config.crowdsourcing_result.get((vs[0], vs[1]), [])),
                                    len(self.config.crowdsourcing_result.get((vs[0], vs[2]), [])),
                                    len(self.config.crowdsourcing_result.get((vs[1], vs[2]), [])),
                                ],
                                [
                                    (vs[0], vs[1]),
                                    (vs[0], vs[2]),
                                    (vs[1], vs[2]),
                                ],
                            )
                        ),
                        key=lambda x: (x[0], x[1]),
                    )

                    if target[0][2] not in query_for_crowd_list:
                        query_for_crowd_list.append(target[0][2])

        # クラウドソーシングで問い合わせる
        human_count = 0
        for indices in query_for_crowd_list:
            if (
                self.config.crowdsourcing_count + human_count >= self.config.crowdsourcing_limit
                or human_count >= cs_task_limit
            ):
                break

            human_score = self.request_crowdsourcing(indices, cs_worker_accuracy, False)
            human_count += cost

            if human_score is not None:
                for i in range(len(indices) - 1):
                    for j in range(i + 1, len(indices)):
                        target = (indices[i], indices[j])

                        # 一致 (クラウドソーシング)
                        if human_score[target] >= 0.5:
                            inf_result[target] += InfResult.SAME * 1000

                        # 不一致 (クラウドソーシング)
                        else:
                            inf_result[target] += InfResult.NOT_SAME * 1000

        if self.config.crowdsourcing_platform != CSPlatform.SIMULATION:
            # シミュレーション以外の場合は、検証不可能なためここでcsvファイルを生成してreturnする
            return WorkflowState(suspend=True, finished=True, required_crowdsourcing=True)

        if self.config.inference_mode == InferenceMode.BAYESIAN:
            cat = {}
            for cor in [3, 201, 300]:
                cat[cor * 1000] = 0

            pair = {}
            for inf in [1, 10, 100]:
                for cor in [1, 100]:
                    pair[inf + cor * 1000] = 0

            # カテゴリごとに集計
            for a, b, c in triangles:
                target = int((inf_result[(a, b)] + inf_result[(a, c)] + inf_result[(b, c)]) // 1000) * 1000
                if target in cat:
                    cat[target] += 1

            # ペアに関して集計
            for qc in query_for_crowd:
                for i in range(len(qc) - 1):
                    for j in range(i + 1, len(qc)):
                        integer = int(inf_result[(qc[i], qc[j])] // 1000 * 1000)
                        decimal = inf_result[(qc[i], qc[j])] - integer
                        target = integer + (InfResult.SAME if decimal > 0.5 else InfResult.NOT_SAME)
                        if target in pair:
                            pair[target] += 1

            result = [
                "",
                "[[ Axiomatic System Result (Bayesian) ]]",
                "   Group Evaluation |     inference |",
                "                    | inconsistency |",
                "-------------------------------------",
                " correct |  3 pairs |      {:>8d} |".format(cat[3000]),
                "            1 pair  |      {:>8d} |".format(cat[201000]),
                "           no pair  |      {:>8d} |".format(cat[300000]),
                "",
                "    Pair Evaluation | inference",
                "                    | Positive | Negative |",
                "-------------------------------------------",
                " correct |     Same | {:>8d} | {:>8d} |".format(pair[1001], pair[1100]),
                "           Not-same | {:>8d} | {:>8d} |".format(pair[100001], pair[100100]),
                " human tasks ... {}\n".format(human_count),
            ]
            result = "\n".join(result)

        elif self.config.inference_mode == InferenceMode.SUN:
            cat = {}
            for inf in [102, 12, 111, 21, 120, 30]:
                for cor in [3, 201, 300]:
                    cat[inf + cor * 1000] = 0

            pair = {}
            for inf in [1, 10, 100]:
                for cor in [1, 100]:
                    pair[inf + cor * 1000] = 0

            # カテゴリごとに集計
            for a, b, c in triangles:
                if inf_result[(a, b)] + inf_result[(a, c)] + inf_result[(b, c)] in cat:
                    cat[inf_result[(a, b)] + inf_result[(a, c)] + inf_result[(b, c)]] += 1

            # ペアに関して集計
            for a, b in query_for_crowd:
                if inf_result[(a, b)] in pair:
                    pair[inf_result[(a, b)]] += 1

            result = "[[ Axiomatic System Result (SUN) ]]\n"
            result += "   Group Evaluation | inference\n"
            result += "                    |  S/S/N |  S/S/U |  S/N/U |  S/U/U |  N/U/U |  U/U/U |\n"
            result += "---------------------------------------------------------------------------\n"
            result += " correct |  3 pairs | {:>6d} | {:>6d} | {:>6d} | {:>6d} | {:>6d} | {:>6d} |\n".format(
                cat[3102], cat[3012], cat[3111], cat[3021], cat[3120], cat[3030]
            )
            result += "            1 pair  | {:>6d} | {:>6d} | {:>6d} | {:>6d} | {:>6d} | {:>6d} |\n".format(
                cat[201102], cat[201012], cat[201111], cat[201021], cat[201120], cat[201030]
            )
            result += "           no pair  | {:>6d} | {:>6d} | {:>6d} | {:>6d} | {:>6d} | {:>6d} |\n".format(
                cat[300102], cat[300012], cat[300111], cat[300021], cat[300120], cat[300030]
            )
            result += "\n"
            result += "    Pair Evaluation | inference\n"
            result += "                    |     Same |  Unknown | Not-same |\n"
            result += "-------------------------------------------------------\n"
            result += " correct |     Same | {:>8d} | {:>8d} | {:>8d} |\n".format(pair[1001], pair[1010], pair[1100])
            result += "           Not-same | {:>8d} | {:>8d} | {:>8d} |\n".format(
                pair[100001], pair[100010], pair[100100]
            )
            result += " human tasks ... {}\n".format(human_count)

        self.logger.info(result)

    def apply_uncertainty_strategy_for_retraining(
        self,
        crowdsourcing_task_limit: int = None,
        cs_worker_accuracy: float = None,
        cs_result_priority: bool = True,
    ):
        """
        現在のグラフに対しUncetaintyの大きいペア順にクラウドソーシングすべきペアを格納する

        - - -

        Params
        ------
        crowdsourcing_task_limit: int, by default None
            クラウドソーシングするペアの上限数
        cs_worker_accuracy: float, by default none
            クラウドソーシングワーカーの回答精度
        cs_result_priority: bool, by default True
            クラウドソーシングの結果を優先して利用するか否か
        """

        if crowdsourcing_task_limit is None:
            crowdsourcing_task_limit = sys.maxsize

        if cs_worker_accuracy is None:
            cs_worker_accuracy = self.config.crowdsourcing_worker_accuracy

        inf_result: "dict[float, list[tuple[int, int]]]" = (
            {}
        )  # uncertaintyの値をキーに、推論した書誌データペアの配列を値に格納
        query_for_crowd: "list[tuple[int, int]]" = []  # クラウドソーシングに投げて縮約を行うペアの組み合わせ

        # ログに関する設定
        LOG_MINUTES = 10
        calltime = datetime.now() + timedelta(minutes=LOG_MINUTES)

        # グラフのエッジに対して、公理系を適用して矛盾したグラフからクラウドソーシングする候補ペアを探す
        count = 0
        all_edges = WorkflowConfig.graph.get_all_edges()
        for i, j in all_edges:
            a, b = sorted([i, j])
            self.update_pairdata(a, b, use_sun_params=True)

            # クラウドソーシング結果を利用し、さらにそのペアの結果を保持しているならその結果を使う
            if (
                cs_result_priority
                and (a, b) in self.config.crowdsourcing_result
                and len(self.config.crowdsourcing_result[(a, b)]) > 0
            ):
                inf_same = sum(self.config.crowdsourcing_result[(a, b)]) / len(self.config.crowdsourcing_result[(a, b)])

            # 結果を利用しない場合は推論結果を利用する
            else:
                inf_sum = self.config.pairdata[(a, b)].inf_same + self.config.pairdata[(a, b)].inf_not
                inf_same = self.config.pairdata[(a, b)].inf_same / inf_sum

            uncertainty = 0.5 - abs(inf_same - 0.5)
            inf_result[uncertainty] = inf_result.get(uncertainty, [])
            inf_result[uncertainty].append((a, b))

            # 定期ログを発行する
            count += 1
            if datetime.now() > calltime:
                sum_edges = len(all_edges)
                self.logger.info(
                    "Number of remaining edges: {} / {} ({}%)".format(count, sum_edges, int(count * 100 / sum_edges))
                )
                calltime += timedelta(minutes=LOG_MINUTES)

        # クラウドソーシングに問い合わせて縮約処理を行う (シミュレーション)
        cr = ContractionResult()

        # スコアの高い順から配列に格納する
        for i in sorted(inf_result.keys(), reverse=True):
            query_for_crowd += inf_result[i]

        finished = set([])
        # クラウドソーシングで問い合わせる
        for a, b in query_for_crowd:
            if (
                self.config.crowdsourcing_count >= self.config.crowdsourcing_limit
                or self.config.crowdsourcing_task_count >= crowdsourcing_task_limit
            ):
                break

            if (a, b) in finished:
                continue

            finished.add((a, b))
            human_score = self.request_crowdsourcing((a, b), cs_worker_accuracy, False)[(a, b)]

            if human_score is None:
                pass

            # 一致 (クラウドソーシング)
            elif human_score >= 0.5:
                if self.config.pairdata[(a, b)].inf_same > self.config.pairdata[(a, b)].inf_not:
                    cr.true_positive += 1
                else:
                    cr.false_negative += 1

            # 不一致 (クラウドソーシング)
            else:
                if self.config.pairdata[(a, b)].inf_same > self.config.pairdata[(a, b)].inf_not:
                    cr.false_positive += 1
                else:
                    cr.true_negative += 1

        if self.config.crowdsourcing_platform != CSPlatform.SIMULATION:
            # シミュレーション以外の場合は、検証不可能なためここでcsvファイルを生成してreturnする
            return WorkflowState(suspend=True, finished=True, required_crowdsourcing=True)

        # シミュレーションの場合は結果を表示する
        result = [
            "",
            "[[ Uncertainty Strategy Result ]]",
            "    Pair Evaluation | inference",
            "                    | Positive | Negative |",
            "-------------------------------------------",
            " correct |     Same | {:>8d} | {:>8d} |".format(cr.true_positive, cr.false_negative),
            "           Not-same | {:>8d} | {:>8d} |".format(cr.false_positive, cr.true_negative),
            " human tasks ... {}\n".format(self.config.crowdsourcing_task_count),
        ]
        result = "\n".join(result)

        # ログ出力
        self.logger.info(result)

    def apply_qbc_strategy_for_retraining(
        self,
        crowdsourcing_task_limit: int = None,
        cs_worker_accuracy: float = None,
        cs_result_priority: bool = True,
    ):
        """
        現在のグラフに対しUnknownの大きいペア順にクラウドソーシングすべきペアを格納する (Query by Committee)

        - - -

        Params
        ------
        crowdsourcing_task_limit: int, by default None
            クラウドソーシングするペアの上限数
        cs_worker_accuracy: float, by default None
            クラウドソーシングワーカーの回答精度
        cs_result_priority: bool, by default True
            クラウドソーシングの結果を優先して利用するか否か
        """

        if crowdsourcing_task_limit is None:
            crowdsourcing_task_limit = sys.maxsize

        if cs_worker_accuracy is None:
            cs_worker_accuracy = self.config.crowdsourcing_worker_accuracy

        inf_result: "dict[float, list[tuple[int, int]]]" = (
            {}
        )  # uncertaintyの値をキーに、推論した書誌データペアの配列を値に格納
        query_for_crowd: "list[tuple[int, int]]" = []  # クラウドソーシングに投げて縮約を行うペアの組み合わせ

        # ログに関する設定
        LOG_MINUTES = 10
        calltime = datetime.now() + timedelta(minutes=LOG_MINUTES)

        # グラフのエッジに対して、Unknown値の高いものからクラウドソーシングする候補ペアを探す
        count = 0
        all_edges = WorkflowConfig.graph.get_all_edges()
        for i, j in all_edges:
            a, b = sorted([i, j])
            self.update_pairdata(a, b, use_sun_params=True)

            # クラウドソーシング結果を利用し、さらにそのペアの結果を保持しているなら、クラウドソーシング結果で拮抗を判定する
            if (
                cs_result_priority
                and (a, b) in self.config.crowdsourcing_result
                and len(self.config.crowdsourcing_result[(a, b)]) > 0
            ):
                # TODO:  Σ|k-μ|/n <= 0.5 であることを証明する必要はありそう
                _result = self.config.crowdsourcing_result[(a, b)]
                _avg = sum(_result) / len(_result)
                unknown = sum([abs(r - _avg) for r in _result]) * 2 / len(_result)

            # 結果を利用しない場合は推論結果を利用する
            else:
                unknown = self.config.pairdata[(a, b)].inf_unknown

            inf_result[unknown] = inf_result.get(unknown, [])
            inf_result[unknown].append((a, b))

            # 定期ログを発行する
            count += 1
            if datetime.now() > calltime:
                sum_edges = len(all_edges)
                self.logger.info(
                    "Number of remaining edges: {} / {} ({}%)".format(count, sum_edges, int(count * 100 / sum_edges))
                )
                calltime += timedelta(minutes=LOG_MINUTES)

        # クラウドソーシングに問い合わせて縮約処理を行う (シミュレーション)
        cr = ContractionResult()

        # スコアの高い順から配列に格納する
        for i in sorted(inf_result.keys(), reverse=True):
            query_for_crowd += inf_result[i]

        finished = set([])
        # クラウドソーシングで問い合わせる
        for a, b in query_for_crowd:
            if (
                self.config.crowdsourcing_count >= self.config.crowdsourcing_limit
                or self.config.crowdsourcing_task_count >= crowdsourcing_task_limit
            ):
                break

            if (a, b) in finished:
                continue

            finished.add((a, b))
            human_score = self.request_crowdsourcing((a, b), cs_worker_accuracy, False)[(a, b)]

            if human_score is None:
                pass

            # 一致 (クラウドソーシング)
            elif human_score:
                if self.config.pairdata[(a, b)].inf_same > self.config.pairdata[(a, b)].inf_not:
                    cr.true_positive += 1
                else:
                    cr.false_negative += 1

            # 不一致 (クラウドソーシング)
            else:
                if self.config.pairdata[(a, b)].inf_same > self.config.pairdata[(a, b)].inf_not:
                    cr.false_positive += 1
                else:
                    cr.true_negative += 1

        if self.config.crowdsourcing_platform != CSPlatform.SIMULATION:
            # シミュレーション以外の場合は、検証不可能なためここでcsvファイルを生成してreturnする
            return WorkflowState(suspend=True, finished=True, required_crowdsourcing=True)

        # シミュレーションの場合は結果を表示する
        result = [
            "",
            "[[ Query by Committee Strategy Result ]]",
            "    Pair Evaluation | inference",
            "                    | Positive | Negative |",
            "-------------------------------------------",
            " correct |     Same | {:>8d} | {:>8d} |".format(cr.true_positive, cr.false_negative),
            "           Not-same | {:>8d} | {:>8d} |".format(cr.false_positive, cr.true_negative),
            " human tasks ... {}\n".format(self.config.crowdsourcing_task_count),
        ]
        result = "\n".join(result)

        self.logger.info(result)

    def apply_diversity_strategy_for_retraining(
        self,
        crowdsourcing_task_limit: int = None,
        cs_worker_accuracy: float = None,
        max_cores: int = None,
        log_minutes: int = 5,
    ):
        """
        現在のグラフに対しDiversity戦略に基づいてクラウドソーシングすべきペアを格納する (Diversity)

        - - -

        Params
        ------
        crowdsourcing_task_limit: int, by default None
            クラウドソーシングするペアの上限数
        cs_worker_accuracy: float, by default None
            クラウドソーシングワーカーの回答精度
        """

        # ペアの候補
        answered = list(self.config.crowdsourcing_result.keys())
        suspend = list(set([(a, b) for a, b in self.config.graph.get_all_edges()]) - set(answered))

        # クラウドソーシングの回答が1つもない場合は、ランダムストラテジーを適用する
        if len(answered) == 0:
            return self.apply_random_strategy_for_retraining(crowdsourcing_task_limit, cs_worker_accuracy)

        # パラメータの初期化
        k = 2048  # faissの制限により最大値を設定

        # recordからそれぞれをベクトルに変換し、行列化する (multiprocessによる並列化)
        self.logger.info("Start make_fasttext_dist_matrix.")
        record_array = self.inference.get_fasttext_vectors_with_mp(
            CacheMode.WRITE,
            self.re_container.filepath,
            self.record,
            max_cores,
            log_minutes,
        )
        self.logger.info("Finished make_fasttext_dist_matrix.")

        # 距離学習機によりベクトルを変換する
        converted_matrix = self.inference.convert_vector_by_metric_learner(np.array(record_array))

        # 各ペアに対して各要素の平均値を取ったベクトルにより、行列を生成する
        target_matrix = np.array(
            [np.mean([converted_matrix[i], converted_matrix[j]], axis=0) for i, j in answered + suspend]
        )

        self.logger.info("Start initializing faiss index.")

        # faissにより距離を計算するためのインデックス構築
        # TODO: FlatL2(総当り)以外にIVF(ボロノイを利用した近似最近傍探索)もあるため、選択できるようにする
        # ただし、現状 Intel MKL FATAL ERROR: Cannot load libmkl_avx2.so or libmkl_def.so. エラーが出てる
        # quatizer = faiss.IndexFlatL2(len(converted_matrix[0]))
        # faiss_index = faiss.IndexIVFFlat(quatizer, len(converted_matrix[0]), 1)
        # faiss_index.train(converted_matrix)
        faiss_index = faiss.IndexFlatL2(len(target_matrix[0]))

        # GPUが利用できる場合は、GPU利用のためのインデックス実装に切り替える
        if self.config.gpu_status.faiss:
            res = faiss.StandardGpuResources()
            self.logger.info(">> Done: faiss.StandardGpuResources")
            faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
            self.logger.info(">> Done: faiss.index_cpu_to_gpu")

        self.logger.info("Start adding faiss vectors.")

        # インデックスにベクトルを追加する
        faiss_index.add(target_matrix)

        # 探索
        _, indices = faiss_index.search(target_matrix, k)

        self.logger.info("Start finding pairs.")

        # faissにより距離計算を行い、近いペアを取得する
        near: dict[tuple[int, int], int] = {}
        for i in range(len(answered)):
            target = [j for j in indices[i] if j > len(answered)]
            for j in target:
                __idx = suspend[int(j - len(answered))]
                near[__idx] = near.get(__idx, 0) + 1

        near_count: dict[int, list[tuple[int, int]]] = {}
        for k, v in near.items():
            near_count[v] = near_count.get(v, [])
            near_count[v].append(k)

        # クラウドソーシングに問い合わせる
        if crowdsourcing_task_limit is None:
            crowdsourcing_task_limit = sys.maxsize

        if cs_worker_accuracy is None:
            cs_worker_accuracy = self.config.crowdsourcing_worker_accuracy

        inf_result: "dict[float, list[tuple[int, int]]]" = (
            {}
        )  # uncertaintyの値をキーに、推論した書誌データペアの配列を値に格納
        query_for_crowd: "list[tuple[int, int]]" = []  # クラウドソーシングに投げて縮約を行うペアの組み合わせ

        # nearのカウント数が少ないペアから問い合わせを行う
        # 1度も出現していないペア
        __pairs = list(set(suspend) - set(near.keys()))
        shuffle(__pairs)
        query_for_crowd = __pairs[0:crowdsourcing_task_limit]

        # 1度以上出現
        for key in sorted(near_count.keys()):
            if len(query_for_crowd) >= crowdsourcing_task_limit:
                break
            __pairs = near_count[key]
            shuffle(__pairs)
            query_for_crowd += __pairs[0 : (crowdsourcing_task_limit - len(query_for_crowd))]

        # クラウドソーシングに問い合わせて縮約処理を行う (シミュレーション)
        cr = ContractionResult()

        # スコアの高い順から配列に格納する
        for i in sorted(inf_result.keys(), reverse=True):
            query_for_crowd += inf_result[i]

        finished = set([])
        # クラウドソーシングで問い合わせる
        for a, b in query_for_crowd:
            if (
                self.config.crowdsourcing_count >= self.config.crowdsourcing_limit
                or self.config.crowdsourcing_task_count >= crowdsourcing_task_limit
            ):
                break

            if (a, b) in finished:
                continue

            finished.add((a, b))
            human_score = self.request_crowdsourcing((a, b), cs_worker_accuracy, False)[(a, b)]

            if human_score is None:
                pass

            # 一致 (クラウドソーシング)
            elif human_score:
                if self.config.pairdata[(a, b)].inf_same > self.config.pairdata[(a, b)].inf_not:
                    cr.true_positive += 1
                else:
                    cr.false_negative += 1

            # 不一致 (クラウドソーシング)
            else:
                if self.config.pairdata[(a, b)].inf_same > self.config.pairdata[(a, b)].inf_not:
                    cr.false_positive += 1
                else:
                    cr.true_negative += 1

        if self.config.crowdsourcing_platform != CSPlatform.SIMULATION:
            # シミュレーション以外の場合は、検証不可能なためここでcsvファイルを生成してreturnする
            return WorkflowState(suspend=True, finished=True, required_crowdsourcing=True)

        # シミュレーションの場合は結果を表示する
        result = [
            "",
            "[[ Diversity Strategy Result ]]",
            "    Pair Evaluation | inference",
            "                    | Positive | Negative |",
            "-------------------------------------------",
            " correct |     Same | {:>8d} | {:>8d} |".format(cr.true_positive, cr.false_negative),
            "           Not-same | {:>8d} | {:>8d} |".format(cr.false_positive, cr.true_negative),
            " human tasks ... {}\n".format(self.config.crowdsourcing_task_count),
        ]
        result = "\n".join(result)

        self.logger.info(result)

    def apply_random_strategy_for_retraining(
        self,
        crowdsourcing_task_limit: int = None,
        cs_worker_accuracy: float = None,
    ):
        """
        現在のグラフに対しRandomにクラウドソーシングすべきペアを格納する (Random)

        - - -

        Params
        ------
        crowdsourcing_task_limit: int, by default None
            クラウドソーシングするペアの上限数
        """

        if crowdsourcing_task_limit is None:
            crowdsourcing_task_limit = sys.maxsize

        if cs_worker_accuracy is None:
            cs_worker_accuracy = self.config.crowdsourcing_worker_accuracy

        query_for_crowd: "list[tuple[int, int]]" = []  # クラウドソーシングに投げて縮約を行うペアの組み合わせ

        # ログに関する設定
        LOG_MINUTES = 10
        calltime = datetime.now() + timedelta(minutes=LOG_MINUTES)

        # グラフのエッジに対して、公理系を適用して矛盾したグラフからクラウドソーシングする候補ペアを探す
        count = 0
        all_edges = WorkflowConfig.graph.get_all_edges()
        for i, j in all_edges:
            a, b = min(i, j), max(i, j)
            self.update_pairdata(a, b, use_sun_params=True)
            query_for_crowd.append((a, b))

            # 定期ログを発行する
            count += 1
            if datetime.now() > calltime:
                sum_edges = len(all_edges)
                self.logger.info(
                    "Number of remaining edges: {} / {} ({}%)".format(count, sum_edges, int(count * 100 / sum_edges))
                )
                calltime += timedelta(minutes=LOG_MINUTES)

        # クラウドソーシングに問い合わせて縮約処理を行う (シミュレーション)
        cr = ContractionResult()

        # ランダム
        shuffle(query_for_crowd)

        finished = set([])
        # クラウドソーシングで問い合わせる
        for a, b in query_for_crowd:
            if (
                self.config.crowdsourcing_count >= self.config.crowdsourcing_limit
                or self.config.crowdsourcing_task_count >= crowdsourcing_task_limit
            ):
                break

            if (a, b) in finished:
                continue

            finished.add((a, b))
            human_score = self.request_crowdsourcing((a, b), cs_worker_accuracy, False)[(a, b)]

            if human_score is None:
                pass

            # 一致 (クラウドソーシング)
            elif human_score >= 0.5:
                if self.config.pairdata[(a, b)].inf_same > self.config.pairdata[(a, b)].inf_not:
                    cr.true_positive += 1
                else:
                    cr.false_negative += 1

            # 不一致 (クラウドソーシング)
            else:
                if self.config.pairdata[(a, b)].inf_same > self.config.pairdata[(a, b)].inf_not:
                    cr.false_positive += 1
                else:
                    cr.true_negative += 1

        if self.config.crowdsourcing_platform != CSPlatform.SIMULATION:
            # シミュレーション以外の場合は、検証不可能なためここでcsvファイルを生成してreturnする
            return WorkflowState(suspend=True, finished=True, required_crowdsourcing=True)

        # シミュレーションの場合は結果を表示する
        result = [
            "",
            "[[ Random Result ]]",
            "    Pair Evaluation | inference",
            "                    | Positive | Negative |",
            "-------------------------------------------",
            " correct |     Same | {:>8d} | {:>8d} |".format(cr.true_positive, cr.false_negative),
            "           Not-same | {:>8d} | {:>8d} |".format(cr.false_positive, cr.true_negative),
            " human tasks ... {}\n".format(self.config.crowdsourcing_task_count),
        ]
        result = "\n".join(result)

        self.logger.info(result)

    def update_ngraph(
        self,
        crowdsourcing_enable: bool = True,
        cs_worker_accuracy: float = None,
        cs_result_priority: bool = True,
        title_blocking: bool = False,
    ):
        """
        近傍グラフを1回分更新する

        - - -

        Params
        ------
        crowdsourcing_enable: bool, by default True
            クラウドソーシングの上限数に達していないかどうか
        cs_worker_accuracy: float, by default 1.0
            クラウドソーシングワーカーの回答精度
        cs_result_priority: bool, by default True
            推論値を計算する際、既存クラウドソーシング結果を優先するか否か

        Return
        ------
        str
            操作内容のログ
        """

        # 共有する被リンク数が最大のペアを取得
        # self.logger.debug(
        #     "self_edge: {}, graph_edge: {}".format(len(self.num_in_edges), WorkflowConfig.graph.num_of_edges)
        # )
        try:
            a, b = WorkflowConfig.graph.get_max_score_edge()
            a, b = int(a), int(b)
        except ValueError as e:
            self.logger.error(WorkflowConfig.graph.get_all_edges())
            self.logger.error(e)
            raise e

        # 初期値の定義
        prob_same = prob_unknown = prob_not = 0

        if (
            cs_result_priority
            and (a, b) in self.config.crowdsourcing_result
            and len(self.config.crowdsourcing_result[(a, b)]) > 0
        ):
            # クラウドソーシング結果が存在する場合は、それを推論値として利用
            prob_same = sum(self.config.crowdsourcing_result[(a, b)]) / len(self.config.crowdsourcing_result[(a, b)])
            prob_not = 1 - prob_same
            prob_unknown = None

        else:
            # ペアに対する推論値が未計算の場合は推論を実行
            if (a, b) not in self.config.pairdata or self.config.pairdata[(a, b)].inf_same is None:
                self.update_pairdata(a, b)

            # 推論値を獲得
            prob_same, prob_unknown, prob_not = (
                self.config.pairdata[(a, b)].inf_same,
                None,
                self.config.pairdata[(a, b)].inf_not,
            )
            if self.config.inference_mode == InferenceMode.SUN:
                prob_unknown = 1 - prob_same - prob_not

        # ### Recordは明確な属性をしていできないため一旦廃止
        # ### 属性型を指定すればOKかも
        """
        title_blocking = title_blocking and len(set(self.record[a].re.title) & set(self.record[b].re.title)) == 0

        # タイトルブロッキング (切断処理)
        if title_blocking:
            # エッジ削除処理
            self.remove_edge(a, b)

            # 操作結果をログに残す
            correct_result = self.re_container.verify_record_pairs(self.record[a].re.id, self.record[b].re.id)
            self.logger.debug(
                "remove(title_blocking): ({}, {}), p: {} <- {}".format(
                    a, b, prob_same, "true" if not correct_result else "false"
                )
            )
            if correct_result:
                self.config.contraction_result.true_positive += 1
                self.config.contraction_machine_result.true_positive += 1
            else:
                self.config.contraction_result.false_positive += 1
                self.config.contraction_machine_result.false_positive += 1
        """

        # 縮約 (計算機)
        if (1 - prob_same < self.tau_same and not crowdsourcing_enable) or (
            prob_same > prob_not and not crowdsourcing_enable
        ):
            # 縮約処理
            self.contraction(a, b)

            # 操作結果をログに残す
            correct_result = self.re_container.verify_record_pairs(a, b)
            self.logger.debug(
                "contraction(machine): ({}, {}), p: {} <- {}".format(
                    a, b, prob_same, "true" if correct_result else "false"
                )
            )
            if correct_result:
                self.config.contraction_result.true_positive += 1
                self.config.contraction_machine_result.true_positive += 1
                self.config.machine_contraction_pair.append((a, b, prob_same, prob_not, prob_unknown))

            else:
                self.config.contraction_result.false_positive += 1
                self.config.contraction_machine_result.false_positive += 1
                self.config.machine_misidentification_pair.append((a, b, prob_same, prob_not, prob_unknown))

        # エッジ削除 (計算機)
        elif (1 - prob_not < self.tau_not and not crowdsourcing_enable) or (
            prob_same <= prob_not and not crowdsourcing_enable
        ):
            # エッジ削除処理
            self.remove_edge(a, b)

            # 操作結果をログに残す
            correct_result = self.re_container.verify_record_pairs(a, b)
            self.logger.debug(
                "remove(machine): ({}, {}), p: {} <- {}".format(
                    a, b, prob_same, "true" if not correct_result else "false"
                )
            )
            if correct_result:
                self.config.contraction_result.false_negative += 1
                self.config.contraction_machine_result.false_negative += 1

            else:
                self.config.contraction_result.true_negative += 1
                self.config.contraction_machine_result.true_negative += 1

        # クラウドソーシングで問い合わせる
        else:
            human_score = self.request_crowdsourcing((a, b), cs_worker_accuracy)

            # クラウドソーシング結果が得られない場合は、Falseを返す
            if human_score is None:
                return False

            # 縮約 (クラウドソーシング)
            if human_score >= 0.5:
                # 縮約処理
                self.contraction(a, b)

                # 操作結果をログに残す
                correct_result = self.re_container.verify_record_pairs(a, b)
                self.logger.debug(
                    "contraction(human): ({}, {}), p: {} <- {}".format(
                        a, b, prob_same, "true" if correct_result else "false"
                    )
                )
                if correct_result:
                    self.config.contraction_result.true_positive += 1
                else:
                    self.config.contraction_result.false_positive += 1

            # エッジ削除 (クラウドソーシング)
            else:
                # エッジ削除処理
                self.remove_edge(a, b)

                # 操作結果をログに残す
                correct_result = self.re_container.verify_record_pairs(a, b)
                self.logger.debug(
                    "remove(human): ({}, {}), p: {} <- {}".format(
                        a, b, prob_same, "true" if not correct_result else "false"
                    )
                )
                if correct_result:
                    self.config.contraction_result.false_negative += 1
                else:
                    self.config.contraction_result.true_negative += 1

        return True

    def update_all_ngraph_limited_strategy(self, crowdsourcing_task_limit: int = None):
        """
        [Deperecated] クラウドソーシング制限があるなかで近傍グラフをすべて更新する

        tauの値が人間側の処理の範囲の場合は、必ず正しい結果を返す(シミュレータ)

        - - -

        Return
        ------
        str
            操作内容のログ
        """

        if crowdsourcing_task_limit is None:
            crowdsourcing_task_limit = sys.maxsize

        LOG_MINUTES = 10
        calltime = datetime.now() + timedelta(minutes=LOG_MINUTES)

        # エッジがなくなるまで縮約処理
        start_edges = WorkflowConfig.graph.num_of_edges
        self.logger.info("start neighbor graph contraction")
        self.logger.info("init number of edges: {}".format(start_edges))

        unknown_value: "dict[float, set[tuple[int, int]]]" = {}

        # 全ペアを対象に推論を行う
        for e in WorkflowConfig.graph.get_all_edges():
            a, b = min(int(e[0]), int(e[1])), max(int(e[0]), int(e[1]))
            self.update_pairdata(a, b)
            inf_unknown = 1 - (self.config.pairdata[(a, b)].inf_same + self.config.pairdata[(a, b)].inf_not)
            unknown_value[inf_unknown] = unknown_value.get(inf_unknown, set([]))
            unknown_value[inf_unknown].add((a, b))

        # Unknown領域の小さい順にソートしたペア配列を作成する
        unknown_list: "list[tuple[int, int]]" = []
        for key in sorted(unknown_value.keys()):
            unknown_list += list(unknown_value[key])

        # === クラウドソーシングが可能な状態 ===
        # エッジが0になるか、クラウドソーシング上限値に到達するまで縮約・切断を行う
        crowdsourcing_enable = True
        for a, b in unknown_list:
            # エッジが0 あるいは クラウドソーシング数が上限値に達した場合は、クラウドソーシングを許可しない
            crowdsourcing_enable = not (
                self.config.crowdsourcing_count >= self.config.crowdsourcing_limit
                or self.config.crowdsourcing_task_count >= crowdsourcing_task_limit
            )

            # エッジがなくなった場合は、ループを抜ける
            if WorkflowConfig.graph.num_of_edges == 0:
                break

            target = self.config.pairdata[(a, b)]

            # 縮約 (計算機)
            if (1 - target.inf_same < self.tau_same and crowdsourcing_enable) or (
                target.inf_same > target.inf_not and not crowdsourcing_enable
            ):
                # 縮約処理
                self.contraction(a, b)
                correct_result = self.re_container.verify_record_pairs(a, b)

                if correct_result:
                    self.config.contraction_result.true_positive += 1
                    self.config.contraction_machine_result.true_positive += 1

                else:
                    self.config.contraction_result.false_positive += 1
                    self.config.contraction_machine_result.false_positive += 1

            # エッジ削除 (計算機)
            elif (1 - target.inf_not < self.tau_not and crowdsourcing_enable) or (
                target.inf_same <= target.inf_not and not crowdsourcing_enable
            ):
                # エッジ削除処理
                self.remove_edge(a, b)
                correct_result = self.re_container.verify_record_pairs(a, b)

                if correct_result:
                    self.config.contraction_result.false_negative += 1
                    self.config.contraction_machine_result.false_negative += 1

                else:
                    self.config.contraction_result.true_negative += 1
                    self.config.contraction_machine_result.true_negative += 1

            # クラウドソーシングで問い合わせる
            else:
                human_score = self.request_crowdsourcing((a, b))

                # 縮約 (クラウドソーシング)
                if human_score >= 0.5:
                    # 縮約処理
                    self.contraction(a, b)
                    correct_result = self.re_container.verify_record_pairs(a, b)

                    if correct_result:
                        self.config.contraction_result.true_positive += 1
                    else:
                        self.config.contraction_result.false_positive += 1

                # エッジ削除 (クラウドソーシング)
                else:
                    # エッジ削除処理
                    self.remove_edge(a, b)
                    correct_result = self.re_container.verify_record_pairs(a, b)

                    if correct_result:
                        self.config.contraction_result.false_negative += 1
                    else:
                        self.config.contraction_result.true_negative += 1

            if datetime.now() > calltime:
                num_of_edges = len(WorkflowConfig.graph.edges())
                self.logger.info(
                    "Number of remaining edges: {} / {} ({}%)".format(
                        num_of_edges, start_edges, int((start_edges - num_of_edges) * 100 / start_edges)
                    )
                )
                calltime += timedelta(minutes=LOG_MINUTES)

        self.output_log_ngraph_result()  # 結果をログに出力する

    def update_all_ngraph(
        self,
        crowdsourcing_task_limit: int = None,
        crowdsourcing_limit_suspend: bool = False,
        cs_worker_accuracy: float = None,
        cs_result_priority: bool = True,
        title_blocking: bool = False,
        log_minutes: int = 5,
    ):
        """
        近傍グラフの更新が停止するまで更新を続け、結果をログに書き込む

        - - -

        Params
        ------
        crowdsourcing_task_limit: int, by default None
            クラウドソーシングに投稿できるタスク上限値
        crowdsourcing_limit_suspend: bool, by default False
            クラウドソーシング上限に達した場合に停止するか否か
        cs_worker_accuracy: float, by default None
            クラウドソーシングワーカーの回答精度
        cs_result_priority: bool, by default True
            推論値を計算する際、既存クラウドソーシング結果を優先するか否か
        """
        start_edges = WorkflowConfig.graph.num_of_edges
        calltime = datetime.now() + timedelta(minutes=log_minutes)

        if crowdsourcing_task_limit is None:
            crowdsourcing_task_limit = sys.maxsize

        # エッジがなくなるまで縮約処理
        self.logger.info("Start neighbor graph contraction.")
        self.logger.info("init number of edges: {}".format(start_edges))

        while WorkflowConfig.graph.num_of_edges > 0:
            # クラウドソーシングが上限値に達しているかを検査
            crowdsourcing_enable = not (
                self.config.crowdsourcing_count >= self.config.crowdsourcing_limit
                or self.config.crowdsourcing_task_count >= crowdsourcing_task_limit
            )

            # クラウドソーシングが上限値に達した場合は一時停止する
            if not crowdsourcing_enable and crowdsourcing_limit_suspend:
                self.logger.info("Suspend neighbor graph contraction due to crowdsourcing task limit reached.")
                break

            # グラフを更新する。Falseが返ってきた場合は、クラウドソーシングを要求する
            if not self.update_ngraph(
                crowdsourcing_enable,
                cs_worker_accuracy,
                cs_result_priority,
                title_blocking,
            ):
                break

            if datetime.now() > calltime:
                num_of_edges = WorkflowConfig.graph.num_of_edges
                self.logger.info(
                    "Number of remaining edges: {} / {} ({}%)".format(
                        num_of_edges, start_edges, int((start_edges - num_of_edges) * 100 / start_edges)
                    )
                )
                calltime += timedelta(minutes=log_minutes)

        self.logger.info("Finished neighbor graph contraction.")

        self.output_log_ngraph_result()  # 結果をログに出力する

        # クラウドソーシングキューに残っているタスクがある場合は一時停止しクラウドソーシングを要求する
        if len(self.config.crowdsourcing_queue) > 0:
            return WorkflowState(suspend=True, required_crowdsourcing=True)
        else:
            return

    def __draw_distribution_in_current_inference(
        self,
        data: np.ndarray,
        cs_data: np.ndarray,
        params: "list[list[BetaParams] | list[GammaParams] | list[GaussianParams]]",
        title: str = "Fit probability density function",
        axs: Axes = None,
    ):
        """
        フィッティング結果を与えられたaxsにグラフ化して格納する

        - - -

        Params
        ------
        data: np.ndarray
            フィッティングするデータ
        params: list[BetaParams] | list[GammaParams] | list[GaussianParams]
            フィッティング結果のパラメータ集合
        title: str
            グラフタイトル
        axs: Axes, by default None
            グラフを格納するAxes
        """

        # 念のため変換
        data = np.array(data)

        ax1 = axs
        ax2 = None if len(params) == 0 else ax1.twinx()

        ax1.hist(
            (cs_data, data),
            color=("#d0a0d0", "#c0c0c0"),
            histtype="barstacked",
            range=(0, max(np.append(data, [1]))),
            bins=50,
            label=(f"Crowdsourcing ({len(cs_data)})", f"Only Inference ({len(data)})"),
        )  # ヒストグラムの追加
        ax1.set_xlabel(r"Similarity ($P(Match | \mathbf{x}_{i,j})$)" if len(params) == 0 else "Distance")
        ax1.set_ylabel("Number of pairs")

        if len(params) > 0:
            ax2.set_ylabel(r"$p(x_{i,j})$")
        else:
            ax1.legend(loc=0)

        max_height = 0
        x = np.linspace(0, max(np.append(data, [1])), 601)

        for p in params:
            x = np.linspace(0, max(np.append(data, [1])), 601)

            if p[0].type == ParamsType.BETA:
                value = stats.beta.pdf(x, *p[0].params_for_draw()[0])
                ax2.plot(x, value, color="#FF4040", label="Beta")

            elif p[0].type == ParamsType.GAMMA:
                value = stats.gamma.pdf(x, *p[0].params_for_draw()[0])
                ax2.plot(x, value, color="#FF4040", label="Gamma")

            elif p[0].type == ParamsType.GAUSSIAN:
                value = 0
                for i in p:
                    value += i.weight * (np.exp(-((x - i.mean) ** 2) / (2 * i.cov)) / (np.sqrt(2 * np.pi * i.cov)))
                ax2.plot(x, value, color="#FF4040", label="Gaussian" if len(p) == 1 else "Mixed Gaussian")

            max_height = max(max_height, value.max())

        if len(params) > 0:
            ax2.legend(loc=0)
            ax2.set_ylim(0, min(10, max_height) * 1.1)

        ax1.set_title(title)

    def verify_current_inference(
        self,
        cs_result_priority: bool = True,
        image_title: str = None,
        image_dirpath: str = None,
    ):
        """
        同定済みペアと近傍グラフ内のペアに対して、推論結果を算出し統計を出力する (縮約操作をしない精度計測)

        - - -

        Params
        ------
        cs_result_priority: bool, by default False
            推論値を計算する際、既存クラウドソーシング結果を優先するか否か
        image_title: str, by default None
            分布画像のタイトル
        image_dirpath: str, by default None
            分布画像の保存先ディレクトリ
        """
        # TODO: re_containerのverify_record_all_pairsに一部置き換えできないかを検討

        # start_edges = WorkflowConfig.graph.num_of_edges
        LOG_MINUTES = 5
        calltime = datetime.now() + timedelta(minutes=LOG_MINUTES)

        target_edges = WorkflowConfig.graph.get_all_edges()
        # num_of_edges = 0
        inference_result = ContractionResult()
        all_result = ContractionResult()

        # ========== 下処理 ==========
        # 全体の総ペアを計算
        all_pairs = len(self.re_container.records) * (len(self.re_container.records) - 1) // 2

        # 正解一致総ペアを計算
        num_of_pairs = {}
        all_correct_pairs = 0
        for key, value in self.re_container.clusters.items():
            num_of_pairs[len(value)] = num_of_pairs.get(len(value), 0) + 1

        for key, value in num_of_pairs.items():
            if key > 1:
                all_correct_pairs += value * (key) * (key - 1) // 2

        # ========== 推論結果の検証 ==========
        # ---------- 縮約済みペアに対する推論結果を計算 ----------
        collection_idx: "list[set[int, int]]" = []
        c_unique = np.unique(WorkflowConfig.graph.collection)
        c_target = [list(np.where(WorkflowConfig.graph.collection == c)[0]) for c in c_unique]
        for c in c_target:
            if len(c) > 1:
                collection_idx += [(c[i], c[j]) for i in range(len(c) - 1) for j in range(i + 1, len(c))]

        for c in collection_idx:
            for i in range(len(c) - 1):
                for j in range(i + 1, len(c)):
                    # 縮約済みであるから、全て positive として計算
                    correct_result = self.re_container.verify_record_pairs(c[i], c[j])
                    if correct_result:
                        inference_result.true_positive += 1
                    else:
                        inference_result.false_positive += 1

        # ---------- 近傍グラフに対する推論結果を計算 ----------
        counter = 0
        record_array_pair = {"fasttext": [], "difflib": [], "leven": [], "jaro": [], "inf": []}
        record_array_not = {"fasttext": [], "difflib": [], "leven": [], "jaro": [], "inf": []}
        record_array_pair_cs = {"fasttext": [], "difflib": [], "leven": [], "jaro": [], "inf": []}
        record_array_not_cs = {"fasttext": [], "difflib": [], "leven": [], "jaro": [], "inf": []}

        for a, b in target_edges:
            value = None  # 推論値

            # 正解の取得
            correct_result = self.re_container.verify_record_pairs(a, b)

            # クラウドソーシング結果を優先し、かつ結果が存在する場合は、それを推論値として取得
            if cs_result_priority and (a, b) in self.config.crowdsourcing_result:
                value = sum(self.config.crowdsourcing_result[(a, b)]) / len(self.config.crowdsourcing_result[(a, b)])

                _, _, __sim = self.update_pairdata(a, b)
                if value >= 0.5:
                    record_array_pair_cs["fasttext"].append(__sim[0])
                    record_array_pair_cs["difflib"].append(__sim[1])
                    record_array_pair_cs["leven"].append(__sim[2])
                    record_array_pair_cs["jaro"].append(__sim[3])
                    record_array_pair_cs["inf"].append(value)
                else:
                    record_array_not_cs["fasttext"].append(__sim[0])
                    record_array_not_cs["difflib"].append(__sim[1])
                    record_array_not_cs["leven"].append(__sim[2])
                    record_array_not_cs["jaro"].append(__sim[3])
                    record_array_not_cs["inf"].append(value)

            # クラウドソーシング結果が存在しない場合は、確率密度関数から得られる値を推論値として取得
            else:
                # ペアに対する推論値が未計算の場合は、推論を実行
                _, _, __sim = self.update_pairdata(a, b)

                _inf_same = self.config.pairdata[(a, b)].inf_same
                _inf_not = self.config.pairdata[(a, b)].inf_not
                value = _inf_same / max((_inf_same + _inf_not), 0.0001)

                if correct_result:
                    record_array_pair["fasttext"].append(__sim[0])
                    record_array_pair["difflib"].append(__sim[1])
                    record_array_pair["leven"].append(__sim[2])
                    record_array_pair["jaro"].append(__sim[3])
                    record_array_pair["inf"].append(value)
                else:
                    record_array_not["fasttext"].append(__sim[0])
                    record_array_not["difflib"].append(__sim[1])
                    record_array_not["leven"].append(__sim[2])
                    record_array_not["jaro"].append(__sim[3])
                    record_array_not["inf"].append(value)

            # 推論値からあてはまる統計に加算
            # positive
            if value >= 0.5:
                if correct_result:
                    inference_result.true_positive += 1
                else:
                    inference_result.false_positive += 1

            # negative
            else:
                if correct_result:
                    inference_result.false_negative += 1
                else:
                    inference_result.true_negative += 1

            counter += 1

            # 定期ログ
            if datetime.now() > calltime:
                msg = "Number of remaining edges: {} / {} ({}%)".format(
                    counter,
                    len(target_edges),
                    int(counter * 100 / len(target_edges)),
                )
                self.logger.info(msg)
                calltime += timedelta(minutes=LOG_MINUTES)

        # 全ペアに対する推論結果を保持
        all_result.true_positive = inference_result.true_positive
        all_result.false_positive = inference_result.false_positive
        all_result.false_negative = all_correct_pairs - inference_result.true_positive
        all_result.true_negative = all_pairs - all_correct_pairs - inference_result.false_positive

        # Potential Recallを使ったF1値の計算
        potential_recall, _, demmed_potential_recall, _ = WorkflowConfig.graph.get_potential_recall(
            self.re_container.get_all_match_pairs_index(),
            self.demmed_pairs,
        )
        potential_recall_weight = potential_recall * inference_result.calc_recall()
        f1_with_potential_recall = (
            2
            * all_result.calc_precision()
            * potential_recall_weight
            / (all_result.calc_precision() + potential_recall_weight)
        )

        demmed_potential_recall = demmed_potential_recall if demmed_potential_recall is not None else -1
        demmed_potential_recall_weight = (
            demmed_potential_recall * inference_result.calc_recall() if demmed_potential_recall >= 0 else -1
        )
        f1_with_demmed_potential_recall = (
            (
                2
                * all_result.calc_precision()
                * demmed_potential_recall_weight
                / (all_result.calc_precision() + demmed_potential_recall_weight)
            )
            if demmed_potential_recall_weight >= 0
            else -1
        )

        # 結果をログに保存
        result = f"""VERIFY CURRENT INFERENCE:
[[ Inference Result ]]
                      | inference                         |
                      | positive        | negative        |
-----------------------------------------------------------
   correct |    match | {inference_result.true_positive:>6d} ({all_result.true_positive:>6d}) | {inference_result.false_negative:>6d} ({all_result.false_negative:>6d}) |
           | mismatch | {inference_result.false_positive:>6d} ({all_result.false_positive:>6d}) | {inference_result.true_negative:>6d} ({all_result.true_negative:>6d}) |
(n) ... Includes edge removal by blocking methods.

[[ Inference Summary ]]
                 Precision: {inference_result.calc_precision():.8f} ({all_result.calc_precision():.8f})
                    Recall: {inference_result.calc_recall():.8f} ({all_result.calc_recall():.8f})
                        F1: {inference_result.calc_f1():.8f} ({all_result.calc_f1():.8f})
          Potential recall:          - ({potential_recall:.8f})
F1 (with Potential recall):          - ({f1_with_potential_recall:.8f})
   Demmed Potential recall:          - ({demmed_potential_recall:.8f})
  F1 (with Demmed PRecall):          - ({f1_with_demmed_potential_recall:.8f})
"""

        self.logger.info(result)

        # ---------- 分布画像を出力 ----------
        if image_title is not None and image_dirpath is not None:

            original_data = [
                np.array(record_array_pair["fasttext"]),
                np.array(record_array_not["fasttext"]),
                np.array(record_array_pair["difflib"]),
                np.array(record_array_not["difflib"]),
                np.array(record_array_pair["leven"]),
                np.array(record_array_not["leven"]),
                np.array(record_array_pair["jaro"]),
                np.array(record_array_not["jaro"]),
                np.array(record_array_pair["inf"]),
                np.array(record_array_not["inf"]),
            ]

            original_data_cs = [
                np.array(record_array_pair_cs["fasttext"]),
                np.array(record_array_not_cs["fasttext"]),
                np.array(record_array_pair_cs["difflib"]),
                np.array(record_array_not_cs["difflib"]),
                np.array(record_array_pair_cs["leven"]),
                np.array(record_array_not_cs["leven"]),
                np.array(record_array_pair_cs["jaro"]),
                np.array(record_array_not_cs["jaro"]),
                np.array(record_array_pair_cs["inf"]),
                np.array(record_array_not_cs["inf"]),
            ]

            params = self.inference.params_container.get_params_list()
            params = params + [None, None]

            axs = None

            plt.clf()  # 初期化
            figure, axs = plt.subplots(2, 5, figsize=(24, 8))
            figure.suptitle(image_title)

            # 画像の生成
            distribution_title = ["fasttext", "difflib", "leven", "jaro", "inference"]
            i = 0
            for d, c, j in zip(original_data, original_data_cs, params):
                self.__draw_distribution_in_current_inference(
                    d,
                    c,
                    [j] if j is not None else [],
                    f"{distribution_title[i // 2]} - {'mismatch' if i & 1 else 'match'}",
                    axs[i % 2, i // 2],
                )
                i += 1

            # ディレクトリ作成
            if not path.exists(image_dirpath):
                os.mkdir(image_dirpath)

            # ファイル名生成
            filepath = None
            for i in range(1, 10000):
                tmpname = path.join(image_dirpath, "static-{:0=4}.png".format(i))
                if not path.exists(tmpname):
                    filepath = tmpname
                    break

            # 既に規定数以上のファイルが保存されている場合はエラーを出す
            if filepath is None:
                raise FileExistsError("Cannot save PNG file because there are already too many files (over 10000).")

            # 画像の出力
            plt.tight_layout()
            plt.legend()
            plt.savefig(filepath)

    def verify_worker_accuracy(self):
        """クラウドソーシングワーカの回答精度を測定する"""

        # 精度を集計
        result = ContractionResult()  # 各回答ごとの結果
        pair_result = ContractionResult()  # 各ペアごとの結果
        dp = {-1: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

        for k, answers in self.config.crowdsourcing_result.items():
            correct = self.re_container.verify_record_pairs(k[0], k[1])
            dp[len(answers)] = dp.get(len(answers), 0) + 1
            if len(answers) >= 10:
                dp[-1] += 1

            # 各回答ごとの結果を集計
            for a in answers:
                if a >= 0.5:
                    if correct:
                        result.true_positive += 1
                    else:
                        result.false_positive += 1

                else:
                    if correct:
                        result.false_negative += 1
                    else:
                        result.true_negative += 1

            # ペアごとの結果を集計
            if sum(answers) / len(answers) >= 0.5:
                if correct:
                    pair_result.true_positive += 1
                else:
                    pair_result.false_positive += 1

            else:
                if correct:
                    pair_result.false_negative += 1
                else:
                    pair_result.true_negative += 1

        dp_sum = sum(dp.values())

        self.logger.info(f"Duplication: {dp}")

        # 結果をログに保存
        result = f"""
CROWD WORKER ACCURACY:
[[ Task Result ]]
                     | answer                |
                     | positive  | negative  |
----------------------------------------------
   correct |   match | {result.true_positive:>9d} | {result.false_negative:>9d} |
            mismatch | {result.false_positive:>9d} | {result.true_negative:>9d} |

    total: {result.sum()}
 accuracy: {result.true_positive + result.true_negative} / {result.sum()} ( {(result.true_positive + result.true_negative) / result.sum() * 100 if result.sum() > 0 else 0} % )
precision: {result.calc_precision()}
   recall: {result.calc_recall()}
       f1: {result.calc_f1()}

[[ Pair Result ]]
                     | answer                |
                     | positive  | negative  |
----------------------------------------------
   correct |   match | {pair_result.true_positive:>9d} | {pair_result.false_negative:>9d} |
            mismatch | {pair_result.false_positive:>9d} | {pair_result.true_negative:>9d} |

    total: {pair_result.sum()}
 accuracy: {pair_result.true_positive + pair_result.true_negative} / {pair_result.sum()} ( {(pair_result.true_positive + pair_result.true_negative) / pair_result.sum() * 100 if result.sum() > 0 else 0} % )
precision: {pair_result.calc_precision()}
   recall: {pair_result.calc_recall()}
       f1: {pair_result.calc_f1()}

[[ Duplication ]]
10 and more | {'@' * int(30*(dp[-1])/dp_sum) if dp_sum > 0 else 0} {dp[-1]}
          9 | {'@' * int(30*(dp[9])/dp_sum) if dp_sum > 0 else 0} {dp[9]}
          8 | {'@' * int(30*(dp[8])/dp_sum) if dp_sum > 0 else 0} {dp[8]}
          7 | {'@' * int(30*(dp[7])/dp_sum) if dp_sum > 0 else 0} {dp[7]}
          6 | {'@' * int(30*(dp[6])/dp_sum) if dp_sum > 0 else 0} {dp[6]}
          5 | {'@' * int(30*(dp[5])/dp_sum) if dp_sum > 0 else 0} {dp[5]}
          4 | {'@' * int(30*(dp[4])/dp_sum) if dp_sum > 0 else 0} {dp[4]}
          3 | {'@' * int(30*(dp[3])/dp_sum) if dp_sum > 0 else 0} {dp[3]}
          2 | {'@' * int(30*(dp[2])/dp_sum) if dp_sum > 0 else 0} {dp[2]}
          1 | {'@' * int(30*(dp[1])/dp_sum) if dp_sum > 0 else 0} {dp[1]}
"""

        self.logger.info(result)

    def verify_all_ngraph(self):
        """[Deprecated] 近傍グラフについてすべてのエッジの推論結果を検証する (縮約操作をしない精度計測)"""

        start_edges = WorkflowConfig.graph.num_of_edges
        LOG_MINUTES = 10
        calltime = datetime.now() + timedelta(minutes=LOG_MINUTES)

        target_edges = WorkflowConfig.graph.get_all_edges()
        num_of_edges = 0
        inference_result = ContractionResult()

        # 総ペアを計算
        all_pairs = len(self.re_container.records) * (len(self.re_container.records) - 1) // 2

        # 一致総ペアを計算
        num_of_pairs = {}
        all_correct_pairs = 0
        for key, value in self.re_container.clusters.items():
            num_of_pairs[len(value)] = num_of_pairs.get(len(value), 0) + 1

        for key, value in num_of_pairs.items():
            if key > 1:
                all_correct_pairs += value * (key) * (key - 1) // 2

        for a, b in target_edges:
            # ペアに対する推論値が未計算の場合は推論を実行
            if (a, b) not in self.config.pairdata or self.config.pairdata[(a, b)].inf_same is None:
                self.update_pairdata(a, b)

            # 推論値からあてはまる統計に加算
            correct_result = self.re_container.verify_record_pairs(a, b)
            # positive
            if self.config.pairdata[(a, b)].inf_same > self.config.pairdata[(a, b)].inf_not:
                if correct_result:
                    inference_result.true_positive += 1

                else:
                    inference_result.false_positive += 1

            # negative
            else:
                if correct_result:
                    inference_result.false_negative += 1

                else:
                    inference_result.true_negative += 1

            # 定期ログ
            if datetime.now() > calltime:
                num_of_edges = WorkflowConfig.graph.num_of_edges
                self.logger.info(
                    "Number of remaining edges: {} / {} ({}%)".format(
                        num_of_edges, start_edges, int((start_edges - num_of_edges) * 100 / start_edges)
                    )
                )
                calltime += timedelta(minutes=LOG_MINUTES)

        # 結果をログに保存
        result = [
            "",
            "VERIFY ALL NGRAPH: ",
            "[[ Inference Result ]]",
            "                      | inference                         |",
            "                      | positive        | negative        |",
            "-----------------------------------------------------------",
            "   correct |    match | {:>6d} ({:>6d}) | {:>6d} ({:>6d}) |".format(
                inference_result.true_positive,
                inference_result.true_positive,
                inference_result.false_negative,
                all_correct_pairs - inference_result.true_positive,
            ),
            "             mismatch | {:>6d} ({:>6d}) | {:>6d} ({:>6d}) |".format(
                inference_result.false_positive,
                inference_result.false_positive,
                inference_result.true_negative,
                all_pairs - all_correct_pairs - inference_result.false_positive,
            ),
            " * (n) ... ",
            "",
            "[[ Inference Summary ]]",
            "precision: {}".format(inference_result.calc_precision()),
            "   recall: {}".format(inference_result.calc_recall()),
            "       f1: {}".format(inference_result.calc_f1()),
            "",
        ]

        # ログ出力
        self.logger.info("\n".join(result))

    def output_log_ngraph_result(self):
        """縮約結果をログに出力する"""

        # 縮約結果から推論群の検証
        g_ev, g_complete = self.re_container.verify_all_record_group(WorkflowConfig.graph.collection)
        g_ev_calc = g_ev.calc_evaluation()

        # 縮約結果からペアの検証
        p_ev = self.re_container.verify_all_record_pairs(WorkflowConfig.graph.collection)

        # 計算機同定数とクラウドソーシング問い合わせ数を計算
        machine_count = self.config.contraction_machine_result.sum()
        human_count = self.config.contraction_result.sum() - machine_count

        # 結果をログに保存
        result = "\n"
        result += "[[ Inference Result ]]\n"
        result += "                      | inference                         |\n"
        result += "                      | contract        | remove          |\n"
        result += "-----------------------------------------------------------\n"
        result += "   correct | contract | {:>6d} ({:>6d}) | {:>6d} ({:>6d}) |\n".format(
            self.config.contraction_result.true_positive,
            self.config.contraction_machine_result.true_positive,
            self.config.contraction_result.false_negative,
            self.config.contraction_machine_result.false_negative,
        )
        result += "               remove | {:>6d} ({:>6d}) | {:>6d} ({:>6d}) |\n".format(
            self.config.contraction_result.false_positive,
            self.config.contraction_machine_result.false_positive,
            self.config.contraction_result.true_negative,
            self.config.contraction_machine_result.true_negative,
        )
        result += " * (n) ... mechanical judgment\n"
        result += "\n"
        result += "[[ Inference Summary ]]\n"
        result += "precision: {}\n".format(self.config.contraction_result.calc_precision())
        result += "   recall: {}\n".format(self.config.contraction_result.calc_recall())
        result += "       f1: {}\n".format(self.config.contraction_result.calc_f1())
        result += "\n"
        result += "[[ machine and human ]]\n"
        result += "         | machine |  human  |\n"
        result += "------------------------------\n"
        result += "   count | {:>7d} | {:>7d} |\n".format(machine_count, human_count)
        result += "\n"
        result += "[[ Group Evaluation ]]\n"
        result += " precision | {:.5} ( {} / {} )\n".format(
            g_ev_calc.precision_nu, g_ev.precision_nu, g_ev.precision_de
        )
        result += "    recall | {:.5} ( {} / {} )\n".format(g_ev_calc.recall_nu, g_ev.recall_nu, g_ev.recall_de)
        result += "  complete | {:.5} ( {} / {} )\n".format(g_ev_calc.complete_nu, g_ev.complete_nu, g_ev.complete_de)
        result += "           | {}\n".format(g_complete)
        result += "\n"
        result += "[[ Pair Evaluation ]]\n"
        result += "                      | inference           |\n"
        result += "                      | match    | mismatch |\n"
        result += "-----------------------------------------------------------\n"
        result += "   correct |    match | {:>8d} | {:>8d} |\n".format(p_ev.true_positive, p_ev.false_negative)
        result += "             mismatch | {:>8d} | {:>8d} |\n".format(p_ev.false_positive, p_ev.true_negative)
        result += "\n"
        result += "precision: {}\n".format(p_ev.calc_precision())
        result += "   recall: {}\n".format(p_ev.calc_recall())
        result += "       f1: {}\n".format(p_ev.calc_f1())
        result += "\n"
        result += "[[ Word memo ]]\n"
        result += "contract ... エッジ縮約\n"
        result += "remove ... エッジ削除\n"
        result += "precision ... 適合率\n"
        result += "recall ... 再現率\n"

        self.logger.info(result)

    def generate_crowdsourcing_tasks_by_dist(
        self,
        match_count: int = 2000,
        mismatch_count: int = 1000,
        match_min_dist: float = None,
        match_max_dist: float = None,
        mismatch_min_dist: float = None,
        mismatch_max_dist: float = None,
        random: bool = True,
        exclude_complete_match_pairs: bool = True,
        max_cores: int = None,
        log_minutes: int = 5,
    ):
        """
        [EXP:N4U] 距離を利用して、クラウドソーシングタスクを生成する

        - - -

        Params
        ------
        match_count: int, by default 2000
            一致ペアの数
        mismatch_count: int, by default 1000
            不一致ペアの数
        match_min_dist: float, by default None
            一致ペアの最小距離
        match_max_dist: float, by default None
            一致ペアの最大距離
        mismatch_min_dist: float, by default None
            不一致ペアの最小距離
        mismatch_max_dist: float, by default None
            不一致ペアの最大距離
        random: bool, by default True
            ランダムに生成するかどうか
        exclude_complete_match_pairs: bool, by default True
            完全一致ペアを除外するかどうか
        max_cores: int, by default None
            並列処理時の最大コア数
        log_minutes: int, by default 5
            ログ出力間隔
        """
        # 初期値追加
        match_min_dist = match_min_dist if match_min_dist is not None else 0
        match_max_dist = match_max_dist if match_max_dist is not None else 1000000
        mismatch_min_dist = mismatch_min_dist if mismatch_min_dist is not None else 0
        mismatch_max_dist = mismatch_max_dist if mismatch_max_dist is not None else 1000000

        # 読み込みが完了していない場合 re_container にファイルを読み込ませる
        if len(self.record) == 0:
            self.re_container.load_file(path.abspath(self.config.target_filepath))
            self.record = self.re_container.get_recordmg()

        # 各ペアの距離を計算する
        # recordからそれぞれをベクトルに変換し、行列化する (multiprocessによる並列化)
        self.logger.info("Start make_fasttext_dist_matrix.")
        record_array = self.inference.get_fasttext_vectors_with_mp(
            CacheMode.WRITE,
            self.re_container.filepath,
            self.record,
            max_cores,
            log_minutes,
        )
        self.logger.info("Finished make_fasttext_dist_matrix.")

        # 距離学習機によりベクトルを変換する
        converted_matrix = self.inference.convert_vector_by_metric_learner(np.array(record_array))

        self.logger.info("Start initializing faiss index.")

        # faissにより距離を計算するためのインデックス構築
        # TODO: FlatL2(総当り)以外にIVF(ボロノイを利用した近似最近傍探索)もあるため、選択できるようにする
        # ただし、現状 Intel MKL FATAL ERROR: Cannot load libmkl_avx2.so or libmkl_def.so. エラーが出てる
        # quatizer = faiss.IndexFlatL2(len(converted_matrix[0]))
        # faiss_index = faiss.IndexIVFFlat(quatizer, len(converted_matrix[0]), 1)
        # faiss_index.train(converted_matrix)
        faiss_index = faiss.IndexFlatL2(len(converted_matrix[0]))

        # GPUが利用できる場合は、GPU利用のためのインデックス実装に切り替える
        if self.config.gpu_status.faiss:
            res = faiss.StandardGpuResources()
            self.logger.info(">>> Done: faiss.StandardGpuResources")
            faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
            self.logger.info(">>> Done: faiss.index_cpu_to_gpu")

        self.logger.info("Start adding faiss vectors.")

        # インデックスにベクトルを追加する
        faiss_index.add(converted_matrix)

        # 探索
        distances, indices = faiss_index.search(converted_matrix, min(len(converted_matrix), 2048))
        # faiss の distances は、2乗のままなので sqrt をかける
        distances = np.sqrt(distances)

        # 一致と不一致ペアをそれぞれ取得する
        match_indices = []
        mismatch_indices = []

        # 条件を満たすペアを取得
        for i in range(len(indices)):
            for j in range(len(indices[i])):
                judge = self.re_container.verify_record_pairs(i, indices[i][j])

                if judge and distances[i][j] >= match_min_dist and distances[i][j] <= match_max_dist:
                    if exclude_complete_match_pairs and self.record[i] == self.record[indices[i][j]]:
                        continue

                    match_indices.append(tuple(sorted((i, indices[i][j]))))

                elif not judge and distances[i][j] >= mismatch_min_dist and distances[i][j] <= mismatch_max_dist:
                    mismatch_indices.append(tuple(sorted((i, int(indices[i][j])))))

        # 統計情報の書き込み
        self.logger.info(
            f"""GENERATE_CROWDSOURCING_TASKS_BY_DISTANCE statics
   Match Count: {len(match_indices)}   ({match_min_dist} <= distance <= {match_max_dist})
Mismatch Count: {len(mismatch_indices)}   ({mismatch_min_dist} <= distance <= {mismatch_max_dist})"""
        )

        # 実際に出力するペアを決定する
        result = []
        if random:
            # 一致ペアと不一致ペアを各々でシャッフルする
            __match_target = match_indices
            __mismatch_target = mismatch_indices
            shuffle(__match_target)
            shuffle(__mismatch_target)

            # 一致ペアと不一致ペアをそれぞれ指定数だけ取得し、もう一度シャッフルする
            result = __match_target[:match_count] + __mismatch_target[:mismatch_count]
            shuffle(result)

        else:
            # 一致ペアと不一致ペアをそれぞれ指定数だけ取得する
            result = match_indices[:match_count] + mismatch_indices[:mismatch_count]

        # クラウドソーシングキューに追加する
        self.config.crowdsourcing_queue.extend(result)

        # クラウドソーシングを要求する
        return WorkflowState(suspend=True, finished=True, required_crowdsourcing=True)

    def request_crowdsourcing(
        self,
        indices: list[int],
        cs_worker_accuracy: float = 1.0,
        reuse: bool = True,
    ) -> dict[tuple[int, int], float] | None:
        """
        クラウドソーシングで問い合わせを行うペアについて、結果を返したりキューに蓄積したりする

        - - -

        Params
        ----------
        indices: list[int]
            レコードのインデックス
        cs_worker_accuracy: float, by default 1.0
            クラウドソーシングワーカーの回答精度

        Return
        ------
        float | None
            ワーカの回答の平均値を返す。回答を待機する場合は None を返す
        """

        indices = tuple(sorted(list(indices)))
        idx_p = [(indices[i], indices[j]) for i in range(len(indices) - 1) for j in range(i + 1, len(indices))]

        # すでに問い合わせが完了しているものについては結果を返す
        if reuse:
            ext = True
            for i in idx_p:
                ext = ext & (i in self.config.crowdsourcing_queue)

            if ext:
                result = {}
                for i in idx_p:
                    result[i] = self.config.crowdsourcing_result[i]
                return result

        # コストを加算する
        self.config.crowdsourcing_count += len(idx_p)
        self.config.crowdsourcing_task_count += len(idx_p)

        # シミュレーションの場合は正解を取得し結果を返す
        if self.config.crowdsourcing_platform == CSPlatform.SIMULATION:
            worker_condition = 0 if random() < cs_worker_accuracy else 1  # ワーカーの回答精度を考慮
            result = {}

            for i in idx_p:
                answer = abs(worker_condition - (1 if self.re_container.verify_record_pairs(*i) else 0))
                self.config.crowdsourcing_result[i] = self.config.crowdsourcing_result.get(i, [])
                self.config.crowdsourcing_result[i].append(answer)
                result[i] = sum(self.config.crowdsourcing_result[i]) / len(self.config.crowdsourcing_result[i])

            return result

        # 実験用
        elif self.config.crowdsourcing_platform == CSPlatform.EXP_WITH_INF:
            worker_condition = 0 if random() < cs_worker_accuracy else 1  # ワーカーの回答精度を考慮
            result: dict[tuple[int, int], float] = {}

            for i in idx_p:
                answer = abs(worker_condition - (1 if self.re_container.verify_record_pairs(*i) else 0))
                self.config.crowdsourcing_result[i] = self.config.crowdsourcing_result.get(i, [])
                self.config.crowdsourcing_result[i].append(answer)
                result[i] = sum(self.config.crowdsourcing_result[i]) / len(self.config.crowdsourcing_result[i])

            self.config.crowdsourcing_queue.append(indices)

            return result

        # 実験用 (CIKM2023)
        elif self.config.crowdsourcing_platform == CSPlatform.EXP_CIKM2023:
            worker_condition = 0 if random() < cs_worker_accuracy else 1  # ワーカーの回答精度を考慮
            result = {}

            for i in idx_p:
                result[i] = abs(worker_condition - (1 if self.re_container.verify_record_pairs(*i) else 0))

            self.config.crowdsourcing_queue.append(indices)
            return result

        # シミュレーション以外のクラウドソーシングプラットフォームを利用する
        else:
            self.config.crowdsourcing_queue.append(indices)
            return None

    def output_result_yaml(self, filepath: str = "result.yaml"):
        """
        現状の縮約結果をyaml形式で出力する

        - - -

        Params
        ------
        filepath: str, by default 'result.yaml'
            出力先のファイルパス
        """
        self.re_container.save_yaml_for_result(self.config, filepath)

    def config_reflection(self, filepath: str, force: bool = False, max_cores: int = None, log_minutes: int = 5):
        """
        [Project] configの内容をインスタンス変数に反映させる

        - - -

        Params
        ------
        filepath: str
            書誌データが格納されたファイルパス
        force: bool, by default False
            読み込み済みの場合でも強制的に読み込む
        max_cores: int, by default None
            並列処理時の最大コア数
        log_minutes: int, by default 5
            ログ出力間隔
        """

        # すでにrecord_containerに読み込み済みの場合は処理を行わない
        if len(self.record) == 0 or force:
            # 距離を取得するためにfasttext用の行列を作成し、正解書誌データインデックスを作成する
            self.re_container.load_file(filepath)
            self.record = self.re_container.get_recordmg()

        # Pairdataを構築する
        if len(self.config.pairdata) == 0 or force:
            if self.config.ngraph_construction == NGraphConstructMode.DIVERSITY_COMPRESSED:
                self.generate_mgraph_and_pairdata_in_diversity(
                    reflect_mgraph=False,
                    reflect_pairdata=True,
                    max_cores=max_cores,
                    contract_edges=False,
                    log_minutes=log_minutes,
                )
            elif self.config.ngraph_construction == NGraphConstructMode.SYNERGY:
                self.generate_mgraph_and_pairdata_in_synergy(
                    reflect_mgraph=False,
                    reflect_pairdata=True,
                    max_cores=max_cores,
                    contract_edges=False,
                    log_minutes=log_minutes,
                )
            else:
                self.generate_mgraph_and_pairdata(
                    reflect_mgraph=False,
                    reflect_pairdata=True,
                    max_cores=max_cores,
                    log_minutes=log_minutes,
                )

    def add_crowdsourcing_for_retraining(self, filepath: str, same: int, not_same: int):
        """
        [Deprecated/Experiment] 再学習用にクラウドソーシング結果を人為的に追加する

        Params
        ------
        filepath: str
            書誌データが格納されたyaml/jsonファイル
        same: int
            一致のデータ数
        not_same: int
            不一致のデータ数
        """

        if len(self.re_container.records) == 0:
            self.load_file(filepath, True)

        same_pair, not_same_pair = self.re_container.get_pairdata(same, not_same)

        for id_1, id_2 in same_pair:
            self.config.crowdsourcing_result[(min(id_1, id_2), max(id_1, id_2))] = True

        for id_1, id_2 in not_same_pair:
            self.config.crowdsourcing_result[(min(id_1, id_2), max(id_1, id_2))] = False

    def add_crowdsourcing_by_inference(self, filepath: str, type: str):
        """
        [Experiment] クラウドソーシングの結果を推論結果に基づいて追加する (SUN推論)

        Params
        ------
        filepath: str
            書誌データを格納したyamlのファイルパス
        type: str
            対象とする推論結果 "true_positive" or "false_positive" or "false_negative" or "true_negative"
        """

        # 要求された型に対して条件を定義する
        if type == "true_positive":
            target = ["Same", True]
        elif type == "false_positive":
            target = ["Same", False]
        elif type == "false_negative":
            target = ["NotSame", True]
        elif type == "true_negative":
            target = ["NotSame", False]
        else:
            return

        # 距離を取得するためにfasttext用の行列を作成し、正解書誌データインデックスを作成する
        self.load_file(filepath, True)
        self.record = self.re_container.get_recordmg()

        # 全ての書誌データペアの距離を計算し、取得する
        # recordからそれぞれをベクトルに変換し、行列化する
        record_array = []
        for data in self.record:
            record_array.append(self.inference.make_fasttext_dist_matrix(data))

        # 距離の算出結果を格納
        pairdata: "dict[tuple[int, int], Pairdata]" = {}

        for i in range(len(record_array)):
            pairs = []

            for j in range(len(record_array)):
                pairs.append([record_array[i], record_array[j]])

            predict_result = np.ravel(self.inference.fasttext_predict(np.array(pairs)))

            record_index = np.argsort(predict_result)
            # ターゲット書誌データ(i)のインデックスを削除
            record_index = record_index[record_index != i]

            for bd_i in record_index:
                pairdata[(min(i, bd_i), max(i, bd_i))] = Pairdata(inf_dist=float(predict_result[bd_i]))

        # 時間がかかる可能性があるため、ログを出力する
        LOG_MINUTES = 10
        calltime = datetime.now() + timedelta(minutes=LOG_MINUTES)
        start_edges = len(self.record) * (len(self.record) - 1) // 2
        count = start_edges
        self.logger.info(f"target edges: {start_edges}")

        # 全ての書誌データペアを対象に検証を行う
        for i in range(len(self.record) - 1):
            for j in range(i + 1, len(self.record)):
                inf_same, _, inf_not, _, _, _, _ = self.inference.greet_sun(
                    self.record[i],
                    self.record[j],
                    pairdata[(i, j)].inf_dist,
                )
                inf_unknown = 1 - inf_same - inf_not
                inf = (
                    "Same"
                    if inf_same > inf_not + inf_unknown
                    else "NotSame" if inf_not > inf_same + inf_unknown else "Unknown"
                )
                correct = self.re_container.verify_record_pairs(i, j)

                # 条件に一致すればクラウドソーシングキューに追加する
                if target[0] == inf and target[1] == correct:
                    id_1, id_2 = self.record[i].re_origin.id, self.record[j].re_origin.id
                    self.config.crowdsourcing_queue.append(tuple(sorted([id_1, id_2])))

                count -= 1

                # 一定時間を超えた場合にログを出力する
                if datetime.now() > calltime:
                    self.logger.info(
                        "Number of remaining edges: {} / {} ({}%)".format(
                            count, start_edges, int((start_edges - count) * 100 / start_edges)
                        )
                    )
                    calltime += timedelta(minutes=LOG_MINUTES)

        return WorkflowState(suspend=True, finished=True, required_crowdsourcing=True)

    def add_crowdsourcing_by_ngraph(self, order_type: str):
        """
        [Experiment] クラウドソーシングの結果を近傍グラフに基づいて追加する

        Params
        ------
        order_type: str
            対象とする推論結果 "order_positive" or "order_negative" or "order_unknown"
        """

        all_edges = WorkflowConfig.graph.get_all_edges()
        start_edges = len(all_edges)
        count = start_edges
        LOG_MINUTES = 10
        calltime = datetime.now() + timedelta(minutes=LOG_MINUTES)

        # エッジがなくなるまで縮約処理
        self.logger.info("Start inference by ngraph")
        self.logger.info("init number of edges: {}".format(start_edges))

        result: "list[list[tuple[int, int], float]]" = []

        for a, b in all_edges:
            # ペアに対する推論値が未計算の場合は推論を実行
            if (a, b) not in self.config.pairdata or self.config.pairdata[(a, b)].inf_same is None:
                self.update_pairdata(a, b)

            # 推論値を獲得
            prob_same, prob_unknown, prob_not = (
                self.config.pairdata[(a, b)].inf_same,
                0,
                self.config.pairdata[(a, b)].inf_not,
            )
            if self.config.inference_mode == InferenceMode.BAYESIAN:
                prob_unknown = 0.5 - abs(0.5 - prob_same)

            elif self.config.inference_mode == InferenceMode.SUN:
                prob_unknown = 1 - prob_same - prob_not

            # オーダー順によって値を格納
            if order_type == "order_positive":
                result.append([(a, b), prob_same])
            elif order_type == "order_negative":
                result.append([(a, b), prob_not])
            elif order_type == "order_unknown":
                result.append([(a, b), prob_unknown])

            # 一定時間を超えた場合にログを出力する
            if datetime.now() > calltime:
                self.logger.info(
                    "Number of remaining edges: {} / {} ({}%)".format(
                        count, start_edges, int((start_edges - count) * 100 / start_edges)
                    )
                )
                calltime += timedelta(minutes=LOG_MINUTES)

        self.logger.info("Finished inference by ngraph")

        # オーダー順にソート
        result = sorted(result, key=lambda x: x[1], reverse=True)

        # クラウドソーシングキューに格納
        for i in result:
            id_1, id_2 = self.record[i[0][0]].re_origin.id, self.record[i[0][1]].re_origin.id
            self.config.crowdsourcing_queue.append(tuple(sorted([id_1, id_2])))

        return WorkflowState(suspend=True, finished=True, required_crowdsourcing=True)

    def output_misidentification_json(self, filepath: str = "machine_misidentification.json"):
        """[inspection] 計算機が誤同定した書誌データをjson形式で出力する"""
        books = []

        for a, b, prob_same, prob_not, prob_unknown in self.config.machine_misidentification_pair:
            data = []
            data.append(self.record[a].re_origin.to_dict())
            data.append(self.record[b].re_origin.to_dict())
            data.append({"prob_same": prob_same, "prob_not": prob_not, "prob_unknown": prob_unknown})

            books.append(data)

        output = {"books": books}

        # 結果をjsonファイルとして書き出す
        with codecs.open(path.join(path.dirname(path.abspath(__file__)), filepath), "w", "utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        self.logger.info("finished to output record pair. (misidentification_pair)")
