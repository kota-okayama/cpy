"""Matrix graph"""

import traceback

from numpy import arange, savetxt, loadtxt, where, zeros, array, int32, float32
from scipy.sparse import load_npz, save_npz, csr_matrix

from .utils import path
from .logger import Logger, LoggerConfig

# cudaが利用できるかどうかを判定し、利用できる場合cupyをimportする
CUDA_AVAILABLE = False


try:
    import cupy as np
    from cupyx.scipy import sparse

    if not np.cuda.is_available():
        raise ImportError()

    CUDA_AVAILABLE = True

except Exception as err:
    print(err)
    import numpy as np
    from scipy import sparse


class MGraph:
    """Cupy(Numpy)の隣接行列により表現されたグラフを管理するクラス"""

    def __init__(self, log_filepath: str = None, log_level: str = "INFO"):
        """コンストラクタ"""

        self.graph = None
        self.collection = None
        self.logger = Logger(
            __name__,
            logger_config=LoggerConfig(level=log_level),
            filepath=log_filepath,
        )

        # statics
        self.num_of_verts = 0
        self.num_of_edges = 0

    def compress(self):
        """0の要素を持つCSR形式のスパース行列を正規化と再圧縮する"""

        # === Tips ===
        # CSR/COO形式の特徴として、0の要素を代入してもメモリ内部に保持し続ける (データが削除されない)
        # self.graph.data により見ることができる
        # これらの行列は、常に順番にメモリの確保が行われるらしく、途中で削除する術がない
        # そのため新たにグラフを作成し、再構築し直すことで、高速に演算や処理を行うことができる
        # 反対に放置すると無駄な0が増えて計算量が増加し、徐々にスピードが低下していくので注意
        #
        # https://note.nkmk.me/python-scipy-sparse-get-element-row-column/
        #
        #
        # ======
        # 疎行列を取り扱える csr_matrix と coo_matrix はbool型やブール条件式をサポートしていない (lil_matrixはサポートされているっぽい)
        # よって、以下みたいな書き方はできない
        # self.graph[:, self.graph[:, 0] == 1] = 0
        #
        # そのため特定の条件を持つ行や列の値を変更したい場合は、
        # np.where や sparse.find で一度整数インデックスに変換する必要がある

        # 新たなグラフの作成とデータ移動
        new_graph = sparse.csr_matrix((self.graph.shape[0], self.graph.shape[0]), dtype=np.float32)
        new_graph[sparse.find(self.graph)[0], sparse.find(self.graph)[1]] = 1

        # メモリの明示的な解放
        del self.graph
        if CUDA_AVAILABLE:
            np.get_default_memory_pool().free_all_blocks()

        self.graph = new_graph

    def update_statics(self):
        """インスタンスが持つグラフの統計情報を更新する"""

        self.num_of_verts = len(self.get_all_vertices())
        self.num_of_edges = len(self.get_all_edges())

    def get_include_edges(self, edges: "list[tuple[int, int]]") -> "list[tuple[int, int]]":
        """与えられたエッジが含まれるエッジのリストを取得する"""

        target = np.array(edges).T
        target_graph = sparse.csr_matrix((self.graph.shape[0], self.graph.shape[0]), dtype=np.float32)
        target_graph[target[0], target[1]] = 1

        result = sparse.find(self.graph.multiply(target_graph))
        result = np.array([result[0], result[1]]).T

        return result.tolist()

    def load(self, graph_filepath: str, collection_filepath: str):
        """
        ファイルから隣接行列グラフと同定結果を読み込む

        - - -

        Params
        ------
        collection_filepath: str
            コレクション保存先のファイルパス
        graph_filepath: str
            グラフ保存先のファイルパス
        """

        if not path.exists(graph_filepath) or not path.exists(collection_filepath):
            self.logger.warning("MGraph.graph or MGraph.collection file is not found, so skip loading.")
            return

        self.graph = load_npz(graph_filepath)
        self.collection = loadtxt(collection_filepath, dtype=np.int32)

        if CUDA_AVAILABLE:
            self.graph = sparse.csr_matrix(self.graph)

        self.update_statics()

    def save(self, graph_filepath: str, collection_filepath: str):
        """
        隣接行列グラフと同定結果をファイルに保存する

        - - -

        Params
        ------
        collection_filepath: str
            コレクション保存先のファイルパス
        graph_filepath: str
            グラフ保存先のファイルパス
        """

        if self.graph is None or self.collection is None:
            self.logger.warning("MGraph.graph or MgGaph.collection is None, so skip saving.")
            return

        save_npz(graph_filepath, self.graph)
        savetxt(collection_filepath, self.collection, fmt="%u")

    def generate(self, num_of_record: int, initial_vertices: bool = False):
        """
        グラフを生成する

        - - -

        Params
        ------
        num_of_record: int
            総レコード数
        initial_vertices: boolean
            初期頂点を生成するか否か
        """

        # 行スライスの得意なCSR形式のスパース行列を採用する
        self.graph = sparse.csr_matrix((num_of_record, num_of_record), dtype=np.float32)
        self.collection = arange(num_of_record, dtype=np.int32)

        self.num_of_verts = 0

        if initial_vertices:
            # 対角要素を1にする
            diag_indices = np.diag_indices(num_of_record)
            self.graph[diag_indices] = 1
            self.num_of_verts = num_of_record

    @classmethod
    def convert_v3_0_graph(cls, num_of_record: int, graph: "dict[int : list[int]]", filepath: str):
        """
        v3.0で保持しているグラフをMGraph形式に変換して出力する

        - - -

        Params
        ------
        num_of_record: int
            総レコード数
        graph: dict[int: list[int]]
            v3.0のグラフ
        filepath: str
            保存先のファイルパス
        """

        # グラフを生成
        mgraph = cls()
        mgraph.generate(num_of_record)

        # エッジを張る
        for base_vertex, target_vertices in graph.items():
            mgraph.connect_edges(base_vertex, target_vertices)

        # グラフファイルとして保存する
        mgraph.save(filepath)

    def connect_edges(self, base_vertex: int, target_vertices: "list[int]", update_statics: bool = True):
        """
        エッジを張る

        - - -

        Params
        ------
        base_vertex: int
            基準となる頂点
        target_vertices: list[int]
            基準点の接続先となる頂点のリスト
        update_statics: bool
            グラフの統計情報を更新する
        """

        self.graph[base_vertex, base_vertex] = 1
        self.graph[base_vertex, target_vertices] = 1
        self.graph[target_vertices, base_vertex] = 1

        if update_statics:
            self.update_statics()

    def contraction_edge(self, primary: int, secondary: int, update_statics: bool = True):
        """
        エッジを縮約する

        - - -

        Params
        ------
        primary: int
            残存する頂点
        secondary: int
            縮約される頂点
        update_statics: bool, default True
            グラフの統計情報を更新する。Falseにした場合は直近でupdate_statics()を呼び出す必要がある
        """

        # 同じ頂点を参照している場合は何もしない
        if self.collection[primary] == self.collection[secondary]:
            return

        # 残存エッジに縮約エッジの情報を渡す
        self.graph[self.collection[primary], :] += self.graph[self.collection[secondary], :]
        self.graph[:, self.collection[primary]] += self.graph[:, self.collection[secondary]]

        # 削除エッジの1が格納されているインデックスを取得し、0を格納する
        self.graph[self.collection[secondary], :] = 0
        self.graph[:, self.collection[secondary]] = 0

        # 圧縮と正規化
        self.compress()

        # 頂点と辺の統計を更新
        if update_statics:
            self.num_of_verts -= 1
            self.num_of_edges = len(self.get_all_edges())

        # コレクションの更新
        self.collection[self.collection == self.collection[secondary]] = self.collection[primary]

    def remove_edge(self, primary: int, secondary: int, compress: bool = False):
        """
        エッジを切断する

        - - -

        Params
        ------
        primary: int
            頂点1
        secondary: int
            頂点2
        compress: bool, by default False
            エッジの切断後に圧縮と正規化を行うか否か
        """

        if self.graph[primary, secondary] == 0:
            return

        self.graph[primary, secondary] = 0
        self.graph[secondary, primary] = 0

        # 圧縮と正規化 (removeではグラフ構造が大きく変化しないと考えられる)
        if compress:
            self.compress()

        # 辺の統計を更新
        self.num_of_edges -= 1

    def get_all_vertices(self) -> "list[int]":
        """
        グラフの頂点をすべて取得する

        - - -

        Return
        ------
        result: list[int]
            頂点のリスト
        """
        diag_indices = np.diag_indices(self.graph.shape[0])
        result = sparse.find(self.graph[diag_indices])[1]

        return result.tolist()

    def get_all_edges(self) -> "list[tuple[int, int]]":
        """
        グラフのエッジをすべて取得する

        - - -

        Return
        ------
        result: list[tuple[int, int]]
            エッジの格納されたリスト
        """

        result = np.array([sparse.find(self.graph)[0], sparse.find(self.graph)[1]]).T

        # 重複を削除
        result = result[result[:, 0] < result[:, 1]]

        return result.tolist()

    def get_max_score_edge(self) -> "tuple[int, int] | None":
        """
        最大スコアを持つエッジを取得する

        - - -

        Return
        ------
        result: tuple[int, int] | None
            最大スコアを持つエッジ。エッジがない場合はNoneを返す
        """

        diag_indices = np.diag_indices(self.graph.shape[0])  # 対角要素のインデックスを取得
        square = self.graph.dot(self.graph)  # 二乗を計算 (スコアの計算)
        square[diag_indices] = 0  # 対角要素を0にする (自身のノードに向かっているエッジを除外)
        square = self.graph.multiply(square)  # アダマール積を計算 (エッジの有無の反映)

        # エッジが存在しない場合は、Noneを返す
        if square.max() == 0:
            return None

        square_find = sparse.find(square)
        index = int(np.where(square_find[2] == int(square.max()))[0][0])
        result = (int(square_find[0][index]), int(square_find[1][index]))

        # メモリの明示的な解放
        del diag_indices
        del square
        del square_find
        if CUDA_AVAILABLE:
            np.get_default_memory_pool().free_all_blocks()

        return result

    def get_triangles_by_vertex(self, base_vertex: int) -> "list[tuple[int, int, int]]":
        """
        ある頂点を起点とした、グラフ内の三角形を形成している頂点を取得する

        - - -

        Params
        ------
        base_vertex: int
            基準となる頂点
        """

        # 基準点からの辺を取得
        vertices = self.graph[base_vertex, :].toarray()
        indices = np.where(vertices == 1)[1]
        indices = indices[base_vertex < indices]
        length = len(indices)
        tmp = sparse.csr_matrix((len(vertices[0]), len(vertices[0])), dtype=np.float32)

        # 基点以外の頂点の組み合わせで作成した行列を生成
        row, col = np.triu_indices(length, 1)
        target_indices = np.stack((indices[row], indices[col]))
        tmp[target_indices[0], target_indices[1]] = 1

        # アダマール積を計算
        triangle = self.graph.multiply(tmp)

        # CSR形式のスパース行列をCOO形式に変換し、非ゼロ要素のインデックスを取得
        triangle_find = sparse.find(triangle)

        # 三角形を形成している頂点の組み合わせを生成
        result = np.array([triangle_find[0], triangle_find[1]]).T
        result = result[(base_vertex < result[:, 0]) & (result[:, 0] < result[:, 1])]
        base_vertex_array = np.full((result.shape[0], 1), base_vertex)
        result = np.hstack((base_vertex_array, result))

        return result.tolist()

    def get_all_triangles(self) -> "list[tuple[int, int, int]]":
        """グラフ内のすべての三角形を取得する"""

        triangles = []
        for i in range(int(self.graph.shape[0])):
            triangles += self.get_triangles_by_vertex(i)

        return triangles

    def get_vertices_in_cluster(self, vertex: int) -> "list[int]":
        """
        ある頂点が所属するクラスタ内の頂点を取得する

        - - -

        Params
        ------
        vertex: int
            頂点
        """

        cluster = self.collection[vertex]
        vertices = where(self.collection == cluster)[0]

        return vertices.tolist()

    def get_recall(
        self,
        match_pairs: "list[tuple[int, int]]",
        demmed_pairs: "list[tuple[int, int]]",
    ) -> "tuple[float, list[tuple[int, int]]]":
        """
        持っているグラフと正解データと比較し、再現率を計算する

        - - -

        Params
        ------
        match_pairs: list[tuple[int, int]]
            正解ペアのリスト
        demmed_pairs: list[tuple[int, int]]
            既存グラフのエッジには存在しないみなし一致ペアのリスト

        Return
        ------
        float
            再現率
        list[tuple[int, int]]
            正解ペアのリスト
        float
            demmed も含めた再現率
        list[tuple[int, int]]
            demmed も含めた正解ペアのリスト
        """

        # 既存グラフを deepcopy して、グラフに demmed_pairs を追加する
        graph = sparse.csr_matrix((self.graph.shape[0], self.graph.shape[0]), dtype=np.float32)
        graph[sparse.find(self.graph)[0], sparse.find(self.graph)[1]] = 1

        # 正解データのエッジが含まれるエッジを取得
        correct = sparse.csr_matrix((self.graph.shape[0], self.graph.shape[0]), dtype=np.float32)
        __pairs = np.array(match_pairs).T
        correct[__pairs[0], __pairs[1]] = 1

        match_demmed_pairs = None
        match_demmed_recall = None

        # グラフに対して純粋な recall と demmed も含めた recall の2つを取得
        for i in range(2 if len(demmed_pairs) > 0 else 1):

            # demmed_pairs が指定されていて、2周目の場合は、グラフに demmed_pairs を追加する
            if i == 1:
                __pairs = np.array(demmed_pairs).T
                graph[__pairs[0], __pairs[1]] = 1

            # グラフと正解データのアダマール積を取得
            __target = graph.multiply(correct)

            if i == 0:
                # ペアのリストの取得と recall 値の取得
                target = np.array([sparse.find(__target)[0], sparse.find(__target)[1]]).T.tolist()
                match_pairs_recall = len(target) / len(match_pairs)

            elif i == 1:
                # demmed_pairs も含めたペアのリストの取得と recall 値を取得
                match_demmed_pairs = np.array([sparse.find(__target)[0], sparse.find(__target)[1]]).T.tolist()
                match_demmed_recall = len(match_demmed_pairs) / len(match_pairs)

        # deepcopy したグラフを削除
        del graph

        return (match_pairs_recall, target, match_demmed_recall, match_demmed_pairs)

    def __get_potential_recall_by_gpu(
        self,
        match_pairs: "list[tuple[int, int]]",
        demmed_pairs: "list[tuple[int, int]]" = [],
    ) -> "tuple[float, list[tuple[int, int]]]":
        """
        持っているグラフと正解データと比較し、クラスタリング時に一致の可能性を持つ潜在的な再現率を計算する

        - - -

        Params
        ------
        match_pairs: list[tuple[int, int]]
            正解ペアのリスト
        demmed_pairs: list[tuple[int, int]]
            既存グラフのエッジには存在しないみなし一致ペアのリスト

        Return
        ------
        float
            潜在的な再現率
        list[tuple[int, int]]
            潜在的な正解ペアのリスト
        float
            demmed も含めた潜在的な再現率
        list[tuple[int, int]]
            demmed も含めた潜在的な正解ペアのリスト
        """

        # 既存グラフを deepcopy して、グラフに demmed_pairs を追加する
        graph = sparse.csr_matrix((self.graph.shape[0], self.graph.shape[0]), dtype=np.float32)
        graph[sparse.find(self.graph)[0], sparse.find(self.graph)[1]] = 1

        # 正解データのエッジが含まれるエッジを取得
        correct = sparse.csr_matrix((self.graph.shape[0], self.graph.shape[0]), dtype=np.float32)
        __pairs = np.array(match_pairs).T
        correct[__pairs[0], __pairs[1]] = 1
        correct[__pairs[1], __pairs[0]] = 1

        potential_demmed_pairs = None
        potential_demmed_recall = None

        # グラフに対して純粋な recall と demmed も含めた recall の2つを取得
        for i in range(2 if len(demmed_pairs) > 0 else 1):

            # demmed_pairs が指定されていて、2周目の場合は、グラフに demmed_pairs を追加する
            if i == 1:
                __pairs = np.array(demmed_pairs).T
                graph[__pairs[0], __pairs[1]] = 1

            # 自身のグラフと正解データのアダマール積を取得
            target = graph.multiply(correct).toarray()

            # 連結成分の抽出 (O(V+E)) を行い、到達可能なベクトルでクラスターを作成する
            visited = zeros(int(graph.shape[0]), dtype=np.int32)
            potential_clusters = []

            # 全ての頂点に対して探索
            for j in range(int(graph.shape[0])):
                if visited[j] == 0:
                    stack = [j]
                    cluster = []
                    while len(stack) > 0:
                        vertex = stack.pop()
                        if visited[vertex] == 0:
                            visited[vertex] = 1
                            cluster.append(vertex)
                            stack += np.where(target[vertex, :] == 1)[0].tolist()
                    potential_clusters.append(cluster)

            # クラスター内のペアを取得
            __potential_pairs = []
            for c in potential_clusters:
                for a in range(len(c) - 1):
                    for b in range(a + 1, len(c)):
                        __potential_pairs.append(tuple(sorted([c[a], c[b]])))

            if i == 0:
                # ペアのリストの取得と recall 値の取得
                potential_pairs = __potential_pairs
                potential_recall = len(set(match_pairs) & set(__potential_pairs)) / len(match_pairs)

            elif i == 1:
                # demmed_pairs も含めたペアのリストの取得と recall 値を取得
                potential_demmed_pairs = __potential_pairs
                potential_demmed_recall = len(set(match_pairs) & set(__potential_pairs)) / len(match_pairs)

        # deepcopy したグラフを削除
        del graph

        # recallを計算
        return (potential_recall, potential_pairs, potential_demmed_recall, potential_demmed_pairs)

    def __get_potential_recall_by_cpu(
        self,
        match_pairs: "list[tuple[int, int]]",
        demmed_pairs: "list[tuple[int, int]]" = [],
    ):
        """
        持っているグラフと正解データと比較し、クラスタリング時に一致の可能性を持つ潜在的な再現率を計算する

        - - -

        Params
        ------
        match_pairs: list[tuple[int, int]]
            正解ペアのリスト
        demmed_pairs: list[tuple[int, int]]
            既存グラフのエッジには存在しないみなし一致ペアのリスト

        Return
        ------
        float
            潜在的な再現率
        list[tuple[int, int]]
            潜在的な正解ペアのリスト
        float
            demmed も含めた潜在的な再現率
        list[tuple[int, int]]
            demmed も含めた潜在的な正解ペアのリスト
        """

        # CPU 版の scipy である場合、既存グラフを deepcopy して、グラフに demmed_pairs を追加する
        # GPU 版である場合は、既存グラフを CPU 版に変換する
        if isinstance(self.graph, csr_matrix):
            # CPU
            graph = csr_matrix((self.graph.shape[0], self.graph.shape[0]), dtype=float32)
            graph[sparse.find(self.graph)[0], sparse.find(self.graph)[1]] = 1
        else:
            # GPU to CPU
            graph = self.graph.get()

        # 正解データのエッジが含まれるエッジを取得
        correct = csr_matrix((graph.shape[0], graph.shape[0]), dtype=float32)
        __pairs = array(match_pairs).T
        correct[__pairs[0], __pairs[1]] = 1
        correct[__pairs[1], __pairs[0]] = 1

        potential_demmed_pairs = None
        potential_demmed_recall = None

        # グラフに対して純粋な recall と demmed も含めた recall の2つを取得
        for i in range(2 if len(demmed_pairs) > 0 else 1):

            # demmed_pairs が指定されていて、2周目の場合は、グラフに demmed_pairs を追加する
            if i == 1:
                __pairs = array(demmed_pairs).T
                graph[__pairs[0], __pairs[1]] = 1

            # 自身のグラフと正解データのアダマール積を取得
            target = graph.multiply(correct).toarray()

            # 連結成分の抽出 (O(V+E)) を行い、到達可能なベクトルでクラスターを作成する
            visited = zeros(int(graph.shape[0]), dtype=int32)
            potential_clusters = []

            # 全ての頂点に対して探索
            for j in range(int(graph.shape[0])):
                if visited[j] == 0:
                    stack = [j]
                    cluster = []
                    while len(stack) > 0:
                        vertex = stack.pop()
                        if visited[vertex] == 0:
                            visited[vertex] = 1
                            cluster.append(vertex)
                            stack += where(target[vertex, :] == 1)[0].tolist()
                    potential_clusters.append(cluster)

            # クラスター内のペアを取得
            __potential_pairs = []
            for c in potential_clusters:
                for a in range(len(c) - 1):
                    for b in range(a + 1, len(c)):
                        __potential_pairs.append(tuple(sorted([c[a], c[b]])))

            if i == 0:
                # ペアのリストの取得と recall 値の取得
                potential_pairs = __potential_pairs
                potential_recall = len(set(match_pairs) & set(__potential_pairs)) / len(match_pairs)

            elif i == 1:
                # demmed_pairs も含めたペアのリストの取得と recall 値を取得
                potential_demmed_pairs = __potential_pairs
                potential_demmed_recall = len(set(match_pairs) & set(__potential_pairs)) / len(match_pairs)

        # deepcopy したグラフを削除
        del graph

        # recallを計算
        return (potential_recall, potential_pairs, potential_demmed_recall, potential_demmed_pairs)

    def get_potential_recall(self, match_pairs: "list[tuple[int, int]]", demmed_pairs: "list[tuple[int, int]]" = []):
        """
        持っているグラフと正解データと比較し、クラスタリング時に一致の可能性を持つ潜在的な再現率を計算する

        - - -

        Params
        ------
        match_pairs: list[tuple[int, int]]
            正解ペアのリスト
        demmed_pairs: list[tuple[int, int]]
            既存グラフのエッジには存在しないみなし一致ペアのリスト

        Return
        ------
        float
            潜在的な再現率
        list[tuple[int, int]]
            潜在的な正解ペアのリスト
        float
            demmed も含めた潜在的な再現率
        list[tuple[int, int]]
            demmed も含めた潜在的な正解ペアのリスト
        """

        try:
            return self.__get_potential_recall_by_gpu(match_pairs, demmed_pairs)

        except MemoryError:
            stderr = traceback.format_exc()
            self.logger.warning(f"{stderr}")
            self.logger.warning("Execute get_potential_recall_by_cpu().")
            return self.__get_potential_recall_by_cpu(match_pairs, demmed_pairs)
