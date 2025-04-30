"""Workflow task type"""

from dataclasses import dataclass, asdict, field
from typing import ClassVar

from uuid import uuid4
from datetime import datetime

from .pairdata import Pairdata
from .result import ContractionResult
from .gpu_status import GPUStatus
from ..metrics import Metrics
from ..mgraph import MGraph


class TaskType:
    """Workflowで呼び出すメソッド名"""

    # Workflow呼び出し時に毎回実行する必要がある場合は`everytime`を、
    # Workflow終了後に必ず中断する場合は`suspend`を、
    # 中断後、サスペンドファイルに保存する場合は`save`を、それぞれconfig.yamlに追記しtrueを指定する

    # filepathの指定があるものは、プロジェクトディレクトリからの相対パスを記述する

    # ===== SPECIAL =====
    # 特殊メソッド。これらはworkflow.pyではなく、file_container.ProjectContainer.complie_workflowで展開され処理される
    # ループ処理を実行する (count ... ループ回数, workflow ... ループ内で実行するワークフロー)
    LOOP = "LOOP"

    # ===== TRAINING =====
    # fasttextモデルの読み込み (filepath ... cc.ja.300.binのパス)
    LOAD_FASTTEXT = "LOAD_FASTTEXT"
    # 距離学習の実行 (filepath_tr ... トレーニング用yamlデータ, filepath_te ... テスト用yamlデータ, use_crowdsourcing ... クラウドソーシング利用フラグ)
    DIST_TRAINING = "DIST_TRAINING"
    # 距離学習モデルの保存 (filepath ... 距離学習モデルの保存先パス)
    SAVE_MODEL = "SAVE_MODEL"
    # 距離学習モデルの読込 (filepath ... 距離学習モデルの読込先パス)
    LOAD_MODEL = "LOAD_MODEL"
    # targetの一致ペアの距離を可視化する
    MATCH_PAIR_ACCURACY = "MATCH_PAIR_ACCURACY"

    # ===== PARAMS =====
    # ベンチマーク用のデータセットの生成 (filepath ... データが格納されたデータセット, , 出力先フォルダ)
    GENERATE_BENCHMARK_DATASET = "GENERATE_BENCHMARK_DATASET"
    # パラメータの生成 (filepath ... 生成用のyaml書誌データ群)
    GENERATE_PARAMS = "GENERATE_PARAMS"
    # パラメータの再生成 (filepath ... 既存パラメータのパス)
    REGENERATE_PARAMS = "REGENERATE_PARAMS"
    # パラメータの保存 (filepath ... パラメータの保存先パス)
    SAVE_PARAMS = "SAVE_PARAMS"
    # パラメータの読込 (filepath ... パラメータの読込先パス)
    LOAD_PARAMS = "LOAD_PARAMS"

    # ===== NGRAPH =====
    # 近傍グラフ構築
    NGRAPH_CONSTRUCTION = "NGRAPH_CONSTRUCTION"
    # configインスタンスをもとに近傍グラフを読込 (すでに近傍グラフが構築されている場合は何もしない)
    NGRAPH_REFLECTION = "NGRAPH_REFLECTION"

    # 再学習用パラメータを使って公理系を適用し、違反グループをクラウドソーシングに問い合わせる (filepath ... 書誌データの保存先パス, crowdsorcing_limit ... このワークフローに費やして良いクラウドソーシング上限数)
    APPLY_AXIOMATIC_SYSTEM_FOR_RETRAINING = "APPLY_AXIOMATIC_SYSTEM_FOR_RETRAINING"
    # Uncertaintyの値の高いペアを優先的に問い合わせる (filepath ... 書誌データの保存先パス, crowdsorcing_limit ... このワークフローに費やして良いクラウドソーシング上限数)
    APPLY_UNCERTAINTY_STRATEGY_FOR_RETRAINING = "APPLY_UNCERTAINTY_STRATEGY_FOR_RETRAINING"
    # Unknownの値の高いペアを優先的に問い合わせる(Query by committee) (filepath ... 書誌データの保存先パス, crowdsorcing_limit ... このワークフローに費やして良いクラウドソーシング上限数)
    APPLY_QBC_STRATEGY_FOR_RETRAINING = "APPLY_QBC_STRATEGY_FOR_RETRAINING"
    # Diversity戦略を適用し、多様性の高いペアを優先的に問い合わせる (crowdsorcing_limit ... このワークフローに費やして良いクラウドソーシング上限数, crowdsourcing_worker_accuracy ... クラウドソーシングワーカの精度)
    APPLY_DIVERSITY_STRATEGY_FOR_RETRAINING = "APPLY_DIVERSITY_STRATEGY_FOR_RETRAINING"
    # グラフ内のランダムにペアを優先的に問い合わせる (filepath ... 書誌データの保存先パス, crowdsorcing_limit ... このワークフローに費やして良いクラウドソーシング上限数)
    APPLY_RANDOM_STRATEGY_FOR_RETRAINING = "APPLY_RANDOM_STRATEGY_FOR_RETRAINING"

    # クラウドソーシングで問い合わせた書誌データ群をファイルとして出力する (filepath ... 書誌データの保存先パス)
    OUTPUT_CROWDSOURCING_PAIRS = "OUTPUT_CROWDSOURCING_PAIRS"
    # 全てノードになるまで縮約処理を実行する
    ALL_CONTRACTION = "ALL_CONTRACTION"
    # クラウドソーシング制限戦略に則り全縮約処理を開始する (crowdsorcing_limit ... このワークフローに費やして良いクラウドソーシング上限数)
    ALL_CONTRACTION_LIMITED_STRATEGY = "ALL_CONTRACTION_LIMITED_STRATEGY"
    # 縮約結果を保存する (filepath ... 結果の保存先パス)
    OUTPUT_RESULT = "OUTPUT_RESULT"

    # 同定済みペアと近傍グラフ内のペアに対して、推論結果を算出し統計を出力する
    VERIFY_CURRENT_INFERENCE = "VERIFY_CURRENT_INFERENCE"
    # クラウドソーシングワーカの精度を測る
    VERIFY_WORKER_ACCURACY = "VERIFY_WORKER_ACCURACY"
    # [Deprecated] 近傍グラフ内のすべてのペアに対して、推論結果を算出し統計を出力する
    VERIFY_ALL_NGRAPH = "VERIFY_ALL_NGRAPH"

    # ===== METRICS =====
    # 現状のメトリクスをログに出力する
    CURRENT_METRICS = "CURRENT_METRICS"

    # ===== RECORD CONTAINER UTILS =====
    # YAMLファイルの読み込み
    LOAD_YAML_FOR_RC = "LOAD_YAML_FOR_RC"  # filepath ... yamlファイルのパス
    # TSVファイルまたはCSVファイルの読み込み
    LOAD_TSV_FOR_RC = "LOAD_TSV_FOR_RC"  # filepath ... csvファイルのパス, delimiter ... 区切り文字
    # レコード作成
    MAKE_RECORD_FOR_RC = "MAKE_RECORD_FOR_RC"
    # 現在のコンテナファイルの出力
    SAVE_YAML_FOR_RC = "SAVE_YAML_FOR_RC"

    # ----- EXEPRIMENT -----
    # 任意のクラウドソーシング結果を追加する
    ADD_CROWDSOURCING_FOR_RETRAINING = "ADD_CROWDSOURCING_FOR_RETRAINING"

    # 推論結果に基づく任意のクラウドソーシング結果を追加する
    ADD_CROWDSOURCING_BY_INFERENCE = "ADD_CROWDSOURCING_BY_INFERENCE"

    # 近傍グラフに基づく任意のクラウドソーシング結果を追加する
    ADD_CROWDSOURCING_BY_NGRAPH = "ADD_CROWDSOURCING_BY_NGRAPH"

    # 距離を利用して、クラウドソーシングタスクを生成する
    GENERATE_CROWDSOURCING_TASKS_BY_DISTANCE = "GENERATE_CROWDSOURCING_TASKS_BY_DISTANCE"


class DiscontinuedTaskType:
    """廃止されたTaskType"""

    # ログに記録すべき内容を記述する

    # ===== TRAINING =====
    # 距離学習用のnumpy行列データセットの作成 (キャッシュ実装によって不要になった)
    CONSTRUCT_TRAIN_DATASET = "Reason: No longer needed due to cache implementation."

    # ===== NGRAPH =====
    # 現在のグラフに対し公理系を適用しグラフの縮約を行う (効果自体が薄そうなのと、しばらくメンテナンスされていないため)
    APPLY_AXIOMATIC_SYSTEM_FOR_CURRENT_GRAPH = "Reason: Not so effective."

    # ===== RECORD CONTAINER UTILS =====
    # トレーニング用データ出力 (ベンチマークデータセットと統合したため)
    SAVE_YAML_FOR_TRAIN = "Reason: Integrated with benchmark dataset."


class RandomResetMode:
    """ランダムシードのリセットモード"""

    NONE = "NONE"  # ランダムシードをリセットしない
    FIRST = "FIRST"  # ランダムシードをワークフローの最初だけリセットする
    EVERYTIME = "EVERYTIME"  # ランダムシードを毎回リセットする


class CSPlatform:
    """クラウドソーシングプラットフォーム"""

    SIMULATION = "SIMULATION"  # (Simulation)
    N4U = "N4U"  # Next Crowd 4U
    AMT = "AMT"  # Amazon Mechanical Turk
    YAHOO = "YAHOO"  # Yahoo Crowdsourcing

    EXP_WITH_INF = "EXP_WITH_INF"  # 推論値もデータに付与する, based on N4U
    EXP_CIKM2023 = "EXP_CIKM2023"  # CIKM2023用にCSVを3つ発行する, based on N4U


class CSAxiomStrategy:
    """クラウドソーシング戦略(Axiom)"""

    BINARY = "BINARY"  # 3つのレコードグループに対してスコアを付与し、高いスコア順に3つのレコードペアで問い合わせる
    BINARY_UNCERTAINTY = "BINARY_UNCERTAINTY"  # 3つのレコードグループに対してスコアを付与し、高いスコア順に3つのレコードペアのうち、より不確実なものを問い合わせる
    TRINARY = "TRINARY"  # 3つのレコードグループを直接問い合わせる


class CacheMode:
    """キャッシュモード"""

    NONE = "NONE"  # キャッシュを利用せず、毎回ファイルを読み込む (保存もしない)
    READ = "READ"  # キャッシュを利用して読み込む (保存はしない)
    WRITE = "WRITE"  # キャッシュを利用して読み込む (保存もする)


class InferenceMode:
    """推論モード"""

    BAYESIAN = "BAYESIAN"  # ベイズ推論
    SUN = "SUN"  # Same / Unknown / Not-same推論
    SVM = "SVM"  # Support Vector Machine 推論 (未実装)


@dataclass
class NGraphParams:
    """グラフに関するパラメータ (利用するパラメータはGraphConstructionModeに依存)"""

    # グラフ構築モードに合わせて抽象クラスを定義しても良さそう
    k: int = 10  # 近傍探索数 (KNN)
    dist: float = 1.0  # 近傍探索距離閾値 (DIST)
    enable_pseudo_dist: bool = False  # グラフに存在しないペア間の距離を近傍探索距離閾値の2倍とするか否か (DIST)


@dataclass
class WorkflowState:
    """Workflowで呼び出したメソッド(Task)がWorkflowに処理を要求するために返却する値"""

    suspend: bool = False  # Workflowを中断するように要請
    finished: bool = False  # Workflowが完了したことを通知
    save: bool = False  # 現状のWorkflowを中断データに保存する (suspendがTrueの場合はこれに依らず保存される)
    required_crowdsourcing: bool = False  # queueにクラウドソーシングタスクがあればファイルを出力
    notification: bool = False  # タスクが完了した場合にメール通知を行う
    message: str = ""  # メッセージ

    def __iadd__(self, el: "WorkflowState | None"):
        """+=演算子。論理演算子のORを取るような形にする"""
        # WorkflowStateであればboolのorを取る
        if isinstance(el, WorkflowState):
            self.suspend = self.suspend or el.suspend
            self.finished = self.finished or el.finished
            self.save = self.save or el.save
            self.required_crowdsourcing = self.required_crowdsourcing or el.required_crowdsourcing
            self.notification = self.notification or el.notification
            if self.message == "":
                self.message = el.message
            else:
                self.message += "" if el.message == "" else f"\n{self.message}"

        # Noneであればfinishedだけフラグを立てる
        elif el is None:
            self.finished = True

        # WorkflowStateかNone以外であればエラーを出す
        else:
            raise TypeError(
                f"Unsupported operand type(s) for +=: '{el.__class__.__name__}', supported for 'WorkflowState' or 'NoneType'"
            )

        return self


@dataclass
class WorkflowConfig:
    """Workflowを維持するための設定クラス"""

    graph: ClassVar[MGraph] = None  # [ClassVar] 隣接行列グラフと同定結果を格納

    config_id: str = str(uuid4())  # 設定ファイルのid
    config_creation_date = ("{}".format(datetime.now()))[:-3]  # 設定ファイルの作成日時
    target_filepath: str = None  # 書誌同定対象が格納されたファイルパス
    log_filepath: str = None  # ログの保存先が格納されたファイルパス
    log_level: str = None  # ログのレベル
    notification_filepath: str = None  # メール通知の設定が格納されたファイルパス
    gpu_status: GPUStatus = field(default_factory=GPUStatus)  # GPU(Cuda)を利用したモジュール起動の可否
    random_reset_mode: RandomResetMode = RandomResetMode.EVERYTIME  # ランダムシードのリセットモード
    random_seed: int = 42  # ランダムシード

    crowdsourcing_strict: bool = False  # Trueの場合、crowdsourcing_queueがなくなるまでシステムを起動しない
    gpu_strict: bool = False  # Trueの場合、GPUを利用するモジュールが1つでも利用できなければシステムを起動しない
    nonexistent_tasktype_strict: bool = (
        False  # Trueの場合、存在しないTaskTypeがWorkflowに含まれていた場合にシステムを起動しない
    )

    crowdsourcing_platform: CSPlatform = CSPlatform.SIMULATION  # クラウドソーシングプラットフォームの選択
    crowdsourcing_limit: int = None  # ワークフローを通して利用できるクラウドソーシング数の上限値
    crowdsourcing_strict: bool = False  # Trueの場合、crowdsourcing_queueがなくなるまでシステムを再開しない
    crowdsourcing_worker_accuracy: float = (
        1.0  # シミュレーション時に使用するクラウドソーシングワーカーの正解率 (0.0, 1.0)
    )
    ngraph_construction: str = None  # グラフ構築モード (KNN or DIST)
    ngraph_params: NGraphParams = None  # グラフ構築に関するパラメータ
    inference_mode: InferenceMode = None  # 推論モード(SUN or BAYESIAN)
    workflow: "list[dict[str, str]]" = None  # ワークフローを格納
    metrics: Metrics = None  # メトリクスを格納

    suspend_id: str = str(uuid4())  # 中断ファイルのid
    suspend_creation_date = ("{}".format(datetime.now()))[:-3]  # 中断ファイルの作成日時
    pairdata: "dict[tuple[int, int], Pairdata]" = field(default_factory=dict)  # ペアの推論データを格納
    current_workflow: int = 0  # 現在のワークフロー
    crowdsourcing_filepath: "dict[str, bool]" = field(default_factory=dict)
    crowdsourcing_queue: "list[list[int]]" = field(default_factory=list)
    crowdsourcing_result: "dict[tuple[int, int], list[float]]" = field(default_factory=dict)  # クラウドソーシング結果
    crowdsourcing_count: int = 0  # いままでに実施された総クラウドソーシング回数
    crowdsourcing_task_count: int = 0  # そのタスク内で実施されたクラウドソーシング回数
    contraction_result: ContractionResult = field(default_factory=ContractionResult)
    contraction_machine_result: ContractionResult = field(default_factory=ContractionResult)
    machine_contraction_pair: "list[tuple[int, int, float, float, float]]" = field(default_factory=list)
    machine_misidentification_pair: "list[tuple[int, int, float, float, float]]" = field(default_factory=list)

    gmail_enabled: bool = False  # メール通知をオンにするか否か
    gmail_finished_notification = False  # ワークフローが全て完了したタイミングで通知するか否か
    gmail_sender: str = None  # 送信元メールアドレス
    gmail_app_pw: str = None  # アプリパスワード
    gmail_receiver: str = None  # 通知先メールアドレス

    def to_dict(self) -> "dict[str, str]":
        """
        データを辞書型に変換する

        - - -

        Return
        ------
        result: dict[str, str]
            WorkflowConfigの持つデータを辞書形式で返す
        """

        return asdict(self)
