"""Project manager"""

import os
import random
import sys
import traceback
from datetime import datetime

import numpy as np
import tensorflow as tf

from .utils import path
from .file_container import RecordContainer, ProjectContainer, CrowdsourcingContainer
from .gptcore import GPTCore
from .inference import Inference
from .ngraph import NGraph

from .observer import Observer
from .logger import Logger, LoggerConfig

from .datatype.workflow import TaskType, WorkflowState, RandomResetMode
from .notification import GmailNotification


class ProjectManager:
    """プロジェクトを円滑に動作させるための管理クラス"""

    def __init__(self, project_dirpath: str):
        """
        コンストラクタ

        - - -

        Params
        ------
        project_dirpath: str
            対象となるプロジェクトディレクトリのフルパス
        base_dirpath: str
            リポジトリのベースディレクトリのフルパス
        """
        self.current_workflow: int = None  # 現在のワークフロー番号
        self.e_start = datetime.now()  # 全実行時間 (開始時刻格納)
        self.d_start: datetime = datetime.now()  # データ読み込み時間 (開始時刻格納)
        self.w_start: datetime = None  # ワークフロー実行時間 (開始時刻格納)

        self.project_dirpath: str = path.abspath(project_dirpath)  # プロジェクトのディレクトリパス
        self.config, err_msg = ProjectContainer.load_project(self.project_dirpath)  # 設定ファイルの読み込み

        if self.config is None or len(err_msg) > 0:
            GmailNotification.config_error(err_msg, self.project_dirpath, self.config)
            raise RuntimeError("\n".join(err_msg))

        # Observerの初期化
        self.observer = Observer(
            self.project_dirpath,
            path.join(self.project_dirpath, self.config.log_filepath),
        )

        self.rc = RecordContainer(log_filepath=path.join(self.project_dirpath, self.config.log_filepath))

        # targetファイルが存在するならば、読み込む
        if self.config.target_filepath is not None:
            self.rc.load_file(path.join(self.project_dirpath, self.config.target_filepath))

        self.logger: Logger = Logger(
            __name__,
            logger_config=LoggerConfig(level=self.config.log_level),
            filepath=path.join(self.project_dirpath, self.config.log_filepath),
        )
        self.core: GPTCore = GPTCore(
            config=self.config,
            target_filepath=path.join(self.project_dirpath, self.config.target_filepath),
            log_filepath=path.join(self.project_dirpath, self.config.log_filepath),
            inf_attr=self.rc.inf_attr,
            api_key=os.environ.get("OPENAI_API_KEY") 
        )
        self.inference: Inference = Inference(
            self.core,
            config=self.config,
            log_filepath=path.join(self.project_dirpath, self.config.log_filepath),
        )
        self.ngraph: NGraph = NGraph(
            self.inference,
            config=self.config,
            log_filepath=path.join(self.project_dirpath, self.config.log_filepath),
        )
        self.crowdsourcing: CrowdsourcingContainer = CrowdsourcingContainer(
            config=self.config,
            req_dirpath=path.join(self.project_dirpath, "request"),
            input_dirpath=path.join(self.project_dirpath, "input"),
            log_filepath=path.join(self.project_dirpath, self.config.log_filepath),
        )

        # 全データ入出力時間を記録
        self.config.metrics.data_io_time.append(int((datetime.now().timestamp() - self.d_start.timestamp()) * 1000))
        self.d_start = None

        # 全実行時間を記録
        self.config.metrics.execution_time += int((datetime.now().timestamp() - self.e_start.timestamp()) * 1000)
        self.e_start = None

    def __reset_random_seed(self, seed: int = 42, cupy_available: bool = False):
        """[Private] 各モジュールのランダムシードをリセットする"""
        self.logger.info(f"Reset random seed to {seed}")

        tf.random.set_seed(seed)  # Tensorflow
        np.random.seed(seed)  # Numpy

        if cupy_available:
            import cupy as cp

            cp.random.seed(seed)  # Cupy

        random.seed(seed)  # Random
        os.environ["PYTHONHASHSEED"] = str(seed)

    def __init_metrics(
        self,
        date: datetime | None = None,
        e_time: bool = False,
        d_time: bool = False,
        w_time: bool = False,
    ):
        """
        [Private] 指定された内容でメトリクスデータを初期化する

        - - -

        Params
        ------
        date: datetime | None
            初期化する内容。Noneの場合は現在時刻で初期化する
        その他: bool
            初期化するか否か
        """
        if date is None:
            date = datetime.now()

        if e_time:
            self.e_start = date

        if d_time:
            self.d_start = date

        if w_time:
            self.w_start = date

    def __record_metrics(self, e_time: bool = False, d_time: bool = False, w_time: bool = False):
        """[Private] メトリクスを記録したあと現在時刻で初期化する"""

        if d_time and self.d_start is not None:
            self.config.metrics.data_io_time.append(int((datetime.now().timestamp() - self.d_start.timestamp()) * 1000))

        if w_time and self.current_workflow is not None and self.w_start is not None:
            self.config.metrics.workflow_time[self.current_workflow] += int(
                (datetime.now().timestamp() - self.w_start.timestamp()) * 1000
            )

        if e_time and self.e_start is not None:
            self.config.metrics.execution_time += int((datetime.now().timestamp() - self.e_start.timestamp()) * 1000)

        # 初期化処理
        self.__init_metrics(e_time=e_time, d_time=d_time, w_time=w_time)

    def __msec_format(self, workflow: int):
        """[Private] ワークフロー番号を渡すと、整形した文字列にして返す"""
        msec = self.config.metrics.msec_2_duration(self.config.metrics.workflow_time[workflow])
        result = "" if msec.day == 0 else f"{msec.day}d"
        result += "" if len(result) == 0 and msec.hour == 0 else f"{msec.hour}h"
        result += "" if len(result) == 0 and msec.minute == 0 else f"{msec.minute}m"
        result += f"{msec.second}.{msec.msec:03}s"
        return result

    def start_workflow(self):
        """プロジェクトのWorkflowを安全に開始する"""

        self.observer.start()  # Observerの監視を開始
        self.__init_metrics(e_time=True)  # 実行時間の記録を開始
        result = None

        try:
            # ワークフローを実行する
            self.__start_workflow()

        except Exception as err:
            # 例外が発生した場合にログに書き込み、メール通知を行う
            stderr = traceback.format_exc()
            _, title, _ = sys.exc_info()
            self.logger.error(stderr)
            print(stderr)

            # 全実行時間とワークフロー実行時間を記録
            self.__record_metrics(e_time=True, w_time=True)

            # Gmail通知を行う
            if self.config.gmail_enabled and self.config.gmail_finished_notification:
                GmailNotification.error(title, self.project_dirpath, self.config, False)

            # 呼び出し元にエラーを返す
            result = err

        except KeyboardInterrupt as err:
            # [Critical] Ctrl+Cにより強制中断される際に呼び出される例外
            # 例外が発生した場合にログに書き込み、メール通知は行わない
            stderr = traceback.format_exc()
            _, title, _ = sys.exc_info()
            self.logger.critical(stderr)
            print(stderr)

            # 全実行時間とワークフロー実行時間を記録
            self.__record_metrics(e_time=True, w_time=True)

            # 呼び出し元にエラーを返す
            result = err

        except BaseException as err:
            # [Critical] システムの異常終了により呼び出される例外
            # 例外が発生した場合にログに書き込み、メール通知を行う
            stderr = traceback.format_exc()
            _, title, _ = sys.exc_info()
            self.logger.critical(stderr)
            print(stderr)

            # 全実行時間とワークフロー実行時間を記録
            self.__record_metrics(e_time=True, w_time=True)

            # Gmail通知を行う
            if self.config.gmail_enabled and self.config.gmail_finished_notification:
                GmailNotification.error(title, self.project_dirpath, self.config, True)

            # 呼び出し元にエラーを返す
            result = err

        self.observer.stop()  # Observerの監視を停止
        return result

    def __start_workflow(self):
        """[private] プロジェクトのWorkflowを開始する"""

        # クラウドソーシングの結果反映
        # TODO: 改修する
        reflect_result = self.crowdsourcing.reflect_crowdsourcing()

        # クラウドソーシングに不足があった場合にエラーする
        if not reflect_result and self.config.crowdsourcing_strict:
            self.crowdsourcing.request_crowdsourcing(path.join(self.project_dirpath, self.config.target_filepath))
            title = "Some crowdsourcing results are missing."
            raise RuntimeError(title)

        else:
            self.config.crowdsourcing_queue = []

        required_crowdsourcing = False
        pd = self.project_dirpath

        # 初回のみランダムシードをリセットする場合はここで実行
        if self.config.random_reset_mode == RandomResetMode.FIRST:
            self.__reset_random_seed(self.config.random_seed, self.config.gpu_status.cupy)

        # === Begin Workflow ===
        for wf_counter, task in enumerate(self.config.workflow):
            self.current_workflow = wf_counter
            self.__init_metrics(w_time=True)  # ワークフロー実行時間の記録を開始

            # 実行するかどうか (everytimeがTrueの場合、必ず実行する)
            execution = (self.config.current_workflow == wf_counter) or (
                task["everytime"] if "everytime" in task.keys() else False
            )
            state = WorkflowState()
            state.suspend = task["suspend"] if "suspend" in task else False
            state.save = task["save"] if "save" in task else False

            # 実行を行う場合にログを残す
            if execution:
                msg = f"Start [{wf_counter}] '{task['name']}' {'(everytime)' if self.config.current_workflow != wf_counter else ''}"
                self.logger.info(msg)

                self.observer.set_workflow(-1 if self.config.current_workflow != wf_counter else wf_counter)

                # 毎回ランダムシードをリセットする、または random_seed が設定されている場合はここで実行
                if self.config.random_reset_mode == RandomResetMode.EVERYTIME or "random_seed" in task:
                    __random_seed = self.config.random_seed
                    # ランダムシードが指定されている場合は、その値を使う
                    if "random_seed" in task and isinstance(task["random_seed"], int):
                        __random_seed = task["random_seed"]
                    elif "random_seed" in task and task["random_seed"] == "wf_counter":
                        __random_seed = wf_counter

                    self.__reset_random_seed(__random_seed, self.config.gpu_status.cupy)

            # 実行しない場合は、次のタスクへ進む
            else:
                continue

            # --- Distance Training ---
            # Fasttextモデルの読み込み
            if task["name"] == TaskType.LOAD_FASTTEXT:
                state += self.core.load_fasttext_dict(path.join(self.project_dirpath, task["filepath"]))

            # トレーニングデータとテストデータを使い距離学習を行う
            if task["name"] == TaskType.DIST_TRAINING:
                if "use_cs_type" in task:
                    # use_cs_type については廃止
                    msg = "[Deprecated] `use_cs_type` has been removed."
                    self.logger.warning(msg)
                    print(msg)

                state += self.core.train(
                    filepath=path.join(self.project_dirpath, task["filepath"]),
                    test_ratio=task["test_ratio"] if "test_ratio" in task else 0.1,
                    data_shuffle=task["data_shuffle"] if "data_shuffle" in task else False,
                    match_num=task["match_num"] if "match_num" in task else None,
                    mismatch_ratio=task["mismatch_ratio"] if "mismatch_ratio" in task else 1,
                    max_in_cluster=task["max_in_cluster"] if "max_in_cluster" in task else 50,
                    basemodel_filepath=(
                        path.join(self.project_dirpath, task["basemodel_filepath"])
                        if "basemodel_filepath" in task and task["basemodel_filepath"] is not None
                        else None
                    ),
                    image_dirpath=(
                        path.join(self.project_dirpath, task["image_dirpath"])
                        if "image_dirpath" in task
                        else path.join(self.project_dirpath, "image")
                    ),
                    use_crowdsourcing=task["use_crowdsourcing"] if "use_crowdsourcing" in task else False,
                    use_best_model=task["use_best_model"] if "use_best_model" in task else True,
                )

            # 距離学習により得られたh5学習データの読み込み
            if task["name"] == TaskType.LOAD_MODEL:
                state += self.core.load_model(path.join(self.project_dirpath, task["filepath"]))

            # 距離学習により得られたh5学習データの保存
            if task["name"] == TaskType.SAVE_MODEL:
                state += self.core.save_model(path.join(self.project_dirpath, task["filepath"]))

            # 読み込まれたtargetの一致ペアについて、距離を計算してグラフで表示する
            if task["name"] == TaskType.MATCH_PAIR_ACCURACY:
                state += self.core.match_pair_accuracy(
                    image_dirpath=(
                        path.join(self.project_dirpath, task["image_dirpath"])
                        if "image_dirpath" in task
                        else path.join(self.project_dirpath, "image")
                    ),
                )

            # --- Params ---
            # ファイルからベンチマークデータセットの生成
            if task["name"] == TaskType.GENERATE_BENCHMARK_DATASET:
                state += self.rc.generate_benchmark_dataset(
                    task["required_records_list"],
                    path.join(self.project_dirpath, task["dirpath"] if "dirpath" in task else "benchmark"),
                )

            # ファイルからフィッティングパラメータ等の生成
            if task["name"] == TaskType.GENERATE_PARAMS:
                state += self.inference.generate_params(
                    path.join(self.project_dirpath, task["filepath"]),
                    (
                        path.join(self.project_dirpath, self.config.target_filepath)
                        if self.config.target_filepath is not None
                        and "use_crowdsourcing" in task
                        and task["use_crowdsourcing"]
                        else None
                    ),
                    distribution_type=task["distribution_type"] if "distribution_type" in task else None,
                    image_title=task["image_title"] if "image_title" in task else None,
                    image_dirpath=(
                        path.join(self.project_dirpath, task["image_dirpath"])
                        if "image_dirpath" in task
                        else path.join(self.project_dirpath, "image")
                    ),
                )

            # ファイルからフィッティングパラメータ等の読み込み
            if task["name"] == TaskType.LOAD_PARAMS:
                state += self.inference.load_params(path.join(self.project_dirpath, task["filepath"]))

            # ファイルからフィッティングパラメータ等の保存
            if task["name"] == TaskType.SAVE_PARAMS:
                state += self.inference.save_params(path.join(self.project_dirpath, task["filepath"]))

            # --- NGraph ---
            # 近傍グラフ新規構築 (既存近傍グラフは削除される)
            if task["name"] == TaskType.NGRAPH_CONSTRUCTION:
                state += self.ngraph.generate_ngraph(
                    filepath=path.join(self.project_dirpath, self.config.target_filepath),
                    k=int(task["k"]) if "k" in task else None,
                    dist=float(task["dist"]) if "dist" in task else None,
                    reflect_crowdsourcing_result=(
                        task["reflect_crowdsourcing_result"] if "reflect_crowdsourcing_result" in task else True
                    ),
                    limited_k=task["limited_k"] if "limited_k" in task else None,
                    connect_edge_by_cs_result=(
                        task["connect_edge_by_crowdsourcing_result"]
                        if "connect_edge_by_crowdsourcing_result" in task
                        else True
                    ),
                    max_cores=task["max_cores"] if "max_cores" in task else None,
                    image_dirpath=(
                        path.join(self.project_dirpath, task["image_dirpath"])
                        if "image_dirpath" in task
                        else path.join(self.project_dirpath, "image")
                    ),
                    image_title=task["image_title"] if "image_title" in task else None,
                    log_minutes=task["log_minutes"] if "log_minutes" in task else 10,
                )

            # suspendに保存されている近傍グラフを再構築
            if task["name"] == TaskType.NGRAPH_REFLECTION:
                state += self.ngraph.config_reflection(
                    path.join(self.project_dirpath, self.config.target_filepath),
                    task["force"] if "force" in task else False,
                    max_cores=task["max_cores"] if "max_cores" in task else None,
                    log_minutes=task["log_minutes"] if "log_minutes" in task else 5,
                )

            # 再学習用のペアを検出するために公理系を適用する
            if task["name"] == TaskType.APPLY_AXIOMATIC_SYSTEM_FOR_RETRAINING:
                state += self.ngraph.apply_axiomatic_system_for_retraining(
                    task["crowdsourcing_limit"] if "crowdsourcing_limit" in task else None,
                    task["crowdsourcing_worker_accuracy"] if "crowdsourcing_worker_accuracy" in task else None,
                    task["crowdsourcing_result_priority"] if "crowdsourcing_result_priority" in task else True,
                    task["crowdsourcing_axiom_strategy"] if "crowdsourcing_axiom_strategy" in task else "BINARY",
                    task["priority_rule"] if "priority_rule" in task else {},
                )

            # Uncertaintyの値の高いペアを優先的に問い合わせる
            if task["name"] == TaskType.APPLY_UNCERTAINTY_STRATEGY_FOR_RETRAINING:
                state += self.ngraph.apply_uncertainty_strategy_for_retraining(
                    task["crowdsourcing_limit"] if "crowdsourcing_limit" in task else None,
                    task["crowdsourcing_worker_accuracy"] if "crowdsourcing_worker_accuracy" in task else None,
                    task["crowdsourcing_result_priority"] if "crowdsourcing_result_priority" in task else True,
                )

            # Unknownの値の高いペアを優先的に問い合わせる (Query by Committee)
            if task["name"] == TaskType.APPLY_QBC_STRATEGY_FOR_RETRAINING:
                state += self.ngraph.apply_qbc_strategy_for_retraining(
                    task["crowdsourcing_limit"] if "crowdsourcing_limit" in task else None,
                    task["crowdsourcing_worker_accuracy"] if "crowdsourcing_worker_accuracy" in task else None,
                )

            # Diversity戦略に基づいて問い合わせる
            if task["name"] == TaskType.APPLY_DIVERSITY_STRATEGY_FOR_RETRAINING:
                state += self.ngraph.apply_diversity_strategy_for_retraining(
                    task["crowdsourcing_limit"] if "crowdsourcing_limit" in task else None,
                    task["crowdsourcing_worker_accuracy"] if "crowdsourcing_worker_accuracy" in task else None,
                )

            # ランダムに問い合わせる
            if task["name"] == TaskType.APPLY_RANDOM_STRATEGY_FOR_RETRAINING:
                state += self.ngraph.apply_random_strategy_for_retraining(
                    task["crowdsourcing_limit"] if "crowdsourcing_limit" in task else None,
                    task["crowdsourcing_worker_accuracy"] if "crowdsourcing_worker_accuracy" in task else None,
                )

            # Bubbleの戦略に基づいて全ての縮約を開始する
            if task["name"] == TaskType.ALL_CONTRACTION:
                state += self.ngraph.update_all_ngraph(
                    task["crowdsourcing_limit"] if "crowdsourcing_limit" in task else None,
                    task["crowdsourcing_limit_suspend"] if "crowdsourcing_limit_suspend" in task else False,
                    task["crowdsourcing_worker_accuracy"] if "crowdsourcing_worker_accuracy" in task else None,
                    task["crowdsourcing_result_priority"] if "crowdsourcing_result_priority" in task else True,
                    title_blocking=task["title_blocking"] if "title_blocking" in task else False,
                    log_minutes=int(task["log_minutes"]) if "log_minutes" in task else 10,
                )

            # クラウドソーシング数が制限されている状況の戦略で全ての縮約を開始する
            if task["name"] == TaskType.ALL_CONTRACTION_LIMITED_STRATEGY:
                state += self.ngraph.update_all_ngraph_limited_strategy(
                    task["crowdsourcing_limit"] if "crowdsourcing_limit" in task else None
                )

            # 同定結果を出力する
            if task["name"] == TaskType.OUTPUT_RESULT:
                state += self.ngraph.output_result_yaml(path.join(self.project_dirpath, "result.yaml"))

            # 同定済みペアと近傍グラフ内のペアに対して、推論結果を算出し統計を出力する
            if task["name"] == TaskType.VERIFY_CURRENT_INFERENCE:
                state += self.ngraph.verify_current_inference(
                    task["crowdsourcing_result_priority"] if "crowdsourcing_result_priority" in task else True,
                    image_title=task["image_title"] if "image_title" in task else None,
                    image_dirpath=(
                        path.join(self.project_dirpath, task["image_dirpath"])
                        if "image_dirpath" in task
                        else path.join(self.project_dirpath, "image")
                    ),
                )

            if task["name"] == TaskType.VERIFY_WORKER_ACCURACY:
                state += self.ngraph.verify_worker_accuracy()

            # [Deprecated] 近傍グラフ内のすべてのペアに対して、推論結果を算出し統計を出力する
            if task["name"] == TaskType.VERIFY_ALL_NGRAPH:
                self.logger.warning("VERIFY_ALL_NGRAPH was deprecated! Use VERIFY_CURRENT_INFERENCE instead.")
                state += self.ngraph.verify_all_ngraph()

            # --- Metrics ---
            # 現在のメトリクスをログに出力する
            if task["name"] == TaskType.CURRENT_METRICS:
                state += self.config.metrics.current_metrics()

            # --- Record Container Utils (Independence) ---
            # TODO: 別のproject.pyではないファイルに切り分けしても良いかも
            # Yamlファイルからデータを読み込む
            if task["name"] == TaskType.LOAD_YAML_FOR_RC:
                state += self.rc.load_file(path.join(self.project_dirpath, task["filepath"]))

            # TSVファイルからデータを読み込む
            if task["name"] == TaskType.LOAD_TSV_FOR_RC:
                state += self.rc.load_tsv(
                    path.join(self.project_dirpath, task["filepath"]),
                    task["cluster_id"] if "cluster_id" in task else None,
                    task["delimiter"] if "delimiter" in task else "\t",
                    task["attributes"] if "attributes" in task else {},
                    task["display_summary"] if "display_summary" in task else True,
                )

            # コンテナに格納されたデータを整形する
            if task["name"] == TaskType.MAKE_RECORD_FOR_RC:
                state += self.rc.make_record(
                    task["option"],  # ペア数とデータ数を辞書型で格納 {1: 100, 2: 500 ...}
                    task["destroy"] if "destroy" in task else False,
                )

            # 現在のコンテナ内のデータをYAMLファイルに保存する
            if task["name"] == TaskType.SAVE_YAML_FOR_RC:
                state += self.rc.save_yaml(
                    path.join(pd, task["filepath"]),
                )

            # --- Experiment ----
            # 正解データを利用しクラウドソーシング結果を追加する (実験用タスク)
            if task["name"] == TaskType.ADD_CROWDSOURCING_FOR_RETRAINING:
                state += self.ngraph.add_crowdsourcing_for_retraining(
                    path.join(self.project_dirpath, self.config.target_filepath),
                    task["same_pair"],
                    task["notsame_pair"],
                )

            # 推論データに基づいてクラウドソーシング結果を追加する (実験用タスク)
            if task["name"] == TaskType.ADD_CROWDSOURCING_BY_INFERENCE:
                state += self.ngraph.add_crowdsourcing_by_inference(
                    path.join(self.project_dirpath, self.config.target_filepath),
                    task["type"],  # "true_positive" or "false_positive" or "false_negative" or "true_negative"
                )

            # 近傍グラフに基づいてクラウドソーシング結果を追加する (実験用タスク)
            if task["name"] == TaskType.ADD_CROWDSOURCING_BY_NGRAPH:
                state += self.ngraph.add_crowdsourcing_by_ngraph(
                    task["type"],  # "order_positive" or "order_negative" or "order_unknown"
                )

            # 距離に基づいてクラウドソーシングタスクを生成する
            if task["name"] == TaskType.GENERATE_CROWDSOURCING_TASKS_BY_DISTANCE:
                state += self.ngraph.generate_crowdsourcing_tasks_by_dist(
                    match_count=task["match_count"] if "match_count" in task else 2000,
                    mismatch_count=task["mismatch_count"] if "mismatch_count" in task else 1000,
                    match_min_dist=task["match_min_dist"] if "match_min_dist" in task else None,
                    match_max_dist=task["match_max_dist"] if "match_max_dist" in task else None,
                    mismatch_min_dist=task["mismatch_min_dist"] if "mismatch_min_dist" in task else None,
                    mismatch_max_dist=task["mismatch_max_dist"] if "mismatch_max_dist" in task else None,
                    random=task["random"] if "random" in task else True,
                    exclude_complete_match_pairs=(
                        task["exclude_complete_match_pairs"] if "exclude_complete_match_pairs" in task else True
                    ),
                )

            # --- --- --- ---

            # 一時停止が必要な場合はこのループを抜ける
            if state.suspend:
                if state.finished:
                    self.logger.info(
                        f"Finished and Suspended [{wf_counter}] '{task['name']}' {'(everytime)' if self.config.current_workflow != wf_counter else ''}"
                    )
                    self.config.current_workflow = max(self.config.current_workflow, wf_counter + 1)
                else:
                    self.logger.info(
                        f"Suspended [{wf_counter}] '{task['name']}' {'(everytime)' if self.config.current_workflow != wf_counter else ''}"
                    )

                self.logger.info(state)
                required_crowdsourcing = state.required_crowdsourcing
                break

            # 全体の実行時間とワークフロー実行時間を記録
            self.__record_metrics(e_time=True, w_time=True)

            # 完了ログを残す
            self.logger.info(
                f"Finished [{wf_counter}] '{task['name']}' ({self.__msec_format(wf_counter)}{', everytime' if self.config.current_workflow != wf_counter else ''})"
            )
            if self.config.current_workflow == wf_counter:
                self.config.crowdsourcing_task_count = 0  # 現在のワークフローで利用されたクラウドソーシング数のリセット

            # Gmail通知が必要な場合は、通知を行う
            if state.notification and self.config.gmail_enabled:
                GmailNotification.send(config=self.config)

            # 次のworkflowに進む
            self.config.current_workflow = max(self.config.current_workflow, wf_counter + 1)

            # 一時ファイルへの保存が必要な場合は保存する
            if state.save:
                self.__init_metrics(d_time=True)  # ファイル入出力時間の記録を開始
                ProjectContainer.save_suspend_yaml(self.project_dirpath, self.config)
                self.__record_metrics(d_time=True)

        # === End workflow ===
        self.observer.set_workflow(-1)

        # suspendを保存
        self.__record_metrics(e_time=True)  # 全実行時間を記録
        ProjectContainer.save_suspend_yaml(self.project_dirpath, self.config)

        # クラウドソーシングを要求する場合は、プラットフォームに合わせたファイルを生成する
        if required_crowdsourcing:
            self.crowdsourcing.request_crowdsourcing(path.join(self.project_dirpath, self.config.target_filepath))

        # Gmail通知を行う
        if self.config.gmail_enabled and self.config.gmail_finished_notification:
            GmailNotification.send(self.project_dirpath, self.config)
