"""File Container"""

from typing import Callable
from dataclasses import asdict

import os
import sys
from datetime import datetime
from copy import deepcopy
from uuid import uuid4
import subprocess

import numpy as np
from csv import reader
import codecs
import yaml
import json
from random import random, randrange, shuffle, sample

import tensorflow as tf
from cupy import cuda
import faiss

from .utils import path
from .logger import Logger, LoggerConfig
from .datatype.record import RecordType, Record, RecordMG
from .datatype.result import ContractionResult
from .datatype.fitting_params import ParamsType, BetaParams, GammaParams, GaussianParams
from .datatype.sun_params import SunParams
from .datatype.group_evaluation import GroupEvaluation
from .datatype.gpu_status import GPUStatus
from .mgraph import MGraph
from .metrics import Metrics
from .datatype.workflow import (
    TaskType,
    DiscontinuedTaskType,
    WorkflowConfig,
    RandomResetMode,
    CSPlatform,
    InferenceMode,
    NGraphParams,
)

# メジャーバージョンについては統一する
CONFIG_LATEST_VERSION = "3.1"
SUSPEND_LATEST_VERSION = "3.2"
NOTIFICATION_LATEST_VERSION = "3.0"
PARAMS_LATEST_VERSION = "3.2"
RECORD_LATEST_VERSION = "3.1"


class ContainerType:
    """保存ファイルに関する型"""

    UNKNOWN = "UNKNOWN"  # 不明なタイプ
    PARAMS = "PARAMS"  # 確率密度関数や推論器のパラメータを格納するタイプ
    NORMAL = "NORMAL"  # (廃止予定) 学習用および検証用に利用するため正解を含んだデータが格納されたタイプ (version 2.0はすべてこの型)
    TARGET = "TARGET"  # 学習用および検証用に利用するため正解を含んだデータが格納されたタイプ
    REAL = "REAL"  # 正解がわからない現実データが格納されたタイプ
    CONFIG = "CONFIG"  # 設定ファイルが格納されたタイプ
    SUSPEND = "SUSPEND"  # 処理を復元するためのタイプ
    FINISHED = "FINISHED"  # 処理を完了したことを示すタイプ
    RESULT = "RESULT"  # 同定結果が格納されたタイプ
    NOTIFICATION = "NOTIFICATION"  # 通知関連の設定が格納されたタイプ


def judge_container_type(filepath: str) -> ContainerType:
    """ファイルパスから、保存ファイルの型を判定する"""

    # 設定ファイルの存在を確認。存在しない場合は、エラーメッセージを返却して終了する
    if not path.isfile(path.join(filepath)):
        Logger(__name__).error(f"The {filepath} does not exist.")

    # ファイルの出力
    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data["type"]


class ProjectContainer:
    """
    NGraphの実行について、設定や処理の中断に関するデータ処理を行い、メインプログラムの補助を行うクラス

    読み込み可能なContainterType: CONFIG, SUSPEND, (FINISHED)
    """

    @classmethod
    def __get_gpu_status(cls) -> GPUStatus:
        """[private/classmethod] GPUの利用可否を取得する (一部subprocessを利用)"""

        # tensorflowがメモリを専有せず、他のプロセスとも共有できるように設定
        physical_devices = tf.config.list_physical_devices("GPU")
        if len(physical_devices) > 0:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)

        gpu_status = GPUStatus()
        # tensorflow
        gpu_status.tensorflow = len(tf.config.list_physical_devices("GPU")) > 0
        # cupy
        gpu_status.cupy = cuda.is_available()
        # faiss (メインプロセスごと持ってかれる可能性があるので、subprocessにより別プロセスで取得できるかを検証した後に実行する)
        result = subprocess.run(["python -c 'import faiss; faiss.get_num_gpus()'"], shell=True)
        if result.returncode == 0:
            gpu_status.faiss = faiss.get_num_gpus() > 0

        return gpu_status

    @classmethod
    def init_config(cls, project_dirpath: str, force: bool = False):
        """
        [classmethod] 初期化用のCONFIGファイルを出力する

        - - -

        Parameters
        ----------
        project_dirpath: str
            プロジェクトが格納されたディレクトリパス
        force: bool, by default False
            フォルダにconfigファイルが存在する場合でも上書きして出力する
        """

        # configファイルが存在する場合は、ログを書き込んで出力しない
        if not force and path.exists(path.join(project_dirpath, "config.yaml")):
            Logger(__name__).warning("The config.yaml file already exists.")
            return

        # configファイルのテンプレート
        date = ("{}".format(datetime.now()))[:-3]
        output = {
            "version": CONFIG_LATEST_VERSION,
            "type": ContainerType.CONFIG,
            "id": str(uuid4()),
            "summary": {
                "creation_date": date,
                "update_date": date,
                "target_filepath": "./data_500.yaml",
                "log_filepath": "./log.log",
                "log_level": "INFO",
                "notification_filepath": "./notification.yaml",
                "random_reset_mode": RandomResetMode.EVERYTIME,
                "random_seed": 42,
                "crowdsourcing_platform": CSPlatform.SIMULATION,
                "crowdsourcing_strict": False,
                "crowdsourcing_limit": None,
                "crowdsourcing_worker_accuracy": 1.0,
                "ngraph_construction": "DIST",
                "ngraph_params": {
                    "k": 10,
                    "dist": 1.0,
                    "enable_pseudo_dist": False,
                },
                "inference_mode": "SUN",
            },
            "workflow": [
                {"name": "LOAD_FASTTEXT", "filepath": "../../cc.ja.300.bin", "everytime": True},
                {"name": "LOAD_MODEL", "filepath": "../model.h5", "everytime": True},
                {"name": "LOAD_PARAMS", "filepath": "../params.yaml", "everytime": True},
                {"name": "NGRAPH_CONSTRUCTION"},
                {"name": "NGRAPH_REFLECTION", "everytime": True},
                {"name": "ALL_CONTRACTION_LIMITED_STRATEGY"},
                {"name": "OUTPUT_RESULT", "filepath": "./result.yaml"},
            ],
        }

        # configファイルの出力
        with codecs.open(path.join(project_dirpath, "config.yaml"), "w", "utf-8") as f:
            yaml.dump(output, f, indent=2, allow_unicode=True, sort_keys=False)

    @classmethod
    def init_notification(cls, project_dirpath: str, force: bool = False):
        """
        [classmethod] 初期化用のNOTIFICATIONファイルを出力する

        - - -

        Parameters
        ----------
        project_dirpath: str
            プロジェクトが格納されたディレクトリパス
        force: bool, by default False
            フォルダにconfigファイルが存在する場合でも上書きして出力する
        """

        # configファイルが存在する場合はこの処理は実行しない
        if not force and path.exists(path.join(project_dirpath, "notification.yaml")):
            Logger(__name__).warning("The notification.yaml file already exists.")
            return

        # notificationファイルのテンプレート
        date = ("{}".format(datetime.now()))[:-3]
        output = {
            "version": NOTIFICATION_LATEST_VERSION,
            "type": ContainerType.NOTIFICATION,
            "id": str(uuid4()),
            "summary": {
                "creation_date": date,
                "update_date": date,
            },
            "gmail": {
                "enabled": True,
                "finished_notification": True,
                "sender": "",
                "app_pw": "",
                "receiver": "",
            },
        }

        # configファイルの出力
        with codecs.open(path.join(project_dirpath, "notification.yaml"), "w", "utf-8") as f:
            yaml.dump(output, f, indent=2, allow_unicode=True, sort_keys=False)

    @classmethod
    def validate_workflow(cls, workflow: "list[dict[str, any]]", logger: Logger, safe_stop: bool = True) -> list[str]:
        """[classmethod] workflowのバリデーションを行う"""
        err_msg: list[str] = []

        # 存在しないワークフローの検出
        for w in workflow:
            if hasattr(DiscontinuedTaskType, w["name"]):
                # 廃止されたワークフローの場合は、エラーメッセージを追加
                logger.warning(f"{w['name']} (Discontinued)")
                logger.warning(f"  {getattr(DiscontinuedTaskType, w['name'])}")

                if safe_stop:
                    err_msg.append(f"{w['name']} (Discontinued)")
                    err_msg.append(f"  {getattr(DiscontinuedTaskType, w['name'])}")

            elif not hasattr(TaskType, w["name"]):
                # 存在しないワークフローの場合は、エラーメッセージを追加
                logger.warning(f"{w['name']} (Unknown)")

                if safe_stop:
                    err_msg.append(f"{w['name']} (Unknown)")

        return err_msg

    @classmethod
    def compile_workflow(cls, workflow: "list[dict[str, any]]") -> "list[dict[str, any]]":
        """[classmethod/recursive] workflow内の解析(主にLOOP処理)を行い、その結果を返す"""

        # ワークフローの解析および変換
        adj = 0  # 調整用変数
        for i in range(len(workflow)):
            # Workflowタイプの解析
            if workflow[i + adj]["name"] == "LOOP":
                # 内部にLOOPが含まれていないか再コンパイル
                compiled_workflow = cls.compile_workflow(workflow[i + adj]["workflow"])

                # ループ回数の設定
                roop_max = workflow[i + adj]["count"] if "count" in workflow[i + adj] else 1
                result = []
                for j in range(roop_max):
                    for cw in compiled_workflow:
                        tmp: dict[str, list[int]] = deepcopy(cw)
                        tmp["loop_count"] = tmp.get("loop_count", [])
                        tmp["loop_count"].insert(0, j)
                        tmp["loop_max"] = tmp.get("loop_max", [])
                        tmp["loop_max"].insert(0, roop_max)
                        result.append(tmp)

                # ループ処理を終えたら、元のワークフローからLOOPを削除し、ループ処理を終えたワークフローを挿入する
                workflow.pop(i + adj)
                for j in range(len(result)):
                    workflow.insert(i + adj + j, result[j])
                adj += len(result) - 1

            else:
                # loop_countとloop_maxを配列で初期化
                workflow[i + adj]["loop_count"] = []
                workflow[i + adj]["loop_max"] = []

        return workflow

    @classmethod
    def __load_current_config(
        cls,
        wc: WorkflowConfig,
        project_dirpath: str,
        config: "dict[str, any]",
    ) -> "tuple[WorkflowConfig, list[str]]":
        """[private/classmethod] 現在のバージョンのConfigを渡されたWorkflowConfigに読み込む"""

        err_msg: list[str] = []

        wc.config_id = config["id"]
        wc.target_filepath = config["summary"]["target_filepath"]
        wc.log_filepath = config["summary"]["log_filepath"]
        wc.log_level = config["summary"]["log_level"] if "log_level" in config["summary"] else "INFO"
        wc.notification_filepath = (
            config["summary"]["notification_filepath"] if "notification_filepath" in config["summary"] else None
        )
        wc.random_reset_mode = (
            config["summary"]["random_reset_mode"]
            if "random_reset_mode" in config["summary"]
            else RandomResetMode.EVERYTIME
        )
        wc.random_seed = config["summary"]["random_seed"] if "random_seed" in config["summary"] else 42
        wc.crowdsourcing_platform = (
            config["summary"]["crowdsourcing_platform"]
            if "crowdsourcing_platform" in config["summary"]
            else CSPlatform.SIMULATION
        )
        wc.crowdsourcing_strict = (
            config["summary"]["crowdsourcing_strict"] if "crowdsourcing_strict" in config["summary"] else False
        )
        wc.crowdsourcing_limit = (
            config["summary"]["crowdsourcing_limit"]
            if "crowdsourcing_limit" in config["summary"] and config["summary"]["crowdsourcing_limit"] is not None
            else sys.maxsize
        )
        wc.crowdsourcing_worker_accuracy = (
            config["summary"]["crowdsourcing_worker_accuracy"]
            if "crowdsourcing_worker_accuracy" in config["summary"]
            else 1.0
        )
        wc.ngraph_construction = config["summary"]["ngraph_construction"]
        wc.ngraph_params = NGraphParams(**config["summary"]["ngraph_params"])
        wc.inference_mode = config["summary"]["inference_mode"]

        # Workflowのバリデートを行う
        logger = Logger(
            __name__,
            logger_config=LoggerConfig(level=wc.log_level),
            filepath=path.join(project_dirpath, wc.log_filepath),
        )
        err_msg += cls.validate_workflow(config["workflow"], logger)

        # WorkflowはLoopの解析をした後、インデックスが付与される
        wc.workflow = cls.compile_workflow(config["workflow"])
        for i, w in enumerate(wc.workflow):
            w["idx"] = i

        # TODO: workflowが何もない場合、エラーでも良いかも

        # メトリクス初期化
        wc.metrics = Metrics(
            workflow=wc.workflow,
            log_filepath=path.join(project_dirpath, wc.log_filepath),
            log_level=wc.log_level,
        )

        # CUDA利用可否フラグ
        wc.gpu_status = cls.__get_gpu_status()

        return wc, err_msg

    @classmethod
    def __load_config_v2_2(cls, config: "dict[str, any]") -> "dict[str, any]":
        """設定ファイルの Version 2.2 および 3.0 の形式のものを 3.1 形式に変換する"""

        # version 2.2 | 3.0 >> 3.1
        config["version"] = "3.1"

        # [add] default value for NGRAPH_PARAMS.
        if "ngraph_params" not in config["summary"]:
            config["summary"]["ngraph_params"] = {
                "k": 10,
                "dist": 1.0,
                "enable_pseudo_dist": False,
            }

        return config

    @classmethod
    def __load_current_suspend(
        cls,
        wc: WorkflowConfig,
        suspend: "dict[str, any]",
        collection_filepath: str,
        graph_filepath: str,
    ) -> WorkflowConfig:
        """[private/classmethod] 現在のバージョンのSuspendを渡されたWorkflowConfigに読み込む"""

        wc.suspend_id = suspend["id"]

        # pairdataの復元は、ngraph読み込み後に行う (suspendファイルが受け持つ範囲外)

        # graphを復元
        wc.graph.load(graph_filepath, collection_filepath)

        # current_workflowを復元
        wc.current_workflow = suspend["summary"]["current_workflow"]

        # crowdsourcing_filepathを復元
        wc.crowdsourcing_filepath = suspend["crowdsourcing_filepath"]

        # crowdsourcing_countとcrowdsourcing_task_countを復元
        wc.crowdsourcing_count = suspend["crowdsourcing_count"]
        wc.crowdsourcing_task_count = suspend["crowdsourcing_task_count"]

        # crowdsourcing_queueを復元
        wc.crowdsourcing_queue = []
        for value in suspend["crowdsourcing_queue"]:
            x = value.split(",")
            wc.crowdsourcing_queue.append(tuple(map(int, x)))

        # crowdsourcing_resultを復元
        for key, value in suspend["crowdsourcing_result"].items():
            a, b = key.split(",")
            wc.crowdsourcing_result[(int(a), int(b))] = value

        # contraction_result/contraction_machine_resultを復元
        wc.contraction_result = ContractionResult(**suspend["evaluation"]["contraction_result"])
        wc.contraction_machine_result = ContractionResult(**suspend["evaluation"]["contraction_machine_result"])

        # machine_contraction_pair/machine_misidentification_pairを復元
        for value in suspend["evaluation"]["machine_contraction_pair"]:
            a, b, c, d, e = value.split(",")
            wc.machine_contraction_pair.append(
                (
                    int(a),
                    int(b),
                    float(c),
                    float(d),
                    None if e == "None" else float(e),
                )
            )
        for value in suspend["evaluation"]["machine_misidentification_pair"]:
            a, b, c, d, e = value.split(",")
            wc.machine_misidentification_pair.append(
                (
                    int(a),
                    int(b),
                    float(c),
                    float(d),
                    None if e == "None" else float(e),
                )
            )

        # メトリクスの復元
        wc.metrics.load_metrics(suspend["metrics"])

        return wc

    @classmethod
    def __load_suspend_v3_1(cls, wc: WorkflowConfig, project_dirpath: str, suspend: "dict[str, any]"):
        """[private/classmethod] version 3.1 から 3.2 へ読み込める形式に変換する"""

        # version 3.1 >> 3.2
        suspend["version"] = "3.2"

        # idの格納方法をuuid(str)からノードインデックス(int)に変更
        rc = RecordContainer()
        rc.load_file(wc.target_filepath)
        record = rc.get_recordmg()

        re_id2index: "dict[str, int]" = {}
        for i, remg in enumerate(record):
            re_id2index[remg.re.id] = i

        # 格納形式の変換
        crowdsourcing_queue: "list[str]" = []
        for value in suspend["crowdsourcing_queue"]:
            a, b = value.split(",")
            crowdsourcing_queue.append(f"{re_id2index[a]},{re_id2index[b]}")
        suspend["crowdsourcing_queue"] = crowdsourcing_queue

        crowdsourcing_result: "dict[str, list[float]]" = {}
        for key, value in suspend["crowdsourcing_result"].items():
            a, b = key.split(",")
            crowdsourcing_result[f"{re_id2index[a]},{re_id2index[b]}"] = value
        suspend["crowdsourcing_result"] = crowdsourcing_result

        # resultについては外部ファイルに保存
        result = np.arange(len(record))
        for r in suspend["result"]:
            # 1つ目の要素をindexとして保存
            index = result[re_id2index[r[0]]]
            for i in r:
                result[re_id2index[r[i]]] = index
        np.savez(path.join(project_dirpath, "collection.txt"), result=result)

        return suspend

    @classmethod
    def __load_suspend_v3_0(cls, wc: WorkflowConfig, project_dirpath: str, suspend: "dict[str, any]"):
        """[private/classmethod] version 3.0 から 3.1 へ読み込める形式に変換する"""

        # version 3.0 >> 3.1
        suspend["version"] = "3.1"

        suspend["metrics"] = None

        # グラフ変換
        rc = RecordContainer()
        rc.load_file(wc.target_filepath)
        MGraph.convert_v3_0_graph(
            len(rc.records),
            suspend["graph"],
            path.join(project_dirpath, "graph.npz"),
        )

        suspend.pop("pairdata")
        suspend.pop("graph")

        return cls.__load_suspend_v3_1(wc, project_dirpath, suspend)

    @classmethod
    def __load_suspend_v2_2(cls, wc: WorkflowConfig, project_dirpath: str, suspend: "dict[str, any]"):
        """[private/classmethod] version 2.2のSuspendをversion 3.0のSuspendに変換する"""

        # version 2.2 >> 3.0
        suspend["version"] = "3.0"
        suspend["crowdsourcing_queue"] = []

        return cls.__load_suspend_v3_0(wc, project_dirpath, suspend)

    @classmethod
    def __load_current_notification(cls, wc: WorkflowConfig, notification: "dict[str, any]") -> WorkflowConfig:
        """[private/classmethod] 現在のバージョンのSuspendを渡されたWorkflowConfigに読み込む"""

        wc.gmail_enabled = notification["gmail"]["enabled"]
        wc.gmail_finished_notification = notification["gmail"]["finished_notification"]
        wc.gmail_sender = notification["gmail"]["sender"]
        wc.gmail_app_pw = notification["gmail"]["app_pw"]
        wc.gmail_receiver = notification["gmail"]["receiver"]

        return wc

    @classmethod
    def load_project(cls, project_dirpath: str) -> "tuple[WorkflowConfig | None, list[str]]":
        """[classmethod] プロジェクトディレクトリ直下のファイルを読み込んで必要なパラメータを返す"""

        # Error message
        err_msg: list[str] = []

        # config
        config = WorkflowConfig()

        config_filename = ""
        # 設定ファイルの存在を確認。存在しない場合は、エラーメッセージを返却して終了する
        for ext in ["yaml", "yml"]:
            __f = path.join(project_dirpath, f"config.{ext}")
            if __f:
                config_filename = __f

        if config_filename == "":
            Logger(__name__).error("The config.yaml file does not exist.")
            err_msg.append("The config.yaml file does not exist.")
            return None, err_msg

        # 設定ファイルから、必要な情報を取得する
        version = "1.0"
        with open(path.join(project_dirpath, config_filename), "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

            if "version" in data:
                version = data["version"]

        # version 3.1
        if version == CONFIG_LATEST_VERSION:
            content = data

        # version 2.2 | 3.0
        elif version == "2.2" or version == "3.0":
            content = cls.__load_config_v2_2(data)

        # version 2.1 以前についてはサポートしない
        else:
            err_msg.append("Failed yaml loading due to unsupported version.")
            return None, err_msg

        # 設定ファイルの読み込み
        data["version"] = CONFIG_LATEST_VERSION
        config, __err_msg = cls.__load_current_config(config, project_dirpath, content)
        err_msg += __err_msg

        # グラフ初期化
        WorkflowConfig.graph = MGraph(path.join(project_dirpath, config.log_filepath), config.log_level)

        logger = Logger(
            __name__,
            logger_config=LoggerConfig(level=config.log_level),
            filepath=path.join(project_dirpath, config.log_filepath),
        )

        # 初期表示出力
        logger.write_start_workflow(
            project_name=(
                path.abspath(project_dirpath).removeprefix(f"{path.abspath('@')}/")
                if path.abspath(project_dirpath).startswith(f"{path.abspath('@')}/project/")
                else path.basename(project_dirpath)
            ),
            start_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            gpu_status=config.gpu_status,
        )

        message = "Start loading config and suspend files."
        print(message)
        logger.info(message)

        # ==============================================================================================================

        suspend_filename = ""
        # 中断ファイルの存在を確認
        if path.isfile(path.join(project_dirpath, "suspend.yaml")):
            suspend_filename = "suspend.yaml"
        elif path.isfile(path.join(project_dirpath, "suspend.yml")):
            suspend_filename = "suspend.yml"

        # 中断ファイルが存在する場合は、中断ファイルを読み込んで値を返す
        if suspend_filename != "":
            with open(path.join(project_dirpath, suspend_filename), "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

                if "version" in data:
                    version = data["version"]

            # version 3.2
            if version == "3.2":
                content = data

            # version 3.1
            elif version == "3.1":
                content = cls.__load_suspend_v3_1(config, project_dirpath, data)

            # version 3.0
            elif version == "3.0":
                content = cls.__load_suspend_v3_0(config, project_dirpath, data)

            # version 2.2
            elif version == "2.2":
                content = cls.__load_suspend_v2_2(config, project_dirpath, data)

            # version 2.1 以前についてはサポートしない
            else:
                raise FileNotFoundError("Failed yaml loading.")

            # 中断ファイルの読み込み
            data["version"] = "3.1"
            config = cls.__load_current_suspend(
                config,
                content,
                path.join(project_dirpath, "collection.txt"),
                path.join(project_dirpath, "graph.npz"),
            )

        # ==============================================================================================================

        notification_filepath = config.notification_filepath
        # 通知設定ファイルの存在を確認
        if notification_filepath is not None and path.isfile(path.join(project_dirpath, notification_filepath)):
            notification_filepath = path.join(project_dirpath, config.notification_filepath)
        else:
            notification_filepath = None

        # メール通知設定が存在する場合は、その設定ファイルを読み込む
        if notification_filepath is not None and path.isfile(notification_filepath):
            with open(notification_filepath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

                if "version" in data:
                    version = data["version"]

            # version 2.2 | 3.0
            if version == "2.2" or version == NOTIFICATION_LATEST_VERSION:
                content = data

            # version 2.1 以前についてはサポートしない
            else:
                raise Exception("Failed notification yaml file loading.")

            # 通知設定の読み込み
            data["version"] = NOTIFICATION_LATEST_VERSION
            config = cls.__load_current_notification(config, content)

        # 読み込み終了ログ
        message = "Finished loading config and suspend files."
        logger.info(message)
        print(message)

        return config, err_msg

    @classmethod
    def update_project_meta(cls, project_dirpath: str, target: "list[str]"):
        """[classmethod] configファイルのメタデータを書き換える

        Params
        ------
        project_dirpath: str
            プロジェクトパス
        target: list[str]
            書き換え対象の設定
        """

        date = ("{}".format(datetime.now()))[:-3]

        if path.isfile(path.join(project_dirpath, "config.yaml")):
            # configファイルを直開きし、書き換える
            with open(path.join(project_dirpath, "config.yaml"), "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

                if "id" in target:
                    data["id"] = str(uuid4())

                if "creation_date" in target:
                    data["summary"]["creation_date"] = date

                if "update_date" in target:
                    data["summary"]["update_date"] = date

            # configファイルの出力
            with codecs.open(path.join(project_dirpath, "config.yaml"), "w", "utf-8") as f:
                yaml.dump(data, f, indent=2, allow_unicode=True, sort_keys=False)

    @classmethod
    def save_suspend_yaml(cls, project_dirpath: str, config: WorkflowConfig):
        """[classmethod] 中断を行うためのファイルを生成する"""

        Logger(
            __name__,
            logger_config=LoggerConfig(level=config.log_level),
            filepath=path.join(project_dirpath, config.log_filepath),
        ).info("Start saving a suspend file.")
        print("Start saving a suspend file.")

        nowdate = ("{}".format(datetime.now()))[:-3]  # 生成日時

        # 初期データの格納
        store = {}
        store["version"] = SUSPEND_LATEST_VERSION
        store["type"] = ContainerType.SUSPEND
        store["id"] = config.suspend_id
        store["summary"] = {
            "creation_date": config.suspend_creation_date,
            "update_date": nowdate,
            "config_id": None,
            "current_workflow": config.current_workflow,
        }
        store["metrics"] = config.metrics.to_dict()
        store["crowdsourcing_filepath"] = config.crowdsourcing_filepath
        store["crowdsourcing_count"] = config.crowdsourcing_count
        store["crowdsourcing_task_count"] = config.crowdsourcing_task_count
        store["crowdsourcing_queue"] = []
        store["crowdsourcing_result"] = {}
        store["evaluation"] = {}

        # crowdsourcing_queueを格納
        crowdsourcing_queue_store = []
        for value in config.crowdsourcing_queue:
            crowdsourcing_queue_store.append(",".join(map(str, value)))
        store["crowdsourcing_queue"] = crowdsourcing_queue_store

        # crowdsourcing_resultを格納
        crowdsourcing_result_store = {}
        for a, b in config.crowdsourcing_result.keys():
            crowdsourcing_result_store["{},{}".format(a, b)] = config.crowdsourcing_result[(a, b)]
        store["crowdsourcing_result"] = crowdsourcing_result_store

        # graphの出力
        WorkflowConfig.graph.save(path.join(project_dirpath, "graph.npz"), path.join(project_dirpath, "collection.txt"))

        if config.current_workflow == len(config.workflow):
            store["type"] = ContainerType.FINISHED

        # evaluationの下処理と格納
        evaluation_store = {}
        evaluation_store["contraction_result"] = config.contraction_result.to_dict()
        evaluation_store["contraction_machine_result"] = config.contraction_machine_result.to_dict()
        evaluation_store["machine_contraction_pair"] = [
            "{},{},{},{},{}".format(a, b, c, d, e) for a, b, c, d, e in config.machine_contraction_pair
        ]
        evaluation_store["machine_misidentification_pair"] = [
            "{},{},{},{},{}".format(a, b, c, d, e) for a, b, c, d, e in config.machine_misidentification_pair
        ]
        store["evaluation"] = evaluation_store

        # suspendファイルの出力
        with codecs.open(path.join(project_dirpath, "suspend.yml"), "w", "utf-8") as f:
            yaml.dump(store, f, indent=2, allow_unicode=True, sort_keys=False)

        Logger(
            __name__,
            logger_config=LoggerConfig(level=config.log_level),
            filepath=path.join(project_dirpath, config.log_filepath),
        ).info("Finished saving a suspend file.")
        print("Finished saving a suspend file.")


class ParamsContainer:
    """パラメータの保存と読み込みを行うコンテナクラス"""

    def __init__(self) -> None:
        """コンストラクタ"""

        self.version = PARAMS_LATEST_VERSION
        self.type: str = ContainerType.PARAMS
        self.id: str = str(uuid4())
        self.creation_date: str = ("{}".format(datetime.now()))[:-3]
        self.weight_match: int = 0
        self.weight_mismatch: int = 0

        self.fasttext_match: "list[BetaParams] | list[GammaParams] | list[GaussianParams]" = None
        self.fasttext_mismatch: "list[BetaParams] | list[GammaParams] | list[GaussianParams]" = None
        self.difflib_match: "list[BetaParams] | list[GammaParams] | list[GaussianParams]" = None
        self.difflib_mismatch: "list[BetaParams] | list[GammaParams] | list[GaussianParams]" = None
        self.leven_match: "list[BetaParams] | list[GammaParams] | list[GaussianParams]" = None
        self.leven_mismatch: "list[BetaParams] | list[GammaParams] | list[GaussianParams]" = None
        self.jaro_match: "list[BetaParams] | list[GammaParams] | list[GaussianParams]" = None
        self.jaro_mismatch: "list[BetaParams] | list[GammaParams] | list[GaussianParams]" = None
        self.sun_params: SunParams = None

    def update_params(
        self,
        fasttext_match=None,
        fasttext_mismatch=None,
        difflib_match=None,
        difflib_mismatch=None,
        leven_match=None,
        leven_mismatch=None,
        jaro_match=None,
        jaro_mismatch=None,
        weight_match=None,
        weight_mismatch=None,
    ):
        """パラメータを更新する"""

        self.fasttext_match = fasttext_match if fasttext_match is not None else self.fasttext_match
        self.fasttext_mismatch = fasttext_mismatch if fasttext_mismatch is not None else self.fasttext_mismatch
        self.difflib_match = difflib_match if difflib_match is not None else self.difflib_match
        self.difflib_mismatch = difflib_mismatch if difflib_mismatch is not None else self.difflib_mismatch
        self.leven_match = leven_match if leven_match is not None else self.leven_match
        self.leven_mismatch = leven_mismatch if leven_mismatch is not None else self.leven_mismatch
        self.jaro_match = jaro_match if jaro_match is not None else self.jaro_match
        self.jaro_mismatch = jaro_mismatch if jaro_mismatch is not None else self.jaro_mismatch
        self.weight_match = weight_match if weight_match is not None else self.weight_match
        self.weight_mismatch = weight_mismatch if weight_mismatch is not None else self.weight_mismatch

    def get_params_list(self) -> "list[list[BetaParams] | list[GammaParams] | list[GaussianParams]]":
        """パラメータをリストで取得する"""

        return [
            self.fasttext_match,
            self.fasttext_mismatch,
            self.difflib_match,
            self.difflib_mismatch,
            self.leven_match,
            self.leven_mismatch,
            self.jaro_match,
            self.jaro_mismatch,
        ]

    @classmethod
    def _load_yaml_v3_1(cls, content: "dict[str, any]" = {}):
        # version 3.1 >> 3.2への変換
        # weightの追加
        content["summary"]["weight_match"] = 5000
        content["summary"]["weight_mismatch"] = 5000

        content["version"] = "3.2"

        return content

    @classmethod
    def _load_yaml_v2_2(cls, content: "dict[str, any]" = {}):
        # version 2.2, 3.0 >> 3.1への変換
        for s in ["same", "not_same"]:
            for m in ["fasttext", "leven"]:
                content[s][m]["type"] = ParamsType.GAMMA
                content[s][m].pop("scale")
            for m in ["difflib", "jaro"]:
                content[s][m]["type"] = ParamsType.BETA

        content["fasttext_match"] = [content["same"]["fasttext"]]
        content["fasttext_mismatch"] = [content["not_same"]["fasttext"]]
        content["difflib_match"] = [content["same"]["difflib"]]
        content["difflib_mismatch"] = [content["not_same"]["difflib"]]
        content["leven_match"] = [content["same"]["leven"]]
        content["leven_mismatch"] = [content["not_same"]["leven"]]
        content["jaro_match"] = [content["same"]["jaro"]]
        content["jaro_mismatch"] = [content["not_same"]["jaro"]]
        content["sun_params"] = content["sun"]

        content["version"] = "3.1"

        return cls._load_yaml_v3_1(content)

    def _load_current_version(self, content: "dict[str, str]" = {}):
        """読み込んだデータを変換してコンテナに格納する"""

        def convert_params(data: "dict[str, any]"):
            if data[0]["type"] == ParamsType.GAMMA:
                return [GammaParams(**d) for d in data]
            elif data[0]["type"] == ParamsType.BETA:
                return [BetaParams(**d) for d in data]
            elif data[0]["type"] == ParamsType.GAUSSIAN:
                return [GaussianParams(**d) for d in data]

        self.id = content["id"]
        self.creation_date = content["summary"]["creation_date"]

        self.fasttext_match = convert_params(content["fasttext_match"])
        self.fasttext_mismatch = convert_params(content["fasttext_mismatch"])
        self.difflib_match = convert_params(content["difflib_match"])
        self.difflib_mismatch = convert_params(content["difflib_mismatch"])
        self.leven_match = convert_params(content["leven_match"])
        self.leven_mismatch = convert_params(content["leven_mismatch"])
        self.jaro_match = convert_params(content["jaro_match"])
        self.jaro_mismatch = convert_params(content["jaro_mismatch"])
        self.weight_match = content["summary"]["weight_match"]
        self.weight_mismatch = content["summary"]["weight_mismatch"]

        self.sun_params = SunParams(**content["sun_params"])

    @classmethod
    def load_params(cls, filepath: str):
        """
        yaml形式を読み込んでコンテナを構築する

        - - -

        フィッティングに関するパラメータファイルを読み込む
        """
        # 拡張子からyamlかymlかを判定する
        _, ext = path.splitext(filepath)

        # version 情報を取得する
        version = "1.0"
        with open(filepath, "r", encoding="utf-8") as f:
            if ext == ".yml" or ext == ".yaml":
                data = yaml.safe_load(f)

                if "version" in data:
                    version = data["version"]

        # version 3.2
        if version == "3.2":
            content = data

        # version 3.1
        elif version == "3.1":
            content = cls._load_yaml_v3_1(data)

        # version 2.2 | 3.0
        elif version == "2.2" or version == "3.0":
            content = cls._load_yaml_v2_2(data)

        # version 2.1 以前についてはサポートしない
        else:
            raise Exception("failed yaml loading")

        # 読み込んだ内容を返却する
        params_container = ParamsContainer()
        params_container._load_current_version(content)
        return params_container

    def save_params_yaml(self, filepath: str):
        """
        パラメータファイルを保存する

        - - -

        Params
        ------
        filepath: str
            パラメータファイルの保存先パス
        """

        # データの格納
        store = {}
        store["version"] = PARAMS_LATEST_VERSION
        store["type"] = self.type
        store["id"] = self.id
        store["summary"] = {
            "creation_date": self.creation_date,
            "update_date": ("{}".format(datetime.now()))[:-3],  # 生成日時
            "weight_match": self.weight_match,
            "weight_mismatch": self.weight_mismatch,
        }

        store["fasttext_match"] = [asdict(p) for p in self.fasttext_match]
        store["fasttext_mismatch"] = [asdict(p) for p in self.fasttext_mismatch]
        store["difflib_match"] = [asdict(p) for p in self.difflib_match]
        store["difflib_mismatch"] = [asdict(p) for p in self.difflib_mismatch]
        store["leven_match"] = [asdict(p) for p in self.leven_match]
        store["leven_mismatch"] = [asdict(p) for p in self.leven_mismatch]
        store["jaro_match"] = [asdict(p) for p in self.jaro_match]
        store["jaro_mismatch"] = [asdict(p) for p in self.jaro_mismatch]

        store["sun_params"] = self.sun_params.to_dict()

        # パラメータファイルの出力
        with codecs.open(filepath, "w", "utf-8") as f:
            yaml.dump(store, f, indent=2, allow_unicode=True, sort_keys=False)


class RecordContainer:
    """
    Recordの読み込みや書き込み等の処理を行い、メインプログラムの補助を行うコンテナクラス

    読み込み可能なContainterType: NORMAL, (一部PAIR)
    """

    def __init__(self, log_filepath: str = None, log_level: str = "INFO") -> None:
        """コンストラクタ"""

        self.records: "list[Record]" = []  # ファイルから順番にレコードを格納する
        self.clusters: "dict[str, list[Record]]" = {}  # cluster_idをキーにして書誌データ群を格納する
        self.clusters_amount: "dict[int, list[str]]" = {}  # グループ数をキーにして値にはcluster_idのリストを格納する
        self.filepath: str = None  # 読み込みに利用されたファイルのパス
        self.logger: Logger = Logger(__name__, logger_config=LoggerConfig(level=log_level), filepath=log_filepath)

        self.id: str = str(uuid4())
        self.inf_attr: "dict[str, RecordType]" = {}
        self.type: str = ContainerType.UNKNOWN  # 目的別に応じた型
        self.creation_date = ("{}".format(datetime.now()))[:-3]  # 生成日時
        self.config_match: int | None = None  # 書誌データ作成設定時の「一致」のペア数 (学習用データ作成時に作成／任意)
        self.config_mismatch: int | None = (
            None  # 書誌データ作成設定時の「不一致」のペア数 (学習用データ作成時に作成／任意)
        )

        self.version = RECORD_LATEST_VERSION

    def __reindexing_record(self):
        """現在格納されているRecord各々に対して、インデックスを振り直す"""

        for i, r in enumerate(self.records):
            r.idx = i

    def __calc_clusters_amount(self):
        """現在格納されているレコードを基にrecord_amount_groupを再構成する"""
        self.clusters_amount = {}  # 初期化

        # 統計情報を取得
        for key, value in self.clusters.items():
            self.clusters_amount[len(value)] = self.clusters_amount.get(len(value), [])  # 総数順に辞書型に格納
            self.clusters_amount[len(value)].append(key)

    def construct_by_records(
        self,
        records: "list[Record]",
        inf_attr: "dict[str, RecordType]",
    ):
        """
        Recordのリストからコンテナを構築する

        - - -

        Parameter
        ---------
        records: list[Record]
            Recordのリスト
        inf_attr: dict[str, RecordType]
            Recordの属性値の型を指定する
        """

        self.records = records
        self.clusters = {}
        self.clusters_amount = {}

        self.id = str(uuid4())
        self.type = ContainerType.TARGET
        self.inf_attr: "dict[str, RecordType]" = inf_attr

        for record in records:
            # クラスターを構成する
            if record.cluster_id is not None:
                self.clusters[record.cluster_id] = self.clusters.get(record.cluster_id, [])
                self.clusters[record.cluster_id].append(record)

        self.__reindexing_record()
        self.__calc_clusters_amount()

    def load_tsv(
        self,
        filepath: str,
        cluster_id: str = None,
        delimiter: str = "\t",
        attributes: "dict[str, RecordType]" = {},
        display_summary: bool = True,
    ):
        """
        tsv形式で格納された書誌データファイルを読み込んでコンテナを構築する

        - - -

        Parameter
        ---------
        filepath: str
            tsvファイルの場所を指定
        cluster_id: str
            グループを識別するための属性名
        delimiter: str
            区切り文字
        attributes: dict[str, RecordType]
            属性値の型を指定する (IGNOREはデータは所持するが推論時に利用しない。指定がないものはデータそのものも格納しない)
        display_summary: bool
            読み込み後にサマリーを表示するかどうか
        """

        # 初期化
        self.records = []
        self.clusters = {}
        self.clusters_amount = {}
        self.filepath = filepath

        self.id = str(uuid4())
        self.type = ContainerType.TARGET
        self.inf_attr: "dict[str, RecordType]" = {}
        self.config_match = None
        self.config_mismatch = None

        self.version = RECORD_LATEST_VERSION
        attr = {}

        with open(filepath, "r", encoding="utf-8") as f:
            tsv = reader(f, delimiter=delimiter)

            for i, row in enumerate(tsv):
                # 1行目は属性値として格納する
                if i == 0:
                    for j, r in enumerate(row):
                        # 引数で与えられたattributesを参考に、属性値を格納する
                        if r in attributes or r == cluster_id:
                            attr[r] = j
                            if r in attributes and attributes[r] != RecordType.IGNORE:
                                self.inf_attr[r] = attributes[r]

                # dataが欠損してRecordを構成できない場合は、スキップする
                else:
                    try:
                        data = {}
                        _cluster_id = ""
                        for key in attr.keys():
                            if key == cluster_id:
                                _cluster_id = row[attr[key]]
                            else:
                                data[key] = row[attr[key]]

                        record = Record(id=str(uuid4()), idx=-1, cluster_id=_cluster_id, data=data)

                        # 順にレコードを格納
                        self.records.append(record)

                        # clusterごとに書誌データを格納
                        if cluster_id is not None:
                            self.clusters[record.cluster_id] = self.clusters.get(record.cluster_id, [])
                            self.clusters[record.cluster_id].append(record)

                    except IndexError:
                        continue

        self.__reindexing_record()  # Recordにインデックスを振り直す
        self.__calc_clusters_amount()  # clusters_amount を構築する

        # サマリーの表示
        if display_summary:
            self.summary()

    def load_yaml(self, filepath: str):
        """
        yaml形式を読み込んでコンテナを構築する

        - - -
        yamlファイルで構成された書誌データファイル
        """
        # 拡張子からymlかyamlかを判定する
        _, ext = path.splitext(filepath)

        # version 情報を取得する
        version = "1.0"
        with open(filepath, "r", encoding="utf-8") as f:
            if ext == ".yml" or ext == ".yaml":
                data = yaml.safe_load(f)

            if "version" in data:
                version = data["version"]

        # version 3.1 (yaml)
        if version == "3.1":
            self.__load_current_version(data)

        # version 3.0 (yaml)
        elif version == "3.0":
            self.__load_yaml_v3_0(data)

        # version 2.2 (yaml)
        elif version == "2.2":
            self.__load_yaml_v2_2(data)

        # version 2.1 以前についてはサポートしない
        else:
            raise RuntimeError("Failed yaml loading.")

        self.filepath = filepath

    def load_file(self, filepath: str):
        """
        ファイルの拡張子によって読み込むファイル形式を判別し、読み込む

        - - -

        Parameter
        ---------
        filepath: str
            ファイルのパス

        """

        # 拡張子自動判別
        _, ext = path.splitext(filepath)

        if ext == ".tsv":
            self.load_tsv(filepath)
        elif ext == ".csv":
            self.load_tsv(filepath, delimiter=",")
        elif ext == ".yml" or ext == ".yaml":
            self.load_yaml(filepath)
        else:
            raise Exception("Failed loading")

    def __load_current_version(self, content: "dict[str, str]"):
        """現在のバージョンのyaml形式を読み込む"""

        self.id = content["id"]
        self.type = content["type"]
        self.creation_date = content["summary"]["creation_date"]
        self.config_match = content["summary"]["config_match"]
        self.config_mismatch = content["summary"]["config_mismatch"]
        self.inf_attr = content["inf_attr"]

        self.records = []
        for k, v in content["records"].items():
            records = []
            for r in v:
                record = Record(idx=-1, **r)
                self.records.append(record)
                records.append(record)
            self.clusters[k] = records

        self.__reindexing_record()  # Recordにインデックスを振り直す
        self.__calc_clusters_amount()  # clusters_amount を構築する

        self.version = RECORD_LATEST_VERSION

    def __load_yaml_v3_0(self, content: "dict[str, any]" = {}, filepath: str = None):
        """[private] version 3.0 から 3.1 へ読み込める形式に変換する"""

        # 属性値の変換
        inf_attr = {}
        for k, v in content["inf_attr"].items():
            if v == "NORMAL":
                inf_attr[k] = "TEXT"  # RecordType.TEXT
            else:
                inf_attr[k] = v
        content["inf_attr"] = inf_attr

        content["records"] = content["data"]  # リネーム
        content.pop("data")

        for k, v in content["records"].items():
            for i, _ in enumerate(v):
                content["records"][k][i]["cluster_id"] = k
                content["records"][k][i].pop("group")

        content["version"] = "3.1"

        return self.__load_current_version(content)

    def __load_yaml_v2_2(self, content: "dict[str, any]"):
        """[private] version 2.2 から 3.0 へ読み込める形式に変換する"""

        content["summary"]["num_of_records"] = content["summary"]["num_of_bookdata"]
        content["summary"].pop("num_of_bookdata")
        content["summary"]["type"] = "TARGET"  # ContainerType.TARGET
        content["inf_attr"] = {
            "title": "NORMAL",  # RecordType.NORMAL
            "author": "COMPLEMENT_JA",  # RecordType.COMPLEMENT_JA
            "publisher": "COMPLEMENT_JA",  # RecordType.COMPLEMENT_JA
            "pubdate": "COMPLEMENT_DATE",  # RecordType.COMPLEMENT_DATE
        }

        content["data"] = content["books"]
        content.pop("books")

        for k, v in content["data"].items():
            records = []
            for r in v:
                records.append(
                    {
                        "id": r["id"],
                        "group": r["group"],
                        "data": {
                            "title": r["title"],
                            "author": r["author"],
                            "publisher": r["publisher"],
                            "pubdate": r["pubdate"],
                        },
                    }
                )
            content["data"][k] = records

        content["version"] = "3.0"

        return self.__load_yaml_v3_0(content)

    def __save_record_to_yaml(
        self,
        filepath: str,
        target: "dict[str, list[Record]]" = None,
        config: "dict[str, any]" = {},
    ):
        """コンテナに格納されているデータ群をyaml形式で保存する"""

        if target is None:
            target = self.clusters

        # 推論属性が存在しない場合は全ての属性をTEXT属性として扱う
        if len(self.inf_attr.keys()) == 0:
            for key in target[target.keys()[0]][0].data.keys():
                self.inf_attr[key] = RecordType.TEXT

        clusters: "dict[str, dict[str, str]]" = {}
        num_of_pairs = {}  # ペア数の計算
        counter = 0  # データ数

        # データを辞書型に変換
        for key, value in target.items():
            data = []
            for v in value:
                data.append(v.to_dict_for_save())
                counter += 1
            clusters[key] = data
            num_of_pairs[len(value)] = num_of_pairs.get(len(value), 0) + 1

        # 概要情報を作成
        summary = {
            "creation_date": self.creation_date,
            "update_date": ("{}".format(datetime.now()))[:-3],
            "num_of_records": counter,
            "num_of_pairs": num_of_pairs,
            "config_match": config["match"] if "match" in config else None,
            "config_mismatch": config["mismatch"] if "mismatch" in config else None,
        }

        output = {
            "version": self.version,
            "type": self.type,
            "id": self.id,
            "summary": summary,
            "inf_attr": self.inf_attr,
            "records": clusters,
        }

        # 結果をyamlファイルとして書き出す
        with codecs.open(filepath, "w", "utf-8") as f:
            yaml.dump(output, f, indent=2, allow_unicode=True, sort_keys=False)

    def save_yaml(self, filepath: str):
        """
        このコンテナをyaml形式で保存する

        - - -

        Params
        ------
        filepath: str
            保存先のファイルパス
        """

        self.__save_record_to_yaml(filepath)

    def get_pairdata(self, same: int, not_same: int):
        """
        一致データと不一致データのidペアを返す

        - - -

        Params
        ------
        same: int
            一致データの数
        not_same: int
            不一致データの数
        """

        record_same: "list[tuple[str, str]]" = []
        record_not_same: "list[tuple[str, str]]" = []

        # 一致ペアの候補を生成
        candidate = set([x for num, v in self.clusters_amount.items() if num >= 2 for x in v])

        # TODO: 候補がない場合はログにエラーを出すようにするとよいかも

        # 訓練用一致ペアデータの生成
        counter = 0
        while counter < same and len(candidate) > 0:
            cluster_id = list(candidate)[int(randrange(0, len(candidate)))]

            # 同一書誌データからn数を使って、nC2ペアを生成する
            num_of_bd = len(self.clusters[cluster_id])
            counter += int(num_of_bd * (num_of_bd - 1) / 2)

            for i in range(num_of_bd - 1):
                for j in range(i + 1, num_of_bd):
                    record_same.append([self.clusters[cluster_id][i].id, self.clusters[cluster_id][j].id])

            candidate = candidate - set([cluster_id])

        if same < len(record_same):
            record_same = record_same[:same]

        # 不一致ペアの候補を生成
        cluster_id = set([x for _, v in self.clusters_amount.items() for x in v])
        candidate = []
        for i in range(len(cluster_id) - 1):
            for j in range(1, len(cluster_id)):
                _group = list(cluster_id)
                candidate.append((_group[i], _group[j]))
        shuffle(candidate)

        for c in candidate:
            num_of_bd1 = len(self.clusters[c[0]])
            num_of_bd2 = len(self.clusters[c[1]])

            for i in range(num_of_bd1):
                for j in range(num_of_bd2):
                    record_not_same.append([self.clusters[c[0]][i].id, self.clusters[c[1]][j].id])

            if not_same <= len(record_not_same):
                record_not_same = record_not_same[:not_same]
                break

        return record_same, record_not_same

    def save_yaml_for_result(
        self,
        config: WorkflowConfig,
        filepath: str = "./result.yaml",
        target: "dict[str, list[Record]]" = None,
    ):
        """
        同定結果をyaml形式で保存する

        - - -

        Parameters
        ----------
        inference: list[set[str]]
            近傍グラフを縮約して得られた推論結果
        filepath: str, by default './result.json'
            テスト用データをyaml形式で出力する際のファイルパス
        target: dict[str, list[Record]], by default None
            グループをキーに、そのグループに属するまとめた書誌データを値とした辞書型
        """

        if target is None:
            target = self.clusters

        # 一致と推論した全てのペアをリスト形式に変換
        c_unique = np.unique(WorkflowConfig.graph.collection)
        collection: list[list[int]] = [list(np.where(WorkflowConfig.graph.collection == c)[0]) for c in c_unique]

        # 推論群順に格納
        counter = 0
        num_of_pairs = 0
        group = []

        for col in collection:
            records = []
            record_idx = set([])
            correct: "set[tuple[str]]" = set([])
            for c in col:
                records.append(self.records[c].to_dict())
                record_idx.add(self.records[c].idx)
                correct.add(tuple(sorted([re.idx for re in self.clusters[self.records[c].cluster_id]])))
                counter += 1
            record_idx = tuple(sorted(list(record_idx)))
            group.append(
                {
                    "perfect_match": len(correct) == 1 and list(correct)[0] == record_idx,
                    "records": records,
                    "correct": [list(c) for c in correct],
                }
            )
            num_of_pairs += 1

        # 推論群の評価値を計算
        g_ev, g_complete = self.verify_all_record_group(WorkflowConfig.graph.collection)
        g_ev_calc = g_ev.calc_evaluation()

        p_ev = self.verify_all_record_pairs(WorkflowConfig.graph.collection)

        # 概要情報を作成
        summary = {
            "type": ContainerType.RESULT,
            "num_of_record": counter,
            "num_of_groups(correct)": len(target.keys()),
            "num_of_groups(inference)": num_of_pairs,
            "config_match": None,
            "config_mismatch": None,
            "crowdsourcing_count": config.crowdsourcing_count,
            "f1(pair)": "{:.5}".format(p_ev.calc_f1()),
            "precision(pair)": "{:.5}".format(p_ev.calc_precision()),
            "recall(pair)": "{:.5}".format(p_ev.calc_recall()),
            "complete(group)": "{:.5} ( {} / {} )".format(g_ev_calc.complete_nu, g_ev.complete_nu, g_ev.complete_de),
            "precision(group)": "{:.5} ( {} / {} )".format(
                g_ev_calc.precision_nu, g_ev.precision_nu, g_ev.precision_de
            ),
            "recall(group)": "{:.5} ( {} / {} )".format(g_ev_calc.recall_nu, g_ev.recall_nu, g_ev.recall_de),
            "complete_group": g_complete,
        }

        # 結果を格納
        output = {
            "version": self.version,
            "type": ContainerType.RESULT,
            "id": self.id,
            "summary": summary,
            "group": group,
        }

        # 結果をyamlファイルとして書き出す
        with codecs.open(filepath, "w", "utf-8") as f:
            yaml.dump(output, f, indent=2, allow_unicode=True, sort_keys=False)

    def get_recordmg(self) -> "list[RecordMG]":
        """RecordMG形式に変換して返す"""

        return [RecordMG(re, self.inf_attr) for re in self.records]

    def get_all_match_pairs_index(self):
        """全ての一致ペアのインデックスを返す"""

        # 一致ペアのインデックスを作成
        idx_pairs: "list[tuple[int, int]]" = []
        for c in self.clusters.values():
            for a in range(len(c) - 1):
                for b in range(a + 1, len(c)):
                    idx_pairs.append(tuple(sorted([c[a].idx, c[b].idx])))

        return idx_pairs

    def generate_benchmark_dataset(self, required_records_list: list[int], dirpath: str = "./benchmark"):
        """
        ベンチマーク用のデータセットを生成する

        - - -

        Params
        ------
        required_records_list: list[int]
            生成するレコード数を含めたリスト
            各要素間は排他的で、レコード数が指定された数を超えるように生成する
        dirpath: str, by default './benchmark'
            生成したデータセットを保存するディレクトリパス
        """

        # ディレクトリが存在しない場合は作成
        if not path.exists(dirpath):
            os.makedirs(dirpath)

        # クラスターidの取得
        cluster_ids = list(self.clusters.keys())
        shuffle(cluster_ids)

        # 各クラスターをランダムに選択してレコード群を生成
        records: list[list[Record]] = [[] for _ in range(len(required_records_list))]
        i = len(required_records_list) - 1
        for cluster_id in cluster_ids:
            i = (len(required_records_list) if i == 0 else i) - 1

            if len(records[i]) >= min(required_records_list):
                # 条件を満たした場合は出力を行う
                _rc = RecordContainer()
                _rc.construct_by_records(records[i], self.inf_attr)
                _rc.save_yaml(path.join(dirpath, f"benchmark_{len(_rc.records)}_{uuid4()}.yml"))

                # records[i] を pop し required_records_list の最小値も pop する
                records.pop(i)
                required_records_list.pop(required_records_list.index(min(required_records_list)))

                # records が空になった場合は終了
                if len(records) == 0:
                    break

            else:
                records[i] += self.clusters[cluster_id]

        # 残存しているレコード群を出力
        for r in records:
            _rc = RecordContainer()
            _rc.construct_by_records(r, self.inf_attr)
            _rc.save_yaml(path.join(dirpath, f"benchmark_{len(_rc.records)}_{uuid4()}.yml"))

    def get_recordmg_for_train(
        self,
        match_num: "int | None" = None,
        mismatch_ratio: float = 1,
        max_in_cluster: int = 50,
        labeling_function: "Callable[[RecordMG, RecordMG, int], float]" = None,
    ) -> "tuple[list[RecordMG], list[RecordMG], list[int], list[int]]":
        """
        格納されている書誌データを学習用フォーマットに直して返す

        - - -

        Params
        ------
        match_num: int | None, by default None
            一致ペア数, Noneが指定された場合は、このコンテナが持つペア数が指定される
        mismatch_ratio: float, by default 1
            一致ペア数に対して不一致ペア数の比率
        max_in_cluster: int, by default 50
            クラスター内の最大ペア数

        Return
        ------
        list[RecordMG]
            一致のペア順で格納されたRecordMGのリスト
        list[RecordMG]
            不一致のペア順で格納されたRecordMGのリスト
        list[int]
            一致ペアのラベルが格納されたリスト
        list[int]
            不一致ペアのラベルが格納されたリスト
        """

        # 一致ペア数の下処理
        if match_num is None:
            if self.config_match is not None:
                match_num = self.config_match  # 特に指定がなくconfigに設定を持っている場合は、configの設定を使う
            else:
                match_num = sys.maxsize // 2  # 特に指定がなくconfigに設定もない場合は、システム上の最大値を使う

        # ratioから不一致データの数を計算
        mismatch_num = int(match_num * mismatch_ratio)

        match_pairs: "list[RecordMG]" = []
        match_labels: "list[int | float]" = []
        mismatch_pairs: "list[RecordMG]" = []
        mismatch_labels: "list[int | float]" = []

        # 一致書誌データの生成
        for k in self.clusters.keys():
            num_of_pair = len(self.clusters[k])
            comb_array = [(x, y) for x in range(num_of_pair - 1) for y in range(x + 1, num_of_pair)]

            # シャッフルし、最大数を調整
            shuffle(comb_array)
            comb_array = comb_array[: min(max_in_cluster, len(comb_array))]

            for x, y in comb_array:
                match_pairs.append(RecordMG(self.clusters[k][x], self.inf_attr))
                match_pairs.append(RecordMG(self.clusters[k][y], self.inf_attr))
                # labeling_function が定義されている場合はラベルを動的に計算する
                if labeling_function is not None:
                    match_labels.append(labeling_function(match_pairs[-2], match_pairs[-1], 1))
                else:
                    match_labels.append(1)

            # 上限値に到達した場合、そこで格納を終了しデータを整形する
            if match_num * 2 <= len(match_pairs):
                match_pairs = match_pairs[: (match_num * 2)]
                match_labels = [ml for ml in match_labels[:match_num]]
                break

        # 不一致ペア数の下処理
        mismatch_num = int(len(match_labels) * mismatch_ratio)

        # 不一致書誌データの生成 (重複の可能性あり)
        keys = list(self.clusters.keys())
        candidate = list(range(len(keys)))
        shuffle(candidate)

        while len(mismatch_pairs) < mismatch_num * 2:
            re_c = self.clusters[keys[candidate[0]]]
            mismatch_pairs.append(RecordMG(re_c[randrange(0, len(re_c))], self.inf_attr))
            re_c = self.clusters[keys[candidate[1]]]
            mismatch_pairs.append(RecordMG(re_c[randrange(0, len(re_c))], self.inf_attr))

            # labeling_function が定義されている場合はラベルを動的に計算する
            if labeling_function is not None:
                mismatch_labels.append(labeling_function(mismatch_pairs[-2], mismatch_pairs[-1], 0))
            else:
                mismatch_labels.append(0)

            # ペアの組み合わせ候補から削除する
            candidate.pop(1)
            candidate.pop(0)

            if len(candidate) < 2:
                candidate = list(range(len(keys)))
                shuffle(candidate)

        # データの整形
        mismatch_pairs = mismatch_pairs[: (mismatch_num * 2)]
        mismatch_labels = mismatch_labels[:mismatch_num]

        return (match_pairs, mismatch_pairs, match_labels, mismatch_labels)

    def get_crowdsourcing_recordmg_for_train(self, crowdsourcing_result: "dict[tuple[int, int], list[float]]"):
        """渡されたクラウドソーシング結果に基づき、学習用データを生成する

        - - -

        Params
        ------
        crowdsourcing_result: dict[tuple[int, int], list[float]]
            クラウドソーシング結果
        """

        match_pairs: "list[RecordMG]" = []
        mismatch_pairs: "list[RecordMG]" = []
        match_labels: "list[int]" = []
        mismatch_labels: "list[int]" = []

        for key, value in crowdsourcing_result.items():
            if sum(value) / len(value) >= 0.5:
                match_pairs.append(RecordMG(self.records[key[0]], self.inf_attr))
                match_pairs.append(RecordMG(self.records[key[1]], self.inf_attr))
                # TODO: 一致ペアのラベルを動的に設定する
                match_labels.append(1)

            else:
                mismatch_pairs.append(RecordMG(self.records[key[0]], self.inf_attr))
                mismatch_pairs.append(RecordMG(self.records[key[1]], self.inf_attr))
                # TODO: 不一致ペアのラベルを動的に設定する
                mismatch_labels.append(0)

        return (match_pairs, mismatch_pairs, match_labels, mismatch_labels)

    def make_record(self, option: "dict[int, int]", destroy: bool = False) -> "dict[str, list[Record]]":
        """
        格納された書誌データを整形する

        - - -

        Params
        ------
        option: dict[int, int]
            キーに一致書誌群の書誌数、値にその群数を指定
        destroy: bool, by default False
            破壊的にインスタンス変数にも書誌データを格納する
        """

        records: "list[Record]" = []
        clusters: "dict[str, dict[str, str]]" = {}
        keys = reversed(option.keys())
        candidate: "set[int]"
        selected: "set[int]" = set([])

        for k in keys:
            candidate = set([x for num, v in self.clusters_amount.items() if num >= k for x in v])
            candidate = candidate - selected

            for _ in range(option[k]):
                # 候補ペアが一つもない場合、探索を終了する
                if len(candidate) == 0:
                    break

                group = list(candidate)[int(random() * len(candidate))]
                re = self.__random_record(group, k)
                clusters[group] = re
                for r in re:
                    records.append(r)
                selected.add(group)
                candidate.remove(group)

        if destroy:
            self.records = records
            self.clusters = clusters
            self.__calc_clusters_amount()

        # return record_group

    def make_record_random(
        self,
        limit_of_records: int,
        limit_of_candidates: int = sys.maxsize,
        destroy: bool = False,
    ) -> "dict[str, list[Record]]":
        """
        格納された書誌データのクラスターからランダムに抽出して取得する

        - - -

        Params
        ------
        limit_of_records: int
            レコード上限数
        destroy: bool, by default False
            破壊的にインスタンス変数にも書誌データを格納する
        """

        records: "list[Record]" = []
        clusters: "dict[str, dict[str, str]]" = {}

        candidate = list(self.clusters.keys())
        candidate = [id for id, v in self.clusters.items() if len(v) <= limit_of_candidates]
        shuffle(candidate)

        for c in candidate:
            if len(records) > limit_of_records:
                break

            clusters[c] = self.clusters[c]
            for r in clusters[c]:
                records.append(r)

        if destroy:
            self.records = records
            self.clusters = clusters
            self.__reindexing_record()
            self.__calc_clusters_amount()

    def verify_record_pairs(self, idx_1: int, idx_2: int):
        """
        渡された2つのデータが同じグループに属しているか否かを判定する

        - - -

        Params
        ---------
        idx_1: int
            Recordのインデックス
        idx_2: int
            Recordのインデックス
        """

        return self.records[idx_1].cluster_id != "" and self.records[idx_1].cluster_id == self.records[idx_2].cluster_id

    def verify_all_record_pairs(self, collection: np.ndarray) -> ContractionResult:
        """
        推論データと正解データを全て突合させてペアの精度を計る

        - - -

        Params
        ------
        inference: list[set[str]]
            推論したデータ
        """

        result = ContractionResult()

        # 総ペアを計算
        all_pairs = len(self.records) * (len(self.records) - 1) // 2

        # 一致総ペアを計算
        num_of_pairs = {}
        all_correct_pairs = 0
        for key, value in self.clusters.items():
            num_of_pairs[len(value)] = num_of_pairs.get(len(value), 0) + 1

        for key, value in num_of_pairs.items():
            if key > 1:
                all_correct_pairs += value * (key) * (key - 1) // 2

        # 一致と推論した全てのペアを列挙
        collection_idx: "list[set[int, int]]" = []
        c_unique = np.unique(collection)
        c_target = [list(np.where(collection == c)[0]) for c in c_unique]
        for c in c_target:
            if len(c) > 1:
                collection_idx += [(c[i], c[j]) for i in range(len(c) - 1) for j in range(i + 1, len(c))]

        # 推論したペアの一致と誤りを算出
        for a, b in collection_idx:
            # 正解データも一致
            if self.verify_record_pairs(a, b):
                result.true_positive += 1
            # 正解データは不一致 (書誌誤同定)
            else:
                result.false_positive += 1

        result.false_negative = all_correct_pairs - result.true_positive
        result.true_negative = all_pairs - (result.true_positive + result.false_negative + result.false_positive)

        return result

    def verify_record_group(self, re_idx: int, collection: "set[int]") -> GroupEvaluation:
        """
        ある書誌データ1つに着目して、その書誌データが判定されたグループを検証する

        - - -

        Params
        ------

        record_idx: int
            ターゲットの書誌データについてのインデックス
        collecion: set[int]
            ターゲットの書誌データについて同定と推論されたRecordインデックスの集合
        """

        collection = collection - {re_idx}

        # ターゲット書誌データが属する正解群を算出する
        real = set([re_idx])
        if self.records[re_idx].cluster_id in self.clusters.keys():
            real = set([re_g.idx for re_g in self.clusters[self.records[re_idx].cluster_id]])
        real = real - {re_idx}

        # 指標の計算
        # TODO: 分子を(分母-分子)にすれば、従来のprecision/recallのように見やすいかも
        g_ev = GroupEvaluation(
            len(real & collection),
            len(real | collection),
            len((real - collection) | (collection - real)),
            len(real | collection),
            len(collection - real),
            len(collection),
            len(real - collection),
            len(real),
            max(0, len(real) + 1 - len((real - collection) | (collection - real))),
            len(real) + 1,
            0,
            0,
        )

        return g_ev

    def verify_all_record_group(self, collection: np.ndarray) -> "tuple[GroupEvaluation, str]":
        """
        推論データと正解データを全て突合させて群に関する精度を計る

        - - -

        Params
        ------
        collection: np.ndarray
            一致と推論したデータのnumpy配列
        """

        # 下処理
        collection_idx: "list[set[int, int]]" = []
        c_unique = list(np.unique(collection))
        c_target: list[list[int]] = [list(np.where(collection == c)[0]) for c in c_unique]
        for c in c_target:
            if len(c) > 1:
                collection_idx += [(c[i], c[j]) for i in range(len(c) - 1) for j in range(i + 1, len(c))]
            else:
                collection_idx

        # Recordのidxをキーにした辞書を作成
        collection_dict: "dict[int, set[int]]" = {}
        for target in c_target:
            for idx in target:
                collection_dict[idx] = set(target)

        # 推論で対象になったキーを全て探索
        g_ev = GroupEvaluation()
        for key, value in collection_dict.items():
            g_ev += self.verify_record_group(key, value)

        # 群の完全一致件数を探す
        complete_group = {}
        complete_group_count = 0

        for target in c_target:
            if set(target) == set([re_g.idx for re_g in self.clusters[self.records[target[0]].cluster_id]]):
                complete_group[len(target)] = complete_group.get(len(target), 0) + 1
                complete_group_count += 1

        g_ev.complete_nu = complete_group_count
        g_ev.complete_de = len(self.clusters)

        complete_group_str = ""
        for k in sorted(complete_group.keys()):
            complete_group_str += " {}: {},".format(k, complete_group[k])
        complete_group_str = "{" + f"{complete_group_str[:-1]}" + " }"

        return (g_ev, complete_group_str)

    def __random_record(self, cluster_id: str, num: int) -> "list[Record]":
        """定められた個数のデータをランダムに選んで返す"""

        result = []

        if len(self.clusters[cluster_id]) <= num:
            # 保有書誌データ数以上の数が指定された場合は、そのまま返す
            result = self.clusters[cluster_id]

        else:
            # ランダムに選択し個数分の書誌データを返す
            candidate = list(range(len(self.clusters[cluster_id])))

            for _ in range(num):
                index = candidate[int(random() * len(candidate))]
                result.append(self.clusters[cluster_id][index])
                candidate.remove(index)

        return result

    def summary(self):
        """統計情報を表示する"""

        target = self.clusters

        num_of_pairs = {}  # ペア数の計算
        counter = 0  # 書誌データ数
        library: "dict[str, int]" = {}

        # 書誌データを辞書型に変換
        for value in target.values():
            counter += len(value)
            num_of_pairs[len(value)] = num_of_pairs.get(len(value), 0) + 1

        # 概要情報を作成
        summary = {
            "#record": counter,
            "#pairs": num_of_pairs,
            "#clusters": len(target.keys()),
            "library": library,
        }

        self.logger.info(summary)
        print(summary)


class CrowdsourcingContainer:
    """クラウドソーシングプラットフォームとの連携を行うコンテナクラス"""

    def __init__(self, config: WorkflowConfig, req_dirpath: str, input_dirpath: str, log_filepath: str = None) -> None:
        """コンストラクタ"""

        self.config = config
        self.req_dirpath = req_dirpath
        self.input_dirpath = input_dirpath
        self.log_filepath = log_filepath
        self.logger: Logger = Logger(
            __name__,
            logger_config=LoggerConfig(level=self.config.log_level),
            filepath=self.log_filepath,
        )

    def _escape_csv(self, content: "int|str") -> str:
        """
        CSVファイルとして出力するための文字列をエスケープする

        - - -

        Params
        ------
        content: int|str
            エスケープ対象の数値または文字列

        Return
        ------
        str
            エスケープ後の文字列
        """
        result = str(content).replace('"', '""')

        return f'"{result}"'

    def reflect_crowdsourcing(self) -> bool:
        """クラウドソーシング結果を反映させる"""

        # inputフォルダが存在しない場合、
        if not path.exists(self.input_dirpath):
            return len(self.config.crowdsourcing_queue) == 0

        # inputフォルダ内のファイル一覧を取得
        files = os.listdir(self.input_dirpath)
        target_files = [f for f in files if path.isfile(path.join(self.input_dirpath, f))]

        msg = "Start reflecting CSV file for crowdsourcing."
        self.logger.info(msg)
        print(msg)

        # inputフォルダ内の全てのCSVファイルを対象とし、suspendに反映させる
        for file in target_files:
            filepath = path.join(self.input_dirpath, file)

            # CSVファイルではない場合は、除外
            if not filepath.endswith(".csv"):
                continue

            with open(filepath, "r", encoding="utf-8") as f:
                csv = reader(f)
                result_index = None

                for i, row in enumerate(csv):
                    # 1行目の属性値から結果のindexを探す
                    if i == 0:
                        result_index = row.index("result")
                        continue

                    try:
                        _target: "dict[str, bool]" = json.loads(
                            row[result_index]
                        )  # JSON形式で結果が格納されるため文字列をデコードする
                        for k, v in _target.items():
                            # 結果に格納する
                            id_1, id_2 = k.split(",")
                            id_p = tuple(sorted([int(id_1), int(id_2)]))
                            # TODO: workerの重み付けを考慮する
                            self.config.crowdsourcing_result[id_p] = self.config.crowdsourcing_result.get(id_p, [])
                            self.config.crowdsourcing_result[id_p].append(1 if v else 0)

                            # タスクキューから除外する
                            if id_p in self.config.crowdsourcing_queue:
                                self.config.crowdsourcing_queue.remove(id_p)

                    except IndexError:
                        continue

        msg = "Finished reflecting CSV file for crowdsourcing."
        self.logger.info(msg)
        print(msg)

        return len(self.config.crowdsourcing_queue) == 0

    def _request_crowdsourcing_by_n4u(self, re_container: RecordContainer) -> bool:
        # TODO: docstringを整備する
        # クラウドソーシングキューからタスクを作成する
        result = []
        max_task = 0
        for task in self.config.crowdsourcing_queue:
            _r = []
            max_task = max(max_task, len(task))
            _r.append(f"{len(task)}")

            for idx in task:
                re = re_container.records[idx]
                _r.append(self._escape_csv(re.id))
                for attr in re_container.inf_attr.keys():
                    _r.append(self._escape_csv(re.data[attr]))

            result.append(",".join(map(str, _r)))

        # タスクタイトル付与
        title = ["count"]
        for i in range(1, max_task + 1):
            title.append(f"id{i}")
            for attr in re_container.inf_attr:
                title.append(f"{attr}{i}")
        result = [",".join(title)] + result
        result = "\n".join(result)

        return result

    def _request_crowdsourcing_with_inference(self, re_container: RecordContainer) -> bool:
        # クラウドソーシングキューから、推論結果を含んだタスクを作成する (axiom順)
        result = []
        max_task = 0
        for task in self.config.crowdsourcing_queue:
            # レコード数の格納
            _r = []
            max_task = max(max_task, len(task))
            _r.append(f"{len(task)}")

            # データの格納
            for idx in task:
                re = re_container.records[idx]
                _r.append(self._escape_csv(re.idx))
                for attr in re_container.inf_attr.keys():
                    _r.append(self._escape_csv(re.data[attr]))

            # 推論値の格納
            for i in range(len(task) - 1):
                for j in range(i + 1, len(task)):
                    pairdata = self.config.pairdata[tuple(sorted([task[i], task[j]]))]
                    _r.append(self._escape_csv("{:.5}".format(pairdata.inf_same)))
                    _r.append(
                        self._escape_csv("{:.5}".format(pairdata.inf_not))
                        if self.config.inference_mode != InferenceMode.BAYESIAN
                        else self._escape_csv("{:.5}".format(1 - pairdata.inf_same))
                    )

            result.append(",".join(_r))

        # タスクタイトル付与
        title = ["count"]
        for i in range(1, max_task + 1):
            title.append(f"idx{i}")
            for attr in re_container.inf_attr:
                title.append(f"{attr}{i}")
        title += ["inf_same", "inf_not"]
        result = [",".join(title)] + result
        result = "\n".join(result)

        return result

    def _request_crowdsourcing_order_uncertainty(self, re_container: RecordContainer) -> bool:
        """[Experiment] Uncertainty順に並び替えてクラウドソーシングタスクを発行する"""

        # クラウドソーシングキューから、推論結果を含んだタスクを作成する (uncertainty順)

        semi_result: "dict[float, list[str]]" = {}
        max_task = 0
        for task in self.config.crowdsourcing_queue:
            # 書誌数の格納
            _r = []
            max_task = max(max_task, len(task))
            _r.append(f"{len(task)}")

            # データの格納
            for idx in task:
                re = re_container.records[idx]
                _r.append(self._escape_csv(re.idx))
                for attr in re_container.inf_attr.keys():
                    _r.append(self._escape_csv(re.data[attr]))

            # デフォルト
            uncertainty = 0

            # 推論値の格納
            for i in range(len(task) - 1):
                for j in range(i + 1, len(task)):
                    pairdata = self.config.pairdata[tuple(sorted([task[i], task[j]]))]
                    _r.append(self._escape_csv("{:.5}".format(pairdata.inf_same)))
                    _r.append(
                        self._escape_csv("{:.5}".format(pairdata.inf_not))
                        if self.config.inference_mode != InferenceMode.BAYESIAN
                        else self._escape_csv("{:.5}".format(1 - pairdata.inf_same))
                    )
                    uncertainty = 0.5 - abs(pairdata.inf_same - 0.5)

            # TODO: 3件時の処理も追加すべき
            semi_result[uncertainty] = semi_result.get(uncertainty, [])
            semi_result[uncertainty].append(",".join(_r))

        # 整列
        result = []
        keys = sorted(list(semi_result.keys()), reverse=True)

        for k in keys:
            result += semi_result[k]

        # タスクタイトル付与
        title = ["count"]
        for i in range(1, max_task + 1):
            title.append(f"idx{i}")
            for attr in re_container.inf_attr:
                title.append(f"{attr}{i}")
        title += ["inf_same", "inf_not"]
        result = [",".join(title)] + result
        result = "\n".join(result)

        return result

    def _request_crowdsourcing_order_unknown(self, re_container: RecordContainer) -> bool:
        """[Experiment] Uncertainty順に並び替えてクラウドソーシングタスクを発行する"""

        # クラウドソーシングキューから、推論結果を含んだタスクを作成する (unknown順)

        semi_result: "dict[float, list[str]]" = {}
        max_task = 0
        for task in self.config.crowdsourcing_queue:
            # 書誌数の格納
            _r = []
            max_task = max(max_task, len(task))
            _r.append(f"{len(task)}")

            # データの格納
            for idx in task:
                re = re_container.records[idx]
                _r.append(self._escape_csv(re.idx))
                for attr in re_container.inf_attr.keys():
                    _r.append(self._escape_csv(re.data[attr]))

            # デフォルト
            unknown = 0

            # 推論値の格納
            for i in range(len(task) - 1):
                for j in range(i + 1, len(task)):
                    pairdata = self.config.pairdata[tuple(sorted([task[i], task[j]]))]
                    _r.append(self._escape_csv("{:.5}".format(pairdata.inf_same)))
                    _r.append(
                        self._escape_csv("{:.5}".format(pairdata.inf_not))
                        if self.config.inference_mode != InferenceMode.BAYESIAN
                        else self._escape_csv("{:.5}".format(1 - pairdata.inf_same))
                    )
                    unknown = (
                        1 - (pairdata.inf_same + pairdata.inf_not)
                        if self.config.inference_mode != InferenceMode.BAYESIAN
                        else pairdata.inf_unknown
                    )

            # TODO: 3件時の処理も追加すべき
            semi_result[unknown] = semi_result.get(unknown, [])
            semi_result[unknown].append(",".join(_r))

        # 整列
        result = []
        keys = sorted(list(semi_result.keys()), reverse=True)

        for k in keys:
            result += semi_result[k]

        # タスクタイトル付与
        title = ["count"]
        for i in range(1, max_task + 1):
            title.append(f"idx{i}")
            for attr in re_container.inf_attr:
                title.append(f"{attr}{i}")
        title += ["inf_same", "inf_not"]
        result = [",".join(title)] + result
        result = "\n".join(result)

        return result

    def _request_crowdsourcing_random_sampling(self, re_container: RecordContainer) -> bool:
        # クラウドソーシングキューから、推論結果を含んだタスクを作成する (ランダムサンプリング)

        result = []
        max_task = 0
        self.config.crowdsourcing_queue

        for task in sample(self.config.crowdsourcing_queue, len(self.config.crowdsourcing_queue)):
            # 書誌数の格納
            _r = []
            max_task = max(max_task, len(task))
            _r.append(f"{len(task)}")

            # データの格納
            for idx in task:
                re = re_container.records[idx]
                _r.append(self._escape_csv(re.idx))
                for attr in re_container.inf_attr.keys():
                    _r.append(self._escape_csv(re.data[attr]))

            # 推論値の格納
            for i in range(len(task) - 1):
                for j in range(i + 1, len(task)):
                    pairdata = self.config.pairdata[tuple(sorted([task[i], task[j]]))]
                    _r.append(self._escape_csv("{:.5}".format(pairdata.inf_same)))
                    _r.append(
                        self._escape_csv("{:.5}".format(pairdata.inf_not))
                        if self.config.inference_mode != InferenceMode.BAYESIAN
                        else self._escape_csv("{:.5}".format(1 - pairdata.inf_same))
                    )

            result.append(",".join(_r))

        # タスクタイトル付与
        title = ["count"]
        for i in range(1, max_task + 1):
            title.append(f"idx{i}")
            for attr in re_container.inf_attr:
                title.append(f"{attr}{i}")
        title += ["inf_same", "inf_not"]
        result = [",".join(title)] + result
        result = "\n".join(result)

        return result

    def request_crowdsourcing(self, target_filepath: str) -> bool:
        """
        CSVファイルを出力し、クラウドソーシングを要求する

        - - -

        Params
        ------
        target_filepath: str
            出力対象のファイルパス
        """

        msg = "Start generating CSV file for crowdsourcing."
        self.logger.info(msg)
        print(msg)

        # 存在しない場合に、クラウドソーシング用ディレクトリを作成する
        if not path.exists(self.req_dirpath):
            os.mkdir(self.req_dirpath)

        if not path.exists(self.input_dirpath):
            os.mkdir(self.input_dirpath)

        # RecordContainerを構築
        re_container = RecordContainer()
        re_container.load_file(target_filepath)

        # N4U
        if self.config.crowdsourcing_platform == CSPlatform.N4U:
            result = [self._request_crowdsourcing_by_n4u(re_container)]

        # 推論値結果付きタスク (based on N4U)
        elif self.config.crowdsourcing_platform == CSPlatform.EXP_WITH_INF:
            result = [self._request_crowdsourcing_with_inference(re_container)]

        # Axiom-driven, Uncertainty, Unknown(Query by comittee), Random-sampling タスク発行 (based on N4U)
        elif self.config.crowdsourcing_platform == CSPlatform.EXP_CIKM2023:
            result = [
                self._request_crowdsourcing_with_inference(re_container),
                self._request_crowdsourcing_order_uncertainty(re_container),
                self._request_crowdsourcing_order_unknown(re_container),
                self._request_crowdsourcing_random_sampling(re_container),
            ]

        # resultごとにファイルを出力
        filepaths = []
        for r in result:
            # CSVファイル名決定
            filepath = None
            for i in range(1, 10000):
                if not path.exists(path.join(self.req_dirpath, "{}-{:0=4}.csv".format(self.config.config_id, i))):
                    filepath = path.join(self.req_dirpath, "{}-{:0=4}.csv".format(self.config.config_id, i))
                    break

            # 既に規定数以上のファイルが保存されている場合はエラーを出す
            if filepath is None:
                raise FileExistsError("Cannot save CSV file because there are already too many files (over 10000).")

            filepaths.append(filepath)

            # CSVファイル出力
            with codecs.open(filepath, "w", "utf-8") as f:
                f.write(r)

        # ロガーの初期化
        self.logger = Logger(
            __name__,
            logger_config=LoggerConfig(level=self.config.log_level),
            filepath=self.log_filepath,
        )

        # ログファイルに出力
        msg = [
            "",
            "===========================================================",
            "                   CROWDSOURCING REQUEST                   ",
            "\n".join([f"- {path.basename(self.req_dirpath)}/{path.basename(fp)}" for fp in filepaths]),
            "===========================================================",
            "",
        ]
        self.logger.info("\n".join(msg))
        print("\n".join(msg))

        msg = "Finished generating CSV file for crowdsourcing."
        self.logger.info(msg)
        print(msg)
