"""Logger"""

import os
import logging
from dataclasses import dataclass
import re

import sys
import psutil
import cpuid
from gpustat.core import GPUStatCollection
from pynvml import NVMLError

from .utils import path
from .datatype.asset import Title, Author
from .datatype.gpu_status import GPUStatus


@dataclass
class LoggerConfig:
    """ロガーの設定"""

    filename: str = path.join("@", "log.log")
    level: str = "INFO"
    format: str = "[%(asctime)s] %(name)s(%(funcName)s): %(lineno)s [%(levelname)s]: %(message)s"

    def get_level(self):
        """文字列からloggingのレベルに変換する"""
        exchange: dict[str, int] = {
            "CRITICAL": logging.CRITICAL,
            "ERROR": logging.ERROR,
            "WARNING": logging.WARNING,
            "INFO": logging.INFO,
            "DEBUG": logging.DEBUG,
        }

        return exchange[self.level]


class Logger:
    """ロガー"""

    def __init__(self, name: str, logger_config: LoggerConfig = LoggerConfig(), filepath: str = None):
        """ロガーの初期化"""

        self.logger = logging.getLogger(name)
        self.logger_config = logger_config
        self.filepath = logger_config.filename if filepath is None else filepath

        # 初期化フィルター追加
        init_filter = self.InitFilter(self.filepath, self.logger_config)
        self.logger.addFilter(init_filter)

        # ロギングの初期化
        init_filter.logger_init()

        # self.__init_loggingを事前に呼び出すメソッドを追加
        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.critical = self.logger.critical

    class InitFilter(logging.Filter):
        """ロガーの初期化フィルター"""

        def __init__(self, filepath: str, logger_config: LoggerConfig):
            super().__init__()
            self.__filepath = filepath
            self.__logger_config = logger_config

        def logger_init(self):
            """ロギングの初期化"""
            # ロギングでは設定が上書きできないようなので、既存ハンドラを削除する
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)

            # ロギングを設定
            logging.basicConfig(
                filename=self.__filepath,
                level=self.__logger_config.get_level(),
                format=self.__logger_config.format,
                encoding="utf-8",
            )

        def filter(self, record):
            """フィルター (logger初期化設定を追加)"""
            self.logger_init()
            return super().filter(record)

    def write_start_workflow(self, project_name: str, start_time: str, gpu_status: GPUStatus):
        """ワークフロー開始時にログファイルに所定の書式を書き込む"""

        def byte_formatter(byte: int):
            """バイト数を整形する"""
            if byte < 1024:
                return f"{byte} B"
            elif byte < 1024**2:
                return f"{byte / 1024:.3f} KB"
            elif byte < 1024**3:
                return f"{byte / 1024 ** 2:.3f} MB"
            elif byte < 1024**4:
                return f"{byte / 1024 ** 3:.3f} GB"
            elif byte < 1024**5:
                return f"{byte / 1024 ** 4:.3f} TB"
            else:
                return f"{byte / 1024 ** 5:.3f} PB"

        def get_gpu_name():
            """GPU名を取得する"""
            result = []
            try:
                _gpu = GPUStatCollection.new_query()
                result = [f"[{i}] {g.name}" for i, g in enumerate(_gpu)]
            except NVMLError:
                self.warning("(Cannot detect GPU.)")

            if len(result) == 0:
                return "(None)"
            else:
                return "\n                 ".join(result)

        cpu_name = re.sub(r"[\x00-\x1F\x7F]", "", cpuid.cpu_name())
        msg = f"""

============================================================{Title(0)}

{Author(14)}
------------------------------------------------------------
   Project Name: {project_name}
     Start time: {start_time}
         Python: {sys.version.split(" ")[0]}
------------------------------------------------------------
            CPU: {psutil.cpu_count()}x {cpu_name}
            GPU: {get_gpu_name()}
            RAM: {byte_formatter(psutil.virtual_memory().total)}
            ROM: {byte_formatter(psutil.disk_usage('/').used)} / {byte_formatter(psutil.disk_usage('/').total)} ({psutil.disk_usage('/').percent}%)
TensorFlow(GPU): {"Enabled" if gpu_status.tensorflow else "Disabled"}
      Cupy(GPU): {"Enabled" if gpu_status.cupy else "Disabled"}
     Faiss(GPU): {"Enabled" if gpu_status.faiss else "Disabled"}
============================================================
"""
        self.info(msg)
