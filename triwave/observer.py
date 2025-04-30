"""Observer"""

import os
import time
import datetime
import subprocess
import multiprocessing as mp
from multiprocessing.sharedctypes import Synchronized
import re

import cpuid
import psutil
import pynvml

from .utils import path
from .logger import Logger


OBSERVER_LATEST_VERSION = "3.0"


class Observer:
    """実行コンピュータの定期監視を行うクラス"""

    def __init__(self, project_dirpath: str, log_filepath: str = None):
        """コンストラクタ"""

        self.__is_running: "Synchronized[bool]" = mp.Value("b", False)
        self.__workflow: "Synchronized[int]" = mp.Value("i", -1)

        self.project_dirpath = project_dirpath
        self.logger = Logger(
            __name__,
            filepath=log_filepath,
        )

        # pynvmlの初期化
        pynvml.nvmlInit()

    @property
    def is_running(self):
        """監視状態を返す"""
        return self.__is_running.value

    def __write(self, filepath: str, data: list[str]):
        """ファイルにデータを追記する"""
        with open(filepath, "a", encoding="utf-8") as f:
            f.write("\n".join(data) + "\n")

    def __get_now(self):
        """現在時刻を ISO 8601 形式で取得する"""
        now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
        return now.isoformat(timespec="seconds")

    # def __get_all_process(self, main_pid: int = os.getpid()):
    #     """メインとサブのすべてのプロセスを取得する"""

    #     # メインプロセスの情報を取得
    #     result = [psutil.Process(main_pid)]
    #     # サブプロセスの情報を取得
    #     result += psutil.Process(main_pid).children(recursive=True)

    #     return result

    def __get_all_process(self, main_pid: int = os.getpid()):
        """メインとサブのすべてのプロセスのRAM使用量を取得する"""

        # メインプロセスの情報を取得
        processes = [psutil.Process(main_pid)]
        # サブプロセスの情報を取得
        processes += psutil.Process(main_pid).children(recursive=True)

        ram = 0
        for p in processes:
            try:
                ram += p.memory_info().rss
            except psutil.NoSuchProcess:
                # 取得時にすでにプロセスが終了していた場合は無視する
                pass

        return (processes, ram)

    def __get_os_name(self):
        """OS名を取得する"""

        # TODO: Ubuntu限定なので、条件を追加するなど多少留意した方が良い気もする
        result = subprocess.run(["cat", "/etc/issue"], stdout=subprocess.PIPE).stdout.decode("utf-8")
        return re.sub(r"[\\n\\l]", "", result).strip()

    def __get_gpu_name(self):
        # システム内のGPUの数を取得
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_name = []

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            gpu_name.append(pynvml.nvmlDeviceGetName(handle))

        return gpu_name

    def __get_cpu_name(self):
        """CPU名を取得する"""
        return re.sub(r"[\x00-\x1F\x7F]", "", cpuid.cpu_name())

    def __get_dir_filesize(self, dirpath: str) -> int:
        """ディレクトリ内のファイルサイズを取得する"""
        size = 0

        for root, _, files in os.walk(dirpath):
            for file in files:
                size += path.getsize(path.join(root, file))

        return size

    def set_workflow(self, workflow: int):
        """ワークフロー番号の設定"""
        self.__workflow.value = workflow

    def __record(self, csvpath: str, main_pid: int):
        """1回分の記録"""

        # 情報の取得
        # sum([p.cpu_percent() for p in processes])  # CPU使用率のベースライン取得
        processes, ram_process = self.__get_all_process(main_pid)
        ram_use = psutil.virtual_memory().used
        gpus = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            gpus += [
                pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                pynvml.nvmlDeviceGetMemoryInfo(handle).used,
                pynvml.nvmlDeviceGetPowerUsage(handle),
            ]
        storage_use = psutil.disk_usage("/").used
        storage_project = self.__get_dir_filesize(self.project_dirpath)
        # cpu_process = sum([p.cpu_percent() for p in processes])

        # 記録内容整形
        content = []

        # content += [self.__get_now(), self.__workflow, len(processes), cpu_process]
        content += [
            self.__get_now(),
            str(self.__workflow.value) if self.__workflow.value >= 0 else "system",
            len(processes),
        ]
        content += psutil.cpu_percent(percpu=True)
        content += [ram_use, ram_process]
        content += gpus
        content += [storage_use, storage_project]
        content = [str(c) for c in content]

        # 書き込み
        self.__write(csvpath, [",".join(content)])

    def __loop(self, csvpath: str, main_pid: int):
        """[Threading] 並行処理"""

        # pynvmlの初期化
        pynvml.nvmlInit()

        __now = int(time.time())

        while self.__is_running.value:
            if __now < int(time.time()):
                self.__record(csvpath, main_pid)
                __now = int(time.time())

            time.sleep(0.1)

    def start(self):
        """定期監視の開始"""

        # 記録用ディレクトリが存在しない場合、作成する
        if not path.exists(path.join(self.project_dirpath, ".observer")):
            os.makedirs(path.join(self.project_dirpath, ".observer"))

        # マシン情報取得
        os_name = self.__get_os_name()
        cpu_name = f"{psutil.cpu_count()}x {self.__get_cpu_name()}"
        cpu_core_count = psutil.cpu_count()
        gpu_name = [f'"{gn}"' for gn in self.__get_gpu_name()]
        gpu_vram_total = [
            str(pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).total)
            for i in range(pynvml.nvmlDeviceGetCount())
        ]
        ram_total = psutil.virtual_memory().total
        storage_total = psutil.disk_usage("/").total

        # csvファイルを生成し、ヘッダ等を付与する
        csvpath = path.join(self.project_dirpath, ".observer", f"{self.__get_now()}.csv").replace(":", ".")
        __chunk = '"# k1z3-observer"'
        __spec = f'# {{"VERSION":"{OBSERVER_LATEST_VERSION}","OS_NAME":"{os_name}","CPU_NAME":"{cpu_name}","CPU_CORE_COUNT":{cpu_core_count},"GPU_NAME":[{",".join(gpu_name)}],"GPU_VRAM_TOTAL":[{",".join(gpu_vram_total)}],"RAM_TOTAL":{ram_total},"STORAGE_TOTAL":{storage_total}}}'
        # __header = ["time", "#workflow", "#process", "cpu:process"]
        __header = ["time", "#workflow", "#process"]
        __header += [f"cpu:{c}" for c in range(cpu_core_count)]
        __header += ["ram:use", "ram:process"]
        __header += [f"gpu:{i}:{metric}" for i in range(len(gpu_name)) for metric in ["use", "vram", "power"]]
        __header += ["storage:use", "storage:project"]
        self.__write(csvpath, [__chunk, __spec, ",".join(__header)])

        # 定期監視開始
        self.__is_running.value = True
        process = mp.Process(target=self.__loop, args=(csvpath, os.getpid()))
        process.start()

    def stop(self):
        """定期監視の終了"""
        self.__is_running.value = False

    def __del__(self):
        """デストラクタ"""

        # 定期監視が動作中であれば、安全に停止する
        if self.__is_running:
            self.stop()
