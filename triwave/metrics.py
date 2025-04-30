"""Metrics"""

from dataclasses import dataclass

import numpy as np

from .logger import Logger, LoggerConfig


@dataclass
class Duration:
    """時間を表すクラス"""

    day: int
    hour: int
    minute: int
    second: int
    msec: int


@dataclass
class FormattedMetrics:
    """メール通知用に整形したメトリクス形式"""

    execution_time: str
    # TODO: disk_ioに切り替える
    data_io_time: str
    workflow_time: "list[str]"


class Metrics:
    """パフォーマンスの記録・管理を行うクラス"""

    def __init__(self, workflow: "list[dict[str, any]]", log_filepath: str = None, log_level: str = "INFO"):
        """コンストラクタ"""

        # ログファイルのパスを格納
        self.logger = Logger(
            __name__,
            logger_config=LoggerConfig(level=log_level),
            filepath=log_filepath,
        )

        # Workflowについて格納
        self.__workflow = workflow

        # 時間データ格納
        self.execution_time: int = 0
        self.data_io_time: "list[int]" = []
        self.workflow_time: "list[int]" = [0 for _ in range(len(self.__workflow))]

    def msec_2_duration(self, msec: int) -> Duration:
        """ミリ秒から時間単位に変換する"""
        seconds, msec = divmod(msec, 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        days, hours = divmod(hours, 24)

        return Duration(days, hours, minutes, seconds, msec)

    def load_metrics(self, data: "dict[str, any] | None"):
        """辞書型のMetricsを読み込む"""

        if data is not None:
            self.execution_time = data["execution_time"]  # 全実行時間
            self.data_io_time = data["data_io_time"]  # 一時ファイル読込保存時間
            dummy_workflow_time = [0 for _ in range(len(self.__workflow) - len(data["workflow_time"]))]
            self.workflow_time = data["workflow_time"] + dummy_workflow_time  # ワークフロー実行時間 (配列)

    def __analyze_loop_tree(self, workflow: "list[dict[str, any]]", depth: int = 0):
        """[Recursive] ワークフローのループを可視化するためのtreeを作成する"""
        start_idx = workflow[0]["idx"]
        end_idx = workflow[0]["idx"]
        others = None
        indices = []
        loops = []

        count = 0
        while count < len(workflow):
            # 同じ階層であれば、othersに記録
            if len(workflow[count]["loop_count"]) == depth:
                end_idx = workflow[count]["idx"]
                others = (0 if others is None else others) + self.workflow_time[end_idx]
                count += 1

            # 階層が深くなっていれば、再帰的に呼び出す
            else:
                loop_max = workflow[count]["loop_max"][depth]
                target = []
                for i in range(loop_max):
                    while (
                        count < len(workflow)
                        and depth < len(workflow[count]["loop_count"])
                        and workflow[count]["loop_count"][depth] == i
                    ):
                        end_idx = workflow[count]["idx"]
                        target.append(workflow[count])
                        count += 1

                    _loops, _indices = self.__analyze_loop_tree(target, depth + 1)
                    loops.append(_loops)
                    indices.append(_indices)
                    target = []

        # ループが存在していれば、othersを一番はじめに追加
        if len(loops) > 0:
            return [others] + loops, [(start_idx, end_idx)] + indices

        # 存在していなければ、othersのみを返す
        else:
            return others, (start_idx, end_idx)

    def __draw_loop_tree(
        self,
        tree: "int | list[int | list[int]]",
        indices: "tuple[int, int] | list[tuple[int, int]]",
        total_time: int = 1,
        loop: int = 0,
        depth: int = 0,
    ) -> list[str]:
        """
        [Recursive] ワークフローのループを文字列配列化する

        - - -

        Params
        ------
        tree: int | list[int | list[int]]
            ループのツリー構造 (__analyze_loop_treeの出力)
        indices: tuple[int, int] | list[tuple[int, int]]
            ループのインデックス (__analyze_loop_treeの出力)
        total_time: int, by default 1
            ループの合計時間
        loop: int, by default 0
            ループ回数
        depth: int, by default 0
            ループの深さ
        """

        # 0除算防止
        total_time = 1 if total_time < 1 else total_time

        def flatten(lst: int | list[any]):
            result = []
            if isinstance(lst, int):
                result.append(lst)
            else:
                for i in lst:
                    if i is None:
                        result.append(0)
                    elif isinstance(i, list):
                        result.extend(flatten(i))
                    else:
                        result.append(i)
            return result

        def msec_format(msec: Duration) -> str:
            """ミリ秒からフォーマットした文字列に変換する"""
            result = "" if msec.day == 0 else f"{msec.day}d"
            result += "" if len(result) == 0 and msec.hour == 0 else f"{msec.hour}h"
            result += "" if len(result) == 0 and msec.minute == 0 else f"{msec.minute}m"
            result += f"{msec.second}.{msec.msec:03}s"
            return result

        result = []
        # WorkflowやLoop内部にLoopが存在しない場合
        if isinstance(tree, int):
            _title = f"Loop{loop:>2}" if loop > 0 else "Workflow"
            _in = f"[{indices[0]}]" if indices[0] == indices[1] else f"[{indices[0]}-{indices[1]}]"
            _time = msec_format(self.msec_2_duration(tree))
            _rate = f"{(tree / total_time * 100):.3f}"

            result.append(f"{_title} {_in}: {_time} ({_rate}%)")

        # WorkflowやLoop内部にLoopが存在する場合
        else:
            _title = f"Loop{loop:>2}" if loop > 0 else "Workflow"
            _in = f"[{indices[0][0]}]" if indices[0][0] == indices[0][1] else f"[{indices[0][0]}-{indices[0][1]}]"
            _time = msec_format(self.msec_2_duration(sum(flatten(tree))))
            _rate = f"{(sum(flatten(tree)) / total_time * 100):.3f}"

            result.append(f"{_title} {_in}: {_time} ({_rate}%)")

            for i in range(1, len(tree)):
                _label = ["|-", "| "] if i < len(tree) - 1 or tree[0] is not None else ["+-", "  "]
                _idx = flatten(indices[i])[0]
                _result = self.__draw_loop_tree(
                    tree[i],
                    indices[i],
                    sum(flatten(tree)),
                    self.__workflow[_idx if isinstance(_idx, int) else _idx[0]]["loop_count"][depth] + 1,
                    depth + 1,
                )
                result += [f" {_label[0]} {_result[0]}"]
                result += [f" {_label[1]} {r}" for r in _result[1:]]

            if tree[0] is not None:
                result.append(f" +- Others: {msec_format(self.msec_2_duration(tree[0]))}")

        return result

    def format_for_notifications(self, limit: int = None) -> FormattedMetrics:
        """
        メール通知用のフォーマットに変換する

        - - -

        Params
        ------
        limit: int
            カレントワークフローの番号を指定する。指定しない場合は全てのワークフローを通しての結果を返す
        """

        def msec_format(msec: Duration, is_workflow: bool = True) -> str:
            """ミリ秒からフォーマットした文字列に変換する"""
            result = "" if msec.day == 0 else f"{msec.day}d"
            result += "" if len(result) == 0 and msec.hour == 0 else f"{msec.hour}h"
            result += "" if len(result) == 0 and msec.minute == 0 else f"{msec.minute}m"
            result += f"{msec.second}.{msec.msec:03}s"
            return f" ({result})" if is_workflow else result

        execution_time = msec_format(self.msec_2_duration(self.execution_time), False)
        data_io_time = msec_format(self.msec_2_duration(sum(self.data_io_time)), False)
        workflow_time = [
            (msec_format(self.msec_2_duration(t)) if i <= limit else "") for i, t in enumerate(self.workflow_time)
        ]

        return FormattedMetrics(execution_time, data_io_time, workflow_time)

    def current_metrics(self):
        """現在のメトリクスをログに書き込む"""

        def msec_format(msec: Duration) -> str:
            """ミリ秒からフォーマットした文字列に変換する"""
            result = "" if msec.day == 0 else f"{msec.day}d"
            result += "" if len(result) == 0 and msec.hour == 0 else f"{msec.hour}h"
            result += "" if len(result) == 0 and msec.minute == 0 else f"{msec.minute}m"
            result += f"{msec.second}.{msec.msec:03}s"
            return result

        # それぞれのワークフローの時間を格納する
        workflow = {"disk_io": self.data_io_time}

        for i, w in enumerate(self.__workflow):
            workflow[w["name"]] = workflow.get(w["name"], [])
            workflow[w["name"]].append(self.workflow_time[i])

        # 一部のワークフローについては削除
        workflow.pop("CURRENT_METRICS")

        result = []
        for i, k in enumerate(sorted(workflow, key=lambda x: sum(workflow[x]), reverse=True)):
            workflow_time = msec_format(self.msec_2_duration(sum(workflow[k])))
            workflow_time_rate = sum(workflow[k]) / self.execution_time * 100
            average = msec_format(self.msec_2_duration(int(np.average(workflow[k]))))
            std = msec_format(self.msec_2_duration(int(np.std(workflow[k]))))

            result.append(
                f"{i+1:>4}. {k}: {workflow_time} ({workflow_time_rate:.3f}%, call: {len(workflow[k])}, avg: {average}, std: {std})"
            )

        wf_time = "\n".join(result)
        loop_tree = "  " + "\n  ".join(
            self.__draw_loop_tree(
                *self.__analyze_loop_tree(self.__workflow),
                self.execution_time,
            )
        )

        message = f"""
[[ Current Metrics ]]
Execution Time: {msec_format(self.msec_2_duration(self.execution_time))}

Workflow Time:
{wf_time}

Loop Tree:
{loop_tree}
"""

        # ログ出力
        self.logger.info(message)

    def to_dict(self) -> "dict[str, any]":
        """Metricsを辞書型に変換する"""

        return {
            "execution_time": self.execution_time,
            "data_io_time": self.data_io_time,
            "workflow_time": self.workflow_time,
        }
