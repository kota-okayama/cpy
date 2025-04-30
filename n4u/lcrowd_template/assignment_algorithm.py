"""Assignment Algorithm for Bibliographic Matching Task"""

from datetime import datetime
from time import time
import re
import hashlib

from human_plus_ai_crowd.workflow_algorithm import (
    GenerateReturns,
    GenericAICrowd,
    Logger,
    WorkflowAlgorithm,
    WorkflowArguments,
    WorkflowReturns,
)

# Workflow呼び出しURL (最新runのタスク呼び出し)
# https://next.crowd4u.org/runs/wf/<workflow-id>/run

# Run直接呼び出しURL
# https://next.crowd4u.org/runs/<run-id>/get_task

# 一度に発行するタスク数
PUBLISH_AT_ONE_TIME = 500


# クラス名はワークフロー名と同一である必要がある
class BibMatching_06(WorkflowAlgorithm):
    """書誌同定タスクの割り当てアルゴリズム"""

    def __init__(self, parameters: dict, ai_crowd: GenericAICrowd, logger: Logger) -> None:
        super().__init__(parameters, ai_crowd, logger)
        return

    def generate(self, event_type: str, dataset: list, task_assignments: list) -> list[GenerateReturns]:
        """
        タスク生成

        - - -

        Parameters
        ----------
        event_type : str
            イベントタイプ
        dataset : list
            データセット
        task_assignments : list
            現状のタスクアサインメントテーブル

        Return
        ------
        list[GenerateReturns]
            追加するタスクのリスト
        """

        output = []
        # 設定値かデータセット数について、発行するタスク数を歳用
        publish_at_one_time = min(len(dataset), PUBLISH_AT_ONE_TIME)

        if event_type == "initialize":
            for dataitem in dataset[:publish_at_one_time]:
                output.append(
                    GenerateReturns(
                        dataitem_id=dataitem["dataitem_id"],
                        qualified_worker_type="human",
                    )
                )

        elif event_type == "task_completed":
            # 未回答タスクが50個未満の場合，新規にタスクを発行
            not_answered = [x["result"] for x in task_assignments if x["result"] is None]
            if len(not_answered) > 50:
                return output

            # 全てのデータアイテムのIDを取得し、既存のタスクに含まれているIDの検索
            existing_ids = dict.fromkeys([x["dataitem_id"] for x in dataset], 0)
            for task in task_assignments:
                existing_ids[task["dataitem_id"]] += 1

            # タスクが発行された数の最小値を取得し、発行数の少ないタスクのリストとして取得
            task_count = min(existing_ids.values())
            dataitem_ids = [key for key, value in existing_ids.items() if value == task_count]

            # 取得したリストの10個をピックアップし、タスク発行
            output += [
                GenerateReturns(dataitem_id=dataitem_id, qualified_worker_type="human")
                for dataitem_id in dataitem_ids[:publish_at_one_time]
            ]

        return output

    def assign(self, task_assignments: list, datasets, request_args: dict[str, str]) -> tuple[str, WorkflowReturns]:
        """
        アクセスにおけるタスク割当

        - - -

        Parameters
        ----------
        task_assignments : list
            タスクアサインメントテーブル
        datasets : list
            データセット
        request_args : dict[str, str]
            URL クエリを含むリクエストデータ
        """

        # 未回答タスクのものをピックアップ
        picked_assignments = list(filter(lambda t: t["result"] is None, task_assignments))

        # assigned_atがNoneおよび早い順にソートし、割当の重複を避ける
        def custom_sort_key(assignment):
            return (
                datetime.strptime(assignment["assigned_at"], "%Y-%m-%dT%H:%M:%S")
                if assignment["assigned_at"] is not None
                else datetime(1900, 1, 1)
            )

        picked_assignment = sorted(picked_assignments, key=custom_sort_key)[0]

        assignment_item_id: str = picked_assignment["task_assignment_id"]
        # assignment_item_count = int(picked_assignment["content"][0])

        q_params = request_args["query_params"]

        # worker_id
        # C4U から未ログインで流入の場合 worker_id が -1 で指定される
        worker_id = q_params["worker_id"] if "worker_id" in q_params and q_params["worker_id"] != "-1" else ""

        # default_url
        # worker_id の URL クエリを存続させるために設定
        default_url = f"../../runs/{picked_assignment['run_id']}/get_task"
        default_url = f"{default_url}?worker_id={worker_id}" if worker_id != "" else default_url

        # counter
        SALT = "bibK1Z3"
        DATE = datetime.now().strftime("%Y%m%d")
        counter = 0 if "counter" in q_params else -1
        counter_sha = q_params["counter"] if "counter" in q_params else None
        if counter_sha:
            for i in range(1001):
                if counter_sha == hashlib.sha1(f"{SALT}{DATE}{i}".encode("utf-8")).hexdigest()[:10]:
                    counter = i
                    break

        if counter == 1000:
            # 1000に到達している場合は、引き続き1000として表示
            next_counter_sha = counter_sha
        else:
            next_counter_sha = hashlib.sha1(f"{SALT}{DATE}{counter + 1}".encode("utf-8")).hexdigest()[:10]

        # return
        if "target" in q_params and q_params["target"] == "doshisha":
            # 同志社大授業用テンプレート
            cu = q_params["callback"] if "callback" in q_params else default_url

            return assignment_item_id, WorkflowReturns(
                template_name="binary",
                callback_url=cu,
                assigned_worker_id=worker_id,
                template_params={
                    "time": int(time()),
                    "skip_callback_url": cu,
                    "notification_title": "書誌タスクご協力のお願い",
                    "notification_body": "回答後、授業資料ページに遷移します",
                },
            )

        else:
            # 通常テンプレート
            if "callback" in q_params:
                cu = q_params["callback"]
                skip_cu = cu
            else:
                cu = default_url + ("&" if re.match(r".*\?.*", default_url) else "?") + f"counter={next_counter_sha}"
                cu = f"{cu}&target={q_params['target']}" if "target" in q_params else cu

                # スキップに対する href に設定される値 (リロード)
                skip_cu = "javascript:window.location.reload();"

            return assignment_item_id, WorkflowReturns(
                template_name="binary",
                callback_url=cu,
                assigned_worker_id=worker_id,
                template_params={
                    "time": int(time()),
                    "skip_callback_url": skip_cu,
                    "counter_hidden": ("0" if counter + 1 < 10 else "") + ("0" if counter + 1 < 100 else ""),
                    "counter": f"{counter + 1}" if counter + 1 < 1000 else "999+",
                },
            )
        # if assignment_item_count == 2:
        # return assignment_item_id, {"task_template_name": "binary"}
        # else:
        # return assignment_item_id, {"task_template_name": "binary"}
        # return assignment_item_id, {"task_template_name": "multi"}

    def update(self, task_assignment: dict, request_args: WorkflowArguments) -> WorkflowReturns:
        """回答に対するデータアップデート"""

        q_params = request_args["query_params"]
        # C4U から未ログインで流入の場合 worker_id が -1 で指定される
        worker_id = q_params["worker_id"] if "worker_id" in q_params and q_params["worker_id"] != "-1" else ""

        return WorkflowReturns(
            template_name="binary",
            callback_url="",
            assigned_worker_id=worker_id,
            template_params={},
        )

    def post_process(self, task_assignments: list, request_args: dict) -> WorkflowReturns:
        """update 関数後のポストプロセス"""

        q_params = request_args["query_params"]
        # C4U から未ログインで流入の場合 worker_id が -1 で指定される
        worker_id = q_params["worker_id"] if "worker_id" in q_params and q_params["worker_id"] != "-1" else ""

        return WorkflowReturns(
            template_name="binary",
            callback_url="",
            assigned_worker_id=worker_id,
            template_params={},
        )
