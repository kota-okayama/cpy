"""Assignment algorithm for the OpenBook Camera task."""

import numpy as np
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

# Postで無理やり送信する場合は、以下の指定が必須
# {
#   "callback_url": str
#   "result_group": str
# }


class BibCapture_02(WorkflowAlgorithm):
    """書影撮影タスクの割り当てアルゴリズム"""

    def __init__(self, parameters: dict, ai_crowd: GenericAICrowd, logger: Logger) -> None:
        super().__init__(parameters, ai_crowd, logger)
        return

    def generate(self, event_type: str, dataset: list, task_assignments: list) -> list[GenerateReturns]:
        """タスク生成メソッド"""

        self.logger.debug(event_type)

        original_dataitem_ids = list(map(lambda x: x["id"], dataset))
        assigned_dataitem_ids = list(map(lambda x: x["dataitem_id"], task_assignments))
        assignable_dataitem_ids = [i for i in original_dataitem_ids if i not in assigned_dataitem_ids]

        output = []
        if event_type == "initialize":
            output = self._pick_n_items(assignable_dataitem_ids, 10, "human")

        elif event_type == "task_completed":
            task_assignments_wip = [ta_i for ta_i in task_assignments if ta_i["result"] is None]

            if len(task_assignments_wip) == 0:
                output = self._pick_n_items(assignable_dataitem_ids, 10, "human")

        return output

    def assign(self, task_assignments: list, datasets, request_args: dict[str, str]) -> tuple[str, WorkflowReturns]:
        """タスク割り当てメソッド"""

        # 未回答タスクの先頭のものをピックアップ
        picked_assignment = list(filter(lambda t: t["result"] is None, task_assignments))[0]
        assignment_item_id: str = picked_assignment["task_assignment_id"]

        return assignment_item_id, WorkflowReturns(
            template_name="standard", callback_url="", assigned_worker_id="", template_params={}
        )

    def update(self, task_assignment: dict, request_args: WorkflowArguments) -> WorkflowReturns:
        """回答に対するデータアップデート"""

        return WorkflowReturns(template_name="standard", callback_url="", assigned_worker_id="", template_params={})

    def post_process(self, task_assignments: list, request_args: dict) -> WorkflowReturns:
        """update 関数後のポストプロセス"""

        return WorkflowReturns(template_name="standard", callback_url="", assigned_worker_id="", template_params={})

    def _pick_n_items(self, assignable_dataitem_ids: int, size: int, qualified_worker_type: str):
        """データセットからランダムにn個のデータアイテムをピックアップする"""

        _output = []
        size = min(len(assignable_dataitem_ids), size)
        picked_ids = np.random.choice(assignable_dataitem_ids, size=size, replace=False)

        for p_i in picked_ids:
            _output.append(
                GenerateReturns(
                    dataitem_id=p_i,
                    qualified_worker_type=qualified_worker_type,
                )
            )

        return _output
