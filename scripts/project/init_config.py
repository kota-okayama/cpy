"""configの更新時刻等をアップデートするスクリプト"""

import os
from datetime import datetime
import yaml
import codecs
from uuid import uuid4

PROJECT_SELECT_AUTO = True


def update_project_meta(project_dirpath: str, target: "list[str]"):
    """configファイルのメタデータを書き換える

    Params
    ------
    project_dirpath: str
        プロジェクトパス
    target: list[str]
        書き換え対象の設定
    """

    date = ("{}".format(datetime.now()))[:-3]

    if os.path.isfile(os.path.join(project_dirpath, "config.yaml")):
        # configファイルを直開きし、書き換える
        with open(os.path.join(project_dirpath, "config.yaml"), "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

            if "id" in target:
                data["id"] = str(uuid4())

            if "creation_date" in target:
                data["summary"]["creation_date"] = date

            if "update_date" in target:
                data["summary"]["update_date"] = date

        # configファイルの出力
        with codecs.open(os.path.join(project_dirpath, "config.yaml"), "w", "utf-8") as f:
            yaml.dump(data, f, indent=2, allow_unicode=True, sort_keys=False)


if __name__ == "__main__":
    if PROJECT_SELECT_AUTO:
        project_dirpath = os.path.dirname(os.path.abspath(__file__))
        project_name = [d for d in os.listdir(project_dirpath) if os.path.isdir(os.path.join(project_dirpath, d))]
        project_name = [d for d in project_name if os.path.isfile(os.path.join(project_dirpath, d, "config.yaml"))]

    else:
        project_dirpath = os.path.join("project_deim2023_2", "exp0")
        project_name = [
            "exp0-jj-retrain-20",
        ]

    for name in project_name:
        config = update_project_meta(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), project_dirpath, name), ["id", "update_date"]
        )
