"""TriWave検証用スクリプト"""

import os

from triwave.project import ProjectManager

SAFE_STOP = True  # どこかのワークフローでエラーが発生した場合に、それ以降のワークフローを実行しない

if __name__ == "__main__":
    project_dirpath = os.path.join("project", "exp1")
    project_name = [
        "test",
    ]

    for name in project_name:
        project_manager = ProjectManager(os.path.join("@", project_dirpath, name))
        result = project_manager.start_workflow()

        if result is not None and SAFE_STOP:
            break
