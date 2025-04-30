"""ベンチマーク生成用スクリプト"""

import os

from triwave.project import ProjectManager

SAFE_STOP = True  # どこかのワークフローでエラーが発生した場合に、それ以降のワークフローを実行しない
DATE = "20241024"

if __name__ == "__main__":
    project_dirpath = os.path.join("project", "203")
    benchmark_name = [
        f"bib_japan_{DATE}",
        f"bib_kyoto_{DATE}",
        f"music_lepizig_{DATE}",
        f"persons_lepizig_{DATE}",
    ]
    project_name = [
        "1k",
        "2k",
        "5k",
        "10k",
        "30k",
        "50k",
        "100k",
    ]

    for be in benchmark_name:
        for pj in project_name:
            project_manager = ProjectManager(os.path.join("@", project_dirpath, be, pj))
            result = project_manager.start_workflow()

            if result is not None and SAFE_STOP:
                break
