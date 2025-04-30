"""クラウドソーシングから統計画像を出力するためのスクリプト (Bayesian推論用)"""

import os

# import sys

import numpy as np
from csv import reader
import pandas as pd

import altair as alt
import toolz


# Parameter
TAU_SAME = 0.5
TAU_NOT = 0.5
INF_TYPE = "BAYESIAN"  # SUN or BAYESIAN

# GRAPH_TITLE = "Bibliorecords (Worker accuracy: 90%)"
GRAPH_TITLE = "Music (Worker accuracy: 90%)"
# GRAPH_TITLE = "Persons (Worker accuracy: 90%)"
RESOLUTION = 2  # 出力画像の解像度

SCALE = "log"  # linear or log
MAX_RANGE = 10000
STEP = 1000

# 5000行以上のデータに対応するための設定
_t = lambda data: toolz.curried.pipe(data, alt.limit_rows(max_rows=1000000), alt.to_values)  # TODO: E731を回避すべき
alt.data_transformers.register("custom", _t)
alt.data_transformers.enable("custom")

DIRPATH = os.path.dirname(os.path.abspath(__file__))


class F1ScoreGraphContainter:
    """クラウドソーシングタスクファイルからグラフを描画するクラス"""

    # EXP_WITH_INF で出力されたファイルを読み込む必要あり

    def __init__(self, filepath) -> None:
        """コンストラクタ"""

        result = []
        with open(filepath, "r", encoding="utf-8") as f:
            csv = reader(f)

            for i, row in enumerate(csv):
                # 1行目は属性値として取得する
                if i == 0:
                    pass

                # dataが欠損してRecordを構成できない場合は、スキップする
                else:
                    try:
                        tmp = [
                            int(row[0]),
                            float(row[1]),
                            row[2],
                        ]
                        result.append(tmp)

                    except IndexError:
                        continue

        # インスタンス変数に格納
        self.result = result

    def draw_line(
        self,
        title_name,
        is_music,
        is_ex,
    ):
        """altairによる折れ線グラフ描画"""

        # color_lst = [alt.value(color_code) for color_code in color_codes]

        if is_music:
            legend_y = 5
        else:
            if is_ex:
                legend_y = 135
            else:
                legend_y = 150

        if is_ex:
            domain = [
                "Alternate use of axiom driven and uncertainty sampling",
                "Axiom driven sampling",
                "Uncertainty sampling",
                "Query by committee sampling",
                "Random sampling",
            ]
            ranges = [
                "#a000a0",
                "#ff6d60",
                "#45d86e",
                "#00c0c0",
                "#808080",
            ]
            marker = [
                "triangle-down",
                "circle",
                "triangle-up",
                "diamond",
                "square",
            ]
        else:
            domain = [
                "Axiom driven sampling",
                "Uncertainty sampling",
                "Query by committee sampling",
                "Random sampling",
            ]
            ranges = [
                "#ff6d60",
                "#45d86e",
                "#00c0c0",
                "#808080",
            ]
            marker = [
                "circle",
                "triangle-up",
                "diamond",
                "square",
            ]

        # altairで描画
        df = pd.DataFrame(self.result, columns=["counter", "f1_score", "type"])
        chart = (
            alt.Chart(
                df,
                mark="circle",
                title=title_name,
            )
            .mark_line(
                size=2,
                point=alt.OverlayMarkDef(size=60),
            )
            .encode(
                x=alt.X(
                    "counter",
                    title="#iterations",
                ),
                y=alt.Y(
                    "f1_score",
                    title="F1 value",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(values=list(np.arange(0, 1.1, 0.1))),
                ),
                color=alt.Color(
                    "type",
                    title="Method",
                    scale=alt.Scale(
                        domain=domain,
                        range=ranges,
                    ),
                    legend=alt.Legend(
                        orient="none",  # orientをnoneに設定するとlegenX,legendYが指定できる
                        legendX=5,
                        legendY=legend_y,
                        # legendY=135,
                        # legendY=150,
                        # strokeColor="blue",
                        fillColor="white",
                        padding=5,
                        labelLimit=280,
                    ),
                ),
                shape=alt.Shape(
                    "type",
                    scale=alt.Scale(
                        range=marker,
                    ),
                    legend=None,
                ),
            )
            .properties(
                width=400,
                height=240,
            )
        )

        # 保存先ディレクトリ生成
        if not os.path.exists(os.path.join(DIRPATH, "output")):
            os.mkdir(os.path.join(DIRPATH, "output"))

        # ファイル名決定
        filepath = None
        for i in range(1, 100000):
            if not os.path.exists(os.path.join(DIRPATH, "output", "result-{:0=5}.pdf".format(i))):
                filepath = os.path.join(DIRPATH, "output", "result-{:0=5}.pdf".format(i))
                break

        # 既に規定数以上のファイルが保存されている場合はエラーを出す
        if filepath is None:
            raise FileExistsError("Cannot save file because there are already too many files (over 100000).")

        # 保存
        chart.save(filepath, scale_factor=RESOLUTION)
        # altair_saver.save(chart, filepath, vega_cli_options=[f"-s {RESOLUTION}"])


if __name__ == "__main__":
    # コマンドライン引数が2+1でない場合、使い方を表示する
    # if len(sys.argv) != 2:
    #     output = "\n"
    #     output += "[[ Usage ]]\n"
    #     output += "1 command line argument is required.\n"
    #     output += "$ python create_n4u_answer.py [path to csv file]\n"
    #     print(output)
    #     sys.exit(0)

    # target_filepath = os.path.join(DIRPATH, sys.argv[1])

    # # クラウドソーシングCSVファイルの読み込み処理を追加
    # graph = F1ScoreGraphContainter(target_filepath)

    # N4Uが出力するであろう結果をCSV形式で出力
    for title in ["Music", "Bibliorecords", "Persons"]:
        for acc in [100, 95, 90, 0]:
            if acc != 0:
                target_filepath = os.path.join(
                    DIRPATH,
                    f"{title}_w{acc:03}.csv",
                )
                title_name = f"{title} (Worker accuracy: {acc}%)"
            else:
                target_filepath = os.path.join(
                    DIRPATH,
                    f"{title}_w090_ex.csv",
                )
                title_name = f"{title} (Worker accuracy: 90%)"

            # クラウドソーシングCSVファイルの読み込み処理を追加
            graph = F1ScoreGraphContainter(target_filepath)

            graph.draw_line(
                title_name,
                title == "Music",
                acc == 0,
            )
    # graph.draw_step_chart()
    # graph.draw_ridgeline()
