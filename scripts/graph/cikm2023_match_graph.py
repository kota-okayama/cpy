"""クラウドソーシングから統計画像を出力するためのスクリプト (Bayesian推論用)"""

import os
import sys

from csv import reader
import pandas as pd

import altair as alt
import altair_saver
import toolz


# Parameter
TAU_SAME = 0.5
TAU_NOT = 0.5
INF_TYPE = "BAYESIAN"  # SUN or BAYESIAN

GRAPH_TITLE = "Bibliographic data"
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

    def draw_line(self):
        """altairによる折れ線グラフ描画"""

        # color_lst = [alt.value(color_code) for color_code in color_codes]

        # altairで描画
        df = pd.DataFrame(self.result, columns=["counter", "num", "type"])
        chart = (
            alt.Chart(
                df,
                mark="circle",
                title=f"{GRAPH_TITLE}",
            )
            .mark_line(
                size=2,
                point=alt.OverlayMarkDef(size=60),
            )
            .encode(
                x=alt.X(
                    "counter",
                    title="Crowdsourcing Tasks",
                ),
                y=alt.Y(
                    "num",
                    title="Number of true matched pairs presented for active learning",
                    scale=alt.Scale(domain=[0, 1400]),
                ),
                color=alt.Color(
                    "type",
                    title="Sampling Method",
                    scale=alt.Scale(
                        domain=[
                            "axiom-driven-sampling",
                            "uncertainty-sampling",
                            "random-sampling",
                        ],
                        range=[
                            "#ff6d60",
                            "#45d86e",
                            "#808080",
                        ],
                    ),
                    legend=alt.Legend(
                        orient="none",  # orientをnoneに設定するとlegenX,legendYが指定できる
                        legendX=5,
                        legendY=5,
                        # strokeColor="blue",
                        fillColor="white",
                        padding=5,
                    ),
                ),
                shape=alt.Shape(
                    "type",
                    scale=alt.Scale(
                        range=[
                            "circle",
                            "square",
                            "diamond",
                        ],
                    ),
                    legend=None,
                ),
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
        altair_saver.save(chart, filepath, vega_cli_options=[f"-s {RESOLUTION}"])


if __name__ == "__main__":
    # コマンドライン引数が2+1でない場合、使い方を表示する
    if len(sys.argv) != 2:
        output = "\n"
        output += "[[ Usage ]]\n"
        output += "1 command line argument is required.\n"
        output += "$ python create_n4u_answer.py [path to csv file]\n"
        print(output)
        sys.exit(0)

    target_filepath = os.path.join(DIRPATH, sys.argv[1])

    # クラウドソーシングCSVファイルの読み込み処理を追加
    graph = F1ScoreGraphContainter(target_filepath)

    # N4Uが出力するであろう結果をCSV形式で出力
    graph.draw_line()
    # graph.draw_step_chart()
    # graph.draw_ridgeline()
