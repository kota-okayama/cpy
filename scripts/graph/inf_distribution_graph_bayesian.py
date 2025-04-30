"""クラウドソーシングから統計画像を出力するためのスクリプト (Bayesian推論用)"""

import os
import sys
from dataclasses import dataclass, asdict, field

from csv import reader
import yaml
import pandas as pd

import altair as alt
import altair_saver
import toolz


# Parameter
TAU_SAME = 0.5
TAU_NOT = 0.5
INF_TYPE = "BAYESIAN"  # SUN or BAYESIAN

GRAPH_TITLE = "Inference Distribution"
RESOLUTION = 2  # 出力画像の解像度


# 5000行以上のデータに対応するための設定
_t = lambda data: toolz.curried.pipe(data, alt.limit_rows(max_rows=1000000), alt.to_values)  # TODO: E731を回避すべき
alt.data_transformers.register("custom", _t)
alt.data_transformers.enable("custom")

DIRPATH = os.path.dirname(os.path.abspath(__file__))


@dataclass
class RecordType:
    """Record type"""

    NORMAL = "NORMAL"  # 補完等は何もしない
    COMPLEMENT_ASCII = "COMPLEMENT_ASCII"  # ASCII文字で補完する
    COMPLEMENT_JA = "COMPLEMENT_JA"  # 常用漢字等を含めた日本語で補完する
    COMPLEMENT_DATE = "COMPLEMENT_DATE"  # 日付を補完する


@dataclass
class Record:
    """1レコードを格納しておくデータ型"""

    id: str
    group: str = ""
    data: "dict[str, str]" = field(default_factory=dict)

    def to_dict(self) -> "dict[str, str]":
        """
        データを辞書型に変換する

        - - -

        Return
        ------
        result: dict[str, str]
            Recordの持つデータを辞書形式で返す
        """

        return asdict(self)


@dataclass
class Bookdata:
    """[廃止] 書誌情報を格納しておくデータ型"""

    id: str = ""  # 一意に書誌データを識別するためのid (uuidが格納されることを想定)
    group: str = ""  # 書誌データがどの群に所属するのかを識別するためのタグ (isbnが格納されることを想定)
    lib_id: str = ""  # 図書館が書誌に対して固有に振っているid
    title: str = ""  # 書誌名
    author: str = ""  # 著者名
    publisher: str = ""  # 出版社名
    isbn: str = ""  # isbn
    pubdate: str = ""  # 出版日
    source: str = ""  # 所蔵図書館名

    def to_dict(self) -> "dict[str, str]":
        """
        データを辞書型に変換する

        - - -

        Return
        ------
        result: dict[str, str]
            Bookdataの持つデータを辞書形式で返す
        """

        return asdict(self)


class RecordContainerLight:
    """
    Recordの読み込みや書き込み等の処理を行い、メインプログラムの補助を行うコンテナクラス

    読み込み可能なContainterType: NORMAL, (一部PAIR)
    """

    def __init__(self) -> None:
        """コンストラクタ"""

        self.record_id: "dict[str, Record]" = {}  # idをキーにして書誌データを格納する
        self.record_group: "dict[str, list[Record]]" = {}  # groupをキーにして書誌データ群を格納する
        self.record_amount_group: "dict[int, list[str]]" = {}  # グループ数をキーにして値にはgroupを格納する
        self.inf_attr = {}
        self.version = "3.0"

    def _calc_record_amount_group(self):
        """現在格納されている書誌データを基にrecord_amount_groupを再構成する"""
        self.record_amount_group = {}  # 初期化

        # 統計情報を取得
        for key, value in self.record_group.items():
            self.record_amount_group[len(value)] = self.record_amount_group.get(len(value), [])  # 総数順に辞書型に格納
            self.record_amount_group[len(value)].append(key)

    def load_file(self, filepath: str):
        """
        yaml/json形式を読み込んでコンテナを構築する

        - - -
        yamlかjsonファイルで構成された書誌データファイル
        """
        # 拡張子からyamlかjsonかを判定する
        _, ext = os.path.splitext(filepath)

        # version 情報を取得する
        version = "1.0"
        with open(filepath, "r", encoding="utf-8") as f:
            if ext == ".yml" or ext == ".yaml":
                data = yaml.safe_load(f)

            if "version" in data:
                version = data["version"]

        # version 3.0 (yaml)
        if version == "3.0":
            content = self._load_yaml_v3_0({}, filepath)

        # version 2.2 (yaml)
        elif version == "2.2":
            content = self._load_yaml_v2_2({}, filepath)

        # version 2.1 以前についてはサポートしない
        else:
            raise Exception("failed yaml loading")

        # 読み込んだ内容をインスタンス変数に格納する
        self._load_current_version(content)

    def _load_current_version(self, content: "dict[str, str]"):
        """現在のバージョンのyaml形式を読み込む"""

        self.inf_attr = content["inf_attr"]
        self.record_id = content["record_id"]
        self.record_group = content["record_group"]
        self.record_amount_group = content["record_amount_group"]
        self.version = content["version"]

    def _load_yaml_v3_0(self, content: "dict[str, str]" = {}, filepath: str = None):
        """version 2.2からのyaml形式の変換 or version: 3.0からのyaml形式を読み込む"""

        # version 2.2からの変換
        if content.get("version", "") == "2.2" and filepath is None:
            content["record_id"] = {}
            content["record_group"] = {}
            content["record_amount_group"] = content["bookdata_amount_group"]  # 使い回し可能

            # Bookdata を Record に変換
            for key, bookdata in content["bookdata_group"].items():
                records = []
                for bd in bookdata:
                    bd: Bookdata
                    data = {
                        "lib_id": bd.lib_id,
                        "title": bd.title,
                        "author": bd.author,
                        "publisher": bd.publisher,
                        "isbn": bd.isbn,
                        "pubdate": bd.pubdate,
                        "source": bd.source,
                    }
                    record = Record(id=bd.id, group=bd.group, data=data)
                    content["record_id"][record.id] = record
                    records.append(record)

                content["record_group"][key] = records

            content["inf_attr"] = {
                "title": RecordType.NORMAL,
                "author": RecordType.COMPLEMENT_JA,
                "publisher": RecordType.COMPLEMENT_JA,
                "pubdate": RecordType.COMPLEMENT_DATE,
            }
            content["version"] = "3.0"
            content["type"] = "TARGET"

        # version 3.0の読み込み
        else:
            content["version"] = "3.0"
            content["record_id"] = {}
            content["record_group"] = {}
            __record_amount_group: "dict[int, list[str]]" = {}
            content["record_amount_group"] = __record_amount_group

            with open(filepath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

                if data["type"] != "TARGET":
                    raise TypeError('Cannot load this yaml type "{}".'.format(data["type"]))

                for key, value in data["data"].items():
                    content["record_amount_group"][len(value)] = content["record_amount_group"].get(
                        len(value), []
                    )  # 総数順に辞書型に格納
                    content["record_amount_group"][len(value)].append(key)

                    records = []
                    for v in value:
                        r = Record(**v)
                        content["record_id"][r.id] = r
                        records.append(r)

                    content["record_group"][key] = records

                content["id"] = data["id"]
                content["type"] = data["type"]
                content["creation_date"] = data["summary"]["creation_date"]
                content["inf_attr"] = data["inf_attr"]
                content["config_match"] = data["summary"]["config_match"]
                content["config_mismatch"] = data["summary"]["config_mismatch"]

        return content

    def _load_yaml_v2_2(self, content: "dict[str, str]" = {}, filepath: str = None):
        """version: 2.2 のyaml形式を読み込む"""

        content["version"] = "2.2"
        content["bookdata_id"] = {}
        content["bookdata_group"] = {}
        __bookdata_amount_group: "dict[int, list[str]]" = {}
        content["bookdata_amount_group"] = __bookdata_amount_group

        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

            if data["type"] != "NORMAL":
                raise TypeError('Cannot load this yaml type "{}".'.format(data["type"]))

            for key, value in data["books"].items():
                content["bookdata_amount_group"][len(value)] = content["bookdata_amount_group"].get(
                    len(value), []
                )  # 総数順に辞書型に格納
                content["bookdata_amount_group"][len(value)].append(key)

                bookdata = []
                for v in value:
                    b = Bookdata(**v)
                    content["bookdata_id"][b.id] = b
                    bookdata.append(b)

                content["bookdata_group"][key] = bookdata

            content["id"] = data["id"]
            content["type"] = data["type"]
            content["creation_date"] = data["summary"]["creation_date"]
            content["config_match"] = data["summary"]["config_match"]
            content["config_mismatch"] = data["summary"]["config_mismatch"]

        return self._load_yaml_v3_0(content)

    def verify_record_pairs(self, id_1: str | Record, id_2: str | Record):
        """
        渡された2つのデータが同じグループに属しているか否かを判定する

        - - -

        Params
        ---------
        id_1: str | Record
            Recordのuuid, Recordのいずれか
        id_2: str | Record
            Recordのuuid, Recordのいずれか
        """
        # 全てidの文字列に直す
        if isinstance(id_1, Record):
            id_1 = id_1.id

        if isinstance(id_2, Record):
            id_2 = id_2.id

        # record_id が存在しない場合はエラーにする
        if id_1 not in self.record_id or id_2 not in self.record_id:
            raise KeyError(f"No Record exists for the id ({id_1} or {id_2}).")

        # 一致不一致を判定
        result = False

        if self.record_id[id_1].group != "" and self.record_id[id_2].group != "":
            result = self.record_id[id_1].group == self.record_id[id_2].group

        return result


class InferenceDistributuionGraphContainter:
    """クラウドソーシングタスクファイルからグラフを描画するクラス"""

    # EXP_WITH_INF で出力されたファイルを読み込む必要あり

    def __init__(self, rc: RecordContainerLight) -> None:
        """コンストラクタ"""

        self.rc = rc
        self.req_cs: "list[list[str]]" = []

        self.num_of_all = 0  # 合計データ数
        self.statics = {
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0,
            "true_negative": 0,
            "unknown(match)": 0,
            "unknown(mismatch)": 0,
        }  # 各種統計情報

    def make_label(self, id_1: str, id_2: str, inf_same: float, inf_not: float) -> str:
        """
        ラベルを作成する

        - - -

        Params
        ------
        id_1: str
            1つ目のRecordのid
        id_2: str
            2つ目のRecordのid

        Return
        ------
        str
            推論結果と実際の結果に基づく付与されるべきラベル
        """

        label = [
            "true_negative",
            "false_negative",
            "false_positive",
            "true_positive",
            "unknown(mismatch)",
            "unknown(match)",
        ]

        id_1, id_2 = sorted([id_1, id_2])

        score = 1 if self.rc.verify_record_pairs(id_1, id_2) else 0

        # 一致と推論
        if 1 - inf_same < TAU_SAME:
            score += 1 * 2

        # 不一致と推論
        elif 1 - inf_not < TAU_NOT:
            score += 0

        # Unknownだった場合
        else:
            score += 1 * 4

        # 統計情報に追加
        self.statics[label[score]] += 1

        return label[score]

    def load_req_cs(self, filepath: str, max_data: int = 0):
        """
        要求されたクラウドソーシングファイルを読み込む

        - - -

        Params
        ------
        filepath: str
            リクエストされたクラウドソーシングファイルのパス
        max_data: int, by default 0
            読み込む最大データ数
        """

        max_data = max_data if max_data != 0 else sys.maxsize

        self.statics = {
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0,
            "true_negative": 0,
            "unknown(match)": 0,
            "unknown(mismatch)": 0,
        }

        req_cs = []
        with open(filepath, "r", encoding="utf-8") as f:
            csv = reader(f)

            for i, row in enumerate(csv):
                # 上限値を超えたらデータの読み込みを終える
                if i > max_data:
                    break

                # 1行目は属性値として無視する
                if i == 0:
                    continue

                # dataが欠損してRecordを構成できない場合は、スキップする
                else:
                    try:
                        _content = []
                        _content.append(float(row[-2]))  # same
                        _content.append(float(row[-1]))  # not_same
                        _content.append(
                            self.make_label(
                                row[1 + (len(self.rc.inf_attr) + 1) * 0],  # id_1
                                row[1 + (len(self.rc.inf_attr) + 1) * 1],  # id_2
                                float(row[-2]),  # same
                                float(row[-1]),  # not_same
                            )
                        )  # label
                        req_cs.append(_content)

                    except IndexError:
                        continue

        # 合計データ数を取得
        with open(filepath, "r", encoding="utf-8") as f:
            csv = reader(f)
            self.num_of_all = len(list(csv)) - 1

        # インスタンス変数に格納
        self.req_cs = req_cs

    def draw_histogram(self):
        """altairによるヒストグラム描画"""

        # 色の設定
        color_lst = [
            alt.ColorName("lightpink"),
            alt.ColorName("blue"),
            alt.ColorName("red"),
            alt.ColorName("lightblue"),
        ]

        # ラベルの設定
        label = [
            f"True Positive ({self.statics['true_positive']})",
            f"False Positive ({self.statics['false_positive']})",
            f"False Negative ({self.statics['false_negative']})",
            f"True Negative ({self.statics['true_negative']})",
        ]

        replace_target = {
            "true_positive": label[0],
            "false_positive": label[1],
            "false_negative": label[2],
            "true_negative": label[3],
        }

        # altairで描画
        df = pd.DataFrame(self.req_cs, columns=["same", "not_same", "inf_type"])
        df = df.replace(replace_target)
        chart = (
            alt.Chart(
                df,
                title=f"{GRAPH_TITLE} (n = {len(self.req_cs)} / {self.num_of_all})",
            )
            .mark_bar()
            .encode(
                x=alt.X(
                    "same",
                    bin=alt.Bin(step=0.01, extent=[0, 1]),
                    title="Same probability",
                ),
                y=alt.Y(
                    "count(same)",
                    scale=alt.Scale(domain=[0, 25000]),
                    title="Number of pairs",
                ),
                color=alt.Color(
                    "inf_type",
                    title="Inference Result",
                    scale=alt.Scale(domain=label, range=color_lst),
                ),
            )
        )

        # 保存先ディレクトリ生成
        if not os.path.exists(os.path.join(DIRPATH, "output")):
            os.mkdir(os.path.join(DIRPATH, "output"))

        # ファイル名決定
        filepath = None
        for i in range(1, 100000):
            if not os.path.exists(os.path.join(DIRPATH, "output", "result-{:0=5}.png".format(i))):
                filepath = os.path.join(DIRPATH, "output", "result-{:0=5}.png".format(i))
                break

        # 既に規定数以上のファイルが保存されている場合はエラーを出す
        if filepath is None:
            raise FileExistsError("Cannot save file because there are already too many files (over 100000).")

        # 保存
        altair_saver.save(chart, filepath, vega_cli_options=[f"-s {RESOLUTION}"])

    def draw_step_chart(self):
        """altairによるチャート形式描画"""

        # ラベルの設定
        label = [
            f"True Positive ({self.statics['true_positive']})",
            f"False Positive ({self.statics['false_positive']})",
            f"False Negative ({self.statics['false_negative']})",
            f"True Negative ({self.statics['true_negative']})",
        ]

        replace_target = {
            "true_positive": label[0],
            "false_positive": label[1],
            "false_negative": label[2],
            "true_negative": label[3],
        }

        dummy_req_cs = []

        for i in range(101):
            _content = []
            _content.append(i / 100)  # same
            _content.append((100 - i) / 100)  # not_same
            _content.append("dummy")  # label
            dummy_req_cs.append(_content)

        # altairで描画
        df = pd.DataFrame(self.req_cs + dummy_req_cs, columns=["same", "not_same", "inf_type"])
        df = df.replace(replace_target)
        chart = (
            alt.Chart(
                df,
                title=f"{GRAPH_TITLE} (n = {len(self.req_cs)} / {self.num_of_all})",
            )
            .mark_area(
                opacity=0.5,
                color=alt.ColorName("orange"),
                line={"color": "orange"},
                lineHeight=2,
                clip=True,
            )
            .encode(
                x=alt.X(
                    "same",
                    bin=alt.Bin(step=0.01, extent=[0, 1]),
                    title="Same probability",
                ),
                y=alt.Y(
                    "count(same)",
                    scale=alt.Scale(domain=[0, 6000]),
                    title="Number of pairs",
                ),
            )
        )

        # 保存先ディレクトリ生成
        if not os.path.exists(os.path.join(DIRPATH, "output")):
            os.mkdir(os.path.join(DIRPATH, "output"))

        # ファイル名決定
        filepath = None
        for i in range(1, 100000):
            if not os.path.exists(os.path.join(DIRPATH, "output", "result-{:0=5}.png".format(i))):
                filepath = os.path.join(DIRPATH, "output", "result-{:0=5}.png".format(i))
                break

        # 既に規定数以上のファイルが保存されている場合はエラーを出す
        if filepath is None:
            raise FileExistsError("Cannot save file because there are already too many files (over 100000).")

        # 保存
        altair_saver.save(chart, filepath, vega_cli_options=[f"-s {RESOLUTION}"])


if __name__ == "__main__":
    # コマンドライン引数が2+1でない場合、使い方を表示する
    if len(sys.argv) < 3 or len(sys.argv) > 5:
        output = "\n"
        output += "[[ Usage ]]\n"
        output += "2 ~ 3 command line arguments are required.\n"
        output += "$ python create_n4u_answer.py [path to target yaml file] [path to requied_crowdsourcing csv file] [number of data]\n"
        print(output)
        sys.exit(0)

    target_filepath = os.path.join(DIRPATH, sys.argv[1])
    req_filepath = os.path.join(DIRPATH, sys.argv[2])
    max_data = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    # RecordContainer 作成
    bc = RecordContainerLight()
    bc.load_file(target_filepath)

    # クラウドソーシングCSVファイルの読み込み処理を追加
    graph = InferenceDistributuionGraphContainter(bc)
    graph.load_req_cs(req_filepath, max_data)

    # N4Uが出力するであろう結果をCSV形式で出力
    graph.draw_histogram()
    graph.draw_step_chart()
    # graph.draw_ridgeline()
