"""クラウドソーシングを要求されたCSVファイルから回答を生成するスクリプト"""

import os
import sys
from dataclasses import dataclass, asdict, field

import codecs
from csv import reader
import yaml
from uuid import uuid4


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

    def _load_yaml_v3_0(self, content: "dict[str, any]" = {}, filepath: str = None):
        """version 2.2 からの yaml 形式の変換 or version 3.0 からの yaml 形式を読み込む"""

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


class CrowdSourcingCsvContainter:
    """クラウドソーシングを行うためのCSVを処理する"""

    def __init__(self, rc: RecordContainerLight) -> None:
        """コンストラクタ"""

        self.rc = rc
        self.req_cs: "list[list[str]]" = []

    def _escape_csv(self, content: str) -> str:
        """CSVファイルとして出力するための文字列をエスケープする

        Params
        ------
        content: str
            エスケープ対象の文字列

        Return
        ------
        str
            エスケープ後の文字列
        """
        result = content.replace('"', '""')

        return f'"{result}"'

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
                        _id = []

                        for j in range(int(row[0])):
                            _id.append(row[(len(self.rc.inf_attr) + 1) * j + 1])
                        req_cs.append(_id)

                    except IndexError:
                        continue

        # インスタンス変数に格納
        self.req_cs = req_cs

    def output_result(self, filepath: str, type: str = "N4U"):
        """クラウドソーシング結果を出力する"""

        req_result = []
        output = ""

        # 結果を整形する
        for ids in self.req_cs:
            for i in range(len(ids) - 1):
                for j in range(i + 1, len(ids)):
                    req_result.append(
                        "{"
                        + f"\"{ids[i]},{ids[j]}\":{'true' if self.rc.verify_record_pairs(ids[i], ids[j]) else 'false'}"
                        + "}"
                    )

        # NextCrowd4U用に整形
        if type == "N4U":
            _r = []

            # NextCrowd4Uのカラムに依存
            title = [
                "id",
                "dataitem_id",
                "run_id",
                "qualified_worker_type",
                "qualified_at",
                "assigned_worker_id",
                "assigned_at",
                "result",
                "completed_at",
                "created_at",
                "updated_at",
            ]
            result_index = title.index("result")

            for task in req_result:
                _tmp = ['""' for _ in range(len(title))]
                _tmp[result_index] = self._escape_csv(task)
                _r.append(",".join(_tmp))

            # タスクタイトル付与
            output = "\n".join([",".join(title)] + _r)

        # CSVファイル出力
        with codecs.open(filepath, "w", "utf-8") as f:
            f.write(output)


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
    rc = RecordContainerLight()
    rc.load_file(target_filepath)

    # クラウドソーシングCSVファイルの読み込み処理を追加
    cs = CrowdSourcingCsvContainter(rc)
    cs.load_req_cs(req_filepath, max_data)

    # N4Uが出力するであろう結果をCSV形式で出力
    cs.output_result(os.path.join(DIRPATH, f"result_{uuid4()}.csv"))
