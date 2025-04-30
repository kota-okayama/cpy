"""Record"""

from dataclasses import dataclass, asdict, field

import random
from .char import ja, ascii


@dataclass
class RecordType:
    """Record type"""

    TEXT = "TEXT"  # テキスト (言語自動判別)
    TEXT_JA = "TEXT_JA"  # 日本語テキスト
    TEXT_EN = "TEXT_EN"  # 英語テキスト
    TEXT_KEY = "TEXT_KEY"  # 重みの強いテキスト
    IGNORE = "IGNORE"  # 利用するデータの対象にしない (データとしては格納する)
    COMPLEMENT_JA = "COMPLEMENT_JA"  # 常用漢字等を含めた日本語で補完する
    COMPLEMENT_ASCII = "COMPLEMENT_ASCII"  # ASCII文字で補完する
    COMPLEMENT_DATE = "COMPLEMENT_DATE"  # 日付を補完する


@dataclass
class Record:
    """1レコードを格納しておくデータ型"""

    id: str
    idx: int
    cluster_id: str = ""
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

    def to_dict_for_save(self) -> "dict[str, str]":
        """
        データ保存用にidx除いたデータを辞書型に変換する

        - - -

        Return
        ------
        result: dict[str, str]
            Recordの持つデータを辞書形式で返す
        """
        result = self.to_dict()
        result.pop("idx")

        return result

    def __eq__(self, other):
        """レコード内のデータが一致しているかを判定する"""

        if not isinstance(other, Record):
            raise ValueError("Unsupported operand type(s) for ==: 'Record' and '{}'".format(type(other).__name__))

        # data の中身を比較し、全て一致していれば True を返す
        # キーが一致しない場合は False を返す
        if self.data.keys() != other.data.keys():
            return False

        # 値を比較し、1つでも不一致があれば False を返す
        for key in self.data.keys():
            if self.data[key] != other.data[key]:
                return False

        return True

    def __ne__(self, other):
        """レコード内のデータが一致していないかを判定する"""

        return not self.__eq__(other)


@dataclass
class RecordMG:
    """補完したレコードとオリジナルのレコードを管理するクラス"""

    re: Record = None  # 空データ補完済み
    re_origin: Record = None  # オリジナルデータ
    lang: str = "ja"

    def __init__(self, record: Record, type: "dict[str, str]" = {}):
        """
        書誌データ型を引数にとり、空データのものについて補完したRecordを提供する

        - - -

        Params
        ------
        record: Record
            1行分のレコード
        type: dict[str, str]
            キーを属性名、値を補完方法とする辞書型
        """

        self.re_origin = record
        self.re = Record(**(record.to_dict()))  # deepcopy

        for key in self.re.data.keys():
            # ASCIIのランダム文字列で補完する
            if key in type.keys() and type[key] == RecordType.COMPLEMENT_ASCII:
                if self.re.data[key] == "":
                    self.re.data[key] = self._make_random_ascii_str(random.randrange(4, 24))

            # 日本語のランダム文字列で補完する
            if key in type.keys() and type[key] == RecordType.COMPLEMENT_JA:
                if self.re.data[key] == "":
                    self.re.data[key] = self._make_random_ja_str(random.randrange(4, 12))

            # 日付を補完する
            if key in type.keys() and type[key] == RecordType.COMPLEMENT_DATE:
                if self.re.data[key] == "":
                    self.re.data[key] = self._make_random_date_str()

    def _make_random_ja_str(self, length: int):
        """
        ひらがなや常用漢字等によりランダム文字列を生成する

        - - -

        Params
        ------
        length: int
            文字列の長さ
        """

        result = ""

        for _ in range(length):
            result += ja[random.randrange(0, len(ja) - 1)]

        return result

    def _make_random_ascii_str(self, length: int):
        """
        ASCIIランダム文字列を生成する

        - - -

        Params
        ------
        length: int
            文字列の長さ
        """

        result = ""

        for _ in range(length):
            result += ascii[random.randrange(0, len(ascii) - 1)]

        return result

    def _make_random_date_str(self):
        """300000以上の6桁の適当な数値文字列を代入する"""

        return "{}".format(random.randrange(300000, 1000000 - 1))

    def __eq__(self, other):
        """レコード内のデータが一致しているかを判定する"""

        if not isinstance(other, RecordMG):
            raise ValueError("Unsupported operand type(s) for ==: 'RecordMG' and '{}'".format(type(other).__name__))

        return self.re_origin == other.re_origin

    def __ne__(self, other):
        """レコード内のデータが一致していないかを判定する"""

        return not self.__eq__(other)
