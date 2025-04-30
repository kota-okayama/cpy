"""ユーティリティ"""

import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class path:
    """os.path wrapper"""

    @staticmethod
    def at_parse(*args: "list[str]") -> str:
        """@マークが文字列にあった場合、それまでのパスを切り捨て、@マークを本体ルートディレクトリの絶対パスに置換する"""

        result = []

        if len(args) > 0:
            for arg in args:
                # 文字列型でなければ、その時点で終了する
                if not isinstance(arg, str):
                    return args

                # 文字列をリストに分割
                target = arg.split(os.path.sep)
                __result = []

                for t in target:
                    if t == "@":
                        # @ が含まれていたら、それまでのパスを切り捨てて、ルートディレクトリの絶対パスに置換する
                        result = []
                        __result = [ROOT]
                    else:
                        __result.append(t)

                # 元の文字列に戻して result に追加
                result.append(os.path.sep.join(__result))

            return result

        return args

    @staticmethod
    def join(*args: "list[str]") -> str:
        """os.path.join のラッパー"""
        return os.path.join(*path.at_parse(*args))

    @staticmethod
    def abspath(arg: str) -> str:
        """os.path.abspath のラッパー"""
        return os.path.abspath(*path.at_parse(arg))

    @staticmethod
    def dirname(*args: str) -> str:
        """os.path.dirname のラッパー"""
        return os.path.dirname(*path.at_parse(*args))

    @staticmethod
    def exists(arg: str) -> bool:
        """os.path.exists のラッパー"""
        return os.path.exists(*path.at_parse(arg))

    @staticmethod
    def isdir(arg: str) -> bool:
        """os.path.isdir のラッパー"""
        return os.path.isdir(*path.at_parse(arg))

    @staticmethod
    def isfile(arg: str) -> bool:
        """os.path.isfile のラッパー"""
        return os.path.isfile(*path.at_parse(arg))

    @staticmethod
    def split(arg: str) -> "tuple[str]":
        """os.path.split のラッパー"""
        return os.path.split(*path.at_parse(arg))

    @staticmethod
    def splitext(arg: str) -> "tuple[str]":
        """os.path.splitext のラッパー"""
        return os.path.splitext(*path.at_parse(arg))

    @staticmethod
    def basename(arg: str) -> str:
        """os.path.basename のラッパー"""
        return os.path.basename(*path.at_parse(arg))

    @staticmethod
    def getsize(arg: str) -> int:
        """os.path.getsize のラッパー"""
        return os.path.getsize(*path.at_parse(arg))
