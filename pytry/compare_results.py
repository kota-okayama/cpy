import yaml
import json
from collections import defaultdict
from typing import Dict, Set, Tuple


def load_gpt_matches(filepath: str) -> Set[Tuple[str, str]]:
    """GPTのマッチング結果を読み込む"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    matches = set()
    for match in data["matches"]:
        record1 = match["record1"]
        record2 = match["record2"]
        # 順序に依存しないようにソートしてタプルとして保存
        matches.add(tuple(sorted([record1, record2])))

    return matches


def load_fasttext_results(filepath: str) -> Set[Tuple[str, str]]:
    """FastTextの結果を読み込む"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    matches = set()
    # グループごとにペアを生成
    for group in data["group"]:
        records = group["records"]
        # 同じグループ内の全ペアを生成
        for i in range(len(records) - 1):
            for j in range(i + 1, len(records)):
                record1_id = records[i]["id"]
                record2_id = records[j]["id"]
                matches.add(tuple(sorted([record1_id, record2_id])))

    return matches


def compare_results(gpt_matches: Set[Tuple[str, str]], fasttext_matches: Set[Tuple[str, str]]):
    """結果を比較して統計を出力"""
    # 共通のマッチ
    common_matches = gpt_matches & fasttext_matches

    # GPTのみのマッチ
    gpt_only = gpt_matches - fasttext_matches

    # FastTextのみのマッチ
    fasttext_only = fasttext_matches - gpt_matches

    print("=== マッチング結果の比較 ===")
    print(f"GPTの総マッチ数: {len(gpt_matches)}")
    print(f"FastTextの総マッチ数: {len(fasttext_matches)}")
    print(f"共通のマッチ数: {len(common_matches)}")
    print(f"GPTのみが見つけたマッチ数: {len(gpt_only)}")
    print(f"FastTextのみが見つけたマッチ数: {len(fasttext_only)}")

    if len(gpt_matches) > 0 and len(fasttext_matches) > 0:
        # 一致率の計算
        agreement = len(common_matches) / len(gpt_matches | fasttext_matches)
        print(f"一致率: {agreement:.2%}")

    return {"common": common_matches, "gpt_only": gpt_only, "fasttext_only": fasttext_only}


def main():
    # ファイルパスは適宜調整してください
    gpt_results = load_gpt_matches("full_matches.yaml")
    fasttext_results = load_fasttext_results("result.yaml")

    results = compare_results(gpt_results, fasttext_results)

    # 詳細な違いの分析を保存
    with open("comparison_details.yaml", "w", encoding="utf-8") as f:
        yaml.dump(
            {
                "gpt_only_matches": sorted(list(results["gpt_only"])),
                "fasttext_only_matches": sorted(list(results["fasttext_only"])),
                "common_matches": sorted(list(results["common"])),
            },
            f,
            allow_unicode=True,
        )


if __name__ == "__main__":
    main()
