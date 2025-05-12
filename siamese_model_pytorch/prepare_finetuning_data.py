import csv
import json
import os
import yaml
import sys  # sys.exitのため追加

# --- グローバル設定 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIMULATION_RESULTS_FILENAME = "human_review_simulation_accuracy_100.csv"
SIMULATION_RESULTS_PATH = os.path.join(BASE_DIR, SIMULATION_RESULTS_FILENAME)

PROJECT_ROOT_ASSUMED = os.path.abspath(os.path.join(BASE_DIR, ".."))
BENCHMARK_DIR_RELATIVE_TO_PROJECT_ROOT = "benchmark/bib_japan_20241024/1k"
RECORD_YAML_FILENAME = "record.yml"
RECORD_YAML_PATH = os.path.join(PROJECT_ROOT_ASSUMED, BENCHMARK_DIR_RELATIVE_TO_PROJECT_ROOT, RECORD_YAML_FILENAME)

OUTPUT_JSONL_FILENAME = "finetuning_data.jsonl"
OUTPUT_JSONL_PATH = os.path.join(BASE_DIR, OUTPUT_JSONL_FILENAME)

# グローバル変数として書誌データを保持
BIB_DATA = {}


# --- 書誌データ読み込み関連関数 (evaluate_pairs_with_openai_async.py から拝借・調整) ---
def load_bib_data_for_finetuning(yaml_path):
    global BIB_DATA
    BIB_DATA = {}
    if not os.path.exists(yaml_path):
        print(f"エラー: 書誌データファイルが見つかりません: {yaml_path}")
        sys.exit(1)
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            all_data = yaml.safe_load(f)

        if isinstance(all_data, dict):
            possible_records_dict = all_data
            if "records" in all_data and isinstance(all_data["records"], dict):
                possible_records_dict = all_data["records"]

            if isinstance(possible_records_dict, dict):
                for key, value_list in possible_records_dict.items():  # value_listのtypo修正 value
                    if key in ["version", "type", "id", "summary", "inf_attr"]:
                        continue
                    if isinstance(value_list, list):
                        for record in value_list:
                            if isinstance(record, dict) and "id" in record and "data" in record:
                                BIB_DATA[str(record["id"])] = record["data"]
                            elif isinstance(record, dict) and "id" in record:
                                record_data_candidate = {
                                    k_rec: v_rec for k_rec, v_rec in record.items() if k_rec not in ["id", "cluster_id"]
                                }
                                if record_data_candidate:
                                    BIB_DATA[str(record["id"])] = record_data_candidate

        if not BIB_DATA:
            print(f"エラー: {yaml_path} から書誌データロード不可、または空。YAMLの構造を確認してください。")
            sys.exit(1)
        print(f"{len(BIB_DATA)} 件の書誌データを {yaml_path} からロードしました。")

    except yaml.YAMLError as e:
        print(f"エラー: 書誌データファイル ({yaml_path}) のYAML形式が正しくありません: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"エラー: 書誌データファイル ({yaml_path}) の読み込み中に予期せぬエラー: {e}")
        sys.exit(1)


def get_record_details_for_finetuning_prompt(record_id):
    if not BIB_DATA:
        print("エラー: 書誌データがロードされていません。(get_record_details_for_finetuning_prompt)")
        # この関数が呼ばれる時点ではBIB_DATAはロードされているはずなので、基本的にはここに来ない想定
        return "情報取得エラー: BIB_DATA未ロード"

    bib_details = BIB_DATA.get(str(record_id))
    if not bib_details:
        return f"レコードID {record_id} の書誌情報なし"

    title = bib_details.get("bib1_title", "タイトル不明")
    authors_str = bib_details.get("bib1_author", "著者不明")
    publisher = bib_details.get("bib1_publisher", "出版社不明")
    pubdate = bib_details.get("bib1_pubdate", "出版日不明")
    return f"タイトル: {title}\n著者: {authors_str}\n出版社: {publisher}\n出版日: {pubdate}"


# --- メイン処理 ---
def main():
    print("ファインチューニング用データ作成処理を開始します...")

    load_bib_data_for_finetuning(RECORD_YAML_PATH)  # BIB_DATAをグローバルにロード
    # BIB_DATAがロード失敗した場合は load_bib_data_for_finetuning 内で exit する

    finetuning_samples = []

    if not os.path.exists(SIMULATION_RESULTS_PATH):
        print(f"エラー: シミュレーション結果ファイルが見つかりません: {SIMULATION_RESULTS_PATH}")
        return

    try:
        with open(SIMULATION_RESULTS_PATH, "r", newline="", encoding="utf-8") as infile:
            reader = csv.DictReader(infile)
            if not reader.fieldnames:  # ヘッダーがない場合（空ファイルなど）
                print(f"エラー: {SIMULATION_RESULTS_PATH} からヘッダーが読み取れませんでした。")
                return

            for row in reader:
                is_llm_correct_str = row.get("is_llm_correct_vs_gt", "").strip().lower()

                if is_llm_correct_str == "false":  # LLMが間違っていたケースのみを対象
                    record_id_1 = row.get("record_id_1")
                    record_id_2 = row.get("record_id_2")
                    ground_truth_label_str = row.get("ground_truth_label", "").strip().lower()

                    if not record_id_1 or not record_id_2:
                        print(f"警告: record_idが不足している行があります: {row}。スキップします。")
                        continue

                    if ground_truth_label_str not in ["true", "false"]:
                        print(
                            f"警告: ペア ({record_id_1}, {record_id_2}) の Ground Truth が不正です ('{ground_truth_label_str}')。スキップします。"
                        )
                        continue

                    ground_truth_is_similar = ground_truth_label_str == "true"

                    bib_info_1 = get_record_details_for_finetuning_prompt(record_id_1)
                    bib_info_2 = get_record_details_for_finetuning_prompt(record_id_2)

                    if (
                        "情報取得エラー" in bib_info_1
                        or "書誌情報なし" in bib_info_1
                        or "情報取得エラー" in bib_info_2
                        or "書誌情報なし" in bib_info_2
                    ):
                        print(
                            f"警告: ペア ({record_id_1}, {record_id_2}) の書誌情報取得に失敗。スキップします。詳細1: {bib_info_1}, 詳細2: {bib_info_2}"
                        )
                        continue

                    system_prompt = (
                        "あなたは2つの書誌情報が実質的に同一の文献を指すかどうかを判断する専門家です。\n"
                        "まず、2つの書誌情報が同一の文献と思われる場合は「はい」、そうでない場合は「いいえ」で明確に回答してください。\n"
                        "次に、その判断の確信度を示す類似度スコアを0.0（全く異なる）から1.0（完全に同一）の範囲で提示してください。"
                    )

                    user_prompt = (
                        f"以下の2つの書誌情報が、実質的に同一の文献を指しているかどうかを判断してください。\n\n"
                        f"書誌情報1:\n{bib_info_1}\n\n"
                        f"書誌情報2:\n{bib_info_2}\n\n"
                        "これらは同一の文献ですか？\n回答:"
                    )

                    if ground_truth_is_similar:
                        assistant_response = "はい\n類似度スコア: 1.0"
                    else:
                        assistant_response = "いいえ\n類似度スコア: 0.0"

                    finetuning_samples.append(
                        {
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                                {"role": "assistant", "content": assistant_response},
                            ]
                        }
                    )

    except Exception as e:
        print(f"シミュレーション結果ファイル ({SIMULATION_RESULTS_PATH}) の処理中にエラー: {e}")
        import traceback  # 詳細なエラー表示のため

        traceback.print_exc()
        return

    if not finetuning_samples:
        print("ファインチューニング対象のサンプルが0件でした。処理を終了します。")
        return

    print(f"{len(finetuning_samples)} 件のファインチューニング用サンプルを作成しました。")

    try:
        with open(OUTPUT_JSONL_PATH, "w", encoding="utf-8") as outfile:
            for sample in finetuning_samples:
                outfile.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"ファインチューニング用データを {OUTPUT_JSONL_PATH} に保存しました。")
    except Exception as e:
        print(f"エラー: JSONLファイル書き込み中にエラー: {e}")


if __name__ == "__main__":
    main()
