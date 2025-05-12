import csv
import os
import random

# --- グローバル設定 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REVIEW_CANDIDATE_PAIRS_FILENAME = "review_candidate_pairs.csv"  # detect_inconsistent_triangles.py の出力
REVIEW_CANDIDATE_PAIRS_PATH = os.path.join(BASE_DIR, REVIEW_CANDIDATE_PAIRS_FILENAME)

OUTPUT_DIR = BASE_DIR
SIMULATION_RESULTS_FILENAME_TEMPLATE = "human_review_simulation_accuracy_{}.csv"  # 精度ごとにファイル名を変える

# LLMの類似度スコアから「類似/非類似」を判断するための閾値
LLM_SIMILARITY_THRESHOLD = 0.5  # この値以上なら「類似」と判断


def simulate_human_judgement(ground_truth_label, human_accuracy):
    """
    指定された精度で人間の判断をシミュレートする。
    ground_truth_label: True (類似) または False (非類似)
    human_accuracy: 人間の判断の正解率 (0.0 - 1.0)
    戻り値: True (類似) または False (非類似) のシミュレートされた判断
    """
    if not (0.0 <= human_accuracy <= 1.0):
        raise ValueError("human_accuracy は 0.0 から 1.0 の間の値である必要があります。")

    if random.random() < human_accuracy:
        return ground_truth_label  # 正しく判断
    else:
        return not ground_truth_label  # 誤って判断


def main():
    human_accuracy_levels = [1.0]  # テストしたい人間の精度レベル

    if not os.path.exists(REVIEW_CANDIDATE_PAIRS_PATH):
        print(f"エラー: レビュー対象ペアのファイルが見つかりません: {REVIEW_CANDIDATE_PAIRS_PATH}")
        return

    all_pairs_data = []
    try:
        with open(REVIEW_CANDIDATE_PAIRS_PATH, "r", newline="", encoding="utf-8") as infile:
            reader = csv.DictReader(infile)
            # ヘッダーの存在チェックの修正
            required_headers = ["record_id_1", "record_id_2", "llm_similarity_score", "same_cluster_ground_truth"]
            if not reader.fieldnames:  # reader.fieldnamesがNoneの場合（空ファイルなど）
                print(
                    f"エラー: {REVIEW_CANDIDATE_PAIRS_PATH} からヘッダーが読み取れませんでした。ファイルが空または不正な可能性があります。"
                )
                return
            if not all(header in reader.fieldnames for header in required_headers):
                missing_headers = [h for h in required_headers if h not in reader.fieldnames]
                print(f"エラー: {REVIEW_CANDIDATE_PAIRS_PATH} に必要なヘッダーが不足しています: {missing_headers}")
                print(f"  現在のヘッダー: {reader.fieldnames}")
                return
            for row in reader:
                all_pairs_data.append(row)
    except Exception as e:
        print(f"ファイル読み込みエラー ({REVIEW_CANDIDATE_PAIRS_PATH}): {e}")
        return

    if not all_pairs_data:
        print("レビュー対象のペアデータがありません。")
        return

    print(f"{len(all_pairs_data)} ペアのレビュー候補をロードしました。")

    for acc_level in human_accuracy_levels:
        print(f"\n--- 人間の評価精度 {acc_level*100:.0f}% でシミュレーション ---")

        simulation_results = []
        stats = {
            "total_pairs_processed_for_stats": 0,  # GTがNoneでないペアの数
            "llm_correct_count": 0,
            "human_correct_count": 0,
            "llm_errors_found_by_sampler": 0,  # GTがあってLLMが間違っていたペアの数
            "llm_errors_corrected_by_human": 0,  # 上記のうち人間が正しく判断した数
        }

        for pair_data in all_pairs_data:
            try:
                record_id_1 = pair_data["record_id_1"]
                record_id_2 = pair_data["record_id_2"]

                original_llm_score_str = pair_data.get("llm_similarity_score", "0.0")
                if (
                    not original_llm_score_str
                    or original_llm_score_str.lower() == "none"
                    or original_llm_score_str == "N/A"
                ):
                    original_llm_score = 0.0
                else:
                    original_llm_score = float(original_llm_score_str)

                original_llm_judgement = original_llm_score >= LLM_SIMILARITY_THRESHOLD

                gt_str = pair_data.get("same_cluster_ground_truth", "").strip().lower()
                ground_truth_label = None
                if gt_str == "true":
                    ground_truth_label = True
                elif gt_str == "false":
                    ground_truth_label = False

                simulated_human_judgement = None
                is_llm_correct = None
                is_human_correct = None
                human_corrected_llm_error_flag = False  # このペアでLLMの誤りを人間が修正したか

                if ground_truth_label is not None:
                    stats["total_pairs_processed_for_stats"] += 1
                    simulated_human_judgement = simulate_human_judgement(ground_truth_label, acc_level)

                    is_llm_correct = original_llm_judgement == ground_truth_label
                    if is_llm_correct:
                        stats["llm_correct_count"] += 1
                    else:
                        stats["llm_errors_found_by_sampler"] += 1

                    is_human_correct = simulated_human_judgement == ground_truth_label
                    if is_human_correct:
                        stats["human_correct_count"] += 1

                    if not is_llm_correct and is_human_correct:
                        stats["llm_errors_corrected_by_human"] += 1
                        human_corrected_llm_error_flag = True

                simulation_results.append(
                    {
                        "record_id_1": record_id_1,
                        "record_id_2": record_id_2,
                        "original_llm_score": original_llm_score,
                        "original_llm_judgement": original_llm_judgement,
                        "ground_truth_label": ground_truth_label,
                        "simulated_human_accuracy": acc_level,
                        "simulated_human_judgement": simulated_human_judgement,
                        "is_llm_correct_vs_gt": is_llm_correct,
                        "is_human_correct_vs_gt": is_human_correct,
                        "human_corrected_llm_error": human_corrected_llm_error_flag,  # 修正: Flagとして保存
                    }
                )
            except Exception as e:
                print(f"エラー: ペアデータ処理中にエラー: {pair_data} - {e}")
                continue

        output_filename = SIMULATION_RESULTS_FILENAME_TEMPLATE.format(f"{acc_level*100:.0f}")
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        try:
            if simulation_results:  # 結果がある場合のみ書き出し
                with open(output_path, "w", newline="", encoding="utf-8") as outfile:
                    fieldnames = simulation_results[0].keys()
                    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(simulation_results)
                    print(f"  シミュレーション結果を {output_path} に保存しました。")
            else:
                print("  シミュレーション結果が空のため、ファイルは作成されませんでした。")
        except Exception as e:
            print(f"  エラー: シミュレーション結果のCSV書き込み中にエラー: {e}")

        print("  --- 統計サマリ ---")
        print(f"  処理対象ペア総数 (Ground Truthあり): {stats['total_pairs_processed_for_stats']}")
        if stats["total_pairs_processed_for_stats"] > 0:
            print(
                f"  元のLLMの正解率 (vs GT): {stats['llm_correct_count']/stats['total_pairs_processed_for_stats']*100:.2f}% ({stats['llm_correct_count']}/{stats['total_pairs_processed_for_stats']})"
            )
            print(
                f"  シミュレートされた人間の正解率 (vs GT): {stats['human_correct_count']/stats['total_pairs_processed_for_stats']*100:.2f}% ({stats['human_correct_count']}/{stats['total_pairs_processed_for_stats']})"
            )
            print(f"  サンプラーが発見したLLMの誤り候補数 (GTあり): {stats['llm_errors_found_by_sampler']}")
            if stats["llm_errors_found_by_sampler"] > 0:
                print(
                    f"    うち、人間(精度 {acc_level*100:.0f}%)が正しく修正できた割合: {stats['llm_errors_corrected_by_human']/stats['llm_errors_found_by_sampler']*100:.2f}% ({stats['llm_errors_corrected_by_human']}/{stats['llm_errors_found_by_sampler']})"
                )
            else:
                print(f"    LLMの誤り候補が見つからなかったか、GTがなかったため修正率の計算はスキップします。")
        else:
            print("  Ground Truthが利用可能なペアがなかったため、詳細な統計は計算できませんでした。")

    print("\n全シミュレーション処理が完了しました。")


if __name__ == "__main__":
    main()
