import openai
import os
import csv
import time
import yaml
import sys
import re
import asyncio  # 非同期処理のため追加
import json  # キャッシュのため追加

# プロジェクトルートをPythonパスに追加 (他モジュールをインポートするため)
# api_client.py など、他の場所にあるモジュールをインポートする場合に必要に応じて調整
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, "..")) # この構造だと1つ上がプロジェクトルート
# if project_root not in sys.path:
#     sys.path.append(project_root)

# --- グローバル設定 ---
# OpenAI APIキー (環境変数から取得)
API_KEY = os.environ.get("OPENAI_API_KEY")

# ファイルパス設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV_FILENAME = "evaluation_candidate_pairs.csv"
INPUT_CSV_PATH = os.path.join(BASE_DIR, INPUT_CSV_FILENAME)

PROJECT_ROOT_ASSUMED = os.path.abspath(os.path.join(BASE_DIR, ".."))
BENCHMARK_DIR_RELATIVE_TO_PROJECT_ROOT = "benchmark/bib_japan_20241024/1k"
RECORD_YAML_FILENAME = "record.yml"
RECORD_YAML_PATH = os.path.join(PROJECT_ROOT_ASSUMED, BENCHMARK_DIR_RELATIVE_TO_PROJECT_ROOT, RECORD_YAML_FILENAME)

OUTPUT_CSV_FILENAME = "llm_evaluation_results_async.csv"  # 出力ファイル名を変更
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, OUTPUT_CSV_FILENAME)

# OpenAI API設定
OPENAI_MODEL = "gpt-4o-mini"
REQUEST_TIMEOUT = 60  # タイムアウトを少し延長 (非同期処理で多数のリクエストが滞留する可能性を考慮)
# REQUEST_INTERVAL は非同期処理では直接使わないが、Semaphoreで制御

# --- 新しい設定 ---
CACHE_FILENAME = "llm_evaluation_cache.json"
CACHE_FILE_PATH = os.path.join(BASE_DIR, CACHE_FILENAME)
MAX_CONCURRENT_REQUESTS = 5  # OpenAI APIへの同時リクエスト数の上限 (適宜調整)
PROCESS_LIMIT_PAIRS = 0  # 処理するペア数の上限 (0またはNoneで無制限)
CHUNK_SIZE = 50  # 新しい定数: 1チャンクあたりのタスク数 (例: 50)

# グローバル変数として書誌データとキャッシュを保持
BIB_DATA = {}
LLM_CACHE = {}


# --- キャッシュ関連関数 ---
def load_cache():
    """キャッシュファイルから結果を読み込む"""
    global LLM_CACHE
    if os.path.exists(CACHE_FILE_PATH):
        try:
            with open(CACHE_FILE_PATH, "r", encoding="utf-8") as f:
                LLM_CACHE = json.load(f)
            print(f"{len(LLM_CACHE)} 件のキャッシュエントリを {CACHE_FILE_PATH} からロードしました。")
        except Exception as e:
            print(f"警告: キャッシュファイル {CACHE_FILE_PATH} の読み込みに失敗しました: {e}")
            LLM_CACHE = {}
    else:
        print(f"キャッシュファイル {CACHE_FILE_PATH} が見つかりません。新しいキャッシュを作成します。")
        LLM_CACHE = {}


def save_cache():
    """現在のキャッシュをファイルに保存する"""
    try:
        with open(CACHE_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(LLM_CACHE, f, ensure_ascii=False, indent=4)
        # print(f"キャッシュを {CACHE_FILE_PATH} に保存しました。") # 頻繁なログ出力を避ける
    except Exception as e:
        print(f"警告: キャッシュファイル {CACHE_FILE_PATH} への保存に失敗しました: {e}")


def get_cache_key(id1, id2):
    """キャッシュ用のキーを生成する (IDの順序を問わない文字列キー)"""
    # タプルではなく、ソートされたIDをアンダースコアで結合した文字列をキーとする
    sorted_ids = sorted((str(id1), str(id2)))
    return f"{sorted_ids[0]}_{sorted_ids[1]}"


def load_bib_data(yaml_path):
    """
    record.yml から書誌データをロードし、グローバル変数 BIB_DATA に格納する。
    キーはレコードID、値は書誌情報の辞書。
    (この関数は元のスクリプトからほぼ変更なし)
    """
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
                for key, value in possible_records_dict.items():
                    if key in ["version", "type", "id", "summary", "inf_attr"]:
                        continue
                    if isinstance(value, list):
                        for record in value:
                            if isinstance(record, dict) and "id" in record and "data" in record:
                                BIB_DATA[str(record["id"])] = record["data"]
                            elif isinstance(record, dict) and "id" in record:
                                record_data_candidate = {
                                    k: v for k, v in record.items() if k not in ["id", "cluster_id"]
                                }
                                if record_data_candidate:
                                    BIB_DATA[str(record["id"])] = record_data_candidate
                                else:
                                    print(f"警告: cluster {key} 内のレコードに書誌情報なし: {record}")
                            # else:
                            #     print(f"警告: cluster {key} 内のレコード形式不正: {record}")
                        # else:
                        # print(f"デバッグ: キー {key} の値はリストではない: {type(value)}")
                        pass
            else:
                print(f"エラー: {yaml_path} の 'records' が期待する辞書形式ではない。")
                # sys.exit(1) # コメントアウトして他の可能性も試す

        elif isinstance(all_data, list):
            print("情報: YAMLがレコードの直接リスト形式と解釈試行。")
            for record_item in all_data:
                if isinstance(record_item, dict) and "id" in record_item and "data" in record_item:
                    BIB_DATA[str(record_item["id"])] = record_item["data"]
                elif isinstance(record_item, dict) and "id" in record_item:
                    record_data_candidate = {k: v for k, v in record_item.items() if k not in ["id", "cluster_id"]}
                    if record_data_candidate:
                        BIB_DATA[str(record_item["id"])] = record_data_candidate
                    # else:
                    #     print(f"警告: リスト内レコードに書誌情報なし: {record_item}")
                # else:
                #     print(f"警告: リスト内アイテム形式不正: {record_item}")
        else:
            print(f"エラー: {yaml_path} の書誌データ形式を解釈できません。")
            sys.exit(1)

        if not BIB_DATA:
            print(f"エラー: {yaml_path} から書誌データロード不可、または空。YAML構造確認要。")
            # sys.exit(1) # 一旦コメントアウトして他のエラーを確認できるようにする

        print(f"{len(BIB_DATA)} 件の書誌データを {yaml_path} からロードしました。")
        if len(BIB_DATA) == 0:  # 0件の場合はエラーとして扱う
            print(f"エラー: 書誌データが0件です。処理を続行できません。YAMLファイルを確認してください: {yaml_path}")
            sys.exit(1)

    except yaml.YAMLError as e:
        print(f"エラー: 書誌データファイル ({yaml_path}) のYAML形式が正しくありません: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"エラー: 書誌データファイル ({yaml_path}) の読み込み中に予期せぬエラー: {e}")
        sys.exit(1)


def get_record_details_for_prompt(record_id):
    """
    指定されたレコードIDの書誌情報をプロンプト用に整形して返す。
    (この関数は元のスクリプトからほぼ変更なし)
    """
    if not BIB_DATA:
        # print("エラー: 書誌データ未ロード。load_bib_data() を呼んでください。") # mainでチェック済
        return "書誌データ未ロードエラー"

    bib_details = BIB_DATA.get(str(record_id))
    if not bib_details:
        # print(f"警告: レコードID {record_id} の書誌詳細情報 (dataキー下) が見つかりません。")
        return f"レコードID {record_id} 情報なし"

    title = bib_details.get("bib1_title", "タイトル不明")
    authors_str = bib_details.get("bib1_author", "著者不明")
    publisher = bib_details.get("bib1_publisher", "出版社不明")
    pubdate = bib_details.get("bib1_pubdate", "出版日不明")

    return f"タイトル: {title}\n著者: {authors_str}\n出版社: {publisher}\n出版日: {pubdate}"


def parse_llm_response(response_text):
    """
    LLMからの応答文字列をパースし、判定、類似度スコア、理由を抽出する。
    (この関数は元のスクリプトからほぼ変更なし)
    """
    judgement = "不明"
    score = None
    reason = ""

    if not response_text or not isinstance(response_text, str):
        return judgement, score, reason

    lines = response_text.strip().split("\n")

    if len(lines) > 0:
        first_line = lines[0].strip()
        if first_line.rstrip(".。") in ["はい", "いいえ"]:
            judgement = first_line.rstrip(".。")

    score_pattern = r"類似度スコア:\s*([0-9.]+)"
    score_found = False
    reason_start_index_after_score = 0
    for i, line in enumerate(lines):
        match = re.search(score_pattern, line)
        if match:
            try:
                score = float(match.group(1))
                score_found = True
                reason_start_index_after_score = i + 1
                break
            except ValueError:
                score = None

    reason_lines = []
    reason_prefix = "理由:"
    reason_started = False
    start_line_for_reason_search = 1 if judgement != "不明" else 0
    if score_found:
        start_line_for_reason_search = reason_start_index_after_score

    for i in range(start_line_for_reason_search, len(lines)):
        line_stripped = lines[i].strip()
        if not reason_started and line_stripped.startswith(reason_prefix):
            reason_lines.append(line_stripped[len(reason_prefix) :].strip())
            reason_started = True
        elif reason_started:
            reason_lines.append(line_stripped)
        elif (
            not reason_started
            and i == start_line_for_reason_search
            and not score_found
            and judgement != "不明"
            and not lines[i].strip().startswith("類似度スコア")
        ):
            if i == 1 and len(lines) > 1:  # 判定がlines[0]にあるので、lines[1]からが理由
                reason_lines.append(line_stripped)
                reason_started = True

    reason = "\n".join(reason_lines).strip()

    if judgement == "不明":
        if "はい" in response_text and "いいえ" not in response_text:
            judgement = "はい"
        elif "いいえ" in response_text and "はい" not in response_text:
            judgement = "いいえ"

    return judgement, score, reason


async def call_openai_api_async(prompt_text, client_session):
    """
    OpenAI APIを非同期で呼び出し、応答を取得する。
    """
    if not API_KEY:
        return "APIキー未設定エラー"
    try:
        response = await client_session.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """あなたは2つの書誌情報が実質的に同一の文献を指すかどうかを判断する専門家です。
まず、2つの書誌情報が同一の文献と思われる場合は「はい」、そうでない場合は「いいえ」で明確に回答してください。
次に、その判断の確信度を示す類似度スコアを0.0（全く異なる）から1.0（完全に同一）の範囲で提示してください。
最後に、必要であれば判断の根拠となる簡潔な理由を続けてください。

回答フォーマット例1 (同一の場合):
はい
類似度スコア: 0.95
理由: タイトル、著者、出版年が完全に一致しています。

回答フォーマット例2 (異なる場合):
いいえ
類似度スコア: 0.1
理由: タイトルは似ていますが、著者が異なり、巻数も違うため別の文献です。

回答フォーマット例3 (情報が少なく判断に迷う場合):
いいえ
類似度スコア: 0.5
理由: タイトルと著者は一致していますが、出版年や出版社が不明なため、完全に同一とは断定できません。類似の可能性はあります。
""",
                },
                {"role": "user", "content": prompt_text},
            ],
            timeout=REQUEST_TIMEOUT,
        )
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        else:
            # print(f"OpenAI API応答形式エラー。応答: {response}") # 詳細ログ
            return "API応答形式エラー"
    except openai.APITimeoutError as e:
        # print(f"APIタイムアウト: {e}") # 詳細ログ
        return "APIタイムアウトエラー"
    except openai.APIConnectionError as e:
        # print(f"API接続エラー: {e}") # 詳細ログ
        return "API接続エラー"
    except openai.RateLimitError as e:
        # print(f"APIレート制限エラー: {e}") # 詳細ログ
        # レート制限に達した場合、少し待ってリトライするメカニズムを呼び出し側で検討するか、
        # Semaphoreの数を調整する。ここではエラーを返す。
        await asyncio.sleep(5)  # ここで少し待機するのも手だが、Semaphore側で制御推奨
        return "APIレート制限エラー (リトライ試行後も継続する可能性あり)"
    except openai.APIStatusError as e:
        # print(f"APIステータスエラー: {e.status_code} - {e.response}") # 詳細ログ
        return f"APIステータスエラー (コード: {e.status_code})"
    except openai.APIError as e:  # その他の一般的なAPIエラー
        # print(f"汎用APIエラー: {e}") # 詳細ログ
        return "汎用APIエラー"
    except Exception as e:
        # print(f"API呼び出し中予期せぬエラー: {e}") # 詳細ログ
        # import traceback
        # traceback.print_exc()
        return "予期せぬAPI呼び出しエラー"


async def process_single_pair(row_data, semaphore, async_client):
    """単一ペアを非同期で処理し、結果を返すコルーチン"""
    record_id_1 = row_data["record_id_1"]
    record_id_2 = row_data["record_id_2"]
    cache_key = get_cache_key(record_id_1, record_id_2)

    bib_info_1_csv = get_record_details_for_prompt(record_id_1)
    bib_info_2_csv = get_record_details_for_prompt(record_id_2)

    # プロンプトはキャッシュの有無に関わらず生成（CSV出力用）
    prompt_text = f"""以下の2つの書誌情報が、実質的に同一の文献を指しているかどうかを判断してください。
可能な限り「はい」または「いいえ」で明確に回答し、必要であれば簡潔な理由を続けてください。

書誌情報1:
{bib_info_1_csv}

書誌情報2:
{bib_info_2_csv}

これらは同一の文献ですか？
回答:"""

    if cache_key in LLM_CACHE:
        cached_data = LLM_CACHE[cache_key]
        return {
            "record_id_1": record_id_1,
            "record_id_2": record_id_2,
            "bib_info_1": bib_info_1_csv,
            "bib_info_2": bib_info_2_csv,
            "llm_prompt": cached_data.get("llm_prompt", prompt_text),  # キャッシュになければ現行プロンプト
            "llm_raw_response": cached_data.get("llm_raw_response", "キャッシュ応答なし"),
            "llm_judgement": cached_data.get("llm_judgement", "不明(キャッシュ)"),
            "llm_similarity_score": cached_data.get("llm_similarity_score"),
            "llm_reason": cached_data.get("llm_reason", "理由なし(キャッシュ)"),
            "status": "cached",
        }

    async with semaphore:
        # print(f"API Call: {record_id_1} vs {record_id_2}") # デバッグ用
        raw_response = await call_openai_api_async(prompt_text, async_client)

    judgement, score, reason = parse_llm_response(raw_response)

    # 新しい結果をキャッシュに保存
    LLM_CACHE[cache_key] = {
        "llm_prompt": prompt_text,
        "llm_raw_response": raw_response,
        "llm_judgement": judgement,
        "llm_similarity_score": score,
        "llm_reason": reason,
        "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    return {
        "record_id_1": record_id_1,
        "record_id_2": record_id_2,
        "bib_info_1": bib_info_1_csv,
        "bib_info_2": bib_info_2_csv,
        "llm_prompt": prompt_text,
        "llm_raw_response": raw_response,
        "llm_judgement": judgement,
        "llm_similarity_score": score,
        "llm_reason": reason,
        "status": "api_called",
    }


async def main():
    if not API_KEY:
        print("エラー: 環境変数 OPENAI_API_KEY が設定されていません。")
        sys.exit(1)
    print("OpenAI API キー読み込み完了。")

    load_bib_data(RECORD_YAML_PATH)
    if not BIB_DATA:
        print("書誌データロード失敗。処理終了。")
        return

    load_cache()

    if not os.path.exists(INPUT_CSV_PATH):
        print(f"エラー: 入力CSVファイルが見つかりません: {INPUT_CSV_PATH}")
        sys.exit(1)

    print(f"入力ペアファイル: {INPUT_CSV_PATH}")
    print(f"出力結果ファイル: {OUTPUT_CSV_PATH}")
    print(f"使用モデル: {OPENAI_MODEL}")
    print(f"最大同時リクエスト数: {MAX_CONCURRENT_REQUESTS}")
    print(f"キャッシュファイル: {CACHE_FILE_PATH}")
    if PROCESS_LIMIT_PAIRS is not None and PROCESS_LIMIT_PAIRS > 0:
        print(f"処理ペア数の上限: {PROCESS_LIMIT_PAIRS}")
    print(f"1チャンクあたりのタスク数: {CHUNK_SIZE}")

    all_rows_from_csv = []
    try:
        with open(INPUT_CSV_PATH, "r", newline="", encoding="utf-8") as infile:
            reader = csv.DictReader(infile)
            if (
                not reader.fieldnames
                or "record_id_1" not in reader.fieldnames
                or "record_id_2" not in reader.fieldnames
            ):
                print(f"エラー: 入力CSVヘッダー不正 ({INPUT_CSV_PATH})")
                sys.exit(1)
            all_rows_from_csv = list(reader)
    except FileNotFoundError:
        print(f"エラー: 入力ファイル {INPUT_CSV_PATH} が見つかりません。")
        sys.exit(1)
    except Exception as e:
        print(f"入力ファイル {INPUT_CSV_PATH} 読み込みエラー: {e}")
        sys.exit(1)

    total_pairs_in_csv = len(all_rows_from_csv)
    pairs_to_process = all_rows_from_csv
    if PROCESS_LIMIT_PAIRS is not None and PROCESS_LIMIT_PAIRS > 0 and total_pairs_in_csv > PROCESS_LIMIT_PAIRS:
        print(f"CSVに {total_pairs_in_csv} ペアありますが、最初の {PROCESS_LIMIT_PAIRS} ペアのみ処理します。")
        pairs_to_process = all_rows_from_csv[:PROCESS_LIMIT_PAIRS]
    else:
        print(f"{total_pairs_in_csv} ペアを処理対象とします。")

    async_client = openai.AsyncOpenAI(api_key=API_KEY)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    tasks = []
    for row_data in pairs_to_process:
        tasks.append(process_single_pair(row_data, semaphore, async_client))

    results = []
    start_time = time.time()
    processed_count = 0  # 正常に処理できたペアの総数
    cached_count = 0
    api_called_count = 0
    total_tasks_to_process = len(tasks)

    actual_chunk_size = CHUNK_SIZE
    if actual_chunk_size <= 0:  # 念のため実際のチャンクサイズが正であることを保証
        actual_chunk_size = 50 if total_tasks_to_process > 50 else total_tasks_to_process
        if actual_chunk_size == 0 and total_tasks_to_process > 0:
            actual_chunk_size = 1  # 0除算を避ける

    num_chunks = (total_tasks_to_process + actual_chunk_size - 1) // actual_chunk_size if actual_chunk_size > 0 else 1
    if total_tasks_to_process == 0:
        num_chunks = 0  # タスクがない場合はチャンクも0

    try:
        if not tasks:
            print("処理対象のタスクがありません。")
        else:
            for i in range(0, total_tasks_to_process, actual_chunk_size):
                task_chunk = tasks[i : i + actual_chunk_size]

                current_chunk_num = (i // actual_chunk_size) + 1
                print(
                    f"\n処理中チャンク: {current_chunk_num} / {num_chunks} (タスク {i+1}～{min(i+actual_chunk_size, total_tasks_to_process)} / {total_tasks_to_process})"
                )

                chunk_results_raw = await asyncio.gather(*task_chunk, return_exceptions=True)

                for res_idx, res in enumerate(chunk_results_raw):
                    original_task_index = i + res_idx
                    if isinstance(res, Exception):
                        print(f"  エラー (タスクインデックス {original_task_index + 1}): {res}")
                    else:
                        results.append(res)
                        if res.get("status") == "cached":
                            cached_count += 1
                        elif res.get("status") == "api_called":
                            api_called_count += 1

                processed_count = len(results)
                print(f"  チャンク完了。現在までの処理済みペア総数: {processed_count} / {total_tasks_to_process}")
                print(f"  統計: キャッシュ利用 = {cached_count}, API呼び出し = {api_called_count}")

                if current_chunk_num % 2 == 0 or current_chunk_num == num_chunks:
                    if total_tasks_to_process > 0:  # タスクがある場合のみ保存
                        save_cache()
                        print(f"  中間キャッシュを {CACHE_FILE_PATH} に保存しました。")

        processed_count = len(results)

    except Exception as e:
        print(f"非同期処理のメインループでエラー: {e}")
        import traceback

        traceback.print_exc()
    finally:
        await async_client.close()
        if total_tasks_to_process > 0:  # タスクがあった場合のみ最終保存
            save_cache()
            print("最終キャッシュを保存しました。")
        end_time = time.time()
        print(f"\n全処理の試行完了。処理時間: {end_time - start_time:.2f} 秒")

    print(f"最終処理結果: 全 {processed_count} ペアの処理が完了しました。")
    print(f"(キャッシュ利用: {cached_count} ペア, API呼び出し: {api_called_count} ペア)")

    # 結果をCSVに書き込み
    if results:
        try:
            with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as outfile:
                output_fieldnames = [
                    "record_id_1",
                    "record_id_2",
                    "bib_info_1",
                    "bib_info_2",
                    "llm_prompt",
                    "llm_raw_response",
                    "llm_judgement",
                    "llm_similarity_score",
                    "llm_reason",
                    "status",  # status列も追加
                ]
                writer = csv.DictWriter(outfile, fieldnames=output_fieldnames)
                writer.writeheader()
                # status列を含まない古いキャッシュデータとの互換性のため、getでデフォルト値設定
                for r in results:
                    r.setdefault("status", "unknown")
                writer.writerows(results)
            print(f"結果は {OUTPUT_CSV_PATH} に保存されました。")
        except Exception as e:
            print(f"CSVファイルへの書き込み中にエラー: {e}")
    else:
        print("処理結果が空のため、CSVファイルは生成されませんでした。")


if __name__ == "__main__":
    asyncio.run(main())
