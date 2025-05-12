import openai
import os
import csv
import time
import yaml
import sys
import re  # 正規表現モジュールをインポート

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
# このスクリプト(evaluate_pairs_with_openai.py)が siamese_model_pytorch ディレクトリにあると仮定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV_FILENAME = "evaluation_candidate_pairs.csv"  # extract_llm_pairs.py の出力
INPUT_CSV_PATH = os.path.join(BASE_DIR, INPUT_CSV_FILENAME)

# record.yml のパスはプロジェクトルートからの相対パスで指定することを想定
# 例: ../benchmark/bib_japan_20241024/1k/record.yml
# 正しいパス構造に合わせて調整してください。
# ここでは仮に、スクリプトから見て2つ上のディレクトリをプロジェクトルートとし、そこからのパスとします。
PROJECT_ROOT_ASSUMED = os.path.abspath(os.path.join(BASE_DIR, ".."))
BENCHMARK_DIR_RELATIVE_TO_PROJECT_ROOT = "benchmark/bib_japan_20241024/1k"
RECORD_YAML_FILENAME = "record.yml"
RECORD_YAML_PATH = os.path.join(PROJECT_ROOT_ASSUMED, BENCHMARK_DIR_RELATIVE_TO_PROJECT_ROOT, RECORD_YAML_FILENAME)


OUTPUT_CSV_FILENAME = "llm_evaluation_results.csv"
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, OUTPUT_CSV_FILENAME)

# OpenAI API設定
OPENAI_MODEL = "gpt-4o-mini"  # 使用するモデル
REQUEST_TIMEOUT = 30  # APIリクエストのタイムアウト（秒）
REQUEST_INTERVAL = 0.1  # APIリクエスト間の待機時間（秒）- レート制限対策

# グローバル変数として書誌データを保持
BIB_DATA = {}


def load_bib_data(yaml_path):
    """
    record.yml から書誌データをロードし、グローバル変数 BIB_DATA に格納する。
    キーはレコードID、値は書誌情報の辞書。
    """
    global BIB_DATA
    BIB_DATA = {}  # 初期化して再ロードに対応
    if not os.path.exists(yaml_path):
        print(f"エラー: 書誌データファイルが見つかりません: {yaml_path}")
        print("RECORD_YAML_PATH の設定を確認してください。")
        sys.exit(1)
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            all_data = yaml.safe_load(f)

        # record.yml の実際の構造に対応する処理
        # ルートが辞書型で、その値がレコードのリスト、またはさらにネストしたデータ構造を持つと想定
        if isinstance(all_data, dict):
            # version, type, id, summary, inf_attr, records といったキーを持つ想定
            if "records" in all_data and isinstance(all_data["records"], dict):
                # all_data['records'] が {cluster_id: [record1, record2]} という形式
                for cluster_id, record_list_in_cluster in all_data["records"].items():
                    if isinstance(record_list_in_cluster, list):
                        for record in record_list_in_cluster:
                            if isinstance(record, dict) and "id" in record and "data" in record:
                                BIB_DATA[str(record["id"])] = record[
                                    "data"
                                ]  # 'data'フィールドに実際の書誌情報があると想定
                            else:
                                print(f"警告: records -> cluster_id 内のレコード形式が不正です: {record}")
                    else:
                        print(f"警告: cluster_id {cluster_id} の値がリストではありません: {record_list_in_cluster}")
            else:
                # もし 'records' キーがない、またはその下の構造が異なる場合、
                # ルートレベルの辞書の各エントリがクラスタIDで、その値がレコードリストかもしれない
                # (ユーザー提供の record.yml の抜粋から推測される構造)
                possible_records_dict = all_data  # all_data 自体が {cluster_id: [record_list]} かもしれない
                if "records" in all_data:  # 'records' キーがあるならそちらを優先
                    possible_records_dict = all_data["records"]

                if isinstance(possible_records_dict, dict):
                    for key, value in possible_records_dict.items():
                        # version, type などのメタ情報キーはスキップ
                        if key in ["version", "type", "id", "summary", "inf_attr"]:
                            continue
                        # 値がレコードのリストであると期待 (key が cluster_id に相当)
                        if isinstance(value, list):
                            for record in value:
                                if isinstance(record, dict) and "id" in record and "data" in record:
                                    BIB_DATA[str(record["id"])] = record["data"]
                                elif (
                                    isinstance(record, dict)
                                    and "id" in record
                                    and not "data" in record
                                    and len(record.keys()) > 2
                                ):  # id と cluster_id 以外のキーがあればそれをdataとみなす
                                    # 'data' キーがなく、直接 bib1_title などが含まれる場合へのフォールバック
                                    record_data_candidate = {
                                        k: v for k, v in record.items() if k not in ["id", "cluster_id"]
                                    }
                                    if record_data_candidate:
                                        BIB_DATA[str(record["id"])] = record_data_candidate
                                        # print(f"DEBUG: 'data'キーなし、直接的な書誌情報を抽出: {record['id']}")
                                    else:
                                        print(f"警告: cluster {key} 内のレコードに書誌情報が見つかりません: {record}")
                                elif isinstance(record, dict) and "id" in record:  # id しかない場合など
                                    print(
                                        f"警告: cluster {key} 内のレコードに 'data' キーがなく、他の書誌情報も見つかりません: {record}"
                                    )
                                else:
                                    print(f"警告: cluster {key} 内のレコード形式が不正です: {record}")
                        else:
                            # print(f"デバッグ: キー {key} の値はレコードリストではありません: {type(value)}")
                            pass  # 他のメタ情報かもしれないので、ここでは警告しない
                else:
                    print(f"エラー: {yaml_path} の 'records' フィールドが期待する辞書形式ではありません。")
                    # sys.exit(1) # 即時終了させず、他の可能性も試すためにコメントアウト

        # 従来のリストベースのYAMLも念のためチェック（ただし今回のケースでは該当しない可能性が高い）
        elif isinstance(all_data, list):
            print("情報: YAMLファイルがレコードの直接的なリスト形式であると解釈しようとしています。")
            for record_item in all_data:
                if isinstance(record_item, dict) and "id" in record_item and "data" in record_item:
                    BIB_DATA[str(record_item["id"])] = record_item["data"]
                elif isinstance(record_item, dict) and "id" in record_item:  # 'data' がない場合
                    record_data_candidate = {k: v for k, v in record_item.items() if k not in ["id", "cluster_id"]}
                    if record_data_candidate:
                        BIB_DATA[str(record_item["id"])] = record_data_candidate
                    else:
                        print(f"警告: リスト内のレコードに書誌情報が見つかりません: {record_item}")
                else:
                    print(f"警告: リスト内のアイテムの形式が不正です: {record_item}")
        else:
            print(f"エラー: {yaml_path} の書誌データ形式を解釈できません。")
            sys.exit(1)

        if not BIB_DATA:
            print(
                f"エラー: {yaml_path} から書誌データをロードできませんでした、またはデータが空です。YAML構造を確認してください。"
            )
            sys.exit(1)

        print(f"{len(BIB_DATA)} 件の書誌データを {yaml_path} からロードしました。")

    except yaml.YAMLError as e:
        print(f"エラー: 書誌データファイル ({yaml_path}) のYAML形式が正しくありません: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"エラー: 書誌データファイル ({yaml_path}) の読み込み中に予期せぬエラーが発生しました: {e}")
        sys.exit(1)


def get_record_details_for_prompt(record_id):
    """
    指定されたレコードIDの書誌情報をプロンプト用に整形して返す。
    BIB_DATA から情報を取得する。
    """
    if not BIB_DATA:
        print("エラー: 書誌データがロードされていません。load_bib_data() を先に呼び出してください。")
        return "情報取得エラー"

    bib_details = BIB_DATA.get(str(record_id))
    if not bib_details:
        raw_record = BIB_DATA.get(str(record_id) + "_raw")
        if raw_record and isinstance(raw_record.get("data"), dict):
            bib_details = raw_record["data"]
        else:
            return f"レコードID {record_id} の書誌詳細情報 (dataキー下) が見つかりません。"

    title = bib_details.get("bib1_title", "タイトル不明")
    authors_str = bib_details.get("bib1_author", "著者不明")
    publisher = bib_details.get("bib1_publisher", "出版社不明")
    pubdate = bib_details.get("bib1_pubdate", "出版日不明")

    return f"タイトル: {title}\n著者: {authors_str}\n出版社: {publisher}\n出版日: {pubdate}"


def parse_llm_response(response_text):
    """
    LLMからの応答文字列をパースし、判定、類似度スコア、理由を抽出する。

    Args:
        response_text (str): LLMの応答文字列。

    Returns:
        tuple: (judgement, score, reason)
               judgement (str): 「はい」、「いいえ」、または抽出失敗時は「不明」など。
               score (float or None): 類似度スコア。抽出失敗時は None。
               reason (str): 理由。抽出失敗時は空文字列など。
    """
    judgement = "不明"
    score = None
    reason = ""

    if not response_text or not isinstance(response_text, str):
        return judgement, score, reason

    lines = response_text.strip().split("\n")

    # 1. 判定の抽出
    if len(lines) > 0:
        first_line = lines[0].strip()
        if first_line in ["はい", "いいえ"]:
            judgement = first_line
        # 時々、判定が「はい。」「いいえ。」のように句点を含む場合も考慮
        elif first_line.rstrip(".。") in ["はい", "いいえ"]:
            judgement = first_line.rstrip(".。")
        else:
            # 1行目に判定がない場合、他の行に含まれる可能性も低いが、念のため
            # ここでは1行目のみを判定行とみなすこととする
            pass

    # 2. 類似度スコアの抽出
    score_pattern = r"類似度スコア:\s*([0-9.]+)"
    score_found = False
    for i, line in enumerate(lines):
        match = re.search(score_pattern, line)
        if match:
            try:
                score = float(match.group(1))
                score_found = True
                # スコアが見つかった行のインデックスを記録 (理由の開始点特定のため)
                reason_start_index_after_score = i + 1
                break
            except ValueError:
                print(f"警告: 類似度スコアの数値変換に失敗しました: {match.group(1)}")
                score = None  # 変換失敗時はNoneのまま

    # 3. 理由の抽出
    # 理由の開始行を特定する。基本は「類似度スコア」の次の行、または「判定」の次の行。
    # ただし、「理由:」というプレフィックスがあればそれを優先する。

    reason_lines = []
    reason_prefix = "理由:"
    reason_started = False

    start_line_for_reason_search = 0
    if judgement != "不明":  # 判定が取れていれば、判定の次の行から理由を探す
        start_line_for_reason_search = 1
    if score_found:  # スコアが取れていれば、スコアの次の行から理由を探す (より優先)
        start_line_for_reason_search = reason_start_index_after_score

    for i in range(start_line_for_reason_search, len(lines)):
        line_stripped = lines[i].strip()
        if not reason_started and line_stripped.startswith(reason_prefix):
            reason_lines.append(line_stripped[len(reason_prefix) :].strip())
            reason_started = True
        elif reason_started:
            reason_lines.append(line_stripped)  # プレフィックスなしで2行目以降の理由
        elif (
            not reason_started
            and i == start_line_for_reason_search
            and judgement != "不明"
            and not score_found
            and not lines[i].strip().startswith("類似度スコア")
        ):
            # フォールバック: 判定のみあり、スコア行がなく、3行目が理由プレフィックスなしで始まる場合
            # (例: はい\nこれは理由です)
            # このケースは、判定が1行目、理由が2行目から始まる場合を想定
            if i == 1 and len(lines) > 1:  # 判定がlines[0]にあるので、lines[1]からが理由
                reason_lines.append(line_stripped)
                reason_started = True  # ここで理由が始まったとみなす

    if not reason_lines and len(lines) > (1 if judgement != "不明" else 0) + (1 if score_found else 0):
        # 上記で見つからなかったが、判定やスコアの後にまだ行が残っている場合、
        # それらを理由とみなす (「理由:」プレフィックスがないパターン)
        # このロジックは、応答が厳密にフォーマットされていない場合に役立つ
        potential_reason_start = 0
        if judgement != "不明":
            potential_reason_start += 1
        if score_found:
            potential_reason_start += 1

        # より安全なのは、判定とスコア行を除いた残りを理由とすること
        # ただし、上のループでreason_start_index_after_scoreを使っているので、ここでの単純な加算は重複する可能性あり
        # 一旦、上記のreason_startedロジックに任せる
        pass

    reason = "\n".join(reason_lines).strip()

    # もし上記で判定が「不明」で、応答全体から「はい」か「いいえ」が見つかればそれを採用（最終手段）
    if judgement == "不明":
        if "はい" in response_text and "いいえ" not in response_text:  # 「いいえ」を含まない「はい」
            judgement = "はい"
        elif "いいえ" in response_text and "はい" not in response_text:  # 「はい」を含まない「いいえ」
            judgement = "いいえ"
        # 「はい」と「いいえ」が両方ある場合は「不明」のまま（混乱を避ける）

    return judgement, score, reason


def call_openai_api(prompt_text):
    """
    OpenAI APIを呼び出し、応答を取得する。
    """
    if not API_KEY:
        print("エラー: OpenAI APIキーがcall_openai_api関数内で未設定です。")
        return "APIキー未設定エラー"

    client = openai.OpenAI(api_key=API_KEY)
    try:
        response = client.chat.completions.create(
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
            print("OpenAI APIからの応答形式が予期したものではありません。")
            print(f"API応答: {response}")
            return "API応答形式エラー"

    except openai.APITimeoutError as e:
        print(f"OpenAI APIリクエストがタイムアウトしました: {e}")
        return "APIタイムアウトエラー"
    except openai.APIConnectionError as e:
        print(f"OpenAI APIへの接続に失敗しました: {e}")
        return "API接続エラー"
    except openai.RateLimitError as e:
        print(f"OpenAI APIのレート制限に達しました: {e}")
        # ここでリトライロジックを入れることも検討できる (例: time.sleepして再帰呼び出しなど)
        # ただし、メインループ側でREQUEST_INTERVALによる待機があるので、頻発する場合はそちらの調整も必要
        return "APIレート制限エラー"
    except openai.APIStatusError as e:
        print(f"OpenAI APIがステータスエラーを返しました (例: 4xx, 5xx): {e}")
        print(f"ステータスコード: {e.status_code}")
        print(f"エラーレスポンス: {e.response}")
        return f"APIステータスエラー (コード: {e.status_code})"
    except openai.APIError as e:  # その他の一般的なAPIエラー
        print(f"OpenAI APIエラーが発生しました: {e}")
        return "汎用APIエラー"
    except Exception as e:
        print(f"OpenAI API呼び出し中に予期せぬエラーが発生しました: {e}")
        return "予期せぬAPI呼び出しエラー"


def main():
    """
    メイン処理。
    """
    if not API_KEY:
        print("エラー: 環境変数 OPENAI_API_KEY が設定されていません。")
        sys.exit(1)

    print("OpenAI API キーが読み込まれました。")

    load_bib_data(RECORD_YAML_PATH)
    if not BIB_DATA:
        return

    if not os.path.exists(INPUT_CSV_PATH):
        print(f"エラー: 入力CSVファイルが見つかりません: {INPUT_CSV_PATH}")
        print(f"{INPUT_CSV_FILENAME} を {BASE_DIR} に配置してください。")
        sys.exit(1)

    print(f"入力ペアファイル: {INPUT_CSV_PATH}")
    print(f"出力結果ファイル: {OUTPUT_CSV_PATH}")
    print(f"使用モデル: {OPENAI_MODEL}")

    processed_pairs = 0
    # テスト用に一時的に True にして、パース関数の動作確認用ダミーデータを流す
    TEST_PARSE_LOGIC_ONLY = False

    try:
        with open(INPUT_CSV_PATH, "r", newline="", encoding="utf-8") as infile, open(
            OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8"
        ) as outfile:

            reader = csv.DictReader(infile)
            if (
                not reader.fieldnames
                or "record_id_1" not in reader.fieldnames
                or "record_id_2" not in reader.fieldnames
            ):
                print(
                    f"エラー: 入力CSVファイル {INPUT_CSV_PATH} に必要なヘッダー 'record_id_1' または 'record_id_2' が見つかりません。"
                )
                sys.exit(1)

            # 新しい列を追加してヘッダーを定義
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
            ]
            writer = csv.DictWriter(outfile, fieldnames=output_fieldnames)
            writer.writeheader()

            print("\nペアの処理を開始します...")

            if TEST_PARSE_LOGIC_ONLY:
                print("*** parse_llm_response 関数のテストモード ***")
                dummy_responses = [
                    "はい\n類似度スコア: 0.95\n理由: 完全に一致します。",
                    "いいえ\n類似度スコア: 0.1\n理由: タイトルが異なります。\n詳細な差分はこちらです。",
                    "はい\n類似度スコア: 1.0",  # 理由なし
                    "いいえ",  # スコア・理由なし
                    "類似度スコア: 0.7",  # 判定なし
                    "理由: これはテストです。",  # 判定・スコアなし
                    "はい。\n類似度スコア:0.88\n理由: ほぼ一致。",  # 句点とスペースの揺れ
                    "データなし",  # 全く異なる形式
                    "はい\n類似度スコア: 不明\n理由: スコアが出せません。",  # スコアが数値でない
                    "はい\n類似度スコア: 0.6\n理由: これは一行目です。\nこれは二行目です。\nそして三行目。",
                ]
                for i, dummy_res in enumerate(dummy_responses):
                    print(f"\n--- ダミー応答 {i+1} ---")
                    print(f"入力: '''{dummy_res}''' ")
                    judgement, score, reason = parse_llm_response(dummy_res)
                    print(f"判定: [{judgement}]")
                    print(f"スコア: [{score}] (型: {type(score)})")
                    print(f"理由:\n'''{reason}''' ")
                print("\n*** parse_llm_response 関数のテストモード終了 ***")
                # main関数から抜けるので、以降の処理は実行されない
            else:
                # TEST_PARSE_LOGIC_ONLY が False の場合 (通常のAPI呼び出し処理)
                for i, row in enumerate(reader):
                    if i >= 10:  # 100ペアに制限 (TEST_PARSE_LOGIC_ONLYがFalseの時のみこのループに入る)
                        break

                    record_id_1 = row["record_id_1"]
                    record_id_2 = row["record_id_2"]

                    bib_info_1 = get_record_details_for_prompt(record_id_1)
                    bib_info_2 = get_record_details_for_prompt(record_id_2)

                    prompt = f"""以下の2つの書誌情報が、実質的に同一の文献を指しているかどうかを判断してください。
可能な限り「はい」または「いいえ」で明確に回答し、必要であれば簡潔な理由を続けてください。

書誌情報1:
{bib_info_1}

書誌情報2:
{bib_info_2}

これらは同一の文献ですか？
回答:"""

                    llm_raw_response = call_openai_api(prompt)

                    judgement, score, reason = parse_llm_response(llm_raw_response)

                    writer.writerow(
                        {
                            "record_id_1": record_id_1,
                            "record_id_2": record_id_2,
                            "bib_info_1": bib_info_1,
                            "bib_info_2": bib_info_2,
                            "llm_prompt": prompt,
                            "llm_raw_response": llm_raw_response,
                            "llm_judgement": judgement,
                            "llm_similarity_score": score,
                            "llm_reason": reason,
                        }
                    )

                    processed_pairs += 1
                    if processed_pairs % 10 == 0:
                        print(f"{processed_pairs} ペア処理完了...")

                    time.sleep(REQUEST_INTERVAL)

                print(f"\n全 {processed_pairs} ペアの処理が完了しました。結果は {OUTPUT_CSV_PATH} に保存されました。")

    except FileNotFoundError:
        print(f"エラー: 入力ファイル {INPUT_CSV_PATH} が見つかりません。")
    except Exception as e:
        print(f"処理中にエラーが発生しました: {e}")


if __name__ == "__main__":
    main()
