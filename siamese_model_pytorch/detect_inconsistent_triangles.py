import csv
import yaml  # record.yml 読み込み用
import os
from collections import defaultdict
from itertools import combinations  # 三角形列挙の候補

# --- グローバル設定 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_RESULTS_FILENAME = "llm_evaluation_results_async.csv"  # LLM評価結果ファイル
LLM_RESULTS_PATH = os.path.join(BASE_DIR, LLM_RESULTS_FILENAME)

PROJECT_ROOT_ASSUMED = os.path.abspath(os.path.join(BASE_DIR, ".."))
BENCHMARK_DIR_RELATIVE_TO_PROJECT_ROOT = "benchmark/bib_japan_20241024/1k"
RECORD_YAML_FILENAME = "record.yml"
RECORD_YAML_PATH = os.path.join(PROJECT_ROOT_ASSUMED, BENCHMARK_DIR_RELATIVE_TO_PROJECT_ROOT, RECORD_YAML_FILENAME)

OUTPUT_DIR = BASE_DIR  # 出力先も同じディレクトリ
INCONSISTENT_TRIANGLES_FILENAME = "inconsistent_triangles.csv"
INCONSISTENT_PAIRS_FILENAME = "review_candidate_pairs.csv"

M_TARGET_PAIRS = 300
NUM_TRIANGLES_TO_SELECT = M_TARGET_PAIRS // 3


def load_llm_evaluations(filepath):
    """LLMの評価結果CSVを読み込み、ペアとその類似度スコアを辞書で返す"""
    pair_scores = {}
    if not os.path.exists(filepath):
        print(f"エラー: LLM評価結果ファイルが見つかりません: {filepath}")
        return pair_scores

    try:
        with open(filepath, "r", newline="", encoding="utf-8") as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                try:
                    id1 = row["record_id_1"]
                    id2 = row["record_id_2"]
                    score_str = row.get("llm_similarity_score", "0.0")  # スコアがない場合は0と見なす（要検討）

                    if not score_str or score_str.lower() == "none" or score_str == "":
                        score = 0.0  # または None を許容し、後でフィルタリング
                    else:
                        score = float(score_str)

                    # キーの順序を正規化
                    key = tuple(sorted((id1, id2)))
                    pair_scores[key] = score
                except ValueError as e:
                    print(f"警告: スコアの数値変換に失敗しました: {score_str} (行: {row}) - エラー: {e}")
                    continue  # この行はスキップ
                except KeyError as e:
                    print(f"警告: CSVに必要なキーが見つかりません: {e} (行: {row})")
                    continue

    except Exception as e:
        print(f"LLM評価結果ファイル ({filepath}) の読み込み中にエラー: {e}")

    print(f"{len(pair_scores)} 件のペア評価を {filepath} からロードしました。")
    return pair_scores


def load_record_clusters(yaml_path):
    """record.yml を読み込み、record_id と cluster_id の対応辞書を返す"""
    record_to_cluster = {}
    if not os.path.exists(yaml_path):
        print(f"エラー: 書誌データファイルが見つかりません: {yaml_path}")
        return record_to_cluster

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            all_data = yaml.safe_load(f)

        if isinstance(all_data, dict):
            possible_records_dict = all_data
            # 'records' キーの下にクラスタごとのリストがある場合 (元々の想定に近い構造)
            if "records" in all_data and isinstance(all_data["records"], dict):
                possible_records_dict = all_data["records"]

            # possible_records_dict が {cluster_id: [record_list]} のような形式であることを期待
            # または、ユーザーが提供した抜粋のように、ルート直下のキーがクラスタIDで、
            # その値がレコードリストであるケースもカバーする
            if isinstance(possible_records_dict, dict):
                for key, value_list in possible_records_dict.items():
                    # version, type などのメタ情報はスキップ
                    if key in ["version", "type", "id", "summary", "inf_attr"]:
                        continue

                    if isinstance(value_list, list):
                        for record in value_list:
                            if isinstance(record, dict) and "id" in record and "cluster_id" in record:
                                record_id_str = str(record["id"])
                                cluster_id_str = str(record["cluster_id"])
                                record_to_cluster[record_id_str] = cluster_id_str
                            # ユーザー提供の抜粋形式で cluster_id が record のトップレベルにない場合、
                            # 親キー(key)がクラスタIDとして機能する可能性がある。
                            # ただし、ユーザー提供の抜粋には record 内に cluster_id があったため、
                            # ここでは record 内の cluster_id を優先。
                            # 親キーを cluster_id とみなす場合のフォールバックは慎重に検討が必要。
                            # 今回は record.get("cluster_id") で明示的に取得する。

        if not record_to_cluster:
            print(f"警告: {yaml_path} からクラスタIDを抽出できませんでした。YAML構造を確認してください。")
            print(f"  (想定: record.yml の中に 'id' と 'cluster_id' を持つレコードエントリがある)")

    except yaml.YAMLError as e:
        print(f"エラー: 書誌データファイル ({yaml_path}) のYAML形式が正しくありません: {e}")
    except Exception as e:
        print(f"エラー: 書誌データファイル ({yaml_path}) の読み込み中に予期せぬエラーが発生しました: {e}")

    print(f"{len(record_to_cluster)} 件のレコードとクラスタIDの対応を {yaml_path} からロードしました。")
    return record_to_cluster


def main():
    print("矛盾検出処理を開始します...")

    pair_scores = load_llm_evaluations(LLM_RESULTS_PATH)
    record_to_cluster = load_record_clusters(RECORD_YAML_PATH)

    if not pair_scores:
        print("LLM評価スコアがロードできなかったため、処理を終了します。")
        return

    all_record_ids = set()
    for id1, id2 in pair_scores.keys():
        all_record_ids.add(id1)
        all_record_ids.add(id2)

    print(f"評価済みペアに含まれるユニークなレコード数: {len(all_record_ids)}")

    # グラフの隣接リスト表現 (id -> set of neighbors)
    adj = defaultdict(set)
    # スコアを保持する辞書 ( (id1,id2) -> score ) キーはソート済みタプル
    # pair_scores が既にこの形式なのでそのまま利用

    for id1_key, id2_key in pair_scores.keys():
        adj[id1_key].add(id2_key)
        adj[id2_key].add(id1_key)

    inconsistent_triangles_data = []

    print("三角形の列挙と非一貫性スコアの計算を開始します...")

    # 三角形 (i, j, k) を見つける。i < j < k となるように処理して重複を避ける。
    # ここでの i, j, k はソート済みのユニークなレコードIDリストのインデックスではなく、IDそのもの。
    sorted_nodes = sorted(list(all_record_ids))

    for i_idx, u_node in enumerate(sorted_nodes):
        # u の隣接ノード v (ただし u < v)
        # adj[u_node] には u と接続している全てのノードが入っている
        # v_candidates = [v for v in adj[u_node] if u_node < v] # この条件だとu-vエッジしか見ない

        # u-v-w-u の三角形を探す
        # u -- v
        # |  /
        # w
        # u と v の共通の隣人を w として探す

        # uの隣接ノードのリストを取得
        # リストをコピーしてイテレート中に変更しないようにする
        u_neighbors = list(adj[u_node])

        for v_node_idx in range(len(u_neighbors)):
            v_node = u_neighbors[v_node_idx]
            # u < v となるようにペアを処理 (重複を避けるための一つの方法)
            if u_node >= v_node:
                continue

            # u と v の両方に隣接するノード w を探す (w > v である必要はない、w != u, w != v)
            # v_neighbors は adj[v_node]
            # w_candidates は u_neighbors と v_neighbors の共通部分

            # vの隣接ノードのリストを取得
            v_neighbors = list(adj[v_node])

            for w_node in v_neighbors:
                # w > v という条件で処理すると (u,v,w) の組み合わせが一意になる (u<v<w)
                # かつ w が u の隣人でもあるか確認
                if u_node < v_node and v_node < w_node and w_node in adj[u_node]:
                    # これで三角形 (u_node, v_node, w_node) が見つかった
                    # 各辺のスコアを取得
                    # pair_scoresのキーはソート済みタプル
                    key_uv = tuple(sorted((u_node, v_node)))
                    key_vw = tuple(sorted((v_node, w_node)))
                    key_wu = tuple(sorted((w_node, u_node)))

                    # 3つの辺全てにスコアがあることを確認（通常はadj構築時に保証されているはずだが念のため）
                    if key_uv not in pair_scores or key_vw not in pair_scores or key_wu not in pair_scores:
                        # print(f"警告: 三角形 {(u_node, v_node, w_node)} の辺のスコアが見つかりません。スキップします。")
                        continue

                    p_uv = pair_scores[key_uv]
                    p_vw = pair_scores[key_vw]
                    p_wu = pair_scores[key_wu]

                    # 非一貫性スコア計算
                    inconsistency = p_uv * p_vw * (1 - p_wu) + p_vw * p_wu * (1 - p_uv) + p_wu * p_uv * (1 - p_vw)

                    # クラスタIDに基づく真偽判定 (オプションだが有用)
                    # True = 同じクラスタ, False = 異なるクラスタ, None = 片方または両方のID不明
                    true_uv, true_vw, true_wu = None, None, None
                    if record_to_cluster:  # クラスタ情報がある場合のみ
                        c_u = record_to_cluster.get(u_node)
                        c_v = record_to_cluster.get(v_node)
                        c_w = record_to_cluster.get(w_node)
                        if c_u and c_v:
                            true_uv = c_u == c_v
                        if c_v and c_w:
                            true_vw = c_v == c_w
                        if c_w and c_u:
                            true_wu = c_w == c_u

                    inconsistent_triangles_data.append(
                        {
                            "triangle": (u_node, v_node, w_node),
                            "inconsistency_score": inconsistency,
                            "p_uv": p_uv,
                            "p_vw": p_vw,
                            "p_wu": p_wu,
                            "true_uv": true_uv,
                            "true_vw": true_vw,
                            "true_wu": true_wu,
                            "c_u": record_to_cluster.get(u_node),
                            "c_v": record_to_cluster.get(v_node),
                            "c_w": record_to_cluster.get(w_node),
                        }
                    )

        if (i_idx + 1) % 100 == 0:  # 100ノード処理するごとに進捗表示
            print(
                f"  {i_idx+1} / {len(sorted_nodes)} ノードの処理完了... 発見済み三角形: {len(inconsistent_triangles_data)}"
            )

    print(f"\n計算が完了した三角形の数: {len(inconsistent_triangles_data)}")

    inconsistent_triangles_data.sort(key=lambda x: x["inconsistency_score"], reverse=True)

    selected_triangles = inconsistent_triangles_data[:NUM_TRIANGLES_TO_SELECT]

    print(f"\n非一貫性スコア上位 {NUM_TRIANGLES_TO_SELECT} 個の三角形を選択しました。")

    # 結果をCSVに出力
    if selected_triangles:
        output_triangles_path = os.path.join(OUTPUT_DIR, INCONSISTENT_TRIANGLES_FILENAME)
        try:
            with open(output_triangles_path, "w", newline="", encoding="utf-8") as outfile:
                fieldnames = [
                    "triangle_node1",
                    "triangle_node2",
                    "triangle_node3",
                    "inconsistency_score",
                    "p_edge12",
                    "p_edge23",
                    "p_edge31",
                    "true_edge12",
                    "true_edge23",
                    "true_edge31",  # クラスタIDベースの真偽
                    "cluster_id1",
                    "cluster_id2",
                    "cluster_id3",
                ]
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()
                for data in selected_triangles:
                    nodes = data["triangle"]
                    writer.writerow(
                        {
                            "triangle_node1": nodes[0],
                            "triangle_node2": nodes[1],
                            "triangle_node3": nodes[2],
                            "inconsistency_score": data["inconsistency_score"],
                            "p_edge12": data["p_uv"],
                            "p_edge23": data["p_vw"],
                            "p_edge31": data["p_wu"],
                            "true_edge12": data["true_uv"],
                            "true_edge23": data["true_vw"],
                            "true_edge31": data["true_wu"],
                            "cluster_id1": data["c_u"],
                            "cluster_id2": data["c_v"],
                            "cluster_id3": data["c_w"],
                        }
                    )
            print(f"選択された三角形の詳細は {output_triangles_path} に保存されました。")
        except Exception as e:
            print(f"エラー: 選択された三角形のCSVファイル書き込み中にエラーが発生しました: {e}")

        # レビュー対象ペアのリストも作成・保存
        review_pairs = set()
        for data in selected_triangles:
            n = data["triangle"]
            review_pairs.add(tuple(sorted((n[0], n[1]))))
            review_pairs.add(tuple(sorted((n[1], n[2]))))
            review_pairs.add(tuple(sorted((n[2], n[0]))))

        output_review_pairs_path = os.path.join(OUTPUT_DIR, INCONSISTENT_PAIRS_FILENAME)
        try:
            with open(output_review_pairs_path, "w", newline="", encoding="utf-8") as outfile:
                writer = csv.writer(outfile)
                writer.writerow(
                    [
                        "record_id_1",
                        "record_id_2",
                        "llm_similarity_score",
                        "cluster_id_1",
                        "cluster_id_2",
                        "same_cluster_ground_truth",
                    ]
                )
                for id1, id2 in sorted(list(review_pairs)):
                    score = pair_scores.get(tuple(sorted((id1, id2))), "N/A")
                    c1 = record_to_cluster.get(id1, "N/A")
                    c2 = record_to_cluster.get(id2, "N/A")
                    same_cluster = None
                    if c1 != "N/A" and c2 != "N/A":
                        same_cluster = c1 == c2
                    writer.writerow([id1, id2, score, c1, c2, same_cluster])
            print(
                f"レビュー対象候補のペアリストは {output_review_pairs_path} に保存されました。({len(review_pairs)}ペア)"
            )
        except Exception as e:
            print(f"エラー: レビュー対象ペアのCSVファイル書き込み中にエラーが発生しました: {e}")

    else:
        print("選択された矛盾三角形はありませんでした。")

    print("処理が完了しました。")


if __name__ == "__main__":
    main()
