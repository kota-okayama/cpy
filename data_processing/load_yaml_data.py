import yaml
import os


def load_bibliographic_data(yaml_file_path):
    """
    書誌情報YAMLファイルを読み込み、レコードのリストを返します。

    各レコードは以下のキーを持つ辞書として表現されます:
    - record_id (str): レコードの一意識別子 (例: '65e694f1-dc96-4c04-bde8-3f924acfb4c6')
    - cluster_id (str): レコードが属するクラスタID
    - data (dict): 書誌データ (例: {'bib1_title': '...', 'bib1_author': '...'})
    """
    if not os.path.exists(yaml_file_path):
        print(f"Error: File not found at {yaml_file_path}")
        return None

    with open(yaml_file_path, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            return None

    records_list = []
    if data and "records" in data:
        for cluster_key, items_in_cluster in data["records"].items():
            if items_in_cluster:  # クラスタ内にアイテムが存在する場合
                for item in items_in_cluster:
                    record = {
                        "record_id": item.get("id"),
                        "cluster_id": item.get("cluster_id"),  # YAML構造に基づき、各アイテムからcluster_idを取得
                        "data": item.get("data", {}),
                    }
                    records_list.append(record)
    else:
        print("No 'records' section found in YAML or file is empty.")
        return None

    return records_list


if __name__ == "__main__":
    # YAMLファイルのパス (ユーザーの環境に合わせて変更してください)
    # 例: 'benchmark/bib_japan_20241024/1k/record.yml'
    #      'F:/lab/Lab2411-archive/benchmark/bib_japan_20241024/1k/record.yml' (WSLからWindowsパスを参照する場合)

    # WSL内の絶対パス、またはこのスクリプトからの相対パスを指定
    yaml_path = "benchmark/bib_japan_20241024/1k/record.yml"

    # WSL環境でWindowsの絶対パスを参照する場合は、パスの先頭に /mnt/ をつけ、ドライブレターを小文字にします。
    # 例: Fドライブの /lab/Lab2411-archive/... の場合
    # windows_absolute_path = 'F:/lab/Lab2411-archive/benchmark/bib_japan_20241024/1k/record.yml'
    # if os.name != 'nt': # WSL (Linux)環境かどうかを簡易的に判定
    #     yaml_path = '/' + windows_absolute_path.replace(':', '').replace('\\', '/').replace('\', '/')
    #     yaml_path = '/mnt/' + yaml_path[1].lower() + yaml_path[2:]
    # else: # Windows環境の場合 (直接実行することは少ないが念のため)
    #     yaml_path = windows_absolute_path

    print(f"Attempting to load data from: {yaml_path}")

    bibliographic_records = load_bibliographic_data(yaml_path)

    if bibliographic_records:
        print(f"Successfully loaded {len(bibliographic_records)} records.")

        # 最初の5件のレコード情報を表示
        print("\nFirst 5 records:")
        for i, record in enumerate(bibliographic_records[:5]):
            print(f"--- Record {i+1} ---")
            print(f"  Record ID: {record.get('record_id')}")
            print(f"  Cluster ID: {record.get('cluster_id')}")
            print(f"  Title: {record.get('data', {}).get('bib1_title')}")
            print(f"  Author: {record.get('data', {}).get('bib1_author')}")
    else:
        print("Failed to load records.")

    # PyYAMLがインストールされていない場合に備えてメッセージ
    try:
        import yaml
    except ImportError:
        print("\n--------------------------------------------------------------------")
        print("PyYAML library is not installed. Please install it by running:")
        print("pip install PyYAML")
        print("--------------------------------------------------------------------")
