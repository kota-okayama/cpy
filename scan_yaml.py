import yaml
import json
import sys

def inspect_yaml(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # 基本情報
        print(f"データ型: {type(data)}")
        if isinstance(data, list):
            print(f"リスト長: {len(data)}")
            
            # 最初の要素を調べる
            if data and len(data) > 0:
                first_item = data[0]
                print(f"最初の要素の型: {type(first_item)}")
                print(f"最初の要素のキー: {list(first_item.keys()) if isinstance(first_item, dict) else 'N/A'}")
                
                # recordsがあるか確認
                if isinstance(first_item, dict) and 'records' in first_item:
                    print(f"最初の要素のrecords長: {len(first_item['records'])}")
                    
                    # recordsの最初の要素を調べる
                    if first_item['records'] and len(first_item['records']) > 0:
                        first_record = first_item['records'][0]
                        print(f"最初のレコードの型: {type(first_record)}")
                        print(f"最初のレコードのキー: {list(first_record.keys()) if isinstance(first_record, dict) else 'N/A'}")
                        
                        # dataフィールドがあるか確認
                        if isinstance(first_record, dict) and 'data' in first_record:
                            print(f"最初のレコードのdataキー: {list(first_record['data'].keys()) if isinstance(first_record['data'], dict) else 'N/A'}")
        
        print("\n簡易構造出力:")
        if isinstance(data, list) and len(data) > 0:
            sample_data = data[0]
            # 最初の要素の構造のみ出力
            print(json.dumps(sample_data, ensure_ascii=False, indent=2)[:500] + "...")
    
    except Exception as e:
        print(f"エラー: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用法: python inspect_yaml.py <yamlファイル>")
        sys.exit(1)
    
    inspect_yaml(sys.argv[1])