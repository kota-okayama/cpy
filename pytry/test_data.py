#!/usr/bin/env python3
"""
マッチングテスト用のデータを生成するスクリプト
明確にマッチするレコードを含むYAMLファイルを作成します
"""

import yaml
import uuid
import random
import argparse
import os

# テスト用の書籍データサンプル
TEST_BOOKS = [
    {
        "title": "1984",
        "author": "George Orwell",
        "publisher": "Penguin Books",
        "pubdate": "1949",
        "variations": [
            {"title": "一九八四年", "author": "ジョージ・オーウェル", "publisher": "ペンギンブックス", "pubdate": "1949.01"},
            {"title": "Nineteen Eighty-Four", "author": "Orwell, George", "publisher": "Penguin Classics", "pubdate": "1949/06"},
            {"title": "1984 (Signet Classics)", "author": "G. Orwell", "publisher": "Signet", "pubdate": "194906"}
        ]
    },
    {
        "title": "Harry Potter and the Philosopher's Stone",
        "author": "J.K. Rowling",
        "publisher": "Bloomsbury Publishing",
        "pubdate": "1997",
        "variations": [
            {"title": "ハリー・ポッターと賢者の石", "author": "J・K・ローリング", "publisher": "静山社", "pubdate": "1997.06"},
            {"title": "Harry Potter and the Sorcerer's Stone", "author": "Rowling, J.K.", "publisher": "Scholastic", "pubdate": "1997/09"},
            {"title": "ハリーポッター 賢者の石", "author": "JKローリング", "publisher": "静山社出版", "pubdate": "199709"}
        ]
    },
    {
        "title": "The Great Gatsby",
        "author": "F. Scott Fitzgerald",
        "publisher": "Scribner",
        "pubdate": "1925",
        "variations": [
            {"title": "グレート・ギャツビー", "author": "F・スコット・フィッツジェラルド", "publisher": "岩波書店", "pubdate": "1925.04"},
            {"title": "The Great Gatsby (Penguin Modern Classics)", "author": "Fitzgerald, F. Scott", "publisher": "Penguin", "pubdate": "1925/04"},
            {"title": "華麗なるギャツビー", "author": "フィッツジェラルド", "publisher": "岩波文庫", "pubdate": "192504"}
        ]
    },
    {
        "title": "Pride and Prejudice",
        "author": "Jane Austen",
        "publisher": "T. Egerton",
        "pubdate": "1813",
        "variations": [
            {"title": "高慢と偏見", "author": "ジェーン・オースティン", "publisher": "岩波書店", "pubdate": "1813.01"},
            {"title": "Pride & Prejudice", "author": "Austen, Jane", "publisher": "Penguin Classics", "pubdate": "1813/01"},
            {"title": "プライドと偏見", "author": "J. オースティン", "publisher": "新潮文庫", "pubdate": "181301"}
        ]
    },
    {
        "title": "The Catcher in the Rye",
        "author": "J.D. Salinger",
        "publisher": "Little, Brown and Company",
        "pubdate": "1951",
        "variations": [
            {"title": "ライ麦畑でつかまえて", "author": "J・D・サリンジャー", "publisher": "白水社", "pubdate": "1951.07"},
            {"title": "The Catcher in the Rye (Modern Classics)", "author": "Salinger, J.D.", "publisher": "Penguin", "pubdate": "1951/07"},
            {"title": "キャッチャー・イン・ザ・ライ", "author": "サリンジャー", "publisher": "白水Uブックス", "pubdate": "195107"}
        ]
    }
]

def generate_test_data(books, prefix="test"):
    """テスト用のクラスター構造データを生成する"""
    data = {
        "version": "3.1",
        "type": "TARGET",
        "id": str(uuid.uuid4()),
        "summary": {
            "creation_date": "2024-06-16 01:13:51.599",
            "update_date": "2024-06-16 01:13:51.609",
            "num_of_records": 0,  # 後で更新
            "num_of_pairs": {},
            "config_match": None,
            "config_mismatch": None
        },
        "inf_attr": {
            "bib1_title": "TEXT",
            "bib1_author": "COMPLEMENT_JA",
            "bib1_publisher": "COMPLEMENT_JA",
            "bib1_pubdate": "COMPLEMENT_DATE"
        },
        "records": {}
    }
    
    # 明確に分離されたクラスターを生成
    original_books = {}
    similar_books = {}
    
    # オリジナルのクラスターを生成
    for i, book in enumerate(books):
        cluster_id = f"{prefix}_original_{i+1}"
        data["records"][cluster_id] = []
        
        # オリジナルの書籍レコードを作成
        record = {
            "id": str(uuid.uuid4()),
            "cluster_id": cluster_id,
            "data": {
                "bib1_title": book["title"],
                "bib1_author": book["author"],
                "bib1_publisher": book["publisher"],
                "bib1_pubdate": book["pubdate"]
            }
        }
        data["records"][cluster_id].append(record)
        original_books[i] = cluster_id
        
        # バリエーションを追加
        for variation in book["variations"]:
            record = {
                "id": str(uuid.uuid4()),
                "cluster_id": cluster_id,
                "data": {
                    "bib1_title": variation["title"],
                    "bib1_author": variation["author"],
                    "bib1_publisher": variation["publisher"],
                    "bib1_pubdate": variation["pubdate"]
                }
            }
            data["records"][cluster_id].append(record)
    
    # 明確に類似したクラスターを生成
    for i, book in enumerate(books):
        if i >= 3:  # 最初の3冊のみ類似クラスターを生成
            continue
            
        cluster_id = f"{prefix}_similar_{i+1}"
        data["records"][cluster_id] = []
        similar_books[i] = cluster_id
        
        # 原著の明確なバリエーション
        record = {
            "id": str(uuid.uuid4()),
            "cluster_id": cluster_id,
            "data": {
                "bib1_title": f"{book['title']} (Annotated Edition)",
                "bib1_author": book["author"],
                "bib1_publisher": f"New {book['publisher']}",
                "bib1_pubdate": f"{book['pubdate']}.revised"
            }
        }
        data["records"][cluster_id].append(record)
        
        # 別のバリエーション
        record = {
            "id": str(uuid.uuid4()),
            "cluster_id": cluster_id,
            "data": {
                "bib1_title": f"{book['title']} - Special Edition",
                "bib1_author": book["author"],
                "bib1_publisher": f"{book['publisher']} International",
                "bib1_pubdate": f"{book['pubdate']}/12"
            }
        }
        data["records"][cluster_id].append(record)
    
    # レコード数を更新
    total_records = 0
    for cluster_records in data["records"].values():
        total_records += len(cluster_records)
    data["summary"]["num_of_records"] = total_records
    
    return data, original_books, similar_books

def main():
    parser = argparse.ArgumentParser(description="マッチングテスト用のデータを生成する")
    parser.add_argument("--output", "-o", type=str, default="test_data.yml", help="出力YAMLファイルパス")
    args = parser.parse_args()
    
    # テストデータを生成
    data, original_books, similar_books = generate_test_data(TEST_BOOKS)
    
    # YAML形式で出力
    with open(args.output, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    print(f"テストデータを {args.output} に保存しました。")
    print(f"レコード総数: {data['summary']['num_of_records']}")
    print(f"クラスター数: {len(data['records'])}")
    
    print("\n期待されるマッチング:")
    for i, (original, similar) in enumerate(zip(original_books.values(), similar_books.values())):
        if i < len(similar_books):
            print(f"クラスター {original} <-> クラスター {similar}")

if __name__ == "__main__":
    main()