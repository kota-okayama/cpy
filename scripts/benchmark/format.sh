#!/bin/sh

# スクリプトが配置されたディレクトリを取得
base_dir=$(dirname "$(readlink -f "$0")")

# 削除対象のファイルおよびディレクトリのパターンを指定
file_patterns="log.log suspend.yml"
dir_patterns=".cache .observer image"

# ファイルを再帰的に削除
for pattern in $file_patterns; do
    find "$base_dir" -type f -name "$pattern" -exec rm -f {} +
done

# ディレクトリを再帰的に削除
for pattern in $dir_patterns; do
    find "$base_dir" -type d -name "$pattern" -exec rm -rf {} +
done

echo "Specified files and directories have been deleted."