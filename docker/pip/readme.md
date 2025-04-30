Docker Image for tensorflow env.
================================

GPU対応tensorflow環境構築のためのDocker Imageをビルドする (実行するホスト環境はUbuntu推奨)


Docker環境の整備
---------------

Docker環境がない場合は、以下を参考に環境を作る

- Ubuntu (apt経由)
    https://www.folklore.place/env/ubuntu/docker


Nvidia環境の構築
----------------

GPUを利用する場合は、ホスト環境側にもNvidia環境が必要となるため、以下を参考にNvidia環境を整える

https://www.folklore.place/techblog/2024/06/30


ビルド
----

1. 以下のコマンドでビルドする

    ```bash
    docker compose build --build-arg UID="$(id -u)" --build-arg GID="$(id -g)" \
    && CMD="pip freeze > Lab2301-triwave/docker/pip/requirements.txt " docker compose up
    ```


実行
----

プロジェクトのトップディレクトリをカレントディレクトリにしていることを前提に話を進める

1. デフォルトでは、プロジェクトのトップディレクトリにある `start_project.py` を実行するようになっている

    ```bash
    docker compose up
    ```

1. 任意のファイルにしたい場合は `CMD` 環境変数に実行したいコマンドを設定する

    ```bash
    CMD="python Lab2301-triwave/exec/start_project.py" docker compose up
    ```

1. デタッチ(バックグラウンド)モードで起動する場合は `-d` オプションをつける

    ```bash
    docker compose up -d
    ```


停止
----

プロジェクトのトップディレクトリをカレントディレクトリにしていることを前提に話を進める

1. 以下のコマンドで停止する

    ```bash
    docker compose down
    ```

cupyのエラーについて
--------------------

1. cupyを含めてビルドするにあたっては、Dockerコンテナのnvidiaドライバに依存するため、以下のように書き換える必要がある

    ``` python
    nvcc -V
    ```

1. 以下のように `base.txt` を書き換える必要があるかもしれない

    ``` diff
    - cupy-cuda11x
    + cupy-cuda12x  # 適したバージョンを導入する
    ```
