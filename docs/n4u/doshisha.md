Triwave | [Home](/readme.md) > Crowd4U 2.0 > 原田先生の授業資料ページ対応
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


Crowd4U 2.0 原田先生の授業資料ページ
===================================

Lcrowd プロジェクトで係わりのある [原田先生の授業資料ページ](http://www.slis.doshisha.ac.jp/~ushi/) では、資料にアクセスする際に Crowd4U 2.0 上の書誌同定タスクを実行することになっています

2024年8月現在の仕様によれば URL のクエリパラメータに `?callback=https://example.com` とすることで、タスク実行後の遷移先 (Callback URL) の変更が可能です

以下のような遷移をします

```
https://lcrowd.folklore.place?target=doshisha&callback=https://example.com
↓
https://next.crowd4u.org/runs/wf/<workflow_id>/run?target=doshisha&callback=hogehoge
↓
(タスク実行)
↓
<Callback URL>
```



冗長化目的のリダイレクト
-----------------------

前段の `https://lcrowd.folklore.place`  は、リダイレクト用 URL であり、原田先生が以下のよう理由で授業用資料ページの URL を変更してもらう必要がないように噛ませてあります

- 何らかの変更により `workflow_id` を変更せざるを得ない場合

- Crowd4U 2.0 が正しく機能せず `callback` に直接遷移する必要が出てきた場合


デプロイは [k1z3/lcrowd-redirect](https://github.com/k1z3/lcrowd-redirect) で行われており、詳しいドキュメントもそちらに添付してあるので `workflow_id` を変更したい場合や `callback` にパススルーしたい場合は、こちらにアクセスしてください

なお、プライベートリポジトリであるので、編集の必要がある場合は、以下から「書誌同定タスクのリダイレクトの設定変更が必要になった」との旨を小泉に伝えてください。権限を差し上げます

- Slack の `2022b-takahiro-koizumi` チャンネルにアクセスし、そこに書かれているメールアドレスから連絡する

- 個人的に設置しているホームページの [お問い合わせ](https://www.folklore.place/contact) から連絡する

これらは暫定的な対応であり Crowd4U 2.0 システムが安定し不要になった場合や、リダイレクトページをこの URL から変更したい場合は、この URL を経由しなくても構いません (むしろ歓迎です)



授業資料用タスクの呼び出し
---------------------------

念のため怪しいサイトでないことの旨を通知するために、サイトにお知らせを表示するようにしています (授業開始時に原田先生から学生へ伝えているようではありますが)

お知らせを表示するには URL クエリパラメータに `?target=doshisha` と付与してください

