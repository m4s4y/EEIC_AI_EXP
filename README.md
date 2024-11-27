# EEIC_AI_EXP

## 実行方法
- データセットを https://zunko.jp/multimodal_dev/login.php からダウンロード
- mp4ファイルの身のフォルダに対して、extract_lip.py でフレームごとに切り出し
- create_data_lists.py で訓練データとテストデータを仕分け
- frame_partition_anno.py でシーン分割したアノテーションファイル
- frame_divide_to_scene.py で画像ファイルについてシーン分け
- 適切に階層構造を作り、

## test.py main.py について
- 適宜改良を加えていったファイルになります。
- 後半については実行が可能な形式になってからの工夫であり、意図とその一個前の反省点を記しています。
- 最終版は test8.py になります。
