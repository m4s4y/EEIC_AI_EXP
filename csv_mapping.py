import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSVファイルの読み込み
csv_file = 'data/picture/ROHAN4600_0001-0400_csv/ROHAN4600_0001_points.csv'  # CSVファイルのパス
data = pd.read_csv(csv_file, header=None)  # ヘッダーがない場合、header=Noneで読み込む

# 画像のサイズを設定（適当なサイズに設定）
width, height = 800, 600
image = 255 * np.ones(shape=[height, width, 3], dtype=np.uint8)  # 白い画像を作成

# CSVのデータをマッピング
for index, row in data.iterrows():
    x = int(row[0])  # CSVの1列目（x座標）
    y = int(row[1])  # CSVの2列目（y座標）
    value = row[2]   # 3列目の数値（必要に応じて使用）

    # 点を描画（青色の点）
    cv2.circle(image, (x, y), 5, (255, 0, 0), -1)  # (x, y)に半径5の青い点を描画

# OpenCVではなく、matplotlibを使って画像を表示
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # BGRからRGBに変換
plt.axis('off')  # 軸を表示しない
plt.show()