# train_model.py
# --------------------------------------------------
# 簡易的な皮膚画像分類AIモデルを作成して保存するサンプル
# （デモ用なのでランダム画像で学習。後で本物のデータに差し替え可）
# --------------------------------------------------

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

# モデル保存先
os.makedirs("model", exist_ok=True)
MODEL_PATH = "model/skin_model.h5"

# -------------------------------
# ダミーデータを生成（デモ用）
# -------------------------------
# 画像100枚分のランダムデータを作る（64x64ピクセル、RGB）
X = np.random.rand(100, 64, 64, 3)
# クラスラベル（0または1）をランダムに割り当て
y = np.random.randint(0, 2, size=(100,))

# -------------------------------
# シンプルなCNNモデル
# -------------------------------
model = models.Sequential([
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(64,64,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')   # 2分類（例：正常/異常）
])

# コンパイル（学習の設定）
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 学習（デモ用に10エポック）
model.fit(X, y, epochs=10, verbose=1)

# -------------------------------
# モデルを保存
# -------------------------------
model.save(MODEL_PATH)
print(f"✅ モデルを保存しました: {MODEL_PATH}")
