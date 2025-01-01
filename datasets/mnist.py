import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist

# MNISTデータセットをロード
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# データを平坦化して、DataFrameに変換
train_data = train_images.reshape((train_images.shape[0], -1))
train_df = pd.DataFrame(train_data)
train_df['label'] = train_labels

# テストデータも同様に変換
test_data = test_images.reshape((test_images.shape[0], -1))
test_df = pd.DataFrame(test_data)
test_df['label'] = test_labels

# CSVとして保存
train_df.to_csv('mnist_train.csv', index=False)
test_df.to_csv('mnist_test.csv', index=False)
