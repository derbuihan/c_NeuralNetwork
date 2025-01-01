import pandas as pd
from sklearn import datasets

# データセットの読み込み
digits = datasets.load_digits()

# データの取得
data = digits.data
target = digits.target

# データとラベルをDataFrameに変換
df_data = pd.DataFrame(data)
df_data['label'] = target

# CSVとして保存
df_data.to_csv('digits.csv', index=False)
