u"""タイタニック号乗客の生存予想モデル."""

# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.ensemble import RandomForestClassifier

# read train data (csv file)
train_df = pd.read_csv("data/train.csv").replace("male", 0).replace("female", 1)

# 欠損値の処理
# 中央値を挿入
train_df["Age"].fillna(train_df.Age.median(), inplace=True)

# データ整形
# 兄弟の数(SibSp)と両親・子供の数(Parch)と自分の数で新しい属性を作成
train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"] + 1
# 不要なカラム（列）を削除
train_df2 = train_df.drop(["Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1)

# 学習データ準備
train_data = train_df2.values
train_x = train_data[:, 2:]  # Pclass以降の変数
train_y = train_data[:, 1]  # 正解データ

# ランダムフォレストの設定
forest = RandomForestClassifier(n_estimators=100)
# 学習モデルの生成
forest = forest.fit(train_x, train_y)

# テスト
test_df = pd.read_csv("data/test.csv").replace("male", 0).replace("female", 1)
# 欠損値の補完
test_df["Age"].fillna(train_df.Age.median(), inplace=True)
test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1
test_df2 = test_df.drop(["Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1)

# 予測データとアウトプット
test_data = test_df2.values
test_x = test_data[:, 1:]
test_y = forest.predict(test_x)

# アウトプットの書き出し用データ作成
zip_data = zip(test_data[:, 0].astype(int), test_y.astype(int))
predict_data = list(zip_data)

# アウトプットの書き出し
with open("result/predict_result_random_forest.csv", "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["PassengerId", "Survived"])
    for pid, survived in zip(test_data[:, 0].astype(int), test_y.astype(int)):
        writer.writerow([pid, survived])


# 精度の算出
answer_df = pd.read_csv("data/gender_submission.csv")
answer_data = answer_df.values
answer_vec = answer_data[:, 1]

# answer_vec（正解データ）とtest_y（予測データ）を比較
count = 0
for i in range(0, len(answer_vec)):
    if answer_vec[i] == test_y[i]:
        count += 1
print("accuracy: " + str(count / len(answer_vec)))
