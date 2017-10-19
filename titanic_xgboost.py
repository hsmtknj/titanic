# coding: utf-8
"""タイタニック号乗客の生存予想モデル."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import xgboost as xgb


def convert_data(df_input):
    '''
    convert features of data to create new features.
        Input  : (pandas.core.frame.DataFrame) <- 用はpandasのデータ
        Output : (pandas.core.frame.DataFrame) <- データ整形後の新しい特徴が追加されたデータ
    '''

    # 欠損値の処理(中央値を挿入)
    df_input["Age"].fillna(df_input.Age.median(), inplace=True)

    # 兄弟の数(SibSp)と両親・子供の数(Parch)と自分の数で新しい特徴を作成
    df_input["FamilySize"] = df_input["SibSp"] + df_input["Parch"] + 1

    # 不要なカラム（列）を削除
    df_output = df_input.drop(["Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1)

    return(df_output)


def split_data(mat_input):
    '''
    split data into Features data and Supervised data
        Input  : (numpy型の行列)
        OUtput : (Features data(numpy), Supervised data(numpy))
    '''

    x_train = mat_input[:, 2:]  # features data
    y_train = mat_input[:, 1]  # supervised data

    return(x_train, y_train)


def main():
    '''main function.'''

    # read train and supervised data (csv file)
    df_train = pd.read_csv("data/train.csv").replace("male", 0).replace("female", 1)
    df_test = pd.read_csv("data/test.csv").replace("male", 0).replace("female", 1)

    # data analysis -----------------------------------------------------------
    # percentage of survived or passed away
    # print (df_train.head())  # 先頭から5行分表示
    # print(df_train.Survived.value_counts())
    # print(df_train.Survived.value_counts(normalize=True))

    # percentage of survived or passed away in terms of Sex
    # print("--- male ---")
    # print(df_train[df_train.Sex == 0].Survived.value_counts())
    # print(df_train[df_train.Sex == 0].Survived.value_counts(normalize=True))
    # print("--- woman ---")
    # print(df_train[df_train.Sex == 1].Survived.value_counts())
    # print(df_train[df_train.Sex == 1].Survived.value_counts(normalize=True))
    # -------------------------------------------------------------------------

    # 学習データ整形
    df_train2 = convert_data(df_train)
    # 学習データ準備
    (x_train, y_train) = split_data(df_train2.values)

    # xgboostによる学習モデルの生成
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(x_train, y_train)

    # テストデータ整形
    df_test2 = convert_data(df_test)
    # テストデータ準備・予想
    mat_test = df_test2.values
    x_test = mat_test[:, 1:]
    y_test = xgb_model.predict(x_test)

    # アウトプットの書き出し用データ作成
    zip_data = zip(mat_test[:, 0].astype(int), y_test.astype(int))
    mat_predict = list(zip_data)

    # アウトプットの書き出し
    with open("result/predict_result.csv", "w") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(["PassengerId", "Survived"])
        for pid, survived in zip(mat_test[:, 0].astype(int), y_test.astype(int)):
            writer.writerow([pid, survived])

if __name__ == '__main__':
    main()
