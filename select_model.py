# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @DATE    : 5/19/2019
# @Author  : xiaotong
# @File    : select_model
# @Project : PyCharm
# @Github  ：https://github.com/isNxt
# @Describ : ...

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.plotting.register_matplotlib_converters()
plt.rcParams['figure.figsize'] = (12, 8)

# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @DATE    : ${DATE}
# @Author  : ${USER}
# @File    : ${NAME}
# @Project : ${PRODUCT_NAME}
# @Github  ：https://github.com/isNxt
# @Describ : ...

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 4.数据处理
"""
根据字段分析的结果，提取出新的特征，做成字段。
分箱形成的新字段通过pd.concat方法连接组成表格赋值给变量cut_df，pd.concat方法返回值数据类型为DataFrame。
新字段表格与原表格继续通过pd.concat方法连接组成表格赋值给new_df。
"""


def data_processing(df):

    field_cut = {
        'CRIM': [0, 10, 20, 100],
        'ZN': [-1, 5, 18, 20, 40, 80, 86, 100],
        'INDUS': [-1, 7, 15, 23, 40],
        'NOX': [0, 0.51, 0.6, 0.7, 0.8, 1],
        'RM': [0, 4, 5, 6, 7, 8, 9],
        'AGE': [0, 60, 80, 100],
        'DIS': [0, 2, 6, 14],
        'RAD': [0, 5, 10, 25],
        'TAX': [0, 200, 400, 500, 800],
        'PTRATIO': [0, 14, 20, 23],
        'B': [0, 100, 350, 450],
        'LSTAT': [0, 5, 10, 20, 40]
    }

    df = df[load_boston().feature_names].copy()
    cut_df = pd.DataFrame()
    for field in field_cut.keys():
        cut_series = pd.cut(df[field], field_cut[field], right=True)
        onehot_df = pd.get_dummies(cut_series, prefix=field)
        cut_df = pd.concat([cut_df, onehot_df], axis=1)
    new_df = pd.concat([df, cut_df], axis=1)
    return new_df


df = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
new_df = data_processing(df)
print()

print(new_df.head())
new_df.to_csv('./out/new_df.csv')
X = new_df.values
y = load_boston().target
# X = X[y != 50]
# y = y[y != 50]

# %% 3.选择最优模型
"""
回归模型中文名
sklearn库中的对应英文类

极致梯度提升回归模型
Xgboost
梯度提升回归模型
GradientBoostingRegressor
打包回归模型
BaggingRegressor
随机森林回归模型
RandomForestRegressor
自适应提升回归模型
AdaBoostRegressor
线性回归模型
LinearRegression
决策树回归模型
DecisionTreeRegressor
多层感知器回归模型
MLPRegressor
支持向量机回归模型
SVR
"""
print()
df_columns = ['Name', 'Parameters', 'train_r2', 'test_r2', 'fit_time']
df = pd.DataFrame(columns=df_columns)
row_index = 0
cv_split = ShuffleSplit(n_splits=6, train_size=0.7, test_size=0.2, random_state=168)

estimator_list = [
    LinearRegression(),
    DecisionTreeRegressor(),
    GradientBoostingRegressor(),
    MLPRegressor(solver='lbfgs'),
    AdaBoostRegressor(),
    BaggingRegressor(),
    RandomForestRegressor(),
    SVR(),
    XGBRegressor()
]
for estimator in estimator_list:
    cv_results = cross_validate(
        estimator, X, y, cv=cv_split,
        return_train_score=True,
        scoring=['r2'])

    df.loc[row_index, 'Name'] = estimator.__class__.__name__
    df.loc[row_index, 'Parameters'] = str(estimator.get_params())
    df.loc[row_index, 'train_r2'] = cv_results['train_r2'].mean()
    df.loc[row_index, 'test_r2'] = cv_results['test_r2'].mean()
    df.loc[row_index, 'fit_time'] = cv_results['fit_time'].mean()
    row_index += 1
print()

df = df.sort_values(by='test_r2', ascending=False)
df.to_csv('./out/estimators.csv')
print(df)

"""
从上图中可以看出，几个集成回归模型都在测试集上取得0.8以上的得分。
决策树回归模型和额外树回归模型在训练集上取得了满分，与测试集结果差距大，说明这2种模型容易过拟合。
从上面的运行结果可以看出，XGBRegressor模型比GradientBoostingRegressor模型略优，测试集预测结果标准差更小，花费时间也减少1/3。
"""

