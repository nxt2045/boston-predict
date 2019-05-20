# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @DATE    : 5/19/2019
# @Author  : xiaotong
# @File    : xgb_model
# @Project : PyCharm
# @Github  ：https://github.com/isNxt
# @Describ : ...


import time

import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import datetime
import sys,os
time_now = datetime.datetime.now()
time_stamp = time_now.strftime('%d-%H-%M-%S')
# f = open('./out/log.txt', 'a')
# sys.stdout = f
out_path = './out/'+time_stamp
if not os.path.exists(out_path):
    os.mkdir(out_path)
# print("\n\n\n\noutpath:\t" + out_path)
# print("fromfile:\t" + "xgb_model.py")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
plt.rcParams['figure.figsize'] = (8, 6)

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
# print(new_df.head())
new_df.to_csv('./out/new_df.csv')
X = new_df.values
y = load_boston().target
# X = X[y != 50]
# y = y[y != 50]

X_train, X_test, y_train, y_test \
    = train_test_split(X, y, test_size=0.2)

# %% 5.搜索模型最优参数
"""
sklearn.model_selection库中有GridSearchCV方法，作用是搜索模型的最优参数。
"""

xgb_model = XGBRegressor(
    learning_rate=0.1,
    n_estimators=200,
    max_depth=6,
    min_child_weight=4,
    gamma=0.4,
    subsample=0.7,
    colsample_bytree=0.7,
    nthread=4,
)
cv_split = ShuffleSplit(n_splits=6, train_size=0.7, test_size=0.2)

param_list = [
    {'max_depth': range(3, 10, 1)},
    {'min_child_weight': range(1, 6, 1)},
    {'gamma': [i / 10.0 for i in range(0, 5)]},
    {'subsample': [i / 10.0 for i in range(6, 10)]},
    {'colsample_bytree': [i / 10.0 for i in range(6, 10)]},
    {'n_estimators': [50, 100, 200, 500, 1000]},
    {'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2]}
]

bst_param = {'silent': 0, 'nthread': 4}

for param_dict in param_list:
    start = time.time()
    clf = GridSearchCV(estimator=xgb_model, param_grid=param_dict, cv=cv_split)
    clf.fit(X_train, y_train)
    # print('\nGridSearchCV process use %.2f seconds' % (time.time() - start))
    # print("Best parameters set:", clf.best_params_)
    bst_param.update(clf.best_params_)
    # print("Grid scores:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #     print("%0.3f (+/-%0.03f) for %r"
    #           % (mean, std * 2, params))

# %% 6.训练模型并测试
"""
利用得到的参数，训练模型并测试，代码如下：
"""

""" 训练
xgboost模型训练
"""
time_0 = time.clock()
# 生成(X,y)
# print('\n>> 开始训练模型')

# 转换数据
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
# 训练模型
# 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本;这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
num_rounds = 1000  # 迭代次数
# early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
print("Best parameters set:")

print(bst_param)
bst = xgb.train(bst_param, dtrain, num_rounds)
y_train_pred = bst.predict(dtrain)
y_test_pred = bst.predict(dtest)
print("train r2:", r2_score(y_train, y_train_pred))
print("test r2:", r2_score(y_test, y_test_pred))


y_train_pred = pd.DataFrame({'REAL': y_train, 'PRED': y_train_pred})
y_train_pred.to_csv(out_path + '/train_pred.csv', index=False)
y_test_pred = pd.DataFrame({'REAL': y_test, 'PRED': y_test_pred})
y_test_pred.to_csv(out_path + '/test_pred.csv', index=False)
# 得分
# feat 重要性
plt.rcParams['figure.figsize'] = (10, 10)
xgb.plot_importance(bst)
plt.savefig(out_path + '/feat_import.png', dpi=300)
f_score = bst.get_fscore()
f_id = pd.DataFrame(list(f_score.keys()))
f_pro = pd.DataFrame(list(f_score.values()))
f_score = pd.concat([f_id, f_pro], axis=1)
f_score.columns = ['f_id', 'f_pro']
f_score.sort_values(by=['f_pro'], ascending=[0], inplace=True)
f_score.to_csv(out_path + '/feat_import.csv', index=False)
# print('<< 完成验证模型, 用时', time.clock() - time_0, 's')

# 7.结论
"""
通过模型的对比，我们在波士顿房价预测项目后面阶段确定使用xgboost库中的XGBRegressor模型。
通过使用GridSearchCV方法做网格搜索，确定XGBRegressor模型使用{'learning_rate': 0.12, 'max_depth': 4, 'n_estimators': 200}的参数组合。
"""
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

plt.figure(figsize=(30, 5))
plt.plot(range(len(y_train)), y_train, 'r', label='真实值')
plt.plot(range(len(y_train_pred)), y_train_pred, 'g--', alpha=.7, label='预测值')
plt.title('预测房价 —— XGBOOST训练集')
plt.savefig(out_path + '/train_pred.png', dpi=200)

plt.figure(figsize=(10, 5))
plt.plot(range(len(y_test)), y_test, 'r', label='真实值')
plt.plot(range(len(y_test_pred)), y_test_pred, 'g--', alpha=.7, label='预测值')
plt.title('预测房价 —— XGBOOST测试集')
plt.savefig(out_path + '/test_pred.png', dpi=200)
