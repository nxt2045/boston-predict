# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @DATE    : 5/19/2019
# @Author  : xiaotong
# @File    : explor_data
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
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import time
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

if not os.path.exists("./img"):
    os.mkdir("./img")


def draw_scatter(x, y, xlabel):
    plt.rcParams['figure.figsize'] = (6, 4)
    plt.figure()
    plt.scatter(x, y, alpha=.5, marker='o', c='steelblue')
    plt.title('%s与房价散点图' % xlabel)
    plt.xlabel(xlabel)
    plt.ylabel('房价')
    plt.yticks(range(0, 60, 5))
    plt.grid()
    plt.tight_layout()
    plt.savefig("./img/%s与房价散点图.png" % xlabel, dpi=150)
    # plt.show()


# %% 1.1 载入数据集
"""
波士顿房价数据集详细中文解释链接：
http://sklearn.apachecn.org/cn/0.19.0/datasets/index.html#boston-house-prices
"""
X = load_boston().data
y = load_boston().target

# %% 1.2 数据观察
X = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
y = pd.DataFrame(load_boston().target, columns=['MEDV'])
data = pd.concat([X,y],axis=1)
print(data.head(5), '\n')
print(data.info())
print(data.describe().T)
data.hist(bins=10,figsize=(10,9),grid=False)
plt.savefig("./img/数据条形图.png", dpi=200)
df = X
# %% 1.3 清除异常值
# X = X[y != 50]
# y = y[y != 50]
# print(X.shape)

# %% 2.特征提取
# 2.1 字段CRIM分析
"""
CRIM表示城镇人均犯罪率，把它作为x轴的数值。
朴素的想法是如果一个城镇犯罪率很高，则社会不稳定，经济不发达，房价不会过高。
绘制城镇人均犯罪率与房价散点图，代码如下：
"""
draw_scatter(df[['CRIM']], y, '城镇人均犯罪率')
"""
分析结论：
1.高房价的房屋都集中在低犯罪率地区；
2.城镇人均犯罪率超过20%的情况下，房价最高不高于20；
3.城镇人均犯罪率处于(10, 20)区间的情况下，房价最高不高于30。
"""
# 2.2 字段ZN分析
"""
ZN表示住宅用地所占比例，把它作为x轴的数值。
绘制住宅用地所占比例与房价散点图，代码如下：
"""
draw_scatter(df['ZN'], y, '住宅用地所占比例')
"""
分析结论：
1.两者之间的线性关系不明显；
2.在住宅用地所占比例等于0的情况下，房价可以为任意值；
3.在住宅用地所占比例大于0的情况下，房价最低不低于15；
4.在住宅用地所占比例处于(40，80)区间的情况下，房价最高不高过40；
5.在住宅用地所占比例超过80的情况下，房价最低不低于30。
"""
# 2.3 字段INDUS分析
"""
INDUS表示城镇中非商业用地的所占比例，把它作为x轴的数值。
plt.yticks方法指定y轴的刻度，plt.grid方法为绘制网格。
绘制城镇中非商业用地所占比例与房价散点图，代码如下：
"""
draw_scatter(df['INDUS'], y, '城镇中非商业用地所占比例')
"""
分析结论：
1.当城镇中非商业用地所占比例处于(0, 5)区间的情况下，房价处于(15, 50)区间；
2.当城镇中非商业用地所占比例处于(7, 15)区间的请况下，房价处于(10, 30)区间；
3.当城镇中非商业用地所占比例高于25的情况下，房价最高不高于25。
"""
# 2.4 字段CHAS分析
"""
CHAS表示地产是否处于查尔斯河边，1表示在河边，0表示不在河边。
绘制是否处于查尔斯河边与房价散点图，代码如下：
"""
plt.xticks([0, 1])
draw_scatter(df['CHAS'], y, '是否处于查尔斯河边')
"""
分析结论：
1.地产不在查尔斯河边的情况下，房价处于(5,55)区间；
2.地产在查尔斯河边的情况下，房价最低不低于10。 
"""
# 2.5 字段NOX分析
"""
NOX表示一氧化氮的浓度，把它作为x轴的数值。
朴素的想法是一氧化氮为有毒气体，浓度过高的地区不适宜人居住，房价不会过高。
或者可以认为，浓度过高的地区靠近工业区，工业区房价比商业区房价低。
绘制一氧化氮浓度与房价散点图，代码如下：
"""
draw_scatter(df['NOX'], y, '一氧化氮浓度')
"""
分析结论：
1.一氧化氮浓度高于0.7的情况下，房价最高不高于30，绝大部分不高于25；
2.一氧化氮处于(0.6, 0.7)区间的情况下，房价可能出现最低值；
3.一氧化氮低于0.5的情况下，房价绝大部分高于15。
"""
# 2.6 字段RM分析
"""
RM表示每栋住宅的房间数，把它作为x轴的数值。
朴素的想法是每栋住宅的房间数越多，则住宅面积越大，房价越高。
绘制住宅房间数与房价散点图，代码如下：
"""
draw_scatter(df['RM'], y, '住宅房间数')
"""
分析结论：
1.两者之间存在较强的线性关系；
2.住宅房间数处于(4, 5)区间的情况下，房价绝大部分最高不超过25；
3.住宅房间数处于(5, 6)区间的情况下，房价绝大部分最高不超过30；
4.住宅房间数处于(6, 7)区间的情况下，房价绝大部分最高不超过40；
5.住宅房间数处于(7, 8)区间的情况下，房价绝大部分最低不低于30。
"""
# 2.7 字段AGE分析
"""
AGE表示1940年以前建成的业主自住单位的占比，把它作为x轴的数值。
绘制1940年以前建成的业主自住单位的占比与房价散点图，代码如下：
"""
draw_scatter(df['AGE'], y, '1940年以前建成的业主自住单位的占比')
"""
分析结论：
1.自住单位的占比处于(0, 60)的情况下，房价最低不会低于15。
"""
# 2.8 字段DIS分析
"""
DIS表示距离5个波士顿就业中心的平均距离，把它作为x轴的数值。
朴素的想法是距离就业中心近则上下班距离近，人更愿意住在上下班距离近的地方，根据市场规律，需求高则房价会高。
绘制距离5个就业中心的平均距离与房价散点图，代码如下：
"""
draw_scatter(df['DIS'], y, '距离5个就业中心的平均距离')
"""
分析结论：
1.平均距离小于2的情况下，房价处于(5, 55)区间；
2.平均距离处于(2, 6)的情况下，房价最低不低于10；
3.平均距离大于6的情况下，房价最低不低于15。
"""
# 2.9 字段RAD分析
"""
RAD表示距离高速公路的便利指数，把它作为x轴的数值。
朴素的想法是距离高速公路的便利越高，则越受欢迎，房价越高。
绘制距离高速公路的便利指数与房价散点图，代码如下：
"""
draw_scatter(df['RAD'], y, '距离高速公路的便利指数')
"""
分析结论：
1.绝大多数房价高于30的房产，都集中在距离高速公路的便利指数低的地区；
2.距离高速公路的便利程度处于(6,10)区间时，房价最低不低于15；
3.朴素的想法与数据分析结果相反。
"""
# 2.10 字段TAX分析
"""
TAX每一万美元的不动产税率，把它作为x轴的数值。
绘制不动产税率与房价散点图，代码如下：
"""
draw_scatter(df['TAX'], y, '不动产税率')
"""
分析结论：
1.不动产税率小于200的情况下，房价最低不低于15；
2.不动产税率小于500的情况下，房价最低不低于10；
3.只有在税率大于600的情况下，房价会低于10。
"""
# 2.11 字段PTRATIO分析
"""
PTRATIO表示城镇中学生教师比例，把它作为x轴的数值。
朴素的想法是教师较多的情况下，则教育资源多，房价也较高。
绘制学生教师比例与房价散点图，如下图所示：
"""
draw_scatter(df['PTRATIO'], y, '学生教师比例')
"""
分析结论：
1.学生教师比例小于14的情况下，房价最低不低于20，绝大部分高于30；
2.学生教师比例处于(14, 20)区间的情况下，房价最低不低于10；
3.只有在学生教师比例大于20的情况下，房价会低于10，绝大部分不高于30。
"""
# 2.12 字段B分析
"""
B表示城镇中黑人比例，把它作为x轴的数值。
绘制黑人比例与房价散点图，如下图所示：
"""
draw_scatter(df['B'], y, '黑人比例')
"""
分析结论：
1.只有在黑人比例高于350的地区，房价会高于30。
2.黑人比例处于(0, 100)区间的情况下，房价最高不高于20；
3.黑人比例处于(100,350)区间的情况下，房价最高不高于30。
"""
# 2.13字段LSTAT分析
"""
LSTAT表示低收入阶层占比，把它作为x轴的数值。
朴素的想法是低收入阶层占比低，则经济发展程度较高，则房价较高。
"""
draw_scatter(df['LSTAT'], y, '低收入阶层占比')
"""
分析结论：
1.只有低收入阶层占比小于10的情况下，房价会高于35；
2.低收入阶层占比小于5的情况下，房价最低不低于20；
3.低收入阶层占比处于(10,20)区间的情况下，房价处于(10, 30)区间；
4.低收入阶层占比大于20的情况下，房价最高不高于25。
"""
