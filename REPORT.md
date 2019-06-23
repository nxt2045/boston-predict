# 波士顿房价作业报告

## 一、数据预处理

### 1.1 检查缺失值

从下图的结果可以看出，数据总共有506行，13列。  
13个字段中都有506个非空的float64类型的数值，即没有空值。

<img src="https://github.com/isnxt/boston-predict/blob/master/img/347bd8ee3cd44226dc992562ec6bc1a2.png" height="35%" width="35%">

### 1.2 异常值检测

可以看出房价50可能是异常值，通过后文的特征提取和模型结果以证明该异常值合理性。

<img src="https://github.com/isnxt/boston-predict/blob/master/img/8e76786e084317e234eee51d16dd3ad9.png" height="60%" width="60%">

## 二、特征工程

### 2.1 整体特征分析

从上图可以看出，只有NOX、RM和LSTAT与目标房价MEDV拥有相似分布。

但从现实意义上，难以排除其他字段与房价的关联性，下一部分将逐个分析。

### 2.2 逐个特征分析

#### 2.2.1 字段CRIM分析

CRIM表示城镇人均犯罪率，把它作为x轴的数值。

朴素的想法是如果一个城镇犯罪率很高，则社会不稳定，经济不发达，房价不会过高。

绘制城镇人均犯罪率与房价散点图：

<img src="https://github.com/isnxt/boston-predict/blob/master/img/af404935a3508d8274dc74331038cb35.png" height="50%" width="50%">

分析结论：

1.高房价的房屋都集中在低犯罪率地区；

2.城镇人均犯罪率超过20%的情况下，房价最高不高于20；

3.城镇人均犯罪率处于(10, 20)区间的情况下，房价最高不高于30。

#### 2.2.2 字段ZN分析

ZN表示住宅用地所占比例，把它作为x轴的数值。

绘制住宅用地所占比例与房价散点图：

<img src="https://github.com/isnxt/boston-predict/blob/master/img/a6483d6ae27223ef391e0adadf7d2aaf.png" height="50%" width="50%">

分析结论：

1.两者之间的线性关系不明显；

2.在住宅用地所占比例等于0的情况下，房价可以为任意值；

3.在住宅用地所占比例大于0的情况下，房价最低不低于15；

4.在住宅用地所占比例处于(40，80)区间的情况下，房价最高不高过40；

5.在住宅用地所占比例超过80的情况下，房价最低不低于30。

#### 2.2.3 字段INDUS分析

INDUS表示城镇中非商业用地的所占比例，把它作为x轴的数值。

plt.yticks方法指定y轴的刻度，plt.grid方法为绘制网格。

绘制城镇中非商业用地所占比例与房价散点图：

<img src="https://github.com/isnxt/boston-predict/blob/master/img/d16c24e5a0e766d169a44ebcca12c526.png" height="50%" width="50%">

分析结论：

1.当城镇中非商业用地所占比例处于(0, 5)区间的情况下，房价处于(15, 50)区间；

2.当城镇中非商业用地所占比例处于(7, 15)区间的请况下，房价处于(10, 30)区间；

3.当城镇中非商业用地所占比例高于25的情况下，房价最高不高于25。

#### 2.2.4 字段CHAS分析

CHAS表示地产是否处于查尔斯河边，1表示在河边，0表示不在河边。

绘制是否处于查尔斯河边与房价散点图：

<img src="https://github.com/isnxt/boston-predict/blob/master/img/bb3d4d95920758c47c2c27016df68cf6.png" height="50%" width="50%">

分析结论：

1.地产不在查尔斯河边的情况下，房价处于(5,55)区间；

2.地产在查尔斯河边的情况下，房价最低不低于10。

#### 2.2.5 字段NOX分析

NOX表示一氧化氮的浓度，把它作为x轴的数值。

朴素的想法是一氧化氮为有毒气体，浓度过高的地区不适宜人居住，房价不会过高。

或者可以认为，浓度过高的地区靠近工业区，工业区房价比商业区房价低。

绘制一氧化氮浓度与房价散点图：

<img src="https://github.com/isnxt/boston-predict/blob/master/img/278d6bb92d79cba3516d2a7aed827bd9.png" height="50%" width="50%">

分析结论：

1.一氧化氮浓度高于0.7的情况下，房价最高不高于30，绝大部分不高于25；

2.一氧化氮处于(0.6, 0.7)区间的情况下，房价可能出现最低值；

3.一氧化氮低于0.5的情况下，房价绝大部分高于15。

#### 2.2.6 字段RM分析

RM表示每栋住宅的房间数，把它作为x轴的数值。

朴素的想法是每栋住宅的房间数越多，则住宅面积越大，房价越高。

绘制住宅房间数与房价散点图：

<img src="https://github.com/isnxt/boston-predict/blob/master/img/d04ac7351dd83fe4498bb1eb64fb1efe.png" height="50%" width="50%">

分析结论：

1.两者之间存在较强的线性关系；

2.住宅房间数处于(4, 5)区间的情况下，房价绝大部分最高不超过25；

3.住宅房间数处于(5, 6)区间的情况下，房价绝大部分最高不超过30；

4.住宅房间数处于(6, 7)区间的情况下，房价绝大部分最高不超过40；

5.住宅房间数处于(7, 8)区间的情况下，房价绝大部分最低不低于30。

#### 2.2.7 字段AGE分析

AGE表示1940年以前建成的业主自住单位的占比，把它作为x轴的数值。

绘制1940年以前建成的业主自住单位的占比与房价散点图：

<img src="https://github.com/isnxt/boston-predict/blob/master/img/e83ec7eb04f2fc338828a5a2db945cb7.png" height="50%" width="50%">

分析结论：

1.自住单位的占比处于(0, 60)的情况下，房价最低不会低于15。

#### 2.2.8 字段DIS分析

DIS表示距离5个波士顿就业中心的平均距离，把它作为x轴的数值。

朴素的想法是距离就业中心近则上下班距离近，人更愿意住在上下班距离近的地方，根据市场规律，需求高则房价会高。

绘制距离5个就业中心的平均距离与房价散点图：

<img src="https://github.com/isnxt/boston-predict/blob/master/img/3d213c93b0f36a90bfbdc7c1639ff736.png" height="50%" width="50%">

分析结论：

1.平均距离小于2的情况下，房价处于(5, 55)区间；

2.平均距离处于(2, 6)的情况下，房价最低不低于10；

3.平均距离大于6的情况下，房价最低不低于15。

#### 2.2.9 字段RAD分析

RAD表示距离高速公路的便利指数，把它作为x轴的数值。

朴素的想法是距离高速公路的便利越高，则越受欢迎，房价越高。

绘制距离高速公路的便利指数与房价散点图：

<img src="https://github.com/isnxt/boston-predict/blob/master/img/40b12cf445ef5c7a52a60d96f05c4ef4.png" height="50%" width="50%">

分析结论：

1.绝大多数房价高于30的房产，都集中在距离高速公路的便利指数低的地区；

2.距离高速公路的便利程度处于(6,10)区间时，房价最低不低于15；

3.朴素的想法与数据分析结果相反。

#### 2.2.10 字段TAX分析

TAX每一万美元的不动产税率，把它作为x轴的数值。

绘制不动产税率与房价散点图：

<img src="https://github.com/isnxt/boston-predict/blob/master/img/850b20a3c6028fca44c64b630823ba4c.png" height="50%" width="50%">

分析结论：

1.不动产税率小于200的情况下，房价最低不低于15；

2.不动产税率小于500的情况下，房价最低不低于10；

3.只有在税率大于600的情况下，房价会低于10。

#### 2.2.11 字段PTRATIO分析

PTRATIO表示城镇中学生教师比例，把它作为x轴的数值。

朴素的想法是教师较多的情况下，则教育资源多，房价也较高。

绘制学生教师比例与房价散点图，如下图所示：

<img src="https://github.com/isnxt/boston-predict/blob/master/img/d16c24e5a0e766d169a44ebcca12c526.png" height="50%" width="50%">

分析结论：

1.学生教师比例小于14的情况下，房价最低不低于20，绝大部分高于30；

2.学生教师比例处于(14, 20)区间的情况下，房价最低不低于10；

3.只有在学生教师比例大于20的情况下，房价会低于10，绝大部分不高于30。

#### 2.2.12 字段B分析

B表示城镇中黑人比例，把它作为x轴的数值。

绘制黑人比例与房价散点图，如下图所示：

<img src="https://github.com/isnxt/boston-predict/blob/master/img/c10bcfdacca5e92ba607a210476611d1.png" height="50%" width="50%">

分析结论：

1.只有在黑人比例高于350的地区，房价会高于30。

2.黑人比例处于(0, 100)区间的情况下，房价最高不高于20；

3.黑人比例处于(100,350)区间的情况下，房价最高不高于30。

#### 2.2.13字段LSTAT分析

LSTAT表示低收入阶层占比，把它作为x轴的数值。

朴素的想法是低收入阶层占比低，则经济发展程度较高，则房价较高。

<img src="https://github.com/isnxt/boston-predict/blob/master/img/74721798cf1749f0eb4618c4232e5f41.png" height="50%" width="50%">

分析结论：

1.只有低收入阶层占比小于10的情况下，房价会高于35；

2.低收入阶层占比小于5的情况下，房价最低不低于20；

3.低收入阶层占比处于(10,20)区间的情况下，房价处于(10, 30)区间；

4.低收入阶层占比大于20的情况下，房价最高不高于25。

### 2.2 特征提取

根据字段分析的结果，提取出新的特征，做成字段。

<img src="https://github.com/isnxt/boston-predict/blob/master/img/217ab1397b8b672500fedc8396735baa.png" height="45%" width="45%">

分箱形成的新字段，先Onehot编码再与原表格连接。

<img src="https://github.com/isnxt/boston-predict/blob/master/img/15177892375e97ba126f19a856ae0fb5.png" height="120%" width="120%">

## 三、算法

### 3.1 选择最优模型

使用sklearn.model_selection库中的cross_validate方法，从如下回归模型中找到最优模型

<img src="https://github.com/isnxt/boston-predict/blob/master/img/71b6b5c91cfd3edf585e5b637838b79b.png" height="25%" width="25%">

<img src="https://github.com/isnxt/boston-predict/blob/master/img/ce37ec0aaecb074330839741dcef7a25.png" height="40%" width="40%">

从下图中可以看出，几个集成回归模型都在测试集上取得0.8以上的得分。XGBRegressor模型测试集预测结果最优。而决策树回归模型在训练集上取得了满分，与测试集结果差距大，说明这种模型容易过拟合，而相比较之下xgboost因为其惩罚模型复杂度的特点具有很好的泛化能力。

<img src="https://github.com/isnxt/boston-predict/blob/master/img/aad42eda47053aadab99587da7887627.png" height="50%" width="50%">

### 3.2 训练模型

用train_test_split划分训练集和测试集

<img src="https://github.com/isnxt/boston-predict/blob/master/img/80d808f6e708e1b88f527c1d5ba1a423.png" height="45%" width="45%">

sklearn.model_selection库中有GridSearchCV方法，搜索模型的最优参数。

<img src="https://github.com/isnxt/boston-predict/blob/master/img/a65cb9ca7583015704da410426c2a9e2.png" height="60%" width="60%">

示例过程如下：

<img src="https://github.com/isnxt/boston-predict/blob/master/img/ff30437344090eefa586c13d1fc70c9a.png" height="50%" width="50%">

利用得到的参数，训练模型

<img src="https://github.com/isnxt/boston-predict/blob/master/img/d0b31b5fce612d523d3addca098460cf.png" height="75%" width="75%">

## 四、实验结果分析

用train_test_split划分得到的测试集，多次实验结果如下，可以看出本模型准确率(R\^2)稳定在0.92以上，

<img src="https://github.com/isnxt/boston-predict/blob/master/img/bf82e394fc8d723b8c70a3237d0b2326.png" height="70%" width="70%">

将预测与真实值画图比较可以看出，本模型准确率高，泛化能力强，并且有能力预测房价为50的异常值

<img src="https://github.com/isnxt/boston-predict/blob/master/img/f291b3e5eacfec1ffa8a6e0790ad2d53.png" height="70%" width="70%">
