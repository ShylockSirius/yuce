'''
75%目前效果最好
'''
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# 导入数据集
df = pd.read_csv('./train1.csv', encoding='ANSI')
df1 = pd.read_csv('./test1.csv', encoding='ANSI')

# 输出数据预览
print(df.head())

# 自变量（该数据集的前11项）
X = df.iloc[:, 2:-1].values

# 因变量（该数据集的最后1项，即第14项）
y = df.iloc[:, -1].values

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.6, random_state=1)
X_train = X
y_train = y
X_test = df1.iloc[:, 2:-1].values
y_test = df1.iloc[:, -1].values

# 评估回归性能
# criterion ：
# 回归树衡量分枝质量的指标，支持的标准有三种：
# 1）输入"mse"使用均方误差mean squared error(MSE)，父节点和叶子节点之间的均方误差的差额将被用来作为特征选择的标准，
# 这种方法通过使用叶子节点的均值来最小化L2损失
# 2）输入“friedman_mse”使用费尔德曼均方误差，这种指标使用弗里德曼针对潜在分枝中的问题改进后的均方误差
# 3）输入"mae"使用绝对平均误差MAE（mean absolute error），这种指标使用叶节点的中值来最小化L1损失

# 此处使用mse
forest = RandomForestRegressor(n_estimators=1000,
                               criterion='mse',
                               random_state=1,
                               n_jobs=-1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)
print(y_test_pred)
y_test_pred1 = []
for i in range(len(y_test_pred)):
    if y_test_pred[i] < 0.41:
        y_test_pred1.append(0)
    else:
        y_test_pred1.append(1)
print(y_test_pred1)
zheng = 0
zong = 0
for i in range(len(y_test_pred1)):
    if y_test_pred1[i] == y_test[i]:
        zheng += 1
    zong += 1
print(zheng)
print(zong)
print(zheng/zong)
id = df1.iloc[:, 0].values
df2 = {"Id": id, "等径长度": y_test_pred1}
df2 = pd.core.frame.DataFrame(df2)
df2.to_csv('./answerzong.csv', index=False)

y_train_pred1 = []
for i in range(len(y_train_pred)):
    if y_train_pred[i] < 0.41:
        y_train_pred1.append(0)
    else:
        y_train_pred1.append(1)
zheng1 = 0
zong1 = 0
for i in range(len(y_train_pred1)):
    if y_train_pred1[i] == y_train[i]:
        zheng1 += 1
    zong1 += 1
print(zheng1/zong1)


print('MSE train: %.3f, test: %.3f' % (
    mean_squared_error(y_train, y_train_pred),
    mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
    r2_score(y_train, y_train_pred),
    r2_score(y_test, y_test_pred)))
