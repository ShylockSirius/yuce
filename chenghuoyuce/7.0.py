'''
k best features are:  ['放肩高度', '引晶稳定CCD亮度差值', '复投次数', '引晶后50长拉速', '稳定时间', '引晶开始CCD亮度值', '引晶开始结束CCD差值', '引晶平均拉速', '引晶熔接CCD差值', '调温最大亮度值', '放肩降温量']
[ 9  4  0  7  2  3  6  8  5  1 10]
使用部分特征值进行预测效果更差
'''

from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import csv

def get_feature_importance(feature_data, label_data, k=4, column=None):
    """
    此处省略 feature_data, label_data 的生成代码。
    如果是 CSV 文件，可通过 read_csv() 函数获得特征和标签。
    """
    # a = ['1.0', '2.0', '3.0', '4.0']  # non-numeric format
    b = np.array(feature_data, dtype=float)  # convert using numpy
    for i in range(len(feature_data[0])):
        b[i] = [float(j) for j in feature_data[i]]  # convert with for loop
    c = np.array(label_data, dtype=int)  # convert using numpy
    c = [int(j) for j in label_data]  # convert with for loop
    # model = SelectKBest(chi2, k=k)  # 选择k个最佳特征
    model = SelectKBest(f_classif, k=k)  # 选择k个最佳特征
    # print(b, c)
    # X_new = model.fit_transform(feature_data, label_data)
    X_new = model.fit_transform(b, c)
    # feature_data是特征数据，label_data是标签数据，该函数可以选择出k个特征
    print('x_new', X_new)
    scores = model.scores_
    # 按重要性排序，选出最重要的 k 个                                            
    indices = np.argsort(scores)[::-1]  # 找到重要K个的下标
    print(indices)
    if column:
        # k_best_features = [column[i + 1] for i in indices[0:k].tolist()]
        k_best_features = [column[i+2] for i in indices[0:k].tolist()]
        print('k best features are: ', k_best_features)
    return X_new, indices[0:k]

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

with open('./train1.csv', 'r') as f:
    csv_data = list(csv.reader(f))
    column = csv_data[0]
    _, col_indices = get_feature_importance(X_train, y_train, 11, column)
    print(col_indices)
# 评估回归性能
# criterion ：
# 回归树衡量分枝质量的指标，支持的标准有三种：
# 1）输入"mse"使用均方误差mean squared error(MSE)，父节点和叶子节点之间的均方误差的差额将被用来作为特征选择的标准，
# 这种方法通过使用叶子节点的均值来最小化L2损失
# 2）输入“friedman_mse”使用费尔德曼均方误差，这种指标使用弗里德曼针对潜在分枝中的问题改进后的均方误差
# 3）输入"mae"使用绝对平均误差MAE（mean absolute error），这种指标使用叶节点的中值来最小化L1损失

X_train = np.delete(X_train, 10, axis=1)
X_test = np.delete(X_test, 10, axis=1)
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
