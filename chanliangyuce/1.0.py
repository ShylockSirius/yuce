'''
使用两层全连接层
'''

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from IPython import display
from matplotlib import pyplot as plt
import sys
print(torch.__version__)
torch.set_default_tensor_type(torch.FloatTensor)

def use_svg_display():
    # ⽤⽮量图显示
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺⼨
    plt.rcParams['figure.figsize'] = figsize

# 作图函数，本函数已保存在d2lzh_pytorch包中⽅便以后使⽤
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None, figsize=(3.5, 2.5)):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.show()

# 形状转换，本函数已保存在d2lzh_pytorch包中⽅便以后使⽤
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)




# 读取数据
# train_data = pd.read_csv('./trainA1.csv', encoding='ANSI')
# test_data = pd.read_csv('./testA1.csv', encoding='ANSI')
# 读取数据
train_data = pd.read_csv('./train.csv', encoding='ANSI')
test_data = pd.read_csv('./test.csv', encoding='ANSI')

# print(train_data.shape)
# print(test_data.shape)
# print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

# 将所有的训练数据和测试数据的特征按样本连结
# all_features = pd.concat((train_data.iloc[:, 3:29], test_data.iloc[:, 3:29]))
all_features = pd.concat((train_data.iloc[:, 2:29], test_data.iloc[:, 2:29]))
lin = pd.concat((train_data.iloc[:, 32:], test_data.iloc[:, 31:]))
all_features = pd.concat((all_features, lin), axis=1)

# print(all_features.iloc[1, :])
# print(all_features.iloc[-1, :])

# 预处理数据
# 对连续数值的特征做标准化
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
# all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.max() - x.min()))
# 标准化后，每个特征的均值变为0，所以可以直接⽤0来替换缺失值
all_features = all_features.fillna(0)
# print(all_features.iloc[1, :])

# 将离散数值转成指示特征
# dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features.shape)  # (53, 33)

# 转成NDArray
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
train_labels = torch.tensor(train_data.今天产量.values, dtype=torch.float).view(-1, 1)
# print(train_labels[:5])

# 训练模型
loss = torch.nn.MSELoss()
def get_net(feature_num):
    # net = nn.Linear(feature_num, 1)
    net = nn.Sequential(
        FlattenLayer(),
        nn.Linear(feature_num, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
    )
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net

# 数均⽅根误差
def log_rmse(net, features, labels):
    with torch.no_grad():
        # 将⼩于1的值设成1，使得取对数时数值更稳定
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        rmse = torch.sqrt(2 * loss(clipped_preds.log(), labels.log()).mean())
    return rmse.item()

# 训练函数
def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls, mae_ls = [], [], []
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    # 这⾥使⽤了Adam优化算法
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    net = net.float()
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X.float()), y.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        loss_func = torch.nn.L1Loss()
        lossss = loss_func(net(train_features), train_labels)
        mae_ls.append(lossss)
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls, mae_ls

# K折交叉验证
def get_k_fold_data(k, i, X, y):
    # 返回第i折交叉验证时所需要的训练和验证数据
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid

# 训练K次并返回训练和验证的平均误差
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum, mae_l_sum = 0, 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net(X_train.shape[1])
        train_ls, valid_ls, mae_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        mae_l_sum += mae_ls[-1]
        if i == 0:
            semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse', range(1, num_epochs + 1), valid_ls, ['train', 'valid'])
        print('fold %d, train rmse %f, valid rmse %f, valid mea %f' % (i, train_ls[-1], valid_ls[-1], mae_ls[-1]))
    return train_l_sum / k, valid_l_sum / k, mae_l_sum / k

# 模型选择
k, num_epochs, lr, weight_decay, batch_size = 10, 200, 5, 0, 64
train_l, valid_l, mae_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print('%d-fold validation: avg train rmse %f, avg valid rmse %f, avg valid mae %f' % (k, train_l, valid_l, mae_l))

# 预测
def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
    net = get_net(train_features.shape[1])
    train_ls, _, _ = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse')
    print('train rmse %f' % train_ls[-1])
    preds = net(test_features).detach().numpy()
    test_data['今天产量'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['今天产量']], axis=1)
    submission.to_csv('./submissionzong.csv', index=False)

train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)