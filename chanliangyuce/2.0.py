'''
使用两层全连接层和adam
效果不好
'''
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from IPython import display
from matplotlib import pyplot as plt
import time
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



def get_data_ch7(): # 不同飞机机翼噪音数据集，本函数已保存在d2lzh_pytorch包中⽅便以后使⽤
    # 读取数据
    # train_data = pd.read_csv('./trainA1.csv', encoding='ANSI')
    # test_data = pd.read_csv('./testA1.csv', encoding='ANSI')
    train_data = pd.read_csv('train.csv', encoding='ANSI')
    test_data = pd.read_csv('test.csv', encoding='ANSI')

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
    return train_features, test_features, train_labels, test_data

# 训练模型
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

# 定义损失函数
def squared_loss(y_hat, y): # 本函数已保存在d2lzh_pytorch包中⽅便以后使⽤
    # 注意这⾥返回的是向量, 另外, pytorch⾥的MSELoss并没有除以 2
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

# 梯度下降优化算法，本函数与原书不同的是这⾥第⼀个参数优化器函数⽽不是优化器的名字
# 例如: optimizer_fn=torch.optim.SGD, optimizer_hyperparams={"lr": 0.05}
def train_pytorch_ch7(optimizer_fn, optimizer_hyperparams, features, labels, test_data, test_features, batch_size=64, num_epochs=200):
    # 初始化模型
    net = get_net(features.shape[1])
    loss = nn.MSELoss()
    optimizer = optimizer_fn(net.parameters(), **optimizer_hyperparams)

    def eval_loss():
        return loss(net(features).view(-1), labels).item() / 2

    ls = [eval_loss()]
    data_iter = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(features, labels), batch_size, shuffle=True)

    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            # 除以2是为了和train_ch7保持⼀致, 因为squared_loss中除了2
            l = loss(net(X).view(-1), y) / 2

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())
    preds = net(test_features).detach().numpy()
    test_data['今天产量'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['今天产量']], axis=1)
    submission.to_csv('./submission1.csv', index=False)
    # 打印结果和作图
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    set_figsize()
    plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.show()

def init_adam_states():
    v_w, v_b = torch.zeros((train_features.shape[1], 1), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
    s_w, s_b = torch.zeros((train_features.shape[1], 1), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
    return ((v_w, s_w), (v_b, s_b))



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
def k_fold(k, X_train, y_train, test_data, test_features):
    for i in range(k):
        X_train1, y_train1, X_valid, y_valid = get_k_fold_data(k, i, X_train, y_train)
        train_pytorch_ch7(torch.optim.Adam, {'lr': 0.01}, X_train1, y_train1, test_data, test_features)


train_features, test_features, train_labels, test_data = get_data_ch7()
k=10

# 模型选择
k_fold(k, train_features, train_labels, test_data, test_features)


