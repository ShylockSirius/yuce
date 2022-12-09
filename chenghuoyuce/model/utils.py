import numpy as np
from torch.utils.data import Dataset, DataLoader
import csv
import torch
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif


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
    print(444)
    print(indices)
    if column:
        # k_best_features = [column[i + 1] for i in indices[0:k].tolist()]
        k_best_features = [column[i + 2] for i in indices[0:k].tolist()]
        print('k best features are: ', k_best_features)
    return X_new, indices[0:k]


class covidDataset(Dataset):
    def __init__(self, path, mode, feature_dim):
        with open(path, 'r') as f:
            csv_data = list(csv.reader(f))
            column = csv_data[0]
            # train_x = np.array(csv_data)[1:][:, 1:-1]
            train_x = np.array(csv_data)[1:][:, 2:-1]
            train_y = np.array(csv_data)[1:][:, -1]
            # print(train_x)
            print(column)
            _, col_indices = get_feature_importance(train_x, train_y, feature_dim, column)
            col_indices = col_indices.tolist()  # 得到重要列的下标
            # csv_data = np.array(csv_data[1:])[:, 1:].astype(float)
            csv_data = np.array(csv_data[1:])[:, 2:].astype(float)
            if mode == 'train':  # 如果读的是训练数据 就逢5取4  indices是数据下标
                indices = [i for i in range(len(csv_data)) if i % 5 != 0]
                self.y = torch.LongTensor(csv_data[indices, -1])
            elif mode == 'val':  # 如果读的是验证数据 就逢5取1  indices是数据下标
                indices = [i for i in range(len(csv_data)) if i % 5 == 0]
                # data = torch.tensor(csv_data[indices,col_indices])
                self.y = torch.LongTensor(csv_data[indices, -1])
            else:  # 如果读的是测试数据 就全取了
                indices = [i for i in range(len(csv_data))]
            data = torch.tensor(csv_data[indices, :])  # 取行
            self.data = data[:, col_indices]  # 取列
            self.mode = mode
            self.data = (self.data - self.data.mean(dim=0, keepdim=True)) / self.data.std(dim=0,
                                                                                          keepdim=True)  # 这里将数据归一化。
            assert feature_dim == self.data.shape[1]

            print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
                  .format(mode, len(self.data), feature_dim))

    def __getitem__(self, item):
        if self.mode == 'test':
            return self.data[item].float()
        else:
            return self.data[item].float(), self.y[item]

    def __len__(self):
        return len(self.data)
