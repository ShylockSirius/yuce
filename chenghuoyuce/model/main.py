from chenghuoyuce.model.model import myNet
from chenghuoyuce.model.utils import covidDataset
from chenghuoyuce.model.train import train_val
from chenghuoyuce.model.evaluate import evaluate
from torch import optim
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_path = '../train1.csv'
test_path = '../test1.csv'

feature_dim = 11
trainset = covidDataset(train_path, 'train', feature_dim=feature_dim)
valset = covidDataset(train_path, 'val', feature_dim=feature_dim)
testset = covidDataset(test_path, 'test', feature_dim=feature_dim)

config = {
    'n_epochs': 2000,  # maximum number of epochs
    'batch_size': 270,  # mini-batch size for dataloader
    'optimizer': 'SGD',  # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {  # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 0.0001,  # learning rate of SGD
        'momentum': 0.9  # momentum for SGD
    },
    'early_stop': 200,  # early stopping epochs (the number epochs since your model's last improvement)
    # 'save_path': 'model_save/model.pth',  # your model will be saved here
    'save_path': './model.pth',  # your model will be saved here
}


def getLoss(pred, target, model):
    loss = nn.MSELoss(reduction='mean')
    ''' Calculate loss '''
    regularization_loss = 0
    for param in model.parameters():
        # TODO: you may implement L1/L2 regularization here
        # 使用L2正则项
        # regularization_loss += torch.sum(abs(param))
        regularization_loss += torch.sum(param ** 2)
    return loss(pred, target) + 0.00075 * regularization_loss


loss = getLoss

model = myNet(feature_dim).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

trainloader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)
valloader = DataLoader(valset, batch_size=config['batch_size'], shuffle=True)

train_val(model, trainloader, valloader, optimizer, loss, config['n_epochs'], device, save_=config['save_path'])
evaluate(config['save_path'], testset, 'pred.csv', device)