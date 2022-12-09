import torch
import time
import matplotlib.pyplot as plt


def train_val(model, trainloader, valloader, optimizer, loss, epoch, device, save_):
    # trainloader = DataLoader(trainset,batch_size=batch,shuffle=True)
    # valloader = DataLoader(valset,batch_size=batch,shuffle=True)
    model = model.to(device)
    plt_train_loss = []
    plt_val_loss = []
    val_rel = []
    min_val_loss = 100000

    for i in range(epoch):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        val_loss = 0.0
        for data in trainloader:
            optimizer.zero_grad()
            x, target = data[0].to(device), data[1].to(torch.float32).to(device)
            pred = model(x)
            bat_loss = loss(pred, target, model)
            bat_loss.backward()
            optimizer.step()
            train_loss += bat_loss.detach().cpu().item()

        plt_train_loss.append(train_loss / trainloader.__len__())

        model.eval()
        with torch.no_grad():  # 验证时 不计算梯度
            for data in valloader:
                val_x, val_target = data[0].to(device), data[1].to(device)
                val_pred = model(val_x)
                val_bat_loss = loss(val_pred, val_target, model)
                val_loss += val_bat_loss
                val_rel.append(val_pred)
        if val_loss < min_val_loss:
            torch.save(model, save_)
            min_val_loss = val_loss
        plt_val_loss.append(val_loss / valloader.__len__())

        print('[%03d/%03d] %2.2f sec(s) TrainLoss : %3.6f | valLoss: %3.6f' % \
              (i, epoch, time.time() - start_time, plt_train_loss[-1], plt_val_loss[-1])
              )

    plt.plot(plt_train_loss)
    plt.plot(plt_val_loss)
    plt.title('loss')
    plt.legend(['train', 'val'])
    plt.show()