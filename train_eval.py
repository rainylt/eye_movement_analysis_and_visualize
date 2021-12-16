import numpy as np
import time
import torch
from model.classfier import Config, Model
from model.dataset import EyeDataset
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
import torch.nn.functional as F

from sklearn import metrics

def train():
    config = Config()
    train_dataset = EyeDataset('data/all_data/train_data.npy','data/all_data/train_label.txt')
    val_dataset = EyeDataset('data/all_data/val_data.npy','data/all_data/val_label.txt')
    train_loader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = config.batch_size, shuffle = True)

    model = Model(config)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    iterations = 0#现在进行的batch数
    last_improve = 0#上次验证机loss下降的batch数
    dev_best_loss = float('inf')
    flag = False
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))

    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, data in enumerate(train_loader):
            inputs, labels = data
            outputs = model(inputs)

            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if(iterations%100==0):
                true_label = labels.data.cpu()
                predic = torch.max(outputs.data,dim=1)[1].cpu()
                train_acc = metrics.accuracy_score(true_label, predic)
                dev_acc, dev_loss = eval(config, model, val_loader)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    last_improve = iterations
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}'
                print(msg.format(iterations, loss.item(), train_acc, dev_loss, dev_acc))
                writer.add_scalar("loss/train", loss.item(), iterations)
                writer.add_scalar("loss/dev", dev_loss, iterations)
                writer.add_scalar("acc/train", train_acc, iterations)
                writer.add_scalar("acc/dev", dev_acc, iterations)
                model.train()
            iterations += 1
            if iterations - last_improve > 1000:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
        writer.close()

def eval(cfg,model, loader):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    acc = metrics.accuracy_score(labels_all, predict_all)
    return acc, loss_total / len(loader)




