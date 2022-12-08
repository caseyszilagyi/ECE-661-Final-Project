import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from augmentations import augment
from utils import pairwise_sim, NTXent

# credit to https://www.zhihu.com/question/523869554 for training function template
def train(net, device, batch_size, train_loader, val_loader, num_epoch, lr, optim='sgd', scheduler='None', temp=0.1):
    net.to(device)
    if optim == 'sgd':
        optimizer = optim.SGD((param for param in net.parameters() if param.requires_grad), lr=lr, weight_decay=0)
    elif optim == 'adam':
        optimizer = optim.Adam((param for param in net.parameters() if param.requires_grad), lr=lr, weight_decay=0)
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    best_tr_acc = 0
    best_val_acc = 0
    model_checkpoint = 'SimCLR_batchsize=' + str(batch_size) + '_lr=' + str(lr) + '_temp=' + str(temp) + '.pt'

    for epoch in range(num_epoch):
        print('Epoch ' + str(epoch+1))
        net.train()
        train_acc = 0
        train_loss = 0
        for batch in tqdm(train_loader):
            image, label = batch
            image, label = image.to(device), label.to(device)
            # perform augmentation & compute cosine similarities
            aug1 = augment(image)
            aug2 = augment(image)
            z1 = net(aug1, inference=False) # batch_size x 2048
            z2 = net(aug2, inference=False) # batch_size x 2048
            z_mat = torch.cat((z1, z2), 0)
            sim_mat = pairwise_sim(z_mat)
            # compute loss
            loss = 0
            for i in range(batch_size):
                loss += NTXent(2*i, 2*i+1, sim_mat, temp) + NTXent(2*i+1, 2*i, sim_mat, temp)
            loss /= 2 * batch_size
            train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # compute training acc
            net.eval()
            with torch.no_grad():
                pred = net(image, inference=True)
            train_acc += torch.sum(pred == label).item() / batch_size
        # print out average training loss and acc over all minibatches
        print("Epoch: {}, Training Loss: {}, Training Acc: {}".format(epoch, train_loss / len(train_loader), train_acc / len(train_loader)))
        if best_tr_acc < train_acc / len(train_loader):
            best_tr_acc = train_acc / len(train_loader)
            torch.save(net.state_dict(), model_checkpoint)
        print('Best Training Acc: ', best_tr_acc)
        train_acc_list.append(train_acc / len(train_loader))
        train_loss_list.append(train_loss / len(train_loader))

        # Question: should we compute validation loss?
        # thinking about just using cross entropy instead of NTXent for validation
        net.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for image, label in val_loader:
                image, label = image.to(device), label.to(device)
                pred = net(image, inference=True)
                loss_func = nn.CrossEntropyLoss()
                val_loss += loss_func(label, pred)
                val_acc += torch.sum(pred == label).item() / batch_size
            print("Validation Loss: {}, Validation Acc: {}".format(val_loss / len(val_loader), val_acc / len(val_loader)))
            if best_val_acc < val_acc / len(val_loader):
                best_val_acc = val_acc / len(val_loader)
            print('Best Val Acc: ', best_val_acc)
            val_acc_list.append(val_acc / len(val_loader))
            val_loss_list.append(val_loss / len(val_loader))
