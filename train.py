import torch
import torch.nn as nn
from tqdm import tqdm
from augmentations import augment
from SimCLR_Loss import NTXent_Loss


# credit to https://www.zhihu.com/question/523869554 for training function template
def SimCLR_Train(net, device, batch_size, train_loader, num_epoch, temp, lr=1e-4, optim='adam', lr_scheduler='None'):
    if optim == 'sgd':
        optimizer = torch.optim.SGD((param for param in net.parameters() if param.requires_grad), lr=lr,
                                    weight_decay=1e-6)
    elif optim == 'adam':
        optimizer = torch.optim.Adam((param for param in net.parameters() if param.requires_grad), lr=lr,
                                    weight_decay=1e-6)
    if lr_scheduler == 'cosine':
        if optim == 'sgd':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=1e-3)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch)

    train_loss_list = []
    model_checkpoint = 'SimCLR_batchsize=' + str(batch_size) + '_lr=' + str(lr) + '_optim=' + optim + '_temp=' + str(temp) + '_epoch=' + str(num_epoch) +'.pt'

    for epoch in range(num_epoch):
        print('Epoch ' + str(epoch+1))
        net.train()
        train_loss = 0
        for batch in tqdm(train_loader):
            image, _ = batch
            actual_batch_size = image.shape[0]  # taking care of the last batch in case it cannot be fully divided
            loss_func = NTXent_Loss(actual_batch_size, temp)
            image = image.to(device)
            # perform augmentation & compute training loss
            aug1, aug2 = augment(image, augment_twice=True)
            h1, z1 = net(aug1)  # h1: batch_size x 10; z1: batch_size x 128
            h2, z2 = net(aug2)
            loss = loss_func(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler != 'None':
                scheduler.step()
            train_loss += loss
        # print out average training loss and acc over all minibatches
        print("Epoch: {}, Training Loss: {}".format(epoch+1, train_loss / len(train_loader)))
        train_loss_list.append(train_loss / len(train_loader))

    print('Finish training, saving model...')
    torch.save(net.state_dict(), model_checkpoint)
    return train_loss_list

def Linear_Eval_Train(net, device, train_loader, num_epoch, lr=0.1, optim='sgd', lr_scheduler='cosine'):
    # freeze base network
    for param in net.base.parameters():
        param.requires_grad = False
    # follow the original simCLR setting
    if optim == 'sgd':
        optimizer = torch.optim.SGD((param for param in net.parameters() if param.requires_grad), lr=lr,
                                    weight_decay=0)
    if optim == 'adam':
        optimizer = torch.optim.Adam((param for param in net.parameters() if param.requires_grad), lr=lr,
                                     weight_decay=1e-6)
    if lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=1e-3)

    train_loss_list = []
    train_acc_list = []
    best_acc = 0

    for epoch in range(num_epoch):
        print('Epoch ', epoch + 1)
        net.train()
        train_loss = 0
        train_acc = 0
        for batch in tqdm(train_loader):
            image, label = batch
            image, label = image.to(device), label.to(device)
            actual_batch_size = image.shape[0]
            loss_func = nn.CrossEntropyLoss()
            pred = net(image)
            loss = loss_func(pred, label)
            train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler != 'None':
                scheduler.step()
            train_acc += torch.sum(torch.argmax(pred, dim=-1) == label) / actual_batch_size
        print("Training Loss: {}, Training Acc: {}".format(train_loss / len(train_loader), train_acc / len(train_loader)))
        if best_acc < train_acc / len(train_loader):
            best_acc = train_acc / len(train_loader)
        print('Best Train Acc: ', best_acc.item())
        train_loss_list.append(train_loss / len(train_loader))
        train_acc_list.append(train_acc / len(train_loader))

    return train_loss_list, train_acc_list


