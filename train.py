import torch
import torch.nn as nn
from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup
from tqdm import tqdm
from augmentations import augment
from SimCLR_Loss import NTXent_Loss
from torchlars import LARS


# credit to https://www.zhihu.com/question/523869554 for training function template
def SimCLR_Train(net, device, batch_size, train_loader, num_epoch, temp, lr=0.5, optim='lars', lr_scheduler='None'):
    net.to(device)
    if optim == 'sgd':
        optimizer = torch.optim.SGD((param for param in net.parameters() if param.requires_grad), lr=lr,
                                    weight_decay=1e-6)
    elif optim == 'adam':
        optimizer = torch.optim.Adam((param for param in net.parameters() if param.requires_grad), lr=lr,
                                    weight_decay=1e-6)
    elif optim == 'lars':
        optimizer = LARS(torch.optim.SGD((param for param in net.parameters() if param.requires_grad), lr=lr,
                                    weight_decay=1e-6))
    if lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch)
        # scheduler = create_lr_scheduler_with_warmup(scheduler, warmup_start_value=0, warmup_duration=10)

    train_loss_list = []
    model_checkpoint = 'SimCLR_batchsize=' + str(batch_size) + '_lr=' + str(lr) + '_optim=' + optim + '_temp=' + str(temp) + '.pt'

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
            scheduler.step()
            train_loss += loss
        # print out average training loss and acc over all minibatches
        print("Epoch: {}, Training Loss: {}".format(epoch+1, train_loss / len(train_loader)))
        train_loss_list.append(train_loss / len(train_loader))

    print('Finish training, saving model...')
    torch.save(net.state_dict(), model_checkpoint)
        # net.eval()
        # val_loss = 0
        # val_acc = 0
        # with torch.no_grad():
        #     for image, label in val_loader:
        #         image, label = image.to(device), label.to(device)
        #         actual_batch_size = image.shape[0]
        #         pred, _ = net(image)
        #         loss_func = nn.CrossEntropyLoss()
        #         val_loss += loss_func(label, pred)
        #         val_acc += torch.sum(pred == label).item() / actual_batch_size
        #     print("Validation Loss: {}, Validation Acc: {}".format(val_loss / len(val_loader), val_acc / len(val_loader)))
        #     if best_val_acc < val_acc / len(val_loader):
        #         best_val_acc = val_acc / len(val_loader)
        #         torch.save(net.state_dict(), model_checkpoint)
        #         print('Saving model...')
        #     print('Best Val Acc: ', best_val_acc)
        #     val_acc_list.append(val_acc / len(val_loader))
        #     val_loss_list.append(val_loss / len(val_loader))

    return train_loss_list
