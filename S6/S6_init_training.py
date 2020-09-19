from torch.optim.lr_scheduler import StepLR
import torch.optim as optim

from EVA5.S6.S6_data_loader import init_train_test_loader
from torch.optim.lr_scheduler import StepLR


train_loader, test_loader = init_train_test_loader()

from EVA5.S6.S6_train_test_function import train
from EVA5.S6.S6_train_test_function import test

def init_training(model, device, train_loader, epochs,train_losses,train_acc,test_losses,test_acc, step_lr=True, l1_lambda=None, l2_en=False):

    if l2_en:
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, dampening=0, weight_decay=1e-5, nesterov=False)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    if step_lr:
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    
    for epoch in range(epochs):

        train(model, device, train_loader, optimizer, epoch,train_losses,train_acc, l1_lambda,train_d)

        if step_lr:
            scheduler.step()

        print('\n Epoch {}, lr {}'.format(epoch, optimizer.param_groups[0]['lr']))
        test(model, device, test_loader,test_losses,test_acc)
