import os, time
import argparse
import numpy as np
import torch, random
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from capsnet import CapsNet
from data_loader import Dataset
from tqdm import tqdm
from scipy.stats import norm as dist_model
from sklearn.metrics import confusion_matrix, classification_report


SCALE = 0.01

'''
Config class to determine the parameters for capsule net
'''

class Config:
    def __init__(self, dataset=''):
        if dataset == 'wafer':
            # CNN (cnn)
            self.cnn_in_channels = 1
            self.cnn_out_channels = 256
            self.cnn_kernel_size = 9

            # Primary Capsule (pc)
            self.pc_num_capsules = 7
            self.pc_in_channels = 256
            self.pc_out_channels = 32
            self.pc_kernel_size = 9
            self.pc_num_routes = 32 * 10 * 10

            # Wafer Capsule (wc)
            self.wc_num_capsules = 10
            self.wc_num_routes = 32 * 10 * 10
            self.wc_in_channels = 7
            self.wc_out_channels = 16

            # Encoder
            self.input_width = 36
            self.input_height = 36
        

        elif dataset == 'MixedWM38':
            pass


def train(model, optimizer, train_loader, epoch):
    capsule_net = model
    capsule_net.train()
    n_batch = len(list(enumerate(train_loader)))
    total_loss = 0
    for batch_id, (data, target) in enumerate(tqdm(train_loader)):

        if USE_CUDA:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = capsule_net(data)
        loss = capsule_net.loss(output, target)
        loss.backward()
        optimizer.step()
        correct = sum(np.argmax(output.data.cpu().numpy(), 1) == np.argmax(target.data.cpu().numpy(),1))
        train_loss = loss.item()
        total_loss += train_loss
        if (batch_id+1) % 100 == 0:
            tqdm.write("Epoch: [{}/{}], Batch: [{}/{}], train accuracy: {:.6f}, loss: {:.6f}".format(
                epoch,
                N_EPOCHS,
                batch_id + 1,
                n_batch,
                correct / float(len(data)),
                train_loss / float(len(data))
                ))
    tqdm.write('Epoch: [{}/{}], train loss: {:.6f}'.format(epoch,N_EPOCHS,total_loss / len(train_loader.dataset)))


def test(capsule_net, test_loader, epoch, OWL = False):
    capsule_net.eval()
    test_loss = 0
    correct = 0
    for batch_id, (data, target) in enumerate(test_loader):

        if USE_CUDA:
            data, target = data.cuda(), target.cuda()

        output = capsule_net(data)
        loss = capsule_net.loss(output, target)

        test_loss += loss.item()
        correct += sum(np.argmax(output.data.cpu().numpy(), 1) == np.argmax(target.data.cpu().numpy(),1))

    if OWL:
        tqdm.write(
        "Epoch: [{}/{}], Validation accuracy: {:.6f}, loss: {:.6f}".format(epoch, N_EPOCHS, correct / len(test_loader.dataset),
                                                                  test_loss / len(test_loader)))
    else:
        tqdm.write(
        "Epoch: [{}/{}], test accuracy: {:.6f}, loss: {:.6f}".format(epoch, N_EPOCHS, correct / len(test_loader.dataset),
                                                                  test_loss / len(test_loader)))



def fit(prob_pos_X):
    prob_pos = [p for p in prob_pos_X]+[2-p for p in prob_pos_X]
    pos_mu, pos_std = dist_model.fit(prob_pos)
    return pos_mu, pos_std
    


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Open world classification.')
    parser.add_argument('--epoch', type = int, help = "Number of epochs.", default = 2)
    parser.add_argument('--batchSize', type = int, help = "Batch Size.", default = 256)
    parser.add_argument('--learningRate', type = float, help = "Learning Rate.", default = 0.001) 
    parser.add_argument('--augment', type = int, help = "Data augmentation or not.", default = 0)
    parser.add_argument('--percentSeenCls', type = int, help = "% of seen classes.", default = 100)
    parser.add_argument('--splitRatio', type = str, help = "Split ratio like 8:2 or 7:3", default = '8:2')
    args = parser.parse_args()
    
    # Fetching values from user's input
    N_EPOCHS = args.epoch
    BATCH_SIZE = args.batchSize
    LEARNING_RATE = args.learningRate
    AUGMENTATION = args.augment
    PERCENT_SEEN_CLS = args.percentSeenCls
    SPLITRATIO = args.splitRatio

    # For reproducibility 
    seed_num = 42
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    np.random.seed(seed_num)
    random.seed(seed_num)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic =True
    
    USE_CUDA = True if torch.cuda.is_available() else False
    MOMENTUM = 0.9

    dataset = 'wafer'
    config = Config(dataset)
    wafer = Dataset(dataset, BATCH_SIZE, AUGMENTATION, PERCENT_SEEN_CLS, SPLITRATIO)
    

    capsule_net = CapsNet(config, percentSeenCls = PERCENT_SEEN_CLS)
        
    capsule_net = torch.nn.DataParallel(capsule_net)
    
    if USE_CUDA:
        capsule_net = capsule_net.cuda()
    capsule_net = capsule_net.module

    optimizer = torch.optim.AdamW(capsule_net.parameters(), lr=LEARNING_RATE)
    
    numOfSeenCls = int(np.round((PERCENT_SEEN_CLS/100) * 9))
    seen_cls = range(numOfSeenCls)
    unseen_cls = len(seen_cls)
    
    for e in range(1, N_EPOCHS + 1):
        train(capsule_net, optimizer, wafer.train_loader, e)
        if numOfSeenCls < 9:
            test(capsule_net, wafer.val_loader, e, True)
        elif SPLITRATIO == "8:1:1":
            test(capsule_net, wafer.val_loader, e, True)
        else:
            test(capsule_net, wafer.test_loader, e, False)

    
    pred = []
    y = []
    max_cls_test = []
    max_cls_p_test = []
    mu_stds = []
       
    if numOfSeenCls < 9:
        
        # Prediction on training examples to calculate standard deviation
        seen_train_pred = []
        y_train_pred = []
        
        for batch_id, (data, target) in enumerate(wafer.train_loader):
            if USE_CUDA:
                data, target = data.cuda(), target.cuda()
                
            seen_train_pred += capsule_net(data).data.cpu().numpy().tolist()
            y_train_pred += np.argmax(target.data.cpu().numpy(), 1).tolist()
                
        y_train_pred = np.array(y_train_pred)
        seen_train_pred = np.array(seen_train_pred)
            
        # Calculating mu and std of each seen class
        for i in range(len(seen_cls)):
            pos_mu, pos_std = fit(seen_train_pred[y_train_pred==i, i])
            mu_stds.append([pos_mu, pos_std])
  
        
        for batch_id, (data, target) in enumerate(wafer.test_loader):
            if USE_CUDA:
                data, target = data.cuda(), target.cuda()
            
            if PERCENT_SEEN_CLS == 25:
                SCALE = 0.3
                
            scale = SCALE
            for p,t in zip(capsule_net(data).data.cpu().numpy(), np.argmax(target.data.cpu().numpy(), 1)):
                max_cls = np.argmax(p)
                max_cls_p = np.max(p)
                
                max_cls_test.append(max_cls)
                max_cls_p_test.append(max_cls_p) 
                
                # Finding threshold for the predicted class
                threshold = max(0.5, 1. - scale * mu_stds[max_cls][1]) 
                if max_cls_p > threshold:
                    pred.append(max_cls)
                    y.append(t)
                else:
                    pred.append(unseen_cls)
                    y.append(t)
    else:
        st_time = time.time()
        for batch_id, (data, target) in enumerate(wafer.test_loader):
            if USE_CUDA:
                data, target = data.cuda(), target.cuda()
                
            pred += np.argmax(capsule_net(data).data.cpu().numpy(), 1).tolist()
            y += np.argmax(target.data.cpu().numpy(), 1).tolist()
        print("\nTotal Inference Time:", time.time() - st_time, " seconds\n")
    
    # Confusion Matrix
    print("Confusion Matrix :")
    print(confusion_matrix(y, pred))
    print()
    print("Classification Report :")
    print(classification_report(y, pred))
    
    
    # Saving the model
    torch.save(capsule_net, os.getcwd() + "/CapsNet_model.pt")
