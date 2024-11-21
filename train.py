import os
import pickle
import numpy as np
import argparse
from random import random

from torch import optim
from sklearn.metrics import f1_score   

import torch
import torch.nn as nn
from torch.nn import functional as F

from sklearn.utils import shuffle
from models import AudioVisualModel

def preprocess(au_mfcc_path):
    data = []
    labels = []
    with open(au_mfcc_path, 'rb') as f:
        au_mfcc = pickle.load(f)
        
    print(len(au_mfcc))
    
    for key in au_mfcc:
        emotion = key.split('-')[2]
        emotion = int(emotion)-1
        labels.append(emotion)
        data.append(au_mfcc[key])
    
    data = np.array(data)
    labels = np.array(labels).reshape(-1, 1)
    
    data = np.hstack((data, labels))
    fdata = shuffle(data)
    
    data = fdata[:, :-1]
    labels = fdata[:, -1].astype(int)
    
    return data, labels

def eval(data, labels, model, criterion, device, mode=None, to_print=False):
    assert mode is not None

    model.eval()
    y_true, y_pred = [], []
    eval_loss = []
    corr = 0

    if mode == "test" and to_print:
        model.load_state_dict(torch.load('checkpoints/model_saved.pth', map_location=device))
    
    with torch.no_grad():
        for i in range(0, len(data), 60):
            d = data[i:i+60]
            l = labels[i:i+60]
            d = np.expand_dims(d, axis=0)
            v = torch.from_numpy(d[:, :, :35]).float().to(device)
            a = torch.from_numpy(d[:, :, 35:]).float().to(device)
            y = torch.from_numpy(l).float().to(device)

            y_tilde = model(v, a)
            loss = criterion(y_tilde, y)

            eval_loss.append(loss.item())
            preds = y_tilde.detach().cpu().numpy()
            y_trues = y.detach().cpu().numpy()
            
            for j in range(len(preds)):
                pred = np.argmax(preds[j])
                y_true = np.argmax(y_trues[j])
                if pred == y_true:
                    corr += 1        

    eval_loss = np.mean(eval_loss)
    accuracy = corr / len(labels)
    return eval_loss, accuracy

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path')
    args = vars(ap.parse_args())

    data, labels = preprocess(args['data_path'])
    
    new_labels = np.zeros((labels.shape[0], np.unique(labels.astype(int)).size))
    for i in range(len(labels)):
        new_labels[i, labels[i]] = 1
    labels = new_labels

    test_data, test_labels = data[-181:-1], labels[-181:-1]
    data, labels = data[:-180], labels[:-180]
    train_data, train_labels = data[:1020], labels[:1020]
    dev_data, dev_labels = data[1020:], labels[1020:]

    # Set device based on CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(336)
    np.random.seed(336)

    model = AudioVisualModel().to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = nn.MSELoss(reduction="mean")
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

    patience, num_trials = 6, 1
    best_valid_loss = float('inf')
    curr_patience = patience

    for e in range(50):
        model.train()
        train_loss = []

        for i in range(0, len(train_data), 60):
            data_batch = train_data[i:i+60]
            label_batch = train_labels[i:i+60]

            data_batch = np.expand_dims(data_batch, axis=0)
            v = torch.from_numpy(data_batch[:, :, :35]).float().to(device)
            a = torch.from_numpy(data_batch[:, :, 35:]).float().to(device)
            y = torch.from_numpy(label_batch).float().to(device)

            optimizer.zero_grad()
            y_tilde = model(v, a)
            loss = criterion(y_tilde, y)

            loss.backward()
            torch.nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad], 1.0)
            optimizer.step()

            train_loss.append(loss.item())

        print(f"Training loss: {round(np.mean(train_loss), 4)}")
        
        valid_loss, valid_acc = eval(dev_data, dev_labels, model, criterion, device, mode="dev")
        print(f'valid_loss: {valid_loss:.4f}, valid_acc: {valid_acc:.4f}')
        
        if valid_loss <= best_valid_loss:
            best_valid_loss = valid_loss
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')
            torch.save(model.state_dict(), 'checkpoints/model_saved.pth')
            torch.save(optimizer.state_dict(), 'checkpoints/optim_saved.pth')
            curr_patience = patience
        else:
            curr_patience -= 1
            if curr_patience <= 0:
                model.load_state_dict(torch.load('checkpoints/model_saved.pth', map_location=device))
                optimizer.load_state_dict(torch.load('checkpoints/optim_saved.pth'))
                lr_scheduler.step()
                curr_patience = patience
                num_trials -= 1
                print(f"Current learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}")
        
        if num_trials <= 0:
            print("Early stopping.")
            break

    test_loss, test_acc = eval(test_data, test_labels, model, criterion, device, mode="test", to_print=True)
    print(f'test_loss: {test_loss:.4f} test_acc: {test_acc:.4f}')
