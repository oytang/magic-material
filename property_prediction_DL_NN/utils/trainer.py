import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score



def evalue_metrics(pred, label):
    eva_metric = []
    for i in range(pred.shape[1]):
        mae = mean_absolute_error(label[:,i], pred[:,i])
        r2 = r2_score(label[:,i], pred[:,i])
        rmse = np.sqrt(((label[:, i] - pred[:,i]) ** 2).mean())
        pccs = pearsonr(label[:, i], pred[:,i])[0]
        mae, r2, rmse, pccs = np.around([mae, r2, rmse, pccs], decimals=4)
        eva_metric.append([mae, r2, rmse, pccs])
    eva_metric = np.array(eva_metric)
    mean_mae, mean_r2, mean_rmse, mean_pccs = eva_metric.mean(0)[0], eva_metric.mean(0)[1], eva_metric.mean(0)[2], eva_metric.mean(0)[3]
    return [mean_mae, mean_r2, mean_rmse, mean_pccs]
        

def evaluate(model, valid_dataloader, criterion, device, use_cat):
    model.eval()
    with torch.no_grad():
        all_label = []
        all_pred = []
        loss_sum = 0
        for batch, (x, y) in enumerate(valid_dataloader):
            x, y = x.to(device), y.to(device)
            if use_cat:
                y_pred = model(x[:,:-12], x[:,-12:].int())
            else:
                y_pred = model(x, None)
            loss = criterion(y_pred, y)
            all_pred.append(y_pred.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())

            loss_sum += loss.item()

        all_pred = np.concatenate(all_pred, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        mean_values = evalue_metrics(all_pred, all_label)
        
        return loss_sum, mean_values
    

def trainer(model, train_dataloader, valid_dataloader, device, max_epochs=2000, early_stop=10, use_cat=True):
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)
    
    best_valid_mae = 99999
    stop_num = 0
    best_model = model
    
    for epoch in range(1, max_epochs + 1):
    
        all_label = []
        all_pred = []
        loss_sum = 0

        for batch, (x, y) in enumerate(train_dataloader):
            x, y = x.to(device), y.to(device)
            if use_cat:
                y_pred = model(x[:,:-12], x[:,-12:].int())
            else:
                y_pred = model(x, None)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item()
            all_pred.append(y_pred.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
            

        all_pred = np.concatenate(all_pred, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        train_metric = evalue_metrics(all_pred, all_label)
        
        print('epoch: {:>4d} | train_loss: {:5.4f} | train_mae: {:5.4f} | train_r2: {:5.4f} | train_rmse: {:5.4f} | train_pccs: {:5.4f}'.format(\
               epoch, loss_sum, train_metric[0], train_metric[1], train_metric[2], train_metric[3]) )

        
        valid_loss_sum, valid_metric = evaluate(model, valid_dataloader, criterion, device, use_cat)
        print('epoch: {:>4d} | valid_loss: {:5.4f} | valid_mae: {:5.4f} | valid_r2: {:5.4f} | valid_rmse: {:5.4f} | valid_pccs: {:5.4f}'.format(\
               epoch, valid_loss_sum, valid_metric[0], valid_metric[1], valid_metric[2], valid_metric[3]) )
        now_mae = valid_metric[0]
        if now_mae < best_valid_mae:
            best_valid_mae = now_mae
            stop_num = 0
            best_model = model
        else:
            stop_num += 1
            if stop_num >= early_stop:
                print(f'!!!!!!!!!!! early stop at epoch{epoch} !!!!!!!!!!!!!!!!!')
                valid_loss_sum, valid_metric = evaluate(best_model, valid_dataloader, criterion, device, use_cat)
                print('Final valid score:')
                print('epoch: {:>4d} | valid_loss: {:5.4f} | valid_mae: {:5.4f} | valid_r2: {:5.4f} | valid_rmse: {:5.4f} | valid_pccs: {:5.4f}'.format(\
                       epoch, valid_loss_sum, valid_metric[0], valid_metric[1], valid_metric[2], valid_metric[3]) )
                break
    return best_model