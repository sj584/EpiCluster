import pickle
with open(f"./example_pyg_surface_{RSA}.pkl", "wb") as f:
    pickle.dump(example_pyg_list, f)

import numpy as np
import time
import copy
import random
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn import metrics

from models.model import JointModel
from models.KNN import KNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_start = time.time()
epochs = 400
batch_size = 8
data = example_pyg_list

k = 15
self_w = 0.25
directory = "Training"

random.seed(42)
random.shuffle(data)

kfold = 5
X = data
kf = KFold(n_splits=kfold) ## shuffle, random_state edited. not tested yet!
kf.get_n_splits(X)
print(kf)

loss_fn = nn.BCELoss()
    
for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
    
    model = JointModel(node1_feat=83, n_edge_feat=1, n_hidden_feat1=64, n_hidden_feat2=128, device=device, attention=True, n_layer1=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-8)
    
    best_val_loss = 100000000
    best_auc_roc = 0
    train_loss_history = []
    val_loss_history = []
    val_auc_history = []
    
    # the dataset cross-validation setup
    
    # dataset looks like this (v, abbreviation for validation set)
    # ex) ㅁㅁㅁ...ㅁㅁ, 
    # 1st fold    vㅁㅁ...ㅁㅁ
    # 2nd fold    ㅁvㅁ...ㅁㅁ
    # kth fold    ㅁㅁㅁ...ㅁv
    
    test_start, test_end = test_index[0], test_index[-1] 
    print("test_index :", test_start,"~", test_end)
    X_test = X[test_start:test_end+1]
    print("len X_test :", len(X_test))
    X_train = list(set(X) - set(X_test))
    print("len X_train :", len(X_train))
    train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(X_test, batch_size=batch_size, shuffle=False)
    
    for epoch in range(epochs):
        since = time.time()
        train_loss_list = []
        model.train()
        model.cuda()
        for batch in train_loader:
            
            optimizer.zero_grad()
            batch.to(device)
            batch.y = batch.y.float()
            knn_out_batch = torch.tensor([]).to("cuda")
            out = model(batch)
            for idx in range(batch_size):
                # since KNN is instance learning, batch learning is not feasible
                # instead, we dissociate the batch and apply KNN algorithm iteratively
                # then we restore the batch size by concatenation
                try:
                    regressor = KNN(k=k, include_self=True, weights="distance", self_weight=self_w)
                    regressor.train(batch[idx].coords[batch[idx].train_mask], out[batch.ptr[idx]:batch.ptr[idx+1]][batch[idx].train_mask])
                    knn_out = regressor.predict(batch[idx].coords[batch[idx].train_mask])
                    knn_out_batch = torch.concat((knn_out_batch, knn_out))
                # the last batch may not have full batch size, in this case, just make the batch with size as (number of dataset % batch_size)
                except:
                    pass

            loss = loss_fn(knn_out_batch.reshape(-1), batch.y[batch.train_mask]) # out[batch.train_mask].reshape(-1)
            loss.backward()
            train_loss_list.append(copy.deepcopy(loss.data.cpu().numpy()))

            optimizer.step()
        epoch_train_avg_loss = np.sum(np.array(train_loss_list))/len(train_loader) 
        train_loss_history.append(epoch_train_avg_loss)

        model.eval()
        val_loss_list = []
        val_auc_list = []
        total_pred_label_list = []
        total_true_label_list = []
        with torch.no_grad():
            for batch in test_loader:
                batch.to(device)
                batch.y = batch.y.float()
                knn_out_batch = torch.tensor([]).to("cuda")
                out = model(batch)
                for idx in range(batch_size):
                # same as above
                    try:
                        regressor = KNN(k=k, include_self=True, weights="distance", self_weight=self_w)
                        regressor.train(batch[idx].coords[batch[idx].train_mask], out[batch.ptr[idx]:batch.ptr[idx+1]][batch[idx].train_mask])
                        knn_out = regressor.predict(batch[idx].coords[batch[idx].train_mask])
                        knn_out_batch = torch.concat((knn_out_batch, knn_out))
                    except:
                        pass
                    
                loss = loss_fn(knn_out_batch.reshape(-1), batch.y[batch.train_mask])

                pred_label_list = []
                for i in knn_out_batch:
                    pred_label_list.append(i.cpu().numpy())

                true_label_list = []
                for j in batch.y[batch.train_mask]:
                    true_label_list.append(j.cpu().numpy())

                fpr, tpr, threshold = metrics.roc_curve(true_label_list, pred_label_list)
                roc_auc = metrics.auc(fpr, tpr)

                val_auc_list.append(copy.deepcopy(roc_auc))
                val_loss_list.append(copy.deepcopy(loss.data.cpu().numpy()))


        epoch_val_avg_loss = np.sum(np.array(val_loss_list))/len(test_loader) 
        val_loss_history.append(epoch_val_avg_loss)

        epoch_val_auc_roc =  np.sum(np.array(val_auc_list))/len(test_loader)    
        val_auc_history.append(epoch_val_avg_loss)

        
        if (epoch_val_avg_loss < best_val_loss) & (epoch_val_auc_roc > best_auc_roc): 
            best_epoch = epoch+1
            best_val_loss = epoch_val_avg_loss
            best_auc_roc = epoch_val_auc_roc
            best_model_wts = copy.deepcopy(model.state_dict())

        end = time.time()
        print(f'{epoch+1}th epoch,')
        print(f'\ttraining loss: {epoch_train_avg_loss:.5f}')
        print(f'\tval loss: {epoch_val_avg_loss:.5f}')
        print(f'\tval auc-roc: {epoch_val_auc_roc:.5f}')
        print(f'\tepoch time: {end-since:.3f}')



    save_dir = directory
    import os
    if os.path.isdir(save_dir):
        pass
    else:
        os.mkdir(save_dir)
    torch.save(best_model_wts, f'{save_dir}/Best_model_{best_epoch}_{fold_idx}.pt')
    print('-'*10)
    print('Train Finished.')
    print(f'Training time: {time.time()-train_start:.2f}s')
    print(f'The best epoch: {best_epoch}')
    print(f'The best val loss: {best_val_loss}')
    print(f"The best auc-roc: {best_auc_roc}")
