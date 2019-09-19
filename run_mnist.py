import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import torchvision.datasets
import torchvision.transforms as T
import numpy as np
import random
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
import torch.optim as optim
import gzip
from sklearn.utils import shuffle
import itertools
import os


class trial_net(nn.Module):
    def __init__(self):
        super(trial_net,self).__init__()
        
        self.lay1 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=32,kernel_size=4,padding=0,stride=1),
                                 nn.ReLU(),
                                 nn.BatchNorm2d(32),
                                 nn.Conv2d(in_channels=32,out_channels=32,kernel_size=4,padding=0,stride=1),
                                 nn.ReLU(),
                                 nn.BatchNorm2d(32),
                                 nn.MaxPool2d(kernel_size=2,padding=0,stride=1),
                                 nn.Dropout(0.25))
        self.lay2 = nn.Sequential(nn.Conv2d(in_channels=32,out_channels=32,kernel_size=4,padding=0,stride=1),
                                 nn.ReLU(),
                                 nn.BatchNorm2d(32),
                                 nn.Conv2d(in_channels=32,out_channels=32,kernel_size=4,padding=0,stride=1),
                                 nn.ReLU(),
                                 nn.BatchNorm2d(32),
                                 nn.MaxPool2d(kernel_size=2,padding=0,stride=1),
                                 nn.Dropout(0.25))
        self.num_fea = 32*14*14
        self.fc1 = nn.Sequential(nn.Linear(in_features=self.num_fea,out_features=128,bias=True),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(in_features=128,out_features=10,bias=False))
    def forward(self, x):
        x = self.lay1(x)
        x = self.lay2(x)
        x = x.view(-1,self.num_fea)
        x = self.fc1(x)
        return F.softmax(x, dim=1)


def ini_train(L,m,X,y):
    X_fog,y_fog=np.zeros((len(L)*m,1,28,28),dtype=float),np.zeros(len(L)*m,dtype=int)
    for l in L:
        X_fog[l*m:(l+1)*m,:,:,:]=X[y==l][:m,:,:,:]
        y_fog[l*m:(l+1)*m]=y[y==l][:m]
    return X_fog,y_fog  

def entropy(acquisition_iterations,X_Pool,y_Pool,pool_subset,dropout_iterations,
                nb_classes,Queries,X_test,y_test,rep,device):
        mod = trial_net().to(device)
        cp = torch.load(rep)
        mod.load_state_dict(cp['model_state_dict'])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(mod.parameters(), lr=0.001,weight_decay=0.5)
        optimizer.load_state_dict(cp['optimizer_state_dict'])
        X_train = np.empty([0,1,28,28])
        y_train = np.empty([0,])
        AA = []
        losses_train = []
        losses_test = []
        mod.eval()
        X_va = torch.from_numpy(X_test).to(device)
        Y_va = torch.from_numpy(y_test).long().to(device)
        output= mod(X_va)
        losses_test.append(criterion(output,Y_va).item())
        preds = torch.max(output,1)[1]
        acc = accuracy_score(Y_va,preds)
        AA.append(acc)  
        print('initial test accuracy: ',acc)
        for i in range(acquisition_iterations):
            pool_subset_dropout = np.asarray(random.sample(range(0,X_Pool.shape[0]), pool_subset))
            X_Pool_Dropout = X_Pool[pool_subset_dropout, :, :, :]
            y_Pool_Dropout = y_Pool[pool_subset_dropout]

            score_All = np.zeros(shape=(pool_subset, nb_classes))
            All_Entropy_Dropout = np.zeros(shape=X_Pool_Dropout.shape[0])
            
            for d in range(dropout_iterations):
                X_Pool_Dropout_ = torch.from_numpy(X_Pool_Dropout).float().to(device)
                dropout_score = mod(X_Pool_Dropout_)
                dropout_score =dropout_score.detach()
                score_All = score_All + dropout_score.data.numpy()

            Avg_Pi = np.divide(score_All, dropout_iterations)
            Log_Avg_Pi = np.log2(Avg_Pi+1e-5)
            Entropy_Avg_Pi = - np.multiply(Avg_Pi, Log_Avg_Pi)
            Entropy_Average_Pi = Entropy_Avg_Pi.sum(axis=1)

            U_X = Entropy_Average_Pi
            a_1d = U_X.flatten()
            x_pool_index = (a_1d).argsort()[-Queries:][::-1]

            Pooled_X = X_Pool_Dropout[x_pool_index, :,:,:]
            Pooled_Y = y_Pool_Dropout[x_pool_index]

            delete_Pool_X = np.delete(X_Pool, (pool_subset_dropout), axis=0)
            delete_Pool_Y = np.delete(y_Pool, (pool_subset_dropout), axis=0)

            delete_Pool_X_Dropout = np.delete(X_Pool_Dropout, (x_pool_index), axis=0)
            delete_Pool_Y_Dropout = np.delete(y_Pool_Dropout, (x_pool_index), axis=0)

            X_Pool = np.concatenate((delete_Pool_X, delete_Pool_X_Dropout), axis=0)
            y_Pool = np.concatenate((delete_Pool_Y, delete_Pool_Y_Dropout), axis=0)
            #print('updated pool size is ',X_Pool.shape[0])
            # print("new arrival size", Pooled_X.shape[0])
            X_train = np.concatenate((X_train, Pooled_X), axis=0)
            y_train = np.concatenate((y_train, Pooled_Y), axis=0)
            # print('number of data points from pool',X_train.shape[0])

            for u in range(30):
                mod.train()
                X_tra = torch.from_numpy(Pooled_X).float().to(device)
                Y_tra = torch.from_numpy(Pooled_Y).long().to(device)
                optimizer.zero_grad()
                y_out = mod(X_tra)
                lloss = criterion(y_out,Y_tra)
                lloss.backward()
                optimizer.step()
            losses_train.append(lloss.item())

            mod.eval()
            X_va = torch.from_numpy(X_test).to(device)
            Y_va = torch.from_numpy(y_test).long().to(device)
            output= mod(X_va)
            losses_test.append(criterion(output,Y_va).item())
            preds = torch.max(output,1)[1]
            acc = accuracy_score(Y_va,preds)  
            print('test accuracy: ',acc)
            AA.append(acc)
        return AA,mod,X_train,y_train,losses_train,losses_test
def random_run(acquisition_iterations,X_Pool,y_Pool,pool_subset,dropout_iterations,
                nb_classes,Queries,X_test,y_test,rep,device):
        mod = trial_net().to(device)
        cp = torch.load(rep)
        optimizer = optim.Adam(mod.parameters(), lr=0.001,weight_decay=0.5)
        mod.load_state_dict(cp['model_state_dict'])
        criterion = nn.CrossEntropyLoss()
        optimizer.load_state_dict(cp['optimizer_state_dict'])
        X_train = np.empty([0,1,28,28])
        y_train = np.empty([0,])
        AA = []
        losses_train = []
        losses_test = []
        mod.eval()
        X_va = torch.from_numpy(X_test).to(device)
        Y_va = torch.from_numpy(y_test).long().to(device)
        output= mod(X_va)
        losses_test.append(criterion(output,Y_va).item())
        preds = torch.max(output,1)[1]
        acc = accuracy_score(Y_va,preds)  
        AA.append(acc) 
        print('initial test accuracy: ',acc)
        for i in range(acquisition_iterations):
            pool_subset_dropout = np.asarray(random.sample(range(0,X_Pool.shape[0]), pool_subset))
            X_Pool_Dropout = X_Pool[pool_subset_dropout, :, :, :]
            y_Pool_Dropout = y_Pool[pool_subset_dropout]
            
            x_pool_index = np.random.choice(X_Pool_Dropout.shape[0], Queries, replace=False)
            Pooled_X = X_Pool_Dropout[x_pool_index, :,:,:]
            Pooled_Y = y_Pool_Dropout[x_pool_index]

            delete_Pool_X = np.delete(X_Pool, (pool_subset_dropout), axis=0)
            delete_Pool_Y = np.delete(y_Pool, (pool_subset_dropout), axis=0)

            delete_Pool_X_Dropout = np.delete(X_Pool_Dropout, (x_pool_index), axis=0)
            delete_Pool_Y_Dropout = np.delete(y_Pool_Dropout, (x_pool_index), axis=0)

            X_Pool = np.concatenate((delete_Pool_X, delete_Pool_X_Dropout), axis=0)
            y_Pool = np.concatenate((delete_Pool_Y, delete_Pool_Y_Dropout), axis=0)


            X_train = np.concatenate((X_train, Pooled_X), axis=0)
            y_train = np.concatenate((y_train, Pooled_Y), axis=0)
        
            # print("new arrival size ",Pooled_X.shape[0])
            for u in range(30):
                mod.train()
                X_tra = torch.from_numpy(Pooled_X).float().to(device)
                Y_tra = torch.from_numpy(Pooled_Y).long().to(device)
                optimizer.zero_grad()
                y_out = mod(X_tra)
                lloss = criterion(y_out,Y_tra)
                lloss.backward()
                optimizer.step()
            losses_train.append(lloss.item())
            mod.eval()
            X_va = torch.from_numpy(X_test).to(device)
            Y_va = torch.from_numpy(y_test).long().to(device)
            output= mod(X_va)
            losses_test.append(criterion(output,Y_va).item())
            preds = torch.max(output,1)[1]
            acc = accuracy_score(Y_va,preds)  
            print('test accuracy: ',acc)
            AA.append(acc)
        return AA,mod,X_train,y_train,losses_train,losses_test

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data = np.load('data/mnist.npz')
nchannels, rows, cols = 1, 28, 28
# load training and test data
X_tr_All = data['X_train'].astype('float32').reshape((-1, nchannels, rows, cols))
y_tr_All = data['y_train'].astype('int32')
X_ini,y_ini,X_tr,y_tr = X_tr_All[:10000,:],y_tr_All[:10000],X_tr_All[10000:,:],y_tr_All[10000:]
# shuffle the training dataset
X_tr, y_tr = shuffle(X_tr, y_tr)
X_ini,y_ini = shuffle(X_ini,y_ini)
print('training data (edge): ',X_tr.shape)
print('training data (fog): ',X_ini.shape)

X_te = data['X_test'].astype('float32').reshape((-1, nchannels, rows, cols))
y_te = data['y_test'].astype('int32')
print('total test data: ',X_te.shape)

L=np.unique(y_ini) 
M = [10,20,30,40,50]
Q = [10,20,30,40]#10,20,30,
# aq = acquisition()
# entropy = aq.entropy()
# random_run = aq.random_run()
T1 = []
T2 = []
batch_size = 50
epoch =30
s1 = 'first_result'
get_slice = lambda i, size: range(i * size, (i + 1) * size)
if not os.path.exists(os.pat.join("result_new",s1)):
    os.makedirs(os.pat.join("result_new",s1))
if not os.path.exists(os.pat.join("result_new",s1,"model")):
    os.makedirs(os.pat.join("result_new",s1,"model"))


for m in M:
    X_fog,y_fog  = ini_train(L,m,X_ini,y_ini)
    np.save("result_new/"+s1+"/fog_x_"+str(m)+".npy",X_fog)
    np.save("result_new/"+s1+"/fog_y_"+str(m)+".npy",y_fog)
    num_batch = X_fog.shape[0]//batch_size
    print('batch number is ',num_batch)
    myNet = trial_net().to(device)
    optimizer = optim.Adam(myNet.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for i in range(epoch):
        #print('-----epoch '+ str(i)+'-----')
        losses = 0
        myNet.train()
        for j in range(num_batch):
            slce = get_slice(j,batch_size)
            X_fog_ = torch.from_numpy(X_fog[slce]).float().to(device)
            y_fog_ = torch.from_numpy(y_fog[slce]).long().to(device)
            optimizer.zero_grad()
            out = myNet(X_fog_)
            train_loss = criterion(out,y_fog_)
            losses += train_loss
            train_loss.backward()
            optimizer.step()
        # print('traning loss is',losses.item()/num_batch)
        if i+1 == epoch:
            torch.save({
                'epoch': i,
                'model_state_dict': myNet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss.item(),}, "result_new/"+s1+"/model/{}perclass_{}epoch.net".format(m,i+1))


for q,m in itertools.product(Q, M):
    
    rep = "result_new/"+s1+"/model/"+str(m)+"perclass_"+str(epoch)+"epoch.net"
    acquisition_iterations = 10
    X_Pool = X_tr[:10000,:,:,:]
    y_Pool = y_tr[:10000]
    print(X_Pool.shape)
    pool_subset = 200
    dropout_iterations = 40
    nb_classes = 10
    Queries = q

#--------entropy--------
    time_start_en = time.clock()
    A_en,mod_en,X_train_en,y_train_en,losses_tr_en,losses_te_en = entropy(acquisition_iterations,X_Pool,y_Pool,pool_subset,
        dropout_iterations,nb_classes,Queries,X_te,y_te,rep,device)
    time_elapsed_en = (time.clock() - time_start_en)
    T1.append(time_elapsed_en)
    print("entropy active learning time ",time_elapsed_en)
    np.savetxt("result_new/"+s1+"/entropy_accuracy_"+str(m)+"_ini_"+str(Queries)+"query.txt",A_en)

#--------random--------
    time_start_ran = time.clock()
    A_ran,mod_ran,X_train_ran,y_train_ran,losses_tr_ran,losses_te_ran = random_run(acquisition_iterations,X_Pool,y_Pool,pool_subset,dropout_iterations,
                                     nb_classes,Queries,X_te,y_te,rep,device)
    time_elapsed_ran = (time.clock() - time_start_ran)
    T2.append(time_elapsed_ran)
    print("randomness time ",time_elapsed_ran)
    np.savetxt("result_new/"+s1+"/random_accuracy_"+str(m)+'_ini_'+str(Queries)+"query.txt",A_ran)

#---------bald---------
    # time_start_bald = time.clock()
    # A_bald,mod_bald,X_train_bald,y_train_bald,losses_tr_bald,losses_te_bald = bald(acquisition_iterations,X_Pool,y_Pool,pool_subset,dropout_iterations,
    #                                  nb_classes,Queries,X_te,y_te,rep)
    # time_elapsed_bald = (time.clock() - time_start_bald)
    # print("bald time ",time_elapsed_bald)
    # np.savetxt("result_new/bald_accuracy_"+str(m)+'_ini_'+str(Queries)+"query.txt",A_bald)
np.savetxt("result_new/"+s1+"/entropytime.txt",T1)
np.savetxt("result_new/"+s1+"/randomtime.txt",T2)