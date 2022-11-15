#basic libary
import copy
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.tensorboard import SummaryWriter

class Matrix(nn.Module):## Model for task 1
    def __init__(self, row_d, col_d,rto=0.01):
        super(Matrix, self).__init__()
        self.lin_mat = nn.Parameter(rto*torch.randn(row_d, col_d, requires_grad=True))

    def forward(self,x):
        return torch.matmul(self.lin_mat, x)

def subspace_error(U,V):
    with torch.no_grad():    
        pu = U@torch.inverse(U.T@U)@U.T
    
        #pu = U.lin_mat@torch.inverse(U.lin_mat.T@U.lin_mat)@U.lin_mat.T
        pv = V.lin_mat@torch.inverse(V.lin_mat.T@V.lin_mat)@V.lin_mat.T
        return torch.norm(pu-pv).item()

def soft(z, lam):  
    with torch.no_grad():
        m=nn.Threshold(lam,0)   
        res = torch.sign(z)*m(torch.abs(z)-lam) 
        z *= 0
        z += res
        return res

def add_sparse_noise(Y_list,p,amplitude=1):
    if isinstance(Y_list, list):
        for i in range(len(Y_list)):
            (n1,n2) = Y_list[i].shape
            noisei = np.random.choice(3, n1*n2, p=[p/2,1-p,p/2])-1
            noisei = np.reshape(noisei,(n1,n2))*amplitude
            Y_list[i] += torch.tensor(noisei)
        return Y_list
    else:
        (n1,n2) = Y_list.shape
        noisei = np.random.choice(3, n1*n2, p=[p/2,1-p,p/2])-1
        noisei = np.reshape(noisei,(n1,n2))*amplitude
        Y_list += torch.tensor(noisei)
        return Y_list


def lg_matrix_factorization_projgd(Y,args):
    if isinstance(Y, list):
        N = len(Y)
        alliters = list(range(N))
    else:
        alliters = Y.keys()
        N = len(alliters)
    n2dict = {}
    for y in alliters:
        (n1,n2dict[y]) = Y[y].shape
        

    
    Ug = {k:Matrix(n1,args["ngc"]) for k in alliters}
    Ug_avg = Matrix(n1,args["ngc"])
    Vg = {k:Matrix(n2dict[k],args["ngc"]) for k in alliters}
    Ul = {k:Matrix(n1,args["nlc"]) for k in alliters}
    Vl = {k:Matrix(n2dict[k],args["nlc"]) for k in alliters}
    parlist = {i:list(Ug[i].parameters())+list(Vg[i].parameters())+   list(Ul[i].parameters())+list(Vl[i].parameters())+list(S[i].parameters()) for i in alliters}
    if args["optim"] == "SGD":
        optim = {k:torch.optim.SGD(parlist[k], lr=args["lr"], weight_decay=args["wd"]) for k in alliters}
    else:
        optim = {k:torch.optim.Adam(parlist[k], lr=args["lr"]) for k in alliters}
        scheduler = {k:MyReduceLROnPlateau(optim[k], verbose=False, patience=1) for k in alliters}
    if "tensorboard" in args.keys():
        writer = SummaryWriter()
    
    for n in range(args["epochs"]):
        time_start = time.time() 
        
        tot_loss = 0
        tot_reg = 0
        for i in alliters:
            pred = Ug[i].lin_mat@Vg[i].lin_mat.T+ Ul[i].lin_mat@Vl[i].lin_mat.T 
           
            lossi = nn.MSELoss()(pred,Y[i])*n1*n2dict[i]
            optim[i].zero_grad()
            lossi.backward()
            optim[i].step()
            if args["optim"] == "Adam":
                scheduler[i].step()
            tot_loss += lossi.item()

          
        with torch.no_grad():
            Ug_avg.lin_mat *= 0
            Ug_avg.lin_mat += sum([Ug[i].lin_mat for i in alliters])/N
            projection = Ug_avg.lin_mat@torch.inverse(Ug_avg.lin_mat.T@Ug_avg.lin_mat)@Ug_avg.lin_mat.T
            for i in alliters:
                Ug[i].lin_mat *= 0
                Ug[i].lin_mat += Ug_avg.lin_mat
                Ul[i].lin_mat -= projection@Ul[i].lin_mat

   
        tot_loss /= N
        tot_reg /= N
        output = "[%s/%s], loss %s"%(n,args["epochs"],tot_loss,)
        if "global_subspace_err_metric" in args.keys():
            output += " gserr %s "%args["global_subspace_err_metric"](Ug_avg)
        if "local_subspace_err_metric" in args.keys():
            output += " lserr %s"%args["local_subspace_err_metric"](Ul)
        time_end = time.time()

        print(output+", time %s"%(time_end-time_start))
        if "tensorboard" in args.keys():
            writer.add_scalar("Loss/train", tot_loss, n)
            writer.add_scalar("Global error", args["global_subspace_err_metric"](Ug_avg), n)
            writer.add_scalar("Local error", args["local_subspace_err_metric"](Ul), n)
            writer.flush()
  
    return Ug, Vg, Ul, Vl


def cluster_plot(X):
    from sklearn.datasets import load_digits
    from sklearn.manifold import MDS
    
    embedding = MDS(n_components=2)
    X_transformed = embedding.fit_transform(X)
    print(X_transformed.shape)
    plt.scatter(X_transformed[:,0],X_transformed[:,1],s=0.4)
    plt.savefig('user_embeddings.png')
    #(100, 2)