import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchmetrics import ConfusionMatrix
from torch.autograd import Variable
import pickle
import itertools
import time
from myoptim import MyReduceLROnPlateau


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        #if isinstance(alpha,(float,int,torch.long)): self.alpha = torch.Tensor([alpha,1-alpha])
        #if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class QSRData(Dataset):
    def __init__(self,  setx, sety):
        # #data x # dimension
        self.setx = setx.float()
        self.sety = sety.long()

    def __getitem__(self, index):
        return self.setx[index], self.sety[index]

    def __len__(self):
        return len(self.setx)

class Net(nn.Module):
    def __init__(self, indim=48,hiddendim=564,expandfac=32):
        super().__init__()  
        #self.bn0 = nn.BatchNorm1d(indim)
        self.fc1 = nn.Linear(indim, indim*expandfac)
        #self.bn1 = nn.BatchNorm1d(indim*2)
        self.fc2 = nn.Linear(indim*expandfac, hiddendim)
        #self.bn2 = nn.BatchNorm1d(hiddendim)
        self.fc3 = nn.Linear(hiddendim, hiddendim)
        self.fc31 = nn.Linear(hiddendim, hiddendim)

        self.fc32 = nn.Linear(hiddendim, hiddendim)

        self.fc4 = nn.Linear(hiddendim, 2)

    def forward(self, x):
        x0 = F.relu(self.fc1(x))
        #x = self.
        x = F.relu(self.fc2(x0))#+x0
        #x = F.relu(self.fc3(x))

        x = F.relu(self.fc3(x))+x
        x = F.relu(self.fc31(x))+x

        x = F.relu(self.fc32(x))+x

        x = self.fc4(x)
        return x


def replace_coordinate_i(target, reference, i):
    res = target*0
    res += target
    res[i] *= 0
    res[i] += reference[i]
    return res


def calculate_statistics_c_m(cm):
    s1 = cm[1,1]/(cm[1,1]+cm[0,1])
    s2 = cm[1,1]/(cm[1,1]+cm[1,0])
    s3 = s1*s2/(s1+s2)*2
    print("scores are %s, %s, %s"%(s1.item(),s2.item(),s3.item()))
            

def load_data_small():
    REGEN = False#True
    if not REGEN:
        accept_data = pd.read_pickle('qsrdata/fullaccept.pkl')
        reject_data = pd.read_pickle('qsrdata/fullreject.pkl')
        print("data loaded")
        return accept_data, reject_data


    # Load each data set (users, movies, and ratings).
    feature_cols = ['id']+['feature_%s'%i for i in range(56)]
    accept_data = pd.read_csv(
        'qsrdata/accept_sample_competition.csv', sep=',', skiprows=1, names=feature_cols, encoding='latin-1')
    reject_data = pd.read_csv(
        'qsrdata/reject_sample_competition.csv', sep=',', skiprows=1, names=feature_cols, encoding='latin-1')
    
    accept_data.to_pickle('qsrdata/fullaccept.pkl')
    reject_data.to_pickle('qsrdata/fullreject.pkl')

    return accept_data, reject_data

def load_data():
    REGEN = False
    if not REGEN:
        #accept_data = pd.read_pickle('qsrdata/fullaccept.pkl')
        #reject_data = pd.read_pickle('qsrdata/fullreject.pkl')
        accept_dict = pd.read_pickle('qsrdata/acceptdict.pkl')
        reject_dict = pd.read_pickle('qsrdata/rejectdict.pkl')

        with open("qsrdata/alltuples.txt", "rb") as fp:   # Unpickling
            all_tuples = pickle.load(fp)        
        print("data loaded")
        print("accept number %s"%len(accept_dict))
        print("reject number %s"%len(reject_dict))
        print("number of tuples %s"%len(all_tuples)) 
        return accept_dict, reject_dict
    feature_cols = ['id']+['feature_%s'%i for i in range(56)]
    from os import listdir
    from os.path import isfile, join
    accpath = "qsrdata/accept"
    acceptfiles = [join(accpath, f) for f in listdir(accpath) if isfile(join(accpath, f))]
    rejpath = "qsrdata/reject"
    rejectfiles = [join(rejpath, f) for f in listdir(rejpath) if isfile(join(rejpath, f))]

    accept_data = None
    reject_data = None
    accept_dict = dict()
    reject_dict = dict()
    x = 0
    all_tuples = []
    for fl in acceptfiles:
        print("parsing %s"%fl)
        dfi = pd.read_csv(
            fl, sep=',', skiprows=1, names=feature_cols, encoding='latin-1')
        dfi["filename"] = fl
        dfi["feature_48"] = dfi["feature_48"].astype('int')
        dfi["feature_49"] = dfi["feature_49"].astype('int')
        dfi["hashid"] = dfi["feature_48"]*1000000+dfi["feature_49"]
        allhash = dfi["hashid"].unique()
        f48 = allhash // 1000000
        f49 = allhash % 1000000
        for i in range(len(f48)):
            
            all_tuples.append((fl,f49[i],f48[i]))
            accept_dict[(fl,f49[i],f48[i])] = dfi[dfi["hashid"] == allhash[i]]

        if x == 0:
            accept_data = dfi
        else:
            accept_data = pd.concat((accept_data,dfi))
        x+=1
        #break
    x = 0
    for fl in rejectfiles:
        print("parsing %s"%fl)
        dfi = pd.read_csv(
            fl, sep=',', skiprows=1, names=feature_cols, encoding='latin-1')
        dfi["feature_48"] = dfi["feature_48"].astype('int')
        dfi["feature_49"] = dfi["feature_49"].astype('int')

        dfi["filename"] = fl
        dfi["hashid"] = dfi["feature_48"]*1000000+dfi["feature_49"]
        f48 = allhash // 1000000
        f49 = allhash % 1000000
        for i in range(len(f48)):
            
            all_tuples.append((fl,f49[i],f48[i]))
            reject_dict[(fl,f49[i],f48[i])] = dfi[dfi["hashid"] == allhash[i]]

        if x == 0:
            reject_data = dfi
            
        else:
            reject_data = pd.concat((reject_data, dfi))

        x+= 1
        #break
    #print(all_tuples)
    accept_data.to_pickle('qsrdata/fullaccept.pkl')
    reject_data.to_pickle('qsrdata/fullreject.pkl')

    print("accept number %s"%accept_data.shape[0])
    print("reject number %s"%reject_data.shape[0])
    print("data generated")

    f = open("qsrdata/alltuples.txt", "w")
    with open("qsrdata/alltuples.txt", "wb") as f:   #Pickling
        pickle.dump(all_tuples, f)
    with open("qsrdata/acceptdict.pkl", "wb") as f:   #Pickling
        pickle.dump(accept_dict, f)
    with open("qsrdata/rejectdict.pkl", "wb") as f:   #Pickling
        pickle.dump(reject_dict, f)
    
    return accept_dict, reject_dict


def split_dataframe(df, holdout_fraction=0.1):
    """Splits a DataFrame into training and test sets.
    Args:
        df: a dataframe.
        holdout_fraction: fraction of dataframe rows to use in the test set.
    Returns:
        train: dataframe for training
        test: dataframe for testing
    """
    test = df.sample(frac=holdout_fraction, replace=False)
    train = df[~df.index.isin(test.index)]
    return train, test


def split_by_file_phase_step_id(dflist, all_tuples):
    df = pd.concat(dflist)
    df = df.dropna()
    #all_steps = df["feature_48"].unique()
    #all_phases = df["feature_49"].unique()
    #all_files = df["filename"].unique()
    #all_tuples = pd.concat([df["feature_48"],df["feature_49"],df["filename"]], dim=1).unique()

    res_acc = dict()
    res_rej = dict()

    num_dp = []
    #accept_tuple = []
    #reject_tuple = []

    '''
    for step in all_steps:
        for phase in all_phases:
            for filename in all_files:
    '''
    #print(all_tuples)
    for ids,onetp in enumerate(all_tuples):
        #print(ids)
        (step,phase,filename) = onetp
                #print(step,phase)
        print("pt1")
        df2 = df[(df.filename == filename) & (df.feature_48 == step) & (df.feature_49 == phase)]
                #df1 = df0[df0.feature_48 == step]
                #df2 = df1[df1.feature_49 == phase]
        print("pt2")

        if df2.shape[0] < 10:
            continue
                #print(step,phase,df2.shape[0])

                #df40 = df2.iloc[0:round(df2.shape[0] * (1 - nsplit)), :]
        thr = 0.01  # threshold for clipping data
        thresh = df2.quantile([thr, 1 - thr])
        df2 = df2[df2.columns].clip(lower=thresh.loc[thr], upper=thresh.loc[1 - thr], axis=1)
        num_dp.append(df2.shape[0])
        print("pt3")

        if df2.feature_55.sum()>0.1:
            res_rej[(filename,phase,step)]=df2.copy()
                    #reject_tuple.append((filename,phase,step))
        else:
            res_acc[(filename,phase,step)]=df2.copy()
                    #accept_tuple.append((filename,phase,step))
        print("pt4")

    print("%s acc clients and %s rej clients created,"%(len(res_acc.keys()),len(res_rej.keys())))
    num_dp = np.array(num_dp)
    print("average observation %s, std %s"%(num_dp.mean(), num_dp.std()))
    return res_acc, res_rej


def train_test_split_by_file(all_tuples,ptest=0.2):
    print("splitting train and test files")
    file_names = []
    for tp in all_tuples:
        if not tp[0] in file_names:
            file_names.append(tp[0])
    nfile = len(file_names)
    print(nfile)
    file_names = np.array(file_names)
    #print(file_names)
    all_idx = np.arange(nfile)
    np.random.shuffle(all_idx)
    return file_names[all_idx[int(ptest*nfile):]], file_names[all_idx[:int(ptest*nfile)]]

    

def df2tensor(df):
    data = df.iloc[:,1:49].to_numpy()
    data = torch.tensor(data).float()
    #import torch.nn.functional as F
    #dm = torch.mean(data, dim=0)
    #data = (data-dm).T
    #data = F.normalize(data, p=2, dim=1)
    #print(data.shape)
    return data.T

DOT = 'dot'
COSINE = 'cosine'

def compute_scores(query_embedding, item_embeddings, measure=DOT):
    """Computes the scores of the candidates given a query.
    Args:
        query_embedding: a vector of shape [k], representing the query embedding.
        item_embeddings: a matrix of shape [N, k], such that row i is the embedding
        of item i.
        measure: a string specifying the similarity measure to be used. Can be
        either DOT or COSINE.
    Returns:
        scores: a vector of shape [N], such that scores[i] is the score of item i.
    """
    u = query_embedding
    V = item_embeddings
    if measure == COSINE:
        V = V / torch.norm(V, axis=1, keepdims=True)
        u = u / torch.norm(u)
    scores = u@V.T
    return scores



class QSR_Data():
    def __init__(self):
        self.net = Net()
        self.device = "cuda"
        self.confmat = ConfusionMatrix(num_classes=2).to(self.device)
        self.expandnet = Net(48*4)


    def gen_y(self):
        REGEN = False
        if REGEN:
            accept_data, reject_data = load_data()        
            
            self.full_list = accept_data
            self.full_list.update(reject_data)
            
            # remove uncommon observations:
            #self.full_list = {dfi:self.full_list[dfi] for dfi in self.full_list if self.full_list[dfi].shape[0]>100}
            # column normalize
            cpd = pd.concat(list(self.full_list.values()))
            for i in range(48):
                mn = cpd["feature_%s"%i].mean()
                stnd = cpd["feature_%s"%i].std()
                print("normalizing feature %s"%i)
                for ii in self.full_list:
                    self.full_list[ii]["feature_%s"%i] = (self.full_list[ii]["feature_%s"%i]-mn)/stnd
                #cpd["feature_%s"%i]=(cpd["feature_%s"%i] - cpd["feature_%s"%i].mean()) / cpd["feature_%s"%i].std()    
            self.full_list = {dfi:self.full_list[dfi] for dfi in self.full_list if self.full_list[dfi].shape[0]>100}
            
            for ii in self.full_list:                    
                thr = 0.01  # threshold for clipping data
                thresh = self.full_list[ii].quantile([thr, 1 - thr])
                df2 = self.full_list[ii][self.full_list[ii].columns].clip(lower=thresh.loc[thr], upper=thresh.loc[1 - thr], axis=1)
            
            with open("qsrdata/fulllist.pkl", "wb") as f:   #Pickling
                pickle.dump(self.full_list, f)
            '''
            with open("qsrdata/trainfiles.pkl","wb") as f:
                pickle.dump(self.train_files, f)
            with open("qsrdata/testfiles.pkl","wb") as f:
                pickle.dump(self.test_files, f)
            '''
          
        else:
            with open("qsrdata/fulllist.pkl", "rb") as fp:   # Unpickling
                self.full_list = pickle.load(fp)
            '''
            with open("qsrdata/trainfiles.pkl", "rb") as fp:   # Unpickling
                self.train_files = pickle.load(fp)
            with open("qsrdata/testfiles.pkl", "rb") as fp:   # Unpickling
                self.test_files = pickle.load(fp)
            '''
            print("full data loaded")
        
        full_data_list = {dfi:df2tensor(self.full_list[dfi]) for dfi in self.full_list}
        self.train_files, self.test_files = train_test_split_by_file(self.full_list.keys())

        print("Train files %s, test files %s"%(len(self.train_files), len(self.test_files)))
        return full_data_list
        
    def load_root_cause_lookup(self):
        feature_cols = ["Root Cuase code","Signal","Phase","Step"]
        rcdf = pd.read_csv(
        'qsrdata/Expplanatory_files_updated_09062022/Expplanatory_files/Root_cause_look_up_table.csv', sep=',', skiprows=1, names=feature_cols, encoding='latin-1')
        self.code2featurephasestep = dict()
        for i in range(len(rcdf)):
            self.code2featurephasestep[rcdf.loc[i, "Root Cuase code"]]=(rcdf.loc[i, "Signal"], rcdf.loc[i, "Phase"],rcdf.loc[i, "Step"])
        print("root cause lookup table generated with %s codes"%(i+1))
        #print(self.code2featurephasestep)
        self.file2rootcodes = dict()
        for filetp in self.full_list:
            #rc_list = []
            rca_meta = self.full_list[filetp]
            rca_meta_columns = ["feature_%s"%i for i in range(50,55)]
            rc_list = [int(np.max(rca_meta[r_code].unique())) for r_code in rca_meta_columns if int(np.max(rca_meta[r_code].unique()))!=0]
            if not filetp[0] in self.file2rootcodes.keys():
                self.file2rootcodes[filetp[0]] = []

            self.file2rootcodes[filetp[0]] += rc_list
            #if len(rc_list)>0:
            #    print(rc_list)
        print("file to root codes mapping generated")
        #print(self.file2rootcodes)


    def simple_train(self):
        traindata = pd.concat([self.accept_train, self.reject_train])
        trainsetx = torch.tensor(traindata[["feature_%s"%i for i in range(48)]].to_numpy())
        trainsety = torch.tensor(traindata["feature_55" ].to_numpy())
        trainset = QSRData(trainsetx, trainsety)
        print("train set")
        print(trainset.__len__(), traindata.shape[0])

        testdata = pd.concat([self.accept_test, self.reject_test])
        testsetx = torch.tensor(testdata[["feature_%s"%i for i in range(48)]].to_numpy())
        testsety = torch.tensor(testdata["feature_55" ].to_numpy())
        testset = QSRData(testsetx, testsety)

        batch_size = 128
        device = self.device
        self.net.to(device)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

        
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)

        criterion = nn.CrossEntropyLoss()
        #TORCHVISION.OPS.FOCAL_LOSS
        #from torchvision.ops import focal_loss
        #criterion = FocalLoss()
        optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9, weight_decay = 1e-6)

        for epoch in range(200):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)
                # zero the parameter gradients
                self.net.train()
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
            cm = self.test_acc(testloader)
            if True:    # print every 2000 mini-batches
                #print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f} test acc [{a:.3f}, {b:.3f}, {c:.3f}, {d:.3f}, {e:.3f}]')
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                print(cm)
                with torch.no_grad():
                    s1 = cm[0,0]/(cm[0,0]+cm[0,1])
                    s2 = cm[0,0]/(cm[0,0]+cm[1,0])
                    s3 = s1*s2/(s1+s2)*2
                    print("scores are %s, %s, %s"%(s1.item(),s2.item(),s3.item()))

                running_loss = 0.0
  
    def simple_test_acc(sefl, testx, testy, expand=0):
        images, labels = testx.to(self.device), testy.to(self.device)
        # calculate outputs by running images through the network
        if expand == 1:                    
            outputs = self.expandnet(images)
        else:
            outputs = self.net(images)

        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        cm = self.confmat(predicted,labels)
        return cm


    def test_acc(self, testloader, expand=0):
        with torch.no_grad():
            if expand == 1:
                self.expandnet.eval()
            else:
                self.net.eval()
            total = 0
            cm = None
            for data in testloader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                # calculate outputs by running images through the network
                if expand == 1:                    
                    outputs = self.expandnet(images)
                else:
                    outputs = self.net(images)

                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                if cm == None:
                    cm = self.confmat(predicted,labels)
                else:
                    cm += self.confmat(predicted,labels)
                bothone = ((predicted == labels) * (labels == 1)).sum().item()
                onezero = ((1- predicted == labels)*(labels == 1)).sum().item()
                zeroone = ((1- predicted == labels)*(labels == 0)).sum().item()
                zerozero = ((predicted == labels)*(labels == 0)).sum().item()

                #correct += (predicted == labels).sum().item()
            return cm

    def is_the_step_abnormal(self, tpname, expand=0, thresh=0.01):
        with torch.no_grad():
            if expand == 1:
                self.expandnet.eval()
            else:
                self.net.eval()
            idxs = self.test_tuple2idx[tpname]
            inputdata = self.fullx[idxs]
            outputdata = self.fully[idxs]            
            testset = QSRData(inputdata, outputdata)
            batch_size = 128

            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)
            tottimestep = 0
            totones = 0
            for data in testloader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                # calculate outputs by running images through the network
                if expand == 1:                    
                    outputs = self.expandnet(images)
                else:
                    outputs = self.net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                totones += predicted.sum()
                tottimestep += len(predicted)
        return totones/tottimestep > thresh + 0.0

    def is_the_file_abnormal(self,filename,expand=0):
        abnormal_dict = dict()
        for filetuple in self.test_tuple2idx:
            if not filename == filetuple[0]:
                continue
            tpabnormal = self.is_the_step_abnormal(filetuple, expand=expand)
            if tpabnormal > 0.5:
                abnormal_dict[filetuple] = 1
                allfeatures = self.fullx[self.test_tuple2idx[filetuple]]
                meanfeatures = torch.mean(allfeatures, dim=0)
                print(filetuple)
                print("predicted causes")
                self.abnormal_neighbors(meanfeatures)
                print("actual causes")

                
        return abnormal_dict

    def file_test_performance(self, expand=0):
        with torch.no_grad():
            cm = torch.zeros(2,2)
            for filename in self.test_files:
                abnormal_dict = self.is_the_file_abnormal(filename,expand=expand)
                predict = 0
                if len(abnormal_dict)>0:
                    predict = 1
                if "reject" in filename:
                    truth = 1
                else: 
                    truth = 0
                cm[truth, predict] += 1
        return cm

    def _set_uvs(self,Ug,Vg,Ul,Vl,S):
        self.Ug = Ug
        self.Vg = Vg
        self.Ul = Ul
        self.Vl = Vl
        self.S = S

    def _gen_anomaly_score(self):
        normal_s = []
        abnormal_s = []

        n_feature1 = []
        a_feature1 = []
        n_feature2 = []
        a_feature2 = []
        filedict = dict()
        with torch.no_grad():
            for filename in self.train_files:
                if not filename in filedict:
                    isabnormal = int('reject' in filename)
                    filedict[filename] = [isabnormal, 0, 0]
                for fid in self.full_list:
                    if not fid[0] == filename:
                        continue
                    if fid[1] < 0.1 or fid[2]< 0.1:
                        continue
                    feature0 = torch.sum(torch.abs(self.S[fid].lin_mat),dim=0)
                    feature1,idx = torch.max(torch.abs(self.S[fid].lin_mat),dim=0)
                    feature2 = torch.mean((self.Ul[fid].lin_mat@self.Vl[fid].lin_mat.T)**2,dim=0)
                    feature3, idx = torch.max((self.Ul[fid].lin_mat@self.Vl[fid].lin_mat.T)**2,dim=0)
                    feature4 = torch.mean((self.Ug[fid].lin_mat@self.Vg[fid].lin_mat.T)**2,dim=0)

                    is_abs = self.full_list[fid]["feature_55"].to_numpy()
                    agf0 = torch.max(feature0).item()
                    agf2 = torch.max(feature2).item()
                    if agf0 > filedict[filename][1]:
                        filedict[filename][1] = agf0
                    if agf2 > filedict[filename][2]:
                        filedict[filename][2] = agf2
                    '''
                    for j in range(len(is_abs)):
                        if is_abs[j] == 0:
                            normal_s.append(feature1[j].item())
                            n_feature1.append(feature2[j].item())
                        else:
                            abnormal_s.append(feature1[j].item())
                            a_feature1.append(feature2[j].item())
                    '''
        alldata = list(filedict.values())
        nxlist = [data[1] for data in alldata if data[0]==0]
        nylist = [data[2] for data in alldata if data[0]==0]

        abnxlist = [data[1] for data in alldata if data[0]==1]
        abnylist = [data[2] for data in alldata if data[0]==1]

        plt.clf()
        nxlist = np.array(nxlist)
        nylist = np.array(nylist)
        abnxlist = np.array(abnxlist)
        abnylist = np.array(abnylist)
        #plt.scatter(np.ones(n_normal),normal_s,s=0.4,color="blue")
        #plt.scatter(np.ones(n_abnormal)+1,abnormal_s,s=0.4,color="red")

        plt.scatter(abnxlist,abnylist,s=0.9,color="red")
        plt.scatter(nxlist,nylist,s=0.9,alpha=0.9,color="blue")

        plt.savefig('normal_plot.png',bbox_inches='tight')
    

    def _gen_expanded_score(self):
      
        totlen = 0
        for fid in self.full_list:
            if fid[1]< 0.5 or fid[2] < 0.5:
                    continue
            totlen += self.full_list[fid].shape[0]
        fullx = torch.zeros(totlen, 48*4)
        fully = torch.zeros(totlen)
        fullhash = torch.zeros(totlen)
        fullhash = fullhash.long()


        self.train_tuple2idx = dict()
        self.test_tuple2idx = dict()


        train_ids = []
        test_ids = [] 
        
        with torch.no_grad():
            sttid = 0
            endid = 0
            print("processing augmented features")
           
            for fid in self.full_list:
                if fid[1]< 0.5 or fid[2] < 0.5:
                    continue
                #print(fid)
                original_feature = df2tensor(self.full_list[fid]).T
                feature_sparse = torch.abs(self.S[fid].lin_mat).T
                #feature1,idx = torch.max(torch.abs(self.S[id].lin_mat),dim=0)
                #print(feature_sparse.shape)
                feature_local = (self.Ul[fid].lin_mat@self.Vl[fid].lin_mat.T).T
                #feature3, idx = torch.max((self.Ul[id].lin_mat@self.Vl[id].lin_mat.T)**2,dim=0)
                feature_global = (self.Ug[fid].lin_mat@self.Vg[fid].lin_mat.T).T
                #print(feature_local.shape)
                #print(feature_global.shape)

                augmented_feature = torch.cat((original_feature,feature_sparse,feature_local,feature_global), dim=1)
                ys = torch.tensor(self.full_list[fid]["feature_55"].to_numpy())#.long()
                sttid = endid
                #print(ys.shape)
                fullx[sttid:(sttid+augmented_feature.shape[0]),:] += augmented_feature
                fully[sttid:(sttid+augmented_feature.shape[0])] += ys
                #fid is name feature 49, feature 48
                # what comes first in hash is actually feature 48
                
                fullhash[sttid:(sttid+augmented_feature.shape[0])] += fid[2]*1000000+fid[1]#torch.tensor(self.full_list[fid]["hashid"].to_numpy())
              

                endid = sttid+augmented_feature.shape[0]
                if fid[0] in self.train_files:
                    self.train_tuple2idx[fid] = list(range(sttid,endid))
                    #train_ids += list(range(sttid,endid))
                else:
                    self.test_tuple2idx[fid] = list(range(sttid,endid))

                    #test_ids += list(range(sttid,endid))
        with open("qsrdata/test_tuple2idx.pkl", "wb") as f:   #Pickling
            pickle.dump(self.test_tuple2idx, f)
        with open("qsrdata/train_tuple2idx.pkl", "wb") as f:   #Pickling
            pickle.dump(self.train_tuple2idx, f)
        torch.save(fullx, 'qsrdata/fullx.pt')
        torch.save(fully, 'qsrdata/fully.pt')
        torch.save(fullhash, 'qsrdata/fullhash.pt')
        
        #torch.save(fullx[test_ids], 'qsrdata/fullxtest.pt')
        #torch.save(fully[test_ids], 'qsrdata/fullytest.pt')
        self.fullx = fullx
        self.fully = fully
        self.fullhash = fullhash
        '''
        for i in range(3):
            print(self.fullhash[i].item(), int(self.fullhash[i].item())//1000000,int(self.fullhash[i].item())%1000000)
        for i,fid in enumerate(self.full_list):
            print(fid)
            if i >=3:
                break
        '''


        '''
        self.fullx_train = fullx[train_ids]
        self.fully_train = fully[train_ids]
        self.fullx_test = fullx[test_ids]
        self.fully_test = fully[test_ids]
        '''

        #plt.scatter(np.ones(n_normal),normal_s,s=0.4,color="blue")
        #plt.scatter(np.ones(n_abnormal)+1,abnormal_s,s=0.4,color="red")

        #plt.scatter(n_feature1,normal_s,s=0.4,color="blue")
        #plt.scatter(a_feature1,abnormal_s,s=0.4,alpha=0.4,color="red")

        #plt.savefig('normal_plot.png',bbox_inches='tight')
    
    def _load_full_data(self):
        self.fullx = torch.load('qsrdata/fullx.pt')
        self.fully = torch.load('qsrdata/fully.pt')
        self.fullhash = torch.load('qsrdata/fullhash.pt')

        with open("qsrdata/test_tuple2idx.pkl", "rb") as fp:   # Unpickling
            self.test_tuple2idx = pickle.load(fp)
            print("test tuple 2 idx loaded")
        with open("qsrdata/train_tuple2idx.pkl", "rb") as fp:   # Unpickling
            self.train_tuple2idx = pickle.load(fp) 
            print("train tuple 2 idx loaded")
 
        #self.expandnet.load_state_dict(torch.load("qsrdata/expandnet.pkl"))
        #self.fullx_test = torch.load('qsrdata/fullxtest.pt')
        #self.fully_test = torch.load('qsrdata/fullytest.pt')

    def _gen_file2tuple(self):
        self.file2tuple= dict()
        for filename in np.concatenate((self.train_files,self.test_files)):
            if not filename in list(self.file2tuple.keys()):
                self.file2tuple[filename] = []
            for tps in self.full_list:
                if tps[0] == filename and tps[2]>0.1:
                    self.file2tuple[filename].append(tps)
        print("generated file to tuple dictionary")

    def handle_filelevel_data(self, train_files, train_tuple2idx):
        id2classhashid = dict() 
        id2filename = dict()       
        with torch.no_grad():
            trainsetx = torch.zeros(len(train_files),48*8)
            trainsety = torch.zeros(len(train_files))
            for fid,filename in enumerate(train_files):
                #self.beststudent2classhashid[filename] = dict()
                alltuples = self.file2tuple[filename]
                #print(alltuples)
                #for tpi in alltuples:
                #    print(tpi)
                #    print(tpi in train_tuple2idx.keys())
                ail = [train_tuple2idx[tpi] for tpi in alltuples if tpi[2]>0.1]
                allindices = list(itertools.chain(*ail))
                fulldataforfile = self.fullx[allindices,:]
                #fulldataforfile[:,48:2*48] /= (torch.abs(fulldataforfile[:,3*48:])+1e-2)
                #fulldataforfile[:,2*48:3*48] /= (torch.abs(fulldataforfile[:,3*48:])+1e-2)

                k=5
                top10, idxtop10 = torch.topk(fulldataforfile,k,dim=0)
                bot10, idxbot10 = torch.topk(-fulldataforfile,k,dim=0)
                
                bot10 = -bot10
                avgtop10 = torch.mean(top10,dim=0)
                z = fulldataforfile*0
                z += fulldataforfile
                z1 = torch.norm(z[:,48:48*2], dim=1)
                z2 = torch.norm(z[:,48*2:48*3], dim=1)
                z3 = torch.norm(z[:,48*3:], dim=1)
                dudu,z1max = torch.topk(torch.abs(z1),1,dim=0)
                #z1avgtop10 = torch.mean(z1top10,dim=0)
                dudu,z2max = torch.topk(torch.abs(z2),1,dim=0)
                #z2avgtop10 = torch.mean(z2top10,dim=0)
                dudu,z3max = torch.topk(torch.abs(z3),1,dim=0)
                #z3avgtop10 = torch.mean(z3top10,dim=0)
                #print(z1max,z2max,z3max)
                '''
                # choice 1: 50%-60% f1 score
                trainsetx[fid,:48*4] += avgtop10
                trainsetx[fid,48*4:48*8] += fulldataforfile[z1max[0],:]
                trainsetx[fid,48*8:48*12] += fulldataforfile[z2max[0],:]
                trainsetx[fid,48*12:48*16] += torch.mean(bot10,dim=0)
                '''
                trainsetx[fid,:48*4] += avgtop10
                trainsetx[fid,48*4:48*8] += torch.mean(bot10,dim=0)
                
                #trainsetx[fid,48*4:48*8] += fulldataforfile[z1max,:]
                id2classhashid[fid] = dict()
                for beststudentinmath in range(48*2):
                    id2classhashid[fid][beststudentinmath] = [self.fullhash[allindices[bsid]] for bsid in idxtop10[:,48+beststudentinmath]]
                    id2classhashid[fid][beststudentinmath+48*2] = [self.fullhash[allindices[bsid]] for bsid in idxbot10[:,48+beststudentinmath]]
                id2filename[fid] = filename
                trainsety[fid] = int("reject" in filename)
            print("k=%s"%k)
            return trainsetx,trainsety, id2classhashid, id2filename

    def rc_score(self, predicted_list, expert_list):
        signalroccect = 0
        phasecorrect = 0
        stepcorrect = 0
        score = 0
        total_candidates = 0
        for prdtuple in predicted_list:
            total_candidates += 1
            for expert in expert_list:
                if "feature_%s" %prdtuple[0] in expert[0]:
                    signalroccect += 1
                    score += 0.5
                if np.isnan(expert[1]):
                    phasecorrect += 1
                    score += 0.25
                elif prdtuple[1] == int(expert[1]+0.1):
                    phasecorrect += 1
                    score += 0.25
                if np.isnan(expert[2]):
                    stepcorrect += 1
                    score += 0.25
                elif prdtuple[2] == int(expert[2]+0.1):
                    stepcorrect += 1
                    score += 0.25
        return {
            "signal_correct":signalroccect,
            "phase_correct":phasecorrect,
            "step_correct":stepcorrect,
            "score":score,
            "total_candidates":total_candidates,
        }

    def find_origin(self, abn_input, reference_input, sid, id2classhashid, id2filename,classifier):
        abnormal_score_list = torch.zeros(48*4)
        baseline = classifier.predict_log_proba(abn_input.reshape(1,-1))
        #print("baseline")
        #print(baseline)
        #raise Exception("debug 2")
        median_input, iduseless = torch.median(reference_input, dim=0)
        #print("median")
        #print(median_input)
        #raise Exception("debug 3")
        #(n1,n2) = abn_input.shape
        with torch.no_grad():
            for i in range(48):
                abnormal_score_list[i] = (classifier.predict_log_proba(replace_coordinate_i(abn_input, median_input,48+i).reshape(1,-1))-baseline)[0,1]
                abnormal_score_list[i+48] = (classifier.predict_log_proba(replace_coordinate_i(abn_input, median_input,48*2+i).reshape(1,-1))-baseline)[0,1]
                abnormal_score_list[i+48*2] = (classifier.predict_log_proba(replace_coordinate_i(abn_input, median_input,48*5+i).reshape(1,-1))-baseline)[0,1]
                abnormal_score_list[i+48*3] = (classifier.predict_log_proba(replace_coordinate_i(abn_input, median_input,48*6+i).reshape(1,-1))-baseline)[0,1]

        '''
        with torch.no_grad():
            for i in range(48):
                abnormal_score_list[i] = torch.sum(abn_input[48+i]<reference_input[:,48+i])
                abnormal_score_list[i+48] = torch.sum(abn_input[48*2+i]<reference_input[:,48*2+i])
                #abnormal_score_list.append((output[1]-output[0]).item())
                abnormal_score_list[i+48*2] = 1000000#torch.sum(abn_input[48*5+i]>reference_input[:,48*5+i])
                abnormal_score_list[i+48*3] = torch.sum(abn_input[48*6+i]>reference_input[:,48*6+i])
        '''
        #print(abnormal_score_list)
        print("identifying root cause for file: %s"%id2filename[sid])

        k1 = 40
        k2 = 1
        uselessvalues, most_abnormals = torch.topk(-abnormal_score_list, k1)
        predicted_list = []
        for most_abnormal in most_abnormals:
            abnormalhash = id2classhashid[sid][int(most_abnormal.item())]
            for idk2, mostabnormalhash in enumerate(abnormalhash):
                if idk2 >= k2:
                    break
                #testtp = (id2filename[sid], int(mostabnormalhash.item())%1000000, int(mostabnormalhash.item())//1000000)
                
                #mostabnormalhash = abnormalhash[0]
                #fid is name feature 49: phase, feature 48: step
                #print("root cause: %s, phase ID: %s, step ID: %s"%(int(most_abnormal.item())%48, mostabnormalhash.item()%1000000, mostabnormalhash.item()//1000000 ))
                predicted_list.append((int(most_abnormal.item())%48, mostabnormalhash.item()%1000000, mostabnormalhash.item()//1000000))
                #print(""%(int(most_abnormal.item())%48))
        #print("Experts say the root causes are:")
        predicted_list = list(set(predicted_list))
        truerccds = self.file2rootcodes[id2filename[sid]]
        expert_list = []
        for tcd in truerccds:
            stcd = tcd #str(tcd)
            if stcd in self.code2featurephasestep:
                expert_list.append(self.code2featurephasestep[stcd])
                #print("root cause: %s, phase ID: %s, step ID: %s"%self.code2featurephasestep[stcd])
        res = self.rc_score(predicted_list, expert_list)
        print(res)
        #print("all possible tuples are")
        #print(self.file2tuple[id2filename[sid]])



    def filelevel_SVM(self):
        for tps in self.train_tuple2idx:
            if tps[1] < 0.5 or tps[2]<0.2:
                print("warning!!!!!!!!! Using phase 0 information for classification")
                print(tps)
        with torch.no_grad():
            trainsetx, trainsety, trainid2classhashid, trainid2filename = self.handle_filelevel_data(self.train_files,self.train_tuple2idx)
            print("train set constructed")
            testsetx, testsety, testid2classhashid, testid2filename = self.handle_filelevel_data(self.test_files,self.test_tuple2idx)
            print("test set constructed")

        X_train = trainsetx.detach().numpy()
        Y_train = trainsety.detach().numpy()
        X_test = testsetx.detach().numpy()
        Y_test = testsety.detach().numpy()
        from sklearn import svm
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import AdaBoostClassifier
        classifier = AdaBoostClassifier(n_estimators=1000, random_state=0).fit(X_train, Y_train)
        #classifier = svm.SVC().fit(X_train, Y_train)
        #classifier = RandomForestClassifier(max_depth=4, random_state=0).fit(X_train, Y_train)
        
        for sid in range(len(X_train)):
            if Y_train[sid] == 1:
                self.find_origin(trainsetx[sid], trainsetx, sid, trainid2classhashid, trainid2filename,classifier)

        from sklearn.metrics import ConfusionMatrixDisplay
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        titles_options = [
            ("Confusion matrix, without normalization", None),
            ("Normalized confusion matrix", "true"),
        ]
        for title, normalize in titles_options:
            disp = ConfusionMatrixDisplay.from_estimator(
                classifier,
                X_test,
                Y_test,
                #display_labels=class_names,
                cmap=plt.cm.Blues,
                normalize=normalize,
            )
            disp.ax_.set_title(title)

            print(title)
            print(disp.confusion_matrix)

    def filelevel_train(self):
        with torch.no_grad():
            trainsetx, trainsety, trainid2classhashid, trainid2filename = self.handle_filelevel_data(self.train_files,self.train_tuple2idx)
            testsetx, testsety, testid2classhashid, testid2filename = self.handle_filelevel_data(self.test_files,self.test_tuple2idx)
        indim = trainsetx.shape[1]
    


        #trainsetx = self.fullx[trainidx]
        #trainsety = self.fully[trainidx]
        trainset = QSRData(trainsetx, trainsety)
        self.median_input = torch.median(trainsetx,dim=0)
        print("train set")
        print(trainset.__len__(), trainsetx.shape[0])

        
        testset = QSRData(testsetx, testsety)
        print("test set")
        print(testset.__len__(), testsetx.shape[0])

        batch_size = 128
        device = self.device
        self.expandnet = None
        self.expandnet = Net(indim=indim)
        self.expandnet.to(device)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

        
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
        #criterion = FocalLoss()
        criterion = nn.CrossEntropyLoss()
        #optimizer = optim.SGD(self.expandnet.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-16)
        optimizer = optim.Adam(self.expandnet.parameters(), lr=0.001,  weight_decay=1e-16)
        scheduler = MyReduceLROnPlateau(optimizer, verbose=True, patience=1)

        print("training starts")
        for epoch in range(500):  # loop over the dataset multiple times
            running_loss = 0.0
            running_tot = 0.0
            time_init = time.time()
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)
                # zero the parameter gradients
                self.expandnet.train()
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.expandnet(inputs)
                #print(outputs.shape)
                #print(labels.shape)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()


                # print statistics
                running_loss += loss.item()
                running_tot += inputs.shape[0]

            cm = self.test_acc(testloader,expand=1)
            #cmf = self.file_test_performance(expand=1)
               # print every 2000 mini-batches
                #print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f} test acc [{a:.3f}, {b:.3f}, {c:.3f}, {d:.3f}, {e:.3f}]')
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / running_tot:.9f}')
            print("testing statistics")
            print(cm)
            calculate_statistics_c_m(cm)
            cmt = self.test_acc(trainloader,expand=1)
            print("training statistics")
            print(cmt)
            
            #calculate_statistics_c_m(cmf)
            torch.save(self.expandnet.state_dict(), "qsrdata/expandnet.pkl")
            time_end = time.time()
            #print("time spent %s"%(time_end-time_init))
            running_loss = running_tot
        
    def _some_visualization(self):
        #self.fullx
        #self.fully
        colors = ['blue','red']

        filedict = dict()
        with torch.no_grad():
            for filename in self.train_files:
                alltuples = self.file2tuple[filename]
                ail = [self.train_tuple2idx[tpi] for tpi in alltuples]
                allindices = list(itertools.chain(*ail))
                fulldataforfile = self.fullx[allindices,:]
                k=5
                top10, idxuseless = torch.topk(torch.abs(fulldataforfile),k,dim=0)
                avgtop10 = torch.mean(top10,dim=0)

                z = fulldataforfile*0
                z += fulldataforfile
                z1 = torch.norm(z[:,48:48*2], dim=1)
                z2 = torch.norm(z[:,48*2:48*3], dim=1)
                z3 = torch.norm(z[:,48*3:], dim=1)
                

                z1top10, idxuseless = torch.topk(torch.abs(z1),k,dim=0)
                z1avgtop10 = torch.mean(z1top10,dim=0)
                z2top10, idxuseless = torch.topk(torch.abs(z2),k,dim=0)
                z2avgtop10 = torch.mean(z2top10,dim=0)
                z3top10, idxuseless = torch.topk(torch.abs(z3),k,dim=0)
                z3avgtop10 = torch.mean(z3top10,dim=0)

                
                isabnormal = int('reject' in filename)
                filedict[filename] = [isabnormal, avgtop10[48:48*2].mean(),
                    avgtop10[48*2:48*3].mean(),avgtop10[48*3:].mean(),
                    z1avgtop10,z2avgtop10,z3avgtop10
                    ]
                '''
                for fid in self.train_tuple2idx:
                    if not fid[0] == filename:
                        continue
                    if fid[2] < 0.1:
                        continue
                    feature0 = torch.sum(torch.abs(self.fullx[self.train_tuple2idx[fid],48:48*2]),dim=0)
                    feature1,idx = torch.max(torch.abs(self.fullx[self.train_tuple2idx[fid],48:48*2]),dim=0)
                    feature2 = torch.mean((self.fullx[self.train_tuple2idx[fid],48*2:48*3])**2,dim=0)
                    feature3, idx = torch.max((self.fullx[self.train_tuple2idx[fid],48*2:48*3])**2,dim=0)
                    feature4 = torch.mean((self.fullx[self.train_tuple2idx[fid],48*3:])**2,dim=0)
                    feature5 = feature0/(feature4+1e-8)
                    is_abs = self.full_list[fid]["feature_55"].to_numpy()
                    agf0 = torch.max(feature0).item()
                    agf1 = torch.max(feature1).item()
                    agf2 = torch.max(feature2).item()
                    agf3 = torch.max(feature3).item()
                    agf4 = torch.max(feature4).item()
                    agf5 = torch.max(feature5).item()


                    if agf0 > filedict[filename][1]:
                        filedict[filename][1] = agf0
                    if agf1 > filedict[filename][2]:
                        filedict[filename][2] = agf1
                    if agf2 > filedict[filename][3]:
                        filedict[filename][3] = agf2
                    if agf3 > filedict[filename][4]:
                        filedict[filename][4] = agf3
                    if agf4 > filedict[filename][5]:
                        filedict[filename][5] = agf4
                    if agf5 > filedict[filename][6]:
                        filedict[filename][6] = agf5
                '''
        '''
        # sparsity
        feature0 = torch.sum(torch.abs(self.fullx_train[:,48:48*2]),dim=1)
        feature1,idx = torch.max(torch.abs(self.fullx_train[:,48:48*2]),dim=1)
        
        # local
        feature2 = torch.mean((self.fullx_train[:,48*2:48*3])**2,dim=1)
        feature3, idx = torch.max((self.fullx_train[:,48*2:48*3])**2,dim=1)
        
        # global
        feature4 = torch.mean((self.fullx_train[:,48*3:48*4])**2,dim=1)
        '''
        isnormal = [vs[0] for vs in list(filedict.values())]

        feature0 = [vs[1] for vs in list(filedict.values())]
        feature1 = [vs[2] for vs in list(filedict.values())]

        feature2 = [vs[3] for vs in list(filedict.values())]
        feature3 = [vs[4] for vs in list(filedict.values())]
        feature4 = [vs[5] for vs in list(filedict.values())]

        feature5 = [vs[6] for vs in list(filedict.values())]

        #feature3 = [vs[4] for vs in list(filedict.values())]



        import matplotlib

        plt.clf()
        plt.scatter(feature0,feature1,s=0.4,alpha=1,c=isnormal, cmap=matplotlib.colors.ListedColormap(colors))
        plt.savefig('normal_plot_0.png',bbox_inches='tight')

        plt.clf()
        plt.scatter(feature0,feature2,s=0.4,alpha=1,c=isnormal, cmap=matplotlib.colors.ListedColormap(colors))
        plt.savefig('normal_plot_1.png',bbox_inches='tight')

        plt.clf()
        plt.scatter(feature0,feature3,s=0.4,alpha=1,c=isnormal, cmap=matplotlib.colors.ListedColormap(colors))
        plt.savefig('normal_plot_2.png',bbox_inches='tight')

      
        
        plt.clf()
        plt.scatter(feature0,feature4,s=0.4,alpha=1,c=isnormal, cmap=matplotlib.colors.ListedColormap(colors))
        plt.savefig('normal_plot_2.png',bbox_inches='tight')

        plt.clf()
        plt.scatter(feature0,feature5,s=0.4,alpha=1,c=isnormal, cmap=matplotlib.colors.ListedColormap(colors))
        plt.savefig('normal_plot_3.png',bbox_inches='tight')
        
        plt.clf()
        plt.scatter(feature1,feature2,s=0.4,alpha=1,c=isnormal, cmap=matplotlib.colors.ListedColormap(colors))
        plt.savefig('normal_plot_4.png',bbox_inches='tight')
        plt.clf()
        plt.scatter(feature1,feature3,s=0.4,alpha=1,c=isnormal, cmap=matplotlib.colors.ListedColormap(colors))
        plt.savefig('normal_plot_5.png',bbox_inches='tight')
        plt.clf()
        plt.scatter(feature1,feature4,s=0.4,alpha=1,c=isnormal, cmap=matplotlib.colors.ListedColormap(colors))
        plt.savefig('normal_plot_6.png',bbox_inches='tight')

        plt.clf()
        plt.scatter(feature1,feature5,s=0.4,alpha=1,c=isnormal, cmap=matplotlib.colors.ListedColormap(colors))
        plt.savefig('normal_plot_7.png',bbox_inches='tight')

        plt.clf()
        plt.scatter(feature2,feature3,s=0.4,alpha=1,c=isnormal, cmap=matplotlib.colors.ListedColormap(colors))
        plt.savefig('normal_plot_8.png',bbox_inches='tight')
        plt.clf()
        plt.scatter(feature2,feature4,s=0.4,alpha=1,c=isnormal, cmap=matplotlib.colors.ListedColormap(colors))
        plt.savefig('normal_plot_9.png',bbox_inches='tight')
        plt.clf()
        plt.scatter(feature2,feature5,s=0.4,alpha=1,c=isnormal, cmap=matplotlib.colors.ListedColormap(colors))
        plt.savefig('normal_plot_10.png',bbox_inches='tight')
        
    
    def _gen_root_cause_reference_embeddings(self):
        #self.rc_dict = pd.DataFrame(data={'feature':{}, 'causes':[]})
        feature_list = []
        causes_list = []
        for tps in self.train_tuple2idx:
            #print(self.full_list[tps]["feature_55"])
            #if tps[2]==0:
            #    print(tps,self.full_list[tps]["feature_55"].sum())
           
            if self.full_list[tps]["feature_55"].sum()>1:
                #print("appending")
                all_features = self.fullx[self.train_tuple2idx[tps]]
                all_causes = [[int(j) for j in self.full_list[tps]["feature_%s"%i].unique() if j > 0.1] for i in range(50,55)]
                all_causes = [cause_i[0] for cause_i in all_causes if len(cause_i)>0]
                mean_feature = torch.mean(all_features,dim=0)
                feature_list.append(mean_feature)
                causes_list.append(all_causes)
        self.rc_dict= pd.DataFrame({'feature': feature_list, 'causes': causes_list}) 

        
    def abnormal_neighbors(self, given_feature, measure=DOT, k=6):
        scores = compute_scores(
            given_feature, self.rc_dict['features'],
            measure)
        score_key = measure + ' score'
        df = pd.DataFrame({
            score_key: list(scores),
            'causes': self.rc_dict['causes'],
        })
        print(df.sort_values([score_key], ascending=False).head(k))
        #display.display(df.sort_values([score_key], ascending=False).head(k))

    def tsne_movie_embeddings(self):
        """Visualizes the movie embeddings, projected using t-SNE with Cosine measure.
        Args:
            model: A MFModel object.
        """
        import sklearn.manifold 
        tsne = sklearn.manifold.TSNE(
            n_components=2, perplexity=40, metric='cosine', early_exaggeration=10.0,
            init='pca', verbose=True, n_iter=400)

        print('Running t-SNE...')
        V_proj = tsne.fit_transform(self.embeddings["movie_id"])
        self.movies.loc[:,'x'] = V_proj[:, 0]
        self.movies.loc[:,'y'] = V_proj[:, 1]
        plt.clf()
        plt.scatter(V_proj[:,0],V_proj[:,1],s=0.4)
        plt.savefig('movie_embeddings.png',bbox_inches='tight')
        
        #return visualize_movie_embeddings(movies, 'x', 'y')

    def tsne_user_embeddings(self):
        """Visualizes the user embeddings, projected using t-SNE with Cosine measure.
        Args:
            model: A MFModel object.
        """
        import sklearn.manifold 
        tsne = sklearn.manifold.TSNE(
            n_components=2, perplexity=40, metric='cosine', early_exaggeration=10.0,
            init='pca', verbose=True, n_iter=400)

        print('Running t-SNE...')
        V_proj = tsne.fit_transform(self.embeddings["user_id"])
        #self.movies.loc[:,'x'] = V_proj[:, 0]
        #self.movies.loc[:,'y'] = V_proj[:, 1]
        plt.clf()
        plt.scatter(V_proj[:,0],V_proj[:,1],s=0.4)
        plt.savefig('user_embeddings.png',bbox_inches='tight')
        #return visualize_movie_embeddings(movies, 'x', 'y')


# @title User recommendations and nearest neighbors (run this cell)
def user_recommendations(model, measure=DOT, exclude_rated=False, k=6):
    if USER_RATINGS:
        scores = compute_scores(
            model.embeddings["user_id"][943], model.embeddings["movie_id"], measure)
        score_key = measure + ' score'
        df = pd.DataFrame({
            score_key: list(scores),
            'movie_id': movies['movie_id'],
            'titles': movies['title'],
            'genres': movies['all_genres'],
        })
        if exclude_rated:
            # remove movies that are already rated
            rated_movies = ratings[ratings.user_id == "943"]["movie_id"].values
            df = df[df.movie_id.apply(lambda movie_id: movie_id not in rated_movies)]
        display.display(df.sort_values([score_key], ascending=False).head(k))  

def visualize_movie_embeddings(data, x, y):
    nearest = alt.selection(
        type='single', encodings=['x', 'y'], on='mouseover', nearest=True,
        empty='none')
    base = alt.Chart().mark_circle().encode(
        x=x,
        y=y,
        color=alt.condition(genre_filter, "genre", alt.value("whitesmoke")),
    ).properties(
        width=600,
        height=600,
        selection=nearest)
    text = alt.Chart().mark_text(align='left', dx=5, dy=-5).encode(
        x=x,
        y=y,
        text=alt.condition(nearest, 'title', alt.value('')))
    return alt.hconcat(alt.layer(base, text), genre_chart, data=data)


    
def hist_plot(data):
    fig, axs = plt.subplots(1, 1,
                        #figsize =(10, 7),
                        tight_layout = True)
 
    axs.hist(data, bins = 21, orientation='horizontal') 
    # Show plot
    plt.savefig('occupation.png')

if __name__ == "__main__":
    #print(users['occupation'].describe())
    #print(movies.describe())
    #hist_plot(users['occupation'])
    #print(ratings['movie_id'].describe())
    #Y = create_single_sparse_y(ratings,1682,943)
    #print(Y.shape)
    #print(Y)
    model = QSR_Data() 
    fd = model.gen_y()
    

    #model.simple_train()
    #model.gen_ys_by_occupation(model.ratings)

    '''
    model._load_full_data()
    model._some_vidualization()
    '''
    #ad,rd = load_data()
    #print(ad)
    #print(rd)
    #ed = {(0,1):10}