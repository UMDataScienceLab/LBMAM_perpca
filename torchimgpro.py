import matplotlib.pyplot as plt
from PIL import Image
#img = Image.open('data/stinkbug.png')
import numpy as np
import copy
import os
import cv2
import torch
from scipy import fftpack, ndimage
import torch.nn as nn
import pickle

def save_image(image,addr,num):
    cv2.imwrite(addr+str(num)+".jpg", image)

def cv2_save_img(image, addr):
    cv2.imwrite(addr, image)


def load_thermal_data(resdict):
    from os import listdir
    from os.path import isfile, join
    import re    
    srcpath = r'images_ds/test1_thermal/'
    onlyfiles = [join(srcpath, f) for f in listdir(srcpath) if isfile(join(srcpath, f))]
    name_pattern = re.compile(".Build([0-9]+)_Layer([0-9]+)_([0-9]+).pkl")
    totnum = 0
    for fi in onlyfiles:
        s = name_pattern.search(fi)
        if s:
            bd = int(s.group(1))
            ly = int(s.group(2))
            fm = int(s.group(3))
            
            if not bd in resdict.keys():
                resdict[bd] = dict()
            if not ly in resdict[bd].keys():
                resdict[bd][ly] = dict()
            with open(fi, 'rb') as fin :
                img = pickle.load(fin)
            img = np.array(img)
            #print(img.shape)
            cat = img
            #if len(cat.shape) > 2:
            #    cat = np.mean(img, axis=2)
            ct = cat#.T
            resdict[bd][ly][fm] = torch.tensor(cat).float()
            totnum+=1
            #print(bd,ly,fm)
            #print(bd, ly, len(resdict[bd][ly].keys()))
    print("%s files loaded from %s"%(totnum, srcpath))
    for bd in resdict:
        for ly in resdict[bd]:
            print(bd, ly, len(resdict[bd][ly].keys()))
    return resdict
    
def load_thermal_data_old(resdict):
    from os import listdir
    from os.path import isfile, join
    import re    
    srcpath = r'../../pca/working/perpca/frames/thermal/NIST Build2 Layers251-280/'
    onlyfiles = [join(srcpath, f) for f in listdir(srcpath) if isfile(join(srcpath, f))]
    name_pattern = re.compile(".AMB2018_625_Build([0-9]+)_Layer([0-9]+)_frame([0-9]+).jpg")
    totnum = 0
    for fi in onlyfiles:
        s = name_pattern.search(fi)
        if s:
            bd = int(s.group(1))
            ly = int(s.group(2))
            fm = int(s.group(3))
            if not bd in resdict.keys():
                resdict[bd] = dict()
            if not ly in resdict[bd].keys():
                resdict[bd][ly] = dict()
            img = Image.open(fi)    
            img = np.array(img)
            #print(img.shape)
            cat = img
            if len(cat.shape) > 2:
                cat = np.mean(img, axis=2)
            ct = cat#.T
            resdict[bd][ly][fm] = torch.tensor(cat).float()
            totnum+=1
            #print(bd,ly,fm)
            #print(bd, ly, len(resdict[bd][ly].keys()))
    print("%s files loaded from %s"%(totnum, srcpath))
    for bd in resdict:
        for ly in resdict[bd]:
            print(bd, ly, len(resdict[bd][ly].keys()))
    return resdict

def imgsshow(compose):
    for c in compose:
        plt.imshow(c)
        plt.axis('off')
        plt.show()


def threshold(file):
    img = Image.open(file)
    
    img = np.array(img)
    img = np.mean(img, axis=2)
    #print(np.max(img))
    #print(np.min(img))
    print(img.shape)
    fmask = img>=130
    #print(fmask[150]*255)
    print(fmask*255)
    print(fmask.mean())
    plt.imshow(img[:,0:600],cmap='gray')
    plt.savefig('trexample3.png')


def position2momentum(Y):
    res = dict()
    if isinstance(Y, list):
        N = len(Y)
        alliters = list(range(N))
    else:
        alliters = Y.keys()
    for ki in alliters:
        timage = Y[ki]
        image = timage.numpy()
        fft2 = fftpack.fft2(image)
        fft2r = np.real(fft2)
        fft2img = np.imag(fft2)
        cbd = np.concatenate((fft2r,fft2img), axis=0)
        res[ki]=torch.tensor(cbd)
    return res

def r2handdictionary(Y, dictmult=10, sigma=0.1):
    if isinstance(Y, list):
        N = len(Y)
        alliters = list(range(N))
    else:
        alliters = Y.keys()
    # generate dictionaries
    gap = max(1,len(alliters)//dictmult)
    dictionary = []
    for i,ki in enumerate(alliters):
        if i%gap == 0:
            dictionary.append(Y[ki])
    dictionary = torch.cat(dictionary,dim=1)
    largeidx = torch.where(torch.norm(dictionary, dim=0) > 1e-4)[0]
    dictionary = dictionary[:,largeidx]
    tmax = torch.max(dictionary)
    dictionary = torch.abs(torch.randn(len(Y[ki]),10000))*tmax/3
    print("Dictionary built with dimension %s x %s"%(dictionary.shape[0],dictionary.shape[1]))
    with torch.no_grad():
        res = dict()
        for ki in alliters:
            timage = Y[ki]            
            dist = torch.cdist(dictionary.T, timage.T)
            res[ki]= torch.exp(-0.5*dist**2/sigma**2)
    return res, dictionary

def h2r(h, dictionary, sigma=0.1, eps=1e-3):
    (d,ndic) = dictionary.shape
    (ndic,nsample) = h.shape
    with torch.no_grad():
        z = torch.randn(d,nsample)*1e-2
        beta = 0.5
        for i in range(200):
            zdist = torch.cdist(dictionary.T,z.T) #  ndic x nsample 
            coeff = torch.exp(-0.5*zdist**2/sigma**2) * h #  ndic x nsample 
            coeff = (coeff/(1e-5+torch.sum(coeff, dim=0)))
            znew = dictionary@coeff # d x nsample
            if torch.norm(znew-z)<1e-5:
                break
            #print(i,torch.norm(z),torch.norm(znew-z)) 
            # use exponential averaging to stablize the iterate
            z = (1-beta)*z + beta*znew
            #z += znew       
            
    return znew

def hdict2r(Y, dictionary, sigma=0.1):
    if isinstance(Y, list):
        N = len(Y)
        alliters = list(range(N))
    else:
        alliters = Y.keys()
    res = dict()
    with torch.no_grad():
        res = dict()
        for ki in alliters:                      
            res[ki]=h2r(Y[ki], dictionary, sigma,eps=1e-3)
    return res

def reshuffle(Y,n1,n2):
    (d1,d2) = Y.shape
    k1 = d1 // n1
    k2 = d2 // n2
    truncate = Y[:k1*n1,:k2*n2]
    ufd = nn.Unfold(kernel_size=(k1,k2),dilation=(n1,n2))
    return ufd(truncate.unsqueeze(0).unsqueeze(0))[0]#.T


def shuffleback(Yshuffled, n1, n2, k1, k2):
    Ys = Yshuffled#.T
    fd = nn.Fold(output_size=(n1*k1,n2*k2),kernel_size=(k1,k2),dilation=(n1,n2))
    return fd(torch.tensor(Ys).unsqueeze(0)).numpy()[0][0]
    
def reconstruct_picture(pic,name,cutoff_up=1e6,cutoff_low=-1e6,args={}):
    picture = pic.copy()    
    if "momentum" in args and args["momentum"]:
        (n1,n2) = pic.shape
        n1 = n1//2
        fftrecons = picture[:n1,:]+np.array([1j])*picture[n1:,:]
        fft3 = fftpack.ifft2(fftrecons)
        return abs(fft3)
    elif "reshuffle" in args and args["reshuffle"]:
        if "kernel" in args and args["kernel"]:
            #print(picture.shape)
            #print(torch.tensor(picture).shape)
            #print(args['kerneldict'].shape)
            picture = h2r(torch.tensor(picture), args['kerneldict'], args['sigma']).numpy()
        picture = shuffleback(picture, args["n1"], args["n2"], args["k1"], args["k2"])
        picture[picture>cutoff_up] = cutoff_up
        picture[picture<cutoff_low] = cutoff_low
        return picture
    else:
        picture[picture>cutoff_up] = cutoff_up
        picture[picture<cutoff_low] = cutoff_low
        return picture


def only_show_save(picture,name,cutoff_up=1e6,cutoff_low=-1e6,args={}):
    plt.imshow(picture, cmap='gray')
    plt.axis('off')
    plt.savefig(name, bbox_inches='tight')
    plt.show()


def show_save(pic,name,cutoff_up=1e6,cutoff_low=-1e6,args={}):
    picture = pic.copy()
    
    if "momentum" in args and args["momentum"]:
        (n1,n2) = pic.shape
        n1 = n1//2
        fftrecons = picture[:n1,:]+np.array([1j])*picture[n1:,:]
        fft3 = fftpack.ifft2(fftrecons)
        plt.imshow(abs(fft3), cmap='gray')
        plt.axis('off')
        plt.savefig(name, bbox_inches='tight')
        plt.show()
    elif "reshuffle" in args and args["reshuffle"]:
        if "kernel" in args and args["kernel"]:
            #print(picture.shape)
            #print(torch.tensor(picture).shape)
            #print(args['kerneldict'].shape)
            picture = h2r(torch.tensor(picture), args['kerneldict'], args['sigma']).numpy()
       
        picture = shuffleback(picture, args["n1"], args["n2"], args["k1"], args["k2"])
        picture[picture>cutoff_up] = cutoff_up
        picture[picture<cutoff_low] = cutoff_low
        plt.imshow(picture, cmap='gray')
        plt.axis('off')
        plt.savefig(name, bbox_inches='tight')
        plt.show()
    else:
        picture[picture>cutoff_up] = cutoff_up
        picture[picture<cutoff_low] = cutoff_low
        plt.imshow(picture, cmap='gray')
        plt.axis('off')
        plt.savefig(name, bbox_inches='tight')
        plt.show()

def show_fft():
    import matplotlib.pyplot as plt
    Y = load_girl_data()
    image = Y[0].numpy()
    fft2 = fftpack.fft2(image)
    print(fft2)
    fft2r = np.real(fft2)
    (n1,n2)=fft2r.shape
    fft2img = np.imag(fft2)
    cbd = np.concatenate((fft2r,fft2img))
    u,s,vt = np.linalg.svd(cbd,full_matrices=False)
    print(u.shape,s.shape,vt.shape)
    s[10:]*=0
    smat = np.diag(s)
    recons = u@smat@vt
    fftrecons = recons[:n1,:]+np.array([1j])*recons[n1:,:]


    plt.imshow(np.log10(abs(fftrecons)))
    plt.savefig('fft.png')
    fft3 = fftpack.ifft2(fftrecons)
    plt.imshow(abs(fft3))
    plt.savefig('fft3.png')

    u,s,vt = np.linalg.svd(image,full_matrices=False)
    #print(u.shape,s.shape,vt.shape)
    s[10:]*=0
    smat = np.diag(s)
    recons = u@smat@vt
    plt.imshow(recons)
    plt.savefig('fft4.png')


if __name__ == "__main__":
    load_thermal_data(dict())
