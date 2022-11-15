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

def img_add(front, back, stx, sty):
    w, h = front.shape
    res = copy.deepcopy(back)
    res[stx:stx+w, sty:sty+h] = res[stx:stx+w, sty:sty+h]+front
    return res


def gen_img_data(args={}):
    return load_girl_data(args)
    # return load_car_data()
    # return load_cat_data()


def save_image(image,addr,num):
    cv2.imwrite(addr+str(num)+".jpg", image)

def cv2_save_img(image, addr):
    cv2.imwrite(addr, image)


def load_thermal_data(gdpath, resdict):
    from os import listdir
    from os.path import isfile, join
    import re    
    srcpath = r'images/'
    srcpath = join(gdpath, srcpath)
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
            cat = img
           
            ct = cat
            resdict[bd][ly][fm] = torch.tensor(cat).float()
            totnum+=1
      
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
    elif "reshuffle" in args and args["reshuffle"]:
        picture = shuffleback(picture, args["n1"], args["n2"], args["k1"], args["k2"])
        picture[picture>cutoff_up] = cutoff_up
        picture[picture<cutoff_low] = cutoff_low
        plt.imshow(picture, cmap='gray')
        plt.axis('off')
        plt.savefig(name, bbox_inches='tight')
    else:
        picture[picture>cutoff_up] = cutoff_up
        picture[picture<cutoff_low] = cutoff_low
        plt.imshow(picture, cmap='gray')
        plt.axis('off')
        plt.savefig(name, bbox_inches='tight')

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

    



    '''
    image2 = Y[1].numpy()
    fft22 = fftpack.fft2(image2)

    plt.imshow(np.log10(abs(fft22)))
    plt.savefig('fft22.png')
    '''
    

if __name__ == "__main__":
    #imgsshow(gen_img_data())
    #process_cat_data_xb()
    #process_car_data()
    #process_office_data()
    #threshold(r'processedframes/rpca_cat_6.png')
    #threshold(r'frames/office/0.jpg')
    #gen_ellipses()
    #u = np.random.randn(100,2)
    #v = np.random.randn(100,2)
    #cv2.imwrite("random.jpg",np.abs(u@v.T)*256)
    #show_fft()
    load_thermal_data(dict())
