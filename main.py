#basic libary
import copy
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
#import torch.nn.utils.parametrizations 

from solver import *

class Experiment():

    def video_thermal_example(inputargs):
        args = {
            "ngc":15,
            "nlc":50,
            "optim":"SGD",
            "lr":0.05,
            "epochs":400,
            "seed":100,
            "lbd_s":1e4,
            "wd":0,
            "normalize":1,
            "columnnorm":1,
            "momentum":0,
            "reshuffle":1,
            "kernel":0,
            "sigma":800.,
            "dictsize":10000,

        }
        if args['kernel']:
            args['lr'] = 5e-3
            args['ngc'] = 5
        '''
        # Best parameter for the car data under Fourier transformation:        
        args = {
            "ngc":200,
            "nlc":200,
            "optim":"SGD",
            "lr":0.01,
            "epochs":200,
            "seed":100,
            "lbd_s":1e4,
            "wd":0,
            "normalize":1,
            "columnnorm":1,
            "momentum":1,
            "reshuffle":0,
            "kernel":0,
        }
        
        # Best parameter for the car data:
        1. do not transpose the flat figure
        2. args = {
            "ngc":60,
            "nlc":100,
            "optim":"SGD",
            "lr":0.01,
            "epochs":500,
            "seed":100,
            "lbd_s":"auto",
        }
        '''
        print(args)

        np.random.seed(args["seed"])
        torch.manual_seed(args["seed"])
        #print("a1")
        import torchimgpro 
        Yraw = torchimgpro.load_thermal_data(dict())
        ttlayer = 1
        clayer = 0
        for bd in Yraw:
            for ly in Yraw[bd]:
                Y = Yraw[bd][ly]
                if clayer >= ttlayer:
                    break
                clayer += 1                
        

                if args["momentum"]:
                    Y = torchimgpro.position2momentum(Y)
                if args["reshuffle"]:
                    for i in Y.keys():
                        args["n1"] = 10
                        args["n2"] = 10
                        if  args["kernel"]:
                            args["n1"] = 20
                            args["n2"] = 20
                        n1 = args["n1"]
                        n2 = args["n2"]
                        (d1,d2) = Y[i].shape
                        break
                    args["k1"] = d1 // n1
                    args["k2"] = d2 // n2
                    Y = {ki:torchimgpro.reshuffle(Y[ki], n1, n2) for ki in Y.keys()}
              
                if args['kernel']:
                    Y, imgdict = torchimgpro.r2handdictionary(Y, args['dictsize'], args['sigma'])
                    args['kerneldict'] = imgdict
               
                if isinstance(Y, list):
                    N = len(Y)
                    alliters = list(range(N))
                else:
                    alliters = Y.keys()
                N = len(alliters)
               
                for ki in Y.keys(): 
                    print(Y[ki].shape)
                    break
             

                norms = dict()
                transmat = []
                #from sqrtm import sqrtm
                if args["normalize"]:
                    print("normalize data")
                    with torch.no_grad():
                        for i in Y.keys():
                            ni = []
                            if args["columnnorm"]:
                                for j in range(len(Y[i])):
                                    normy = Y[i][j].norm()+1e-10
                                    Y[i][j] /= normy
                                    ni.append(normy)
                                norms[i]=ni
                            else:
                                for j in range(len(Y[i][0])):
                                    normy = Y[i][:,j].norm()
                                    Y[i][:,j] /= normy
                                    ni.append(normy)
                                norms[i]=ni
              
                Ug,Vg,Ul,Vl= lg_matrix_factorization_projgd(Y, args)
                print("decomposing finished")
                names = [ki for ki in Y.keys()]
                Vlist = [Vl[ki].lin_mat.detach().numpy() for ki in Y.keys()]
                
                reconstruct_bg = {i:(Ug[i].lin_mat@Vg[i].lin_mat.T) for i in alliters}
                reconstruct_cat = {i:(Ul[i].lin_mat@Vl[i].lin_mat.T) for i in alliters}
                reconstruct_full = {i:reconstruct_bg[i]+reconstruct_cat[i] for i in alliters}
                print("images reconstructed")
         
                folder = 'thermaldcpdframes/build_%s_layer_%s/'%(bd,ly)
                print(folder)
                if not os.path.exists(folder):
                    os.mkdir(folder)
                SAVE_ALL = True
                plt.close()
                with torch.no_grad():
                    for i in alliters:
                        print("saving figure [%s/%s]"%(i,N))
                    
                        if args["normalize"]:
                            if args["columnnorm"]:
                                for j in range(len(reconstruct_bg[i])):
                                    reconstruct_bg[i][j] *= norms[i][j]
                                    reconstruct_cat[i][j] *= norms[i][j]
                                    reconstruct_full[i][j] *= norms[i][j]
                                    Y[i][j] *= norms[i][j]
                                    S[i].lin_mat[j] *= norms[i][j]
                            else:
                                for j in range(len(reconstruct_bg[i][0])):
                                    reconstruct_bg[i][:,j] *= norms[i][j]
                                    reconstruct_cat[i][:,j] *= norms[i][j]
                                    reconstruct_full[i][:,j] *= norms[i][j]

                                    Y[i][:,j] *= norms[i][j]
                                    S[i].lin_mat[:,j] *= norms[i][j]
                          
                        original = Y[i].detach().numpy()
                   
                        reconstruct_bg[i] = reconstruct_bg[i].detach().numpy()
                    
                        reconstruct_cat[i] = reconstruct_cat[i].detach().numpy()
                        if SAVE_ALL:
                            if  args['kernel']:
                                reconstruct_full[i] = reconstruct_full[i].detach().numpy()
                                full_img = torchimgpro.reconstruct_picture(reconstruct_full[i],folder+'recons_bg_%s.png'%i,args=args)
                                bg_img = torchimgpro.reconstruct_picture(reconstruct_bg[i],folder+'recons_bg_%s.png'%i,args=args)
                                torchimgpro.only_show_save(bg_img,folder+'recons_bg_%s.png'%i,args=args)
                                torchimgpro.only_show_save(full_img-bg_img,folder+'recons_cat_%s.png'%i,args=args)
                                torchimgpro.show_save(original,folder+'a_original_%s.png'%i,args=args)

                            else:   
                                torchimgpro.show_save(reconstruct_bg[i],folder+'recons_bg_%s.png'%i,args=args)
                                torchimgpro.show_save(reconstruct_cat[i],folder+'recons_cat_%s.png'%i,args=args)
                            torchimgpro.show_save(original,folder+'original_%s.png'%i,args=args)

                    
                        #reconstruct_noise = S[i].lin_mat.detach().numpy()
                        #torchimgpro.show_save(reconstruct_noise,folder+'recons_noise_%s.png'%i,args=args)
                avg_bg = sum([reconstruct_cat[i] for i in reconstruct_bg])
                
                torchimgpro.show_save(avg_bg,folder+'avg_bg.png',args=args)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='personalized pca')
    parser.add_argument('--dataset', type=str, default="video_thermal_example")
    parser.add_argument('--algorithm', type=str, default="dgd")
    parser.add_argument('--logoutput', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--d', type=int, default=15)
    parser.add_argument('--epochs', type=int, default=150)

    parser.add_argument('--num_client', type=int, default=100)
    parser.add_argument('--nlc', type=int, default=1)
    parser.add_argument('--ngc', type=int, default=1)
    parser.add_argument('--num_dp_per_client', type=int, default=1000)
    parser.add_argument('--folderprefix', type=str, default='')

    args = parser.parse_args()
    args = vars(args)
    if args['logoutput']:
        import os
        from misc import Tee
        import time
        import sys
        output_dir = args['folderprefix']+'outputs/{}_'.format(args['dataset'])
        jour = time.strftime("%Y-%m-%d-jour-%H-%M-%S", time.localtime())
        output_dir += jour
        os.makedirs(output_dir, exist_ok=True)
        sys.stdout = Tee(os.path.join(output_dir, 'out.txt')) 

    experiment = getattr(Experiment, args['dataset'])
    experiment(args)
