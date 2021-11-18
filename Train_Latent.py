import sys, os, argparse, time

import numpy as np
import cv2
import matplotlib.pyplot as plt

import utils
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import natsort, glob
from PIL import Image, ImageFilter

import datasets, hopenet
import torch.utils.model_zoo as model_zoo

import pickle

def get_ignored_params(model):
    # Generator function that yields ignored params.
    b = [model.conv1, model.bn1, model.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_yaw, model.fc_pitch, model.fc_roll]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param

def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

if __name__ == "__main__":

    cudnn.enabled = True #enable cuda

    gpu = 0 
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66) #hopenet model to use

    device = torch.device('cuda')

    #torch.cuda.set_device(i)

    #model = nn.parallel.DistributedDataParallel(model)
    #model = model.to(device)


    print('Loading snapshot.')
    #Load trained snapshot
    saved_state_dict = torch.load('hopenet_robust_alpha1.pkl')
    model.load_state_dict(saved_state_dict)

    #model =  nn.DataParallel(model)#.to(device)

    model.cuda(gpu)
    print('Loading data.')

    lr = 0.00001 #learning rate #0.00001
    #lr = lr.to(f'cuda:{model.device_ids[0]}')

    #model.cuda(gpu)
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    reg_criterion = nn.MSELoss().cuda(gpu)
    # Regression loss coefficient
    alpha = 1 

    softmax = nn.Softmax().cuda(gpu)
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)


    # TORCH OPTIMIZER (DO model.cuda(gpu) BEFORE THIS!)
    # Optimizer object, that will hold the current state and will update the parameters based on the computed gradients.
    optimizer = torch.optim.Adam([{'params': get_ignored_params(model), 'lr': 0},
                                    {'params': get_non_ignored_params(model), 'lr': lr},
                                    {'params': get_fc_params(model), 'lr': lr * 5}],
                                    lr = lr)


    #occluded_path = natsort.natsorted(glob.glob('/data/Data_Ocluded_300WLP_FULL/*.jpg'),reverse= False)
    occluded_path = natsort.natsorted(glob.glob('/data/Data_Ocluded_300WLP_FULL_25_2/*.jpg'),reverse= False)
    image_path = natsort.natsorted(glob.glob('/data/300W-LP_Full/*.jpg'),reverse= False)

    #occluded_path = occluded_path[0:61225]

    #POSE
    d_mat1 =  natsort.natsorted(glob.glob('/data/300W_LP/AFW/*.mat'),reverse= False) 
    d_mat2 = natsort.natsorted(glob.glob('/data/300W_LP/HELEN/*.mat'),reverse= False) 
    d_mat3 = natsort.natsorted(glob.glob('/data/300W_LP/IBUG/*.mat'),reverse= False) 
    d_mat4 = natsort.natsorted(glob.glob('/data/300W_LP/LFPW/*.mat'),reverse= False)

    pose_path = d_mat1+d_mat2+d_mat3+d_mat4

    # LOAD PICKLE WITH IMAGE Xi
    with open('X_i_300wlp.pickle', 'rb') as handle:
        Xi = pickle.load(handle)

    Xi_path = natsort.natsorted(glob.glob('/data/Xi/*.mat'),reverse= False)

    print("done")


    training_path = occluded_path[0:10000]+occluded_path[12225:12225+10000]+occluded_path[24450:24450+10000]+occluded_path[36675:36675+10000]+occluded_path[48900:48900+10000]
    #testing_path = occluded_path[10000:10000+2225]+occluded_path[22225:22225+2225]+occluded_path[34450:34450+2225]+occluded_path[46675:46675+2225]+occluded_path[58900:58900+2225]

    training_path_clean = image_path[0:10000]+image_path[12225:12225+10000]+image_path[24450:24450+10000]+image_path[36675:36675+10000]+image_path[48900:48900+10000]

    pose_training_path = pose_path[0:10000]+pose_path[12225:12225+10000]+pose_path[24450:24450+10000]+pose_path[36675:36675+10000]+pose_path[48900:48900+10000]
    #pose_testing_path = pose_path[10000:10000+2225]+pose_path[22225:22225+2225]+pose_path[34450:34450+2225]+pose_path[46675:46675+2225]+pose_path[58900:58900+2225]

    xi_training_path = Xi_path[0:10000]+Xi_path[12225:12225+10000]+Xi_path[24450:24450+10000]+Xi_path[36675:36675+10000]+Xi_path[48900:48900+10000]
    #xi_testing_path = Xi_path[10000:10000+2225]+Xi_path[22225:22225+2225]+Xi_path[34450:34450+2225]+Xi_path[46675:46675+2225]+Xi_path[58900:58900+2225]

    transformations = transforms.Compose([transforms.Resize(240), transforms.RandomCrop(224), transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    train_dataset = datasets.o_new_Pose_300W_LP(
        training_path, pose_training_path, xi_training_path, transformations)

    train_dataset_clean = datasets.o_new_Pose_300W_LP(
        training_path_clean, pose_training_path, xi_training_path, transformations)

    #test_dataset = datasets.o_new_Pose_300W_LP(
    #    testing_path, pose_testing_path, xi_testing_path, transformations)

    batch_size = 96


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=96,
                                                shuffle=True,
                                                num_workers=4)

    train_loader_clean = torch.utils.data.DataLoader(dataset=train_dataset_clean,
                                                batch_size=32,
                                                shuffle=True,
                                                num_workers=4)


    print('Ready to train network.')

    num_epochs = 25
    for epoch in range(num_epochs):
        clean_loader = iter(train_loader_clean)
        for i, (images, labels, cont_labels, xi) in enumerate(train_loader):
            images2, labels2, cont_labels2, xi2 = next(clean_loader)
            images = torch.cat((images, images2), 0) ; labels = torch.cat((labels, labels2), 0)
            cont_labels = torch.cat((cont_labels, cont_labels2), 0); xi = torch.cat((xi, xi2), 0)
            images = Variable(images).cuda(gpu)
            

            label_yaw = Variable(labels[:,0]).cuda(gpu)
            label_pitch = Variable(labels[:,1]).cuda(gpu)
            label_roll = Variable(labels[:,2]).cuda(gpu)

            # Continuous labels
            label_yaw_cont = Variable(cont_labels[:,0]).cuda(gpu)
            label_pitch_cont = Variable(cont_labels[:,1]).cuda(gpu)
            label_roll_cont = Variable(cont_labels[:,2]).cuda(gpu)

            xi = Variable(xi).cuda(gpu)


            # Forward pass
            x, yaw, pitch, roll = model(images)

            # Cross entropy loss
            loss_yaw = criterion(yaw, label_yaw)
            loss_pitch = criterion(pitch, label_pitch)
            loss_roll = criterion(roll, label_roll)

            #print('loss yaw:', loss_yaw)

            # MSE loss
            yaw_predicted = softmax(yaw)#,dim = 1)
            pitch_predicted = softmax(pitch)#,dim = 1)
            roll_predicted = softmax(roll)#,dim =1)

            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 99
            roll_predicted = torch.sum(roll_predicted * idx_tensor, 1) * 3 - 99

            #MSE    
            loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont)
            loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont)
            loss_reg_roll = reg_criterion(roll_predicted, label_roll_cont)
            loss_reg_Xi = reg_criterion(x, xi)


            err_y = yaw_predicted-label_yaw_cont
            err_p = pitch_predicted-label_pitch_cont
            err_r = yaw_predicted-label_yaw_cont
            y_err = sum(abs(err_y))/len(err_y)
            p_err = sum(abs(err_p))/len(err_p)
            r_err = sum(abs(err_r))/len(err_r)

        
            # Total loss
            loss_yaw += alpha * loss_reg_yaw
            loss_pitch += alpha * loss_reg_pitch
            loss_roll += alpha * loss_reg_roll



            optimizer.zero_grad()
            loss = loss_yaw + loss_pitch + loss_roll + 100*loss_reg_Xi
            loss.backward()
            optimizer.step()

        

            if (i+1) % 100 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Losses: Yaw %.4f, Pitch %.4f, Roll %.4f, #Xi %.4f, Total %.4f' #Xi %.4f,
                        #%(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss_yaw.data, loss_pitch.data, loss_roll.data, loss)) # ,loss_reg_Xi.data, loss))
                        %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss_yaw.data, loss_pitch.data, loss_roll.data ,100*loss_reg_Xi.data, loss))
                print('Average Angle Error (degrees): Yaw: %.4f, Pitch %.4f, Roll %.4f' %(y_err,p_err,r_err))
        # Save models at numbered epochs.
        if epoch % 2 == 0 and epoch < num_epochs:
            print('Taking snapshot...')         
            torch.save(model.state_dict(), 'Latent_Train_HOPENET_25_100LATENT' + '_model' + '_epoch_'+ str(epoch) + '.pkl')

