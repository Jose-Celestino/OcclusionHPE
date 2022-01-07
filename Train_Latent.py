import utils
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import torch.backends.cudnn as cudnn
import natsort, glob

import datasets, LantentNet


#FUNCTIONS TO DEFINE DIFFERENT LEARNING RATES FOR DIFFERENT MODEL PARAMETERS
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



if __name__ == "__main__":

    cudnn.enabled = True #enable cuda

    gpu = 0 #GPU ID
    model = LantentNet.LantentNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66) 

    device = torch.device('cuda')


    print('Loading snapshot.')
    #Load trained snapshot
    saved_state_dict = torch.load('hopenet_robust_alpha1.pkl') #load hopenet model to initialize parameters (trained for non-occluded images)
    model.load_state_dict(saved_state_dict)


    model.cuda(gpu)
    print('Loading data.')

    lr = 0.00001 #define learning rate

    #classification and regression losses
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    reg_criterion = nn.MSELoss().cuda(gpu)

    # Regression loss coefficient
    alpha = 2 

    softmax = nn.Softmax().cuda(gpu)

    #For classification vector
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)


    # TORCH OPTIMIZER (DO model.cuda(gpu) BEFORE THIS!)
    # Optimizer object, that will hold the current state and will update the parameters based on the computed gradients.
    optimizer = torch.optim.Adam([{'params': get_ignored_params(model), 'lr': 0},
                                    {'params': get_non_ignored_params(model), 'lr': lr},
                                    {'params': get_fc_params(model), 'lr': lr * 5}],
                                    lr = lr)



    occluded_path = natsort.natsorted(glob.glob('/data/Data_Ocluded_300WLP_FULL_25_2/*.jpg'),reverse= False) #occluded images
    image_path = natsort.natsorted(glob.glob('/data/300W-LP_Full/*.jpg'),reverse= False) #non-occluded images



    #Pose annotations
    d_mat1 =  natsort.natsorted(glob.glob('datasets/300W_LP/AFW/*.mat'),reverse= False) 
    d_mat2 = natsort.natsorted(glob.glob('datasets/300W_LP/HELEN/*.mat'),reverse= False) 
    d_mat3 = natsort.natsorted(glob.glob('datasets/300W_LP/IBUG/*.mat'),reverse= False) 
    d_mat4 = natsort.natsorted(glob.glob('datasets/300W_LP/LFPW/*.mat'),reverse= False)

    pose_path = d_mat1+d_mat2+d_mat3+d_mat4


    #Ground_Truth Latent Space
    Xi_path = natsort.natsorted(glob.glob('datasets/Xi/*.mat'),reverse= False)


    #Image transformation (Pre-Processing)
    transformations = transforms.Compose([transforms.Resize(240), transforms.RandomCrop(224), transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    #occluded dataset
    train_dataset = datasets.o_new_Pose_300W_LP(
        occluded_path, pose_path, Xi_path, transformations)

    #non-occluded dataset
    train_dataset_clean = datasets.o_new_Pose_300W_LP(
        image_path, pose_path, Xi_path, transformations)


    occluded_batch_size = 128
    clean_batch_size = 0

    #occluded_batch_size = 96
    #clean_batch_size = 32

    batch_size = occluded_batch_size + clean_batch_size

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=occluded_batch_size,
                                                shuffle=True,
                                                num_workers=4)

    #train_loader_clean = torch.utils.data.DataLoader(dataset=train_dataset_clean,
    #                                            batch_size=clean_batch_size,
    #                                            shuffle=True,
    #                                            num_workers=4)


    print('Ready to train network.')

    num_epochs = 25 #number of train epochs
    beta = 0.999 #Balances the weight of Latent Space loss and Angle losses 

    for epoch in range(num_epochs):
        #clean_loader = iter(train_loader_clean) #uncomment if non_occluded images desired in training
        for i, (images, labels, cont_labels, xi) in enumerate(train_loader):
            #images2, labels2, cont_labels2, xi2 = next(clean_loader) #uncomment if non_occluded images desired in training
            #images = torch.cat((images, images2), 0) ; labels = torch.cat((labels, labels2), 0) #uncomment if non_occluded images desired in training
            #cont_labels = torch.cat((cont_labels, cont_labels2), 0); xi = torch.cat((xi, xi2), 0) #uncomment if non_occluded images desired in training
            images = Variable(images).cuda(gpu)
            
            # Bin labels
            label_yaw = Variable(labels[:,0]).cuda(gpu)
            label_pitch = Variable(labels[:,1]).cuda(gpu)
            label_roll = Variable(labels[:,2]).cuda(gpu)

            # Continuous labels
            label_yaw_cont = Variable(cont_labels[:,0]).cuda(gpu)
            label_pitch_cont = Variable(cont_labels[:,1]).cuda(gpu)
            label_roll_cont = Variable(cont_labels[:,2]).cuda(gpu)

            xi = Variable(xi).cuda(gpu) #Ground-truth latent space


            # Forward pass
            # Outputs: Latent space (x) and Euler angles (yaw, pitch, roll)
            x, yaw, pitch, roll = model(images)

            # Cross entropy loss
            loss_yaw = criterion(yaw, label_yaw)
            loss_pitch = criterion(pitch, label_pitch)
            loss_roll = criterion(roll, label_roll)

            yaw_predicted = softmax(yaw)#,dim = 1)
            pitch_predicted = softmax(pitch)#,dim = 1)
            roll_predicted = softmax(roll)#,dim =1)

            #Compute angle continuous values through expected value
            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 99
            roll_predicted = torch.sum(roll_predicted * idx_tensor, 1) * 3 - 99

            #MSE loss 
            loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont)
            loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont)
            loss_reg_roll = reg_criterion(roll_predicted, label_roll_cont)
            loss_reg_Xi = reg_criterion(x, xi)


            #calculate batch MAE errors
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
            loss = (loss_yaw + loss_pitch + loss_roll)*(1-beta) + beta*loss_reg_Xi
            loss.backward()
            optimizer.step()


            if (i+1) % 100 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Losses: Yaw %.4f, Pitch %.4f, Roll %.4f, #Xi %.4f, Total %.4f' 
                        %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss_yaw.data, loss_pitch.data, loss_roll.data ,loss_reg_Xi.data, loss))
                print('Average Angle Error (degrees): Yaw: %.4f, Pitch %.4f, Roll %.4f' %(y_err,p_err,r_err))
        # Save models at numbered epochs.
        if epoch % 2 == 0 and epoch < num_epochs:
            print('Taking snapshot...')         
            torch.save(model.state_dict(), 'Latent' + '_model' + '_epoch_'+ str(epoch) + '.pkl')

