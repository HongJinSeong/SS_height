import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm.auto import tqdm
import torch.nn as nn
from models import *

import skimage.io as iio
import skimage.color  as icol
import skimage.transform as skiT

from PIL import Image

import torchvision.transforms.functional as TF
from torchvision import datasets, models, transforms

class CustomDataset_for_embed(Dataset):
    def __init__(self, sem_path_list, depth_path_list, cls_list, TF_train):
        self.sem_path_list = sem_path_list
        self.depth_path_list = depth_path_list
        self.TF_train = TF_train
        self.cls_list = cls_list

        ## 0 ~ 1 로 정규화
        self.transform = transforms.Compose([transforms.Resize(size=(96, 64), interpolation=Image.NEAREST),
                                                 transforms.ToTensor()])
        self.resize_and_crop = True

    def __getitem__(self, index):
        sem_path = self.sem_path_list[index]

        sem_img = iio.imread(sem_path, as_gray=True)
        # sem_img = np.expand_dims(sem_img, axis=-1)
        # sem_img = np.concatenate((sem_img, sem_img, sem_img), axis=2)

        if self.depth_path_list is not None:  ## ==> simulation SEM만 denoise 처리후 진행
            sem_img = cv2.bilateralFilter(sem_img, 5, 50, 50)

        if np.max(sem_img) <= 1:
            sem_img = np.clip((sem_img * 255), 0, 255)
            sem_img = sem_img.astype(np.uint8)
        sem_img = Image.fromarray(sem_img)

        if self.depth_path_list is not None:
            depth_path = self.depth_path_list[index]
            depth_img = iio.imread(depth_path, as_gray=True)
            depth_img = Image.fromarray(depth_img)

            cls = self.cls_list[index]

            sem_img_for_depth = sem_img

            if self.TF_train == True:
                if np.random.rand(1)[0] > 0.5:
                    sem_img_for_depth = TF.hflip(sem_img_for_depth)
                    depth_img = TF.hflip(depth_img)
                if np.random.rand(1)[0] > 0.5:
                    sem_img_for_depth = TF.vflip(sem_img_for_depth)
                    depth_img = TF.vflip(depth_img)

                # 동일영역 자르기 위함
                if self.resize_and_crop==True:
                    if np.random.rand(1)[0] > 0.5:
                        sem_img_for_depth = TF.resize(sem_img_for_depth, size = (120,80))
                        depth_img = TF.resize(depth_img, size = (120,80))

                        top = np.random.choice(np.arange(120 - 96), 1)[0]
                        left = np.random.choice(np.arange(80 - 64), 1)[0]

                        sem_img_for_depth = TF.resized_crop(sem_img_for_depth, top=top, left=left, height=96, width=64, size=(96,64), interpolation = Image.NEAREST)
                        depth_img = TF.resized_crop(depth_img, top=top, left=left, height=96, width=64, size=(96,64), interpolation = Image.NEAREST)


                return self.transform(sem_img), self.transform(sem_img_for_depth), self.transform(depth_img), cls
            else:
                return self.transform(sem_img), self.transform(sem_img), self.transform(depth_img), cls
        else:
            img_name = sem_path.split('/')[-1]
            return self.transform(sem_img), img_name  # B,C,H,W

    def __len__(self):
        return len(self.sem_path_list)

class CustomDataset(Dataset):
    def __init__(self, sem_path_list, depth_path_list, cls_list, TF_train):
        self.sem_path_list = sem_path_list
        self.depth_path_list = depth_path_list
        self.TF_train = TF_train
        self.cls_list = cls_list

        ## 0 ~ 1 로 정규화
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.transform_val = transforms.Compose([transforms.Resize(size=(96, 64), interpolation=Image.NEAREST),
                                                 transforms.ToTensor()])
        self.resize_and_crop = True

    def __getitem__(self, index):
        sem_path = self.sem_path_list[index]

        sem_img = iio.imread(sem_path, as_gray=True)


        if np.max(sem_img) <= 1:
            sem_img = np.clip((sem_img * 255), 0, 255)
            sem_img = sem_img.astype(np.uint8)
        sem_img = Image.fromarray(sem_img)

        if self.depth_path_list is not None:
            depth_path = self.depth_path_list[index]
            depth_img = iio.imread(depth_path, as_gray=True)
            depth_img = Image.fromarray(depth_img)

            cls = self.cls_list[index]

            if self.TF_train == True:
                if np.random.rand(1)[0] > 0.5:
                    sem_img = TF.hflip(sem_img)
                    depth_img = TF.hflip(depth_img)
                if np.random.rand(1)[0] > 0.5:
                    sem_img = TF.vflip(sem_img)
                    depth_img = TF.vflip(depth_img)

                # 동일영역 자르기 위함
                if self.resize_and_crop==True:
                    sem_img = TF.resize(sem_img, size = (120,80))
                    depth_img = TF.resize(depth_img, size = (120,80))

                    top = np.random.choice(np.arange(120 - 96), 1)[0]
                    left = np.random.choice(np.arange(80 - 64), 1)[0]

                    sem_img = TF.resized_crop(sem_img, top=top, left=left, height=96, width=64, size=(96,64), interpolation = Image.NEAREST)
                    depth_img = TF.resized_crop(depth_img, top=top, left=left, height=96, width=64, size=(96,64), interpolation = Image.NEAREST)


                return self.transform(sem_img), self.transform_val(depth_img), cls
            else:
                return self.transform_val(sem_img), self.transform_val(depth_img), cls
        else:
            img_name = sem_path.split('/')[-1]
            return self.transform_val(sem_img), img_name  # B,C,H,W

    def __len__(self):
        return len(self.sem_path_list)

class CustomDataset_prev(Dataset):
    def __init__(self, sem_path_list, depth_path_list, TF_train):
        self.sem_path_list = sem_path_list
        self.depth_path_list = depth_path_list
        self.TF_train = TF_train

    def __getitem__(self, index):
        sem_path = self.sem_path_list[index]
        sem_img = cv2.imread(sem_path, cv2.IMREAD_GRAYSCALE)
        sem_img = cv2.resize(sem_img,(80,48))

        # labled data에는 noise 존재하기 때문에 50% 확률로 noise 중화시키기 각각 다른 kernel 써서 조금 다양성 보충
        if self.TF_train==True:
            if np.random.rand(1)[0]>0.5:
                kernel_ls=[3,5,7]
                k_size = np.random.choice(kernel_ls,size=1)
                sem_img = cv2.GaussianBlur(sem_img, (k_size[0], k_size[0]), 1)


        sem_img = np.expand_dims(sem_img, axis=-1).transpose(2, 0, 1)
        # 0 ~ 1 기준 scailing
        sem_img = sem_img / 255.

        # -1 ~ 1 기준 scalilng
        # sem_img = (sem_img - 127.5) / 127.5

        if self.depth_path_list is not None:
            depth_path = self.depth_path_list[index]
            depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
            depth_img = cv2.resize(depth_img, (80,48))
            depth_img = np.expand_dims(depth_img, axis=-1).transpose(2, 0, 1)
            # 0 ~ 1 기준 scailing
            depth_img = depth_img / 255.

            # -1 ~ 1 기준 scalilng
            # depth_img = (depth_img - 127.5) / 127.5
            return torch.Tensor(sem_img), torch.Tensor(depth_img)  # B,C,H,W
        else:
            img_name = sem_path.split('/')[-1]
            return torch.Tensor(sem_img), img_name  # B,C,H,W

    def __len__(self):
        return len(self.sem_path_list)


def noisy(image):
    row,col= image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = image + gauss
    return noisy


# Unsupervised dataset임으로
#
class US_CustomDataset(Dataset):
    def __init__(self, sem_path_list):
        self.sem_path_list = sem_path_list

    def __getitem__(self, index):
        sem_path = self.sem_path_list[index]
        sem_img = cv2.imread(sem_path, cv2.IMREAD_GRAYSCALE)
        sem_img = cv2.resize(sem_img,(80,48))

        # labled data에는 noise 존재하기 때문에 50% 확률로 noise 중화시키기 각각 다른 kernel 써서 조금 다양성 보충

        N_sem_img = noisy(sem_img)

        sem_img = np.expand_dims(sem_img, axis=-1).transpose(2, 0, 1)
        # 0 ~ 1 기준 scailing
        sem_img = sem_img / 255.


        N_sem_img = np.expand_dims(N_sem_img, axis=-1).transpose(2, 0, 1)
        # 0 ~ 1 기준 scailing
        N_sem_img = N_sem_img / 255.

        # -1 ~ 1 기준 scalilng
        # sem_img = (sem_img - 127.5) / 127.5


        return torch.Tensor(sem_img), torch.Tensor(N_sem_img)  # B,C,H,W

    def __len__(self):
        return len(self.sem_path_list)


# meta pseudo learning 형태로 진행 예정
# teacher model과 student model은 동일하게 두고 teacher model은 pretrain model로 student의 feedback 이외에는 학습진행 안하고
# student는 unlabled만 활용하여 진행(경향봐가면서 바꿀수도있음)
def train_psedo(model, optimizer, train_loader, val_loader, scheduler, device, CFG, outputpath, kfold_idx, u_train_loader, student_model, stu_optimizer, stu_scheduler,train_save):
    teachermodel = model.to(device)
    student_model = student_model.to(device)

    UDA_criterion = nn.MSELoss().to(device)
    # 논문뒤져보다 보다 찾은 loss
    criterion = Scale_invariant_logloss().to(device)

    best_score = 999999
    best_model = None


    for epoch in range(1, CFG['EPOCHS'] + 1):
        teachermodel.train()
        student_model.train()
        train_loss = []
        train_stu_loss = []
        save_num = 0
        iters=0
        for sem, depth in tqdm(iter(train_loader)):
            batch_size = sem.shape[0]
            sem = sem.float().to(device)
            depth = depth.float().to(device)

            ## un ==> unlabeled normal data / us ==> unlabled augmentation data
            un, us = next(iter(u_train_loader))

            un = un.float().to(device)
            us = us.float().to(device)

            teacherinput = torch.cat((un,us))

            toutputs = teachermodel(teacherinput)

            toutputs_un, toutputs_us = toutputs[:batch_size], toutputs[batch_size:]

            pseudo_lbl = toutputs_un.detach() * 255.

            pseudo_lbl = torch.clip(pseudo_lbl, min=0, max=255) / 255.

            train_udaloss = UDA_criterion(toutputs_un, toutputs_us)

            ## student ==>  unlabel 학습
            student_input = torch.cat((sem, us))
            s_logits = student_model(student_input)
            s_logits_l = s_logits[:batch_size]
            s_logits_us = s_logits[batch_size:]
            del s_logits

            s_loss_l_old = criterion(s_logits_l.detach(), depth)
            s_loss = criterion(s_logits_us, pseudo_lbl)

            stu_optimizer.zero_grad()
            s_loss.backward()
            stu_optimizer.step()

            ## student feedback
            with torch.no_grad():
                s_logits_l = student_model(sem)
            s_loss_l_new = criterion(s_logits_l.detach(), depth)

            dot_product = torch.clip(s_loss_l_new - s_loss_l_old,min=0,max=1)

            t_loss_mpl = dot_product * criterion(toutputs_us,pseudo_lbl)
            tloss = train_udaloss*0.001 + t_loss_mpl*0.00001

            optimizer.zero_grad()
            tloss.backward()
            optimizer.step()

            train_loss.append(tloss.item())
            train_stu_loss.append(s_loss.item())

            if iters%100==0:
                model_pred = s_logits_us[0].detach().cpu().numpy() * 255
                # -1~1로 input scailing 했을 때
                # model_pred = (model_pred.detach().cpu().numpy() * 127.5) + 127.5
                model_pred = np.clip(model_pred, 0, 255)
                cv2.imwrite(train_save + '/' + str(epoch)+'_'+str(save_num)+'_'+'stuoutput.png',
                            cv2.resize(model_pred[0].astype(np.uint8), (48, 72)))

                pseudo_vis = pseudo_lbl[0].cpu().numpy() * 255
                pseudo_vis = np.clip(pseudo_vis, 0, 255)

                cv2.imwrite(train_save + '/' + str(epoch) + '_' + str(save_num) + '_' + 'pseudo_vis.png',
                            cv2.resize(pseudo_vis[0].astype(np.uint8), (48, 72)))
                save_num+=1
            iters+=1

        val_loss, val_rmse = validation(student_model, criterion, val_loader, device)
        print(
            f'Epoch : [{epoch}] Train teacher Loss : [{np.mean(train_loss):.5f}] Val Loss : [{val_loss:.5f}] Val RMSE : [{val_rmse:.5f}] Train student Loss : [{np.mean(train_stu_loss)                                                      :.5f}]')

        if best_score > val_rmse:
            best_score = val_rmse
            best_model = student_model

        if scheduler is not None:
            scheduler.step()
            stu_scheduler.step()

        torch.save(best_model.state_dict(), outputpath + kfold_idx + 'fold_best_pseudo.pt')



def train(model, optimizer, train_loader, val_loader, scheduler, device, CFG, outputpath, kfold_idx):
    model = model.to(device)
    # smooth L1 (기존 loss)
    # criterion = nn.SmoothL1Loss().to(device)
    # 논문뒤져보다 보다 찾은 loss
    criterion = Scale_invariant_logloss().to(device)
    criterion_cls = nn.CrossEntropyLoss().to(device)
    best_score = 999999
    best_model = None

    for epoch in range(1, CFG['EPOCHS'] + 1):
        ## 40Epoch 이전까지 resize and crop 하다가 이후에 없앰
        if epoch==30:
            train_loader.dataset.resize_and_crop=False
        model.train()
        # model.apply(deactivate_batchnorm)
        train_loss = []

        for sem, depth, cls in tqdm(iter(train_loader)):
            sem = sem.float().to(device)
            depth = depth.float().to(device)
            cls = cls.type(torch.LongTensor).to(device)

            optimizer.zero_grad()

            model_pred,cls_pred = model(sem)
            loss = criterion(model_pred, depth)
            loss_cls = criterion_cls(cls_pred,cls)

            loss_sum = loss_cls+loss

            loss_sum.backward()
            optimizer.step()

            train_loss.append(loss.item())

        val_loss, val_rmse, val_acc = validation(model, criterion, val_loader, device)
        print(
            f'Epoch : [{epoch}] Train Loss : [{np.mean(train_loss):.5f}] Val Loss : [{val_loss:.5f}] Val RMSE : [{val_rmse:.5f}] Val ACC : [{val_acc:.5f}]')

        if best_score > val_rmse:
            best_score = val_rmse
            best_model = model

        if scheduler is not None:
            scheduler.step()

        torch.save(best_model.state_dict(), outputpath + kfold_idx + 'fold_best.pt')


def train_classifier(model, optimizer, train_loader, val_loader, scheduler, device, CFG, outputpath, kfold_idx):
    model = model.to(device)
    # smooth L1 (기존 loss)
    # criterion = nn.SmoothL1Loss().to(device)
    # 논문뒤져보다 보다 찾은 loss
    criterion_cls = nn.CrossEntropyLoss().to(device)
    best_score = 0
    best_model = None

    for epoch in range(1, CFG['EPOCHS'] + 1):
        ## 40Epoch 이전까지 resize and crop 하다가 이후에 없앰
        if epoch == 30:
            train_loader.dataset.resize_and_crop = False
        model.train()
        train_loss = []
        train_acc = []

        for sem, depth, cls in tqdm(iter(train_loader)):
            sem = sem.float().to(device)
            cls = cls.type(torch.LongTensor).to(device)

            optimizer.zero_grad()

            cls_pred = model(sem)
            loss_cls = criterion_cls(cls_pred, cls)
            loss_cls.backward()
            optimizer.step()

            train_loss.append(loss_cls.item())
            acc = torch.sum(cls_pred.argmax(dim=1) == cls) / cls.shape[0]
            train_acc.append(acc.item())

        val_acc = validation_cls(model, val_loader, device)
        print(
            f'Epoch : [{epoch}] Train Loss : [{np.mean(train_loss):.5f}] Train acc : [{np.mean(train_acc):.5f}] Val ACC : [{val_acc:.5f}]')

        if best_score < val_acc:
            best_score = val_acc
            best_model = model

        if scheduler is not None:
            scheduler.step()

        torch.save(best_model.state_dict(), outputpath + kfold_idx + 'fold_classifier_best.pt')


def train(model, optimizer, train_loader, val_loader, scheduler, device, CFG, outputpath, kfold_idx):
    model = model.to(device)
    # smooth L1 (기존 loss)
    # criterion = nn.SmoothL1Loss().to(device)
    # 논문뒤져보다 보다 찾은 loss
    criterion = Scale_invariant_logloss().to(device)
    criterion_cls = nn.CrossEntropyLoss().to(device)
    best_score = 999999
    best_model = None

    for epoch in range(1, CFG['EPOCHS'] + 1):
        ## 40Epoch 이전까지 resize and crop 하다가 이후에 없앰
        if epoch==30:
            train_loader.dataset.resize_and_crop=False
        model.train()
        # model.apply(deactivate_batchnorm)
        train_loss = []

        for sem, depth, cls in tqdm(iter(train_loader)):
            sem = sem.float().to(device)
            depth = depth.float().to(device)
            cls = cls.type(torch.LongTensor).to(device)

            optimizer.zero_grad()

            model_pred,cls_pred = model(sem)
            loss = criterion(model_pred, depth)
            loss_cls = criterion_cls(cls_pred,cls)

            loss_sum = loss_cls+loss

            loss_sum.backward()
            optimizer.step()

            train_loss.append(loss.item())

        val_loss, val_rmse, val_acc = validation(model, criterion, val_loader, device)
        print(
            f'Epoch : [{epoch}] Train Loss : [{np.mean(train_loss):.5f}] Val Loss : [{val_loss:.5f}] Val RMSE : [{val_rmse:.5f}] Val ACC : [{val_acc:.5f}]')

        if best_score > val_rmse:
            best_score = val_rmse
            best_model = model

        if scheduler is not None:
            scheduler.step()

        torch.save(best_model.state_dict(), outputpath + kfold_idx + 'fold_best.pt')


def train_depth_embed(model, pretrained_cls, optimizer, train_loader, val_loader, scheduler, device, CFG, outputpath, kfold_idx):
    model = model.to(device)
    pretrained_cls = pretrained_cls.to(device)
    # smooth L1 (기존 loss)
    criterion = nn.SmoothL1Loss().to(device)
    best_score = 0
    best_model = None

    for epoch in range(1, CFG['EPOCHS'] + 1):
        ## 30Epoch 이전까지 resize and crop 하다가 이후에 없앰
        if epoch == 30:
            train_loader.dataset.resize_and_crop = False
        model.train()
        train_loss = []
        train_acc = []

        for sem_cls,sem_for_train, depth, cls in tqdm(iter(train_loader)):
            sem_cls = sem_cls.float().to(device)
            sem_for_train = sem_for_train.float().to(device)
            depth = depth.float().to(device)
            cls = cls.type(torch.LongTensor).to(device)

            ## 학습용이 아닌 class 판별이후 one-hot encoding 처리용
            with torch.no_grad():
                cls_pred = pretrained_cls(sem_cls)
                onehot = torch.eye(4, dtype=torch.int)
                cls_onehot = onehot[torch.argmax(cls_pred,1)]

            cls_onehot=cls_onehot.to(device)
            ## 학습
            optimizer.zero_grad()
            model_pred = model(sem_for_train,cls_onehot)
            loss = criterion(model_pred, depth)
            loss.backward()
            optimizer.step()

            train_loss.append(cls_pred.item())
            acc = torch.sum(cls_pred.argmax(dim=1) == cls) / cls.shape[0]
            train_acc.append(acc.item())

        val_RMSE,val_acc = validation_depth_embed(model,pretrained_cls , val_loader, device)
        print(
            f'Epoch : [{epoch}] Train Loss : [{np.mean(train_loss):.5f}] Train acc : [{np.mean(train_acc):.5f}] Val ACC : [{val_acc:.5f}]')

        if best_score < val_RMSE:
            best_score = val_RMSE
            best_model = model

        if scheduler is not None:
            scheduler.step()

        torch.save(best_model.state_dict(), outputpath + kfold_idx + 'fold_embed_unet_best.pt')

def validation_depth_embed(model,pretrained_cls, val_loader, device):
    model.eval()
    l1 = nn.L1Loss().to(device)

    val_rmse = []
    val_acc = []
    with torch.no_grad():
        for sem_cls,sem_for_train, depth, cls in tqdm(iter(val_loader)):
            sem_cls = sem_cls.float().to(device)
            sem_for_train = sem_for_train.float().to(device)
            depth = depth.float().to(device)

            ## 학습용이 아닌 class 판별이후 one-hot encoding 처리용
            with torch.no_grad():
                cls_pred = pretrained_cls(sem_cls)
                onehot = torch.eye(4, dtype=torch.int)
                cls_onehot = onehot[torch.argmax(cls_pred, 1)]

            cls_onehot = cls_onehot.to(device)
            model_pred = model(sem_for_train, cls_onehot)

            ## 0~1 로 input scailing 했을 때
            pred = (model_pred * 255.).type(torch.int8).float()
            true = (depth * 255.).type(torch.int8).float()

            train_acc = torch.sum(cls_pred.argmax(dim=1) == cls) / cls.shape[0]

            # -1~1로 input scailing 했을 때
            # pred = ((model_pred.clamp(-1.0, 1.0) + 1) / 2) * 255.0
            # true = ((depth.clamp(-1.0, 1.0) + 1) / 2) * 255.0

            b_rmse = torch.sqrt(l1(pred, true))

            val_rmse.append(b_rmse.item())
            val_acc.append(train_acc.item())

    return np.mean(val_rmse), np.mean(val_acc)


def validation_cls(model, val_loader, device):
    model.eval()

    val_acc = []
    with torch.no_grad():
        for sem, depth, cls in tqdm(iter(val_loader)):
            sem = sem.float().to(device)
            depth = depth.float().to(device)
            cls = cls.float().to(device)

            cls_pred = model(sem)

            ## 0~1 로 input scailing 했을 때

            train_acc = torch.sum(cls_pred.argmax(dim=1) == cls) / cls.shape[0]

            # -1~1로 input scailing 했을 때
            # pred = ((model_pred.clamp(-1.0, 1.0) + 1) / 2) * 255.0
            # true = ((depth.clamp(-1.0, 1.0) + 1) / 2) * 255.0

            val_acc.append(train_acc.item())

    return np.mean(val_acc)

def validation(model, criterion, val_loader, device):
    model.eval()
    l1 = nn.L1Loss().to(device)

    val_loss = []
    val_rmse = []
    val_acc = []
    with torch.no_grad():
        for sem, depth, cls in tqdm(iter(val_loader)):
            sem = sem.float().to(device)
            depth = depth.float().to(device)
            cls = cls.float().to(device)

            model_pred, cls_pred = model(sem)
            loss = criterion(model_pred, depth)

            ## 0~1 로 input scailing 했을 때
            pred = (model_pred * 255.).type(torch.int8).float()
            true = (depth * 255.).type(torch.int8).float()

            train_acc = torch.sum(cls_pred.argmax(dim=1) == cls) / cls.shape[0]

            # -1~1로 input scailing 했을 때
            # pred = ((model_pred.clamp(-1.0, 1.0) + 1) / 2) * 255.0
            # true = ((depth.clamp(-1.0, 1.0) + 1) / 2) * 255.0

            b_rmse = torch.sqrt(l1(pred, true))

            val_loss.append(loss.item())
            val_rmse.append(b_rmse.item())
            val_acc.append(train_acc.item())

    return np.mean(val_loss), np.mean(val_rmse), np.mean(val_acc)

def test_embed(model,pretrained_cls, savefolder, val_loader, device):
    model = model.to(device)
    model.eval()
    pretrained_cls = pretrained_cls.to(device)
    pretrained_cls.eval()

    with torch.no_grad():
        for sem, img_name in tqdm(iter(val_loader)):
            sem = sem.float().to(device)

            cls_pred = pretrained_cls(sem)
            onehot = torch.eye(4, dtype=torch.int)
            cls_onehot = onehot[torch.argmax(cls_pred, 1)]

            cls_onehot = cls_onehot.to(device)
            model_pred = model(sem, cls_onehot)

            ## 0~1 로 input scailing 했을 때
            model_pred = model_pred.detach().cpu().numpy()*255
            # -1~1로 input scailing 했을 때
            # model_pred = (model_pred.detach().cpu().numpy() * 127.5) + 127.5
            model_pred = np.clip(model_pred, 0, 255)
            model_pred[np.where(model_pred > (np.max(model_pred) - (np.max(model_pred) % 10)))] = (np.max(model_pred) - (np.max(model_pred) % 10))
            cv2.imwrite(savefolder+'/'+img_name[0] ,cv2.resize(model_pred[0][0].astype(np.uint8),(48,72),interpolation=cv2.INTER_AREA))



def test(model, savefolder, val_loader, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        for sem, img_name in tqdm(iter(val_loader)):
            sem = sem.float().to(device)

            model_pred = model(sem)
            ## 0~1 로 input scailing 했을 때
            model_pred = model_pred.detach().cpu().numpy()*255
            # -1~1로 input scailing 했을 때
            # model_pred = (model_pred.detach().cpu().numpy() * 127.5) + 127.5
            model_pred = np.clip(model_pred, 0, 255)
            cv2.imwrite(savefolder+'/'+img_name[0] ,cv2.resize(model_pred[0][0].astype(np.uint8),(48,72),interpolation=cv2.INTER_AREA))


def test_ensemble(model, savefolder, val_loader, device, paths):
    model.to(device)
    model.eval()
    i=0
    with torch.no_grad():
        for sem, img_name in iter(val_loader):
            sem = sem.float().to(device)
            model.load_state_dict(torch.load(paths[0]))
            model_pred = model(sem)
            for i_w in range(1, len(paths)):
                model.load_state_dict(torch.load(paths[i_w]))
                model_pred += model(sem)
                ## 0~1 로 input scailing 했을 때
            model_pred = ((model_pred.detach().cpu().numpy() / len(paths))) * 255
            # -1~1로 input scailing 했을 때
            # model_pred = (model_pred.detach().cpu().numpy() * 127.5) + 127.5
            model_pred = np.clip(model_pred, 0, 255)
            cv2.imwrite(savefolder+'/'+img_name[0] ,cv2.resize(model_pred[0][0].astype(np.uint8),(48,72),interpolation=cv2.INTER_AREA))

            if i%1000==0:
                print(str(i)+' end!')
            i+=1
