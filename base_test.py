import random
import pandas as pd
import numpy as np
import os
import glob
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings(action='ignore')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CFG = {
    'WIDTH':48,
    'HEIGHT':72,
    'EPOCHS':20,
    'LEARNING_RATE':1e-3,
    'BATCH_SIZE':1,
    'SEED':2022
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정


simulation_sem_paths = sorted(glob.glob('test/SEM/*.png'))


data_len = len(simulation_sem_paths)


class CustomDataset(Dataset):
    def __init__(self, sem_path_list, depth_path_list):
        self.sem_path_list = sem_path_list
        self.depth_path_list = depth_path_list

    def __getitem__(self, index):
        sem_path = self.sem_path_list[index]
        sem_img = cv2.imread(sem_path, cv2.IMREAD_GRAYSCALE)
        sem_img = np.expand_dims(sem_img, axis=-1).transpose(2, 0, 1)
        sem_img = sem_img / 255.

        if self.depth_path_list is not None:
            depth_path = self.depth_path_list[index]
            depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
            depth_img = np.expand_dims(depth_img, axis=-1).transpose(2, 0, 1)
            depth_img = depth_img / 255.
            return torch.Tensor(sem_img), torch.Tensor(depth_img)  # B,C,H,W
        else:
            img_name = sem_path.split('/')[-1]
            return torch.Tensor(sem_img), img_name  # B,C,H,W

    def __len__(self):
        return len(self.sem_path_list)


test_dataset = CustomDataset(simulation_sem_paths,None)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(CFG['HEIGHT'] * CFG['WIDTH'], 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, CFG['HEIGHT'] * CFG['WIDTH']),
        )

    def forward(self, x):
        x = x.view(-1, CFG['HEIGHT'] * CFG['WIDTH'])
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(-1, 1, CFG['HEIGHT'], CFG['WIDTH'])
        return x

def test(model, savefolder, val_loader, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        for sem, img_name in tqdm(iter(val_loader)):
            sem = sem.float().to(device)

            model_pred = model(sem)
            model_pred = model_pred.detach().cpu().numpy()*255
            model_pred = np.clip(model_pred, 0, 255)
            cv2.imwrite(savefolder+'/'+str.split(str.replace(img_name[0],'\\','/'),'/')[1] ,model_pred[0][0].astype(np.uint8))



model = BaseModel()
model.eval()
model_path='outputs/baseline_best.pt'
model.load_state_dict(torch.load(model_path))
savefolder='outputs/base'
test(model,savefolder,test_loader,device)
