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
    'BATCH_SIZE':128,
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

simulation_sem_paths = sorted(glob.glob('simulation_data/SEM/*/*/*.png'))
simulation_depth_paths = sorted(glob.glob('simulation_data/Depth/*/*/*.png')+glob.glob('simulation_data/Depth/*/*/*.png'))

data_len = len(simulation_sem_paths) # == len(simulation_depth_paths)


train_sem_paths = simulation_sem_paths[:int(data_len*0.8)]
train_depth_paths = simulation_depth_paths[:int(data_len*0.8)]

val_sem_paths = simulation_sem_paths[int(data_len*0.8):]
val_depth_paths = simulation_depth_paths[int(data_len*0.8):]


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

train_dataset = CustomDataset(train_sem_paths, train_depth_paths)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

val_dataset = CustomDataset(val_sem_paths, val_depth_paths)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)


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


def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)
    criterion = nn.L1Loss().to(device)
    best_score = 999999
    best_model = None

    for epoch in range(1, CFG['EPOCHS'] + 1):
        model.train()
        train_loss = []
        for sem, depth in tqdm(iter(train_loader)):
            sem = sem.float().to(device)
            depth = depth.float().to(device)

            optimizer.zero_grad()

            model_pred = model(sem)
            loss = criterion(model_pred, depth)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        val_loss, val_rmse = validation(model, criterion, val_loader, device)
        print(
            f'Epoch : [{epoch}] Train Loss : [{np.mean(train_loss):.5f}] Val Loss : [{val_loss:.5f}] Val RMSE : [{val_rmse:.5f}]')

        if best_score > val_rmse:
            best_score = val_rmse
            best_model = model

        if scheduler is not None:
            scheduler.step()

    return best_model


def validation(model, criterion, val_loader, device):
    model.eval()
    rmse = nn.MSELoss().to(device)

    val_loss = []
    val_rmse = []
    with torch.no_grad():
        for sem, depth in tqdm(iter(val_loader)):
            sem = sem.float().to(device)
            depth = depth.float().to(device)

            model_pred = model(sem)
            loss = criterion(model_pred, depth)

            pred = (model_pred * 255.).type(torch.int8).float()
            true = (depth * 255.).type(torch.int8).float()

            b_rmse = torch.sqrt(criterion(pred, true))

            val_loss.append(loss.item())
            val_rmse.append(b_rmse.item())

    return np.mean(val_loss), np.mean(val_rmse)

model = BaseModel()
model.eval()
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
scheduler = None

infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)

outputpath='outputs/'

torch.save(infer_model.state_dict(), outputpath + 'baseline_best.pt')