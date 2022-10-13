#### simulation Depth case 별로 Max depth 다름

###
#### CASE_1 ==> 140
#### CASE_2 ==> 150
#### CASE_3 ==> 160
#### CASE_4 ==> 170


import random
import os
import glob

from models import *
from train_utils import *

import warnings

warnings.filterwarnings(action='ignore')
from sklearn.model_selection import StratifiedKFold

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CFG = {
    'WIDTH': 64,
    'HEIGHT': 96,
    'EPOCHS': 30,
    'LEARNING_RATE': 1e-3,
    'BATCH_SIZE': 32,
    'SEED': 2022
}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(CFG['SEED'])  # Seed 고정

uns_Ds = sorted(glob.glob('train/SEM/*/*'))
ds = US_CustomDataset(uns_Ds)

### SEM 2개와 label (depth) 1개가 매칭되기 때문에 depth 기준으로 split 한 이후에 SEM matching 진행
train_paths = []

label_paths = []
lbls_for_split = []
simulation_depth_paths = sorted(glob.glob('train/SEM/*'))
i = 0
for f_paths in simulation_depth_paths:
    paths = sorted(glob.glob(f_paths + '/*/*'))
    train_lbl = paths
    lbls_for_split += (np.ones(shape=(len(train_lbl))) * i).tolist()
    label_paths += train_lbl
    i += 1

# 9:1 비율로 데이터 뽑음
# train 기준으로
# 10-fold 로 setting
kf = StratifiedKFold(n_splits=10)
lbls = np.array(label_paths)
lbls_for_split = np.array(lbls_for_split)
outputpath = 'outputs/unet_CLS/'
for kfold_idx, (index_kf_train, index_kf_validation) in enumerate(kf.split(label_paths, lbls_for_split)):
    tds = lbls[index_kf_train]
    vds = lbls[index_kf_validation]

    ## 분류기를 더 달아서 이후에 후처리할 때나 복원 모델에 추가정보 제공하도록 set
    train_cls_lbl = lbls_for_split[index_kf_train].tolist()
    valid_cls_lbl = lbls_for_split[index_kf_validation].tolist()

    train_dataset = CustomDataset(tds, tds, train_cls_lbl, True)
    train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

    val_dataset = CustomDataset(vds, vds, valid_cls_lbl, False)
    val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

    model = classifier(n_channels=1, n_classes=4)
    model.eval()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=CFG["LEARNING_RATE"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15, gamma=0.1, verbose=False)

    train_classifier(model, optimizer, train_loader, val_loader, scheduler, device, CFG, outputpath, str(kfold_idx))
    break