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
    'WIDTH':48,
    'HEIGHT':72,
    'EPOCHS':60,
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

uns_Ds = sorted(glob.glob('train/SEM/*/*'))
ds = US_CustomDataset(uns_Ds)

### SEM 2개와 label (depth) 1개가 매칭되기 때문에 depth 기준으로 split 한 이후에 SEM matching 진행
train_paths=[]

label_paths=[]
lbls_for_split=[]
simulation_depth_paths = sorted(glob.glob('simulation_data/Depth/*'))
i=0
for f_paths in simulation_depth_paths:
    paths = sorted(glob.glob(f_paths+'/*/*'))
    train_lbl = paths
    lbls_for_split+=(np.ones(shape=(len(train_lbl)))*i).tolist()
    label_paths+=train_lbl
    i+=1

# 9:1 비율로 데이터 뽑음
# train 기준으로
# 10-fold 로 setting
kf = StratifiedKFold(n_splits=10)
lbls = np.array(label_paths)
lbls_for_split = np.array(lbls_for_split)
outputpath='outputs/MFF/'
for kfold_idx,(index_kf_train, index_kf_validation) in enumerate(kf.split(label_paths, lbls_for_split)):
    train_lbl = lbls[index_kf_train]
    train_lbl = train_lbl.tolist()+train_lbl.tolist()
    t = np.char.replace(lbls[index_kf_train], 'Depth', 'SEM')
    train_t0 = np.char.replace(t, '.png', '_itr0.png')
    train_t1 = np.char.replace(t, '.png', '_itr1.png')
    tds = train_t0.tolist() + train_t1.tolist()

    valid_lbl = lbls[index_kf_validation]
    valid_lbl = valid_lbl.tolist() + valid_lbl.tolist()
    v = np.char.replace(lbls[index_kf_validation], 'Depth', 'SEM')
    valid_t0 = np.char.replace(v, '.png', '_itr0.png')
    valid_t1 = np.char.replace(v, '.png', '_itr1.png')
    vds = valid_t0.tolist()+valid_t1.tolist()

    ## 분류기를 더 달아서 이후에 후처리할 때나 복원 모델에 추가정보 제공하도록 set
    train_cls_lbl = lbls_for_split[index_kf_train].tolist() + lbls_for_split[index_kf_train].tolist()
    valid_cls_lbl = lbls_for_split[index_kf_validation].tolist() + lbls_for_split[index_kf_validation].tolist()

    train_dataset = CustomDataset_for_embed(tds, train_lbl,train_cls_lbl, True)
    train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

    val_dataset = CustomDataset_for_embed(vds, valid_lbl,valid_cls_lbl, False)
    val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

    cls_model = classifier(n_channels=1, n_classes=4)
    model_path = 'outputs/unet_last_5x5_kernel/0fold_best.pt'
    # cls_model.load_state_dict(torch.load(model_path))
    cls_model.eval()

    # model = Unet_trans(n_channels=1, n_classes=1, bilinear=False)
    # model = MFF_depth()
    model = UNet_with_embed(n_channels=2, n_classes=1, bilinear=False)

    optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1, verbose=False)

    train_depth_embed(model,cls_model, optimizer, train_loader, val_loader, scheduler, device, CFG,outputpath,str(kfold_idx))
    break