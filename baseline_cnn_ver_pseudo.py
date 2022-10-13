import random
import os
import glob

from models import *
from train_utils import *

import warnings
warnings.filterwarnings(action='ignore')
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import RandomSampler, SequentialSampler
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CFG = {
    'WIDTH':48,
    'HEIGHT':72,
    'EPOCHS':100,
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

uns_Ds = sorted(glob.glob('train/SEM/*/*/*'))
uds = US_CustomDataset(uns_Ds)

u_train_loader = DataLoader(uds,batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=0 , drop_last=True)
### SEM 2개와 label (depth) 1개가 매칭되기 때문에 depth 기준으로 split 한 이후에 SEM matching 진행
train_paths=[]

label_paths=[]
lbls_for_split=[]
simulation_depth_paths = sorted(glob.glob('simulation_data/Depth/*'))
i=0
for f_paths in simulation_depth_paths:
    cases = sorted(glob.glob(f_paths+'/*'))
    for c_paths in cases:
        train_lbl = sorted(glob.glob(c_paths + '/*'))
        lbls_for_split+=(np.ones(shape=(len(train_lbl)))*i).tolist()
        label_paths+=train_lbl
        i+=1

# 9:1 비율로 데이터 뽑음
# train 기준으로
# 10-fold 로 setting
kf = StratifiedKFold(n_splits=10)
lbls = np.array(label_paths)
outputpath='outputs/unet_last_5x5_kernel/'
for kfold_idx,(index_kf_train, index_kf_validation) in enumerate(kf.split(label_paths, lbls_for_split)):
    train_lbl = lbls[index_kf_train]
    train_lbl = sorted(train_lbl.tolist()+train_lbl.tolist())
    t = np.char.replace(lbls[index_kf_train], 'Depth', 'SEM')
    train_t0 = np.char.replace(t, '.png', '_itr0.png')
    train_t1 = np.char.replace(t, '.png', '_itr1.png')
    tds = sorted(train_t0.tolist() + train_t1.tolist())

    valid_lbl = lbls[index_kf_validation]
    valid_lbl = sorted(valid_lbl.tolist() + valid_lbl.tolist())
    v = np.char.replace(lbls[index_kf_validation], 'Depth', 'SEM')
    valid_t0 = np.char.replace(v, '.png', '_itr0.png')
    valid_t1 = np.char.replace(v, '.png', '_itr1.png')
    vds = sorted(valid_t0.tolist()+valid_t1.tolist())

    train_dataset = CustomDataset(tds, train_lbl, True)
    train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=0, drop_last=True)

    val_dataset = CustomDataset(vds, valid_lbl, False)
    val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

    model = UNet(n_channels=1, n_classes=1, bilinear=True)
    studentmodel = UNet(n_channels=1, n_classes=1, bilinear=True)

    #pretrained teacher load
    model.load_state_dict(torch.load(outputpath+str(kfold_idx)+'fold_best.pt'))
    # model = Unet_trans(n_channels=1, n_classes=1, bilinear=False)
    model.eval()

    optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1, verbose=False)

    stu_optimizer = torch.optim.Adam(params=studentmodel.parameters(), lr=CFG["LEARNING_RATE"])
    stu_scheduler = torch.optim.lr_scheduler.StepLR(stu_optimizer, 30, gamma=0.1, verbose=False)

    train_save='outputs/unet_last_5x5_kernel/train/'
    train_psedo(model, optimizer, train_loader, val_loader, scheduler, device, CFG, outputpath, str(kfold_idx), u_train_loader, studentmodel,stu_optimizer,stu_scheduler,train_save)