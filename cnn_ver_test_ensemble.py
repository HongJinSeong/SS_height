import random
import os
import glob
from train_utils import *
from models import *

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
test_dataset = CustomDataset(simulation_sem_paths,None, False)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

# model = UNet(n_channels=1, n_classes=1, bilinear=False)
model = UNet(n_channels=1, n_classes=1, bilinear=True)
model.eval()

simulation_sem_paths = sorted(glob.glob('test/SEM/*.png'))
savefolder='outputs/unet_last_5x5_kernel/submit'
weights_paths = sorted(glob.glob('outputs/unet_last_5x5_kernel/*.pt'))

test_ensemble(model,savefolder,test_loader,device,weights_paths)
