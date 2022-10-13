import torch
from models import *
import numpy as np
from torchvision.models.feature_extraction import get_graph_node_names
import timm

from torchvision.models.feature_extraction import create_feature_extractor

model = timm.create_model('efficientnet_b1', pretrained=True)
train_nodes, eval_nodes = get_graph_node_names(model)
#
return_nodes={
    train_nodes[10]  : 'f1',
    train_nodes[64]  : 'f2',
    train_nodes[176] : 'f3',
    train_nodes[288] : 'f4',
    train_nodes[438] : 'f5',
}
aa = create_feature_extractor(model,return_nodes)
inputs = torch.ones(size=(1,3,80,48))
model = MFF_depth()
ooo = model(inputs)


print('aaa')