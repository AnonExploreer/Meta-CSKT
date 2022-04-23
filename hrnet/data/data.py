import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from hrnet.data.animalweb import AnimalWeb

from hrnet.lib.config import config, update_config

logger = logging.getLogger(__name__)




def get_animalweb(args):
    
    train_labeled_dataset = AnimalWeb(config,args,label=True,is_train=True)
    train_unlabeled_dataset = AnimalWeb(config,args,label=False,is_train=True)
    test_dataset = AnimalWeb(config,args, True,False)
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


DATASET_GETTERS = {'animalweb':get_animalweb}
