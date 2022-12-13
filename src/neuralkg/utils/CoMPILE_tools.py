import torch

from models4 import *
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
from copy import deepcopy

from preprocess_inductive import read_entity_from_id, read_relation_from_id, init_embeddings, build_data
from create_batch_inductive2 import *
from utils import save_model, process_data
from torch.utils.data import DataLoader, Dataset
import random
import argparse
import os
import sys
import logging
import time
import pickle

import multiprocessing as mp
import time
import tqdm

def inductive_subgraph(args):
    Corpus_ = load_data(args)
    datapath = "./data/{}_{}_hop_new_data.pickle".format(args.dataset, hop)
    datapath2 = "./subgraph/{}_{}_hop_undirected_subgraph.pickle".format(args.dataset, hop)
    if os.path.isfile(datapath2):