import os
import sys
import csv
import time
import json
import argparse
from os.path import join as pjoin
import numpy as np
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import logging
import nltk

from seq2seq import PhenomenonEncoder, L2EDecoder

import uuid
import pickle

from onmt.model_builder import build_base_model
from onmt import inputters

model_file_path = "model/dissent_step_80000.pt"
temp_dir = "/tmp/"

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__name__)

encoder = PhenomenonEncoder(model_file_path, temp_dir, logger)

decoder = L2EDecoder(encoder)

def decode_sent(sent):

    # TODO: we do preprocessing here

    decoded_tups = decoder.decode_sentences([sent])
    print(decoded_tups)
    return decoded_tups[0]
