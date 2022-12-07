import argparse
from train_test import *
from utils.util import *
import pickle
from parse_config import ConfigParser

import os

# define constants
DATASET_PATH = "./data/"
SENTENCE_EMBEDDINGS_FOLDER = DATASET_PATH + "sentence_embeddings/"

def main(config):

    if os.path.exists("./data/data_mind_small.pkl"):
        with open("./data/data_mind_small.pkl", 'rb') as f:
            data = pickle.load(f)
    else:
        data = load_data_mind(config, SENTENCE_EMBEDDINGS_FOLDER)

    # Train the KRED model
    if config['trainer']['training_type'] == "single_task":
        single_task_training(config, data)
    else:
        multi_task_training(config, data)

    # Evaluate the KRED model
    test_data = data[-1]
    testing(test_data, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KRED')

    parser.add_argument('-c', '--config', default="./config.json", type=str,
                      help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(parser)

    # Hyperparams
    config['trainer']['epochs'] = 5
    config['data_loader']['batch_size'] = 64
    config['trainer']['training_type'] = "single_task"
    config['trainer']['task'] = "user2item" # task should be within: user2item, item2item, vert_classify, pop_predict

    main(config)