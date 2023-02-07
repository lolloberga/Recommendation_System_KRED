import argparse
from train_test import *
from utils.util import *
import pickle
from parse_config import ConfigParser

import os

def main(config):

    if os.path.exists("./data/data_mind_small.pkl"):
        with open("./data/data_mind_small.pkl", 'rb') as f:
            data = pickle.load(f)
    else:
        data = load_data_mind(config)

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
    config['trainer']['epochs'] = 10
    config['data_loader']['batch_size'] = 64
    config['trainer']['training_type'] = "single_task"
    config['trainer']['task'] = "user2item" # task should be within: user2item, item2item, vert_classify, pop_predict
    # The following parameters define which of the extensions are used, 
    # by setting them to False the original KRED model is executed 
    config['model']['use_mh_attention'] = False
    config['model']['mh_number_of_heads'] = 12
    config['data']['use_entity_category'] = True
    config['data']['use_second_entity_category'] = False

    main(config)