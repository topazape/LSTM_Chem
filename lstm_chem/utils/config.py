import os
import time
import json
from bunch import Bunch


def get_config_from_json(json_file):
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
    config = Bunch(config_dict)
    return config


def process_config(json_file):
    config = get_config_from_json(json_file)
    config.config_file = json_file
    config.exp_dir = os.path.join(
        'experiments', time.strftime('%Y-%m-%d/', time.localtime()),
        config.exp_name)
    config.tensorboard_log_dir = os.path.join(
        'experiments', time.strftime('%Y-%m-%d/', time.localtime()),
        config.exp_name, 'logs/')
    config.checkpoint_dir = os.path.join(
        'experiments', time.strftime('%Y-%m-%d/', time.localtime()),
        config.exp_name, 'checkpoints/')
    return config
