# @Time   : 2020/6/28

"""
################################
"""

import re
import os
import yaml
import torch
from logging import getLogger
from utils.utils import get_model_type


class Config(object):
    """ Configurator module that load the defined parameters.

    External input supports three kind of forms: config file, command line and parameter dictionaries.

    - config file: It's a file that record the parameters to be modified or added. It should be in ``yaml`` format,
      e.g. a config file is 'example.yaml', the content is:

        learning_rate: 0.001

        train_batch_size: 2048

    - command line: It should be in the format as '---learning_rate=0.001'

    - parameter dictionaries: It should be a dict, where the key is parameter name and the value is parameter value,
      e.g. config_dict = {'learning_rate': 0.001}

    Configuration module allows the above three kind of external input format to be used together,
    the priority order is as following:

    command line > parameter dictionaries > config file

    e.g. If we set learning_rate=0.01 in config file, learning_rate=0.02 in command line,
    learning_rate=0.03 in parameter dictionaries.

    Finally the learning_rate is equal to 0.02.
    """

    def __init__(self, model=None, dataset=None, config_dict=None):
        """
        Args:
            model (str/AbstractRecommender): the model name or the model class, default is None, if it is None, config
            will search the parameter 'model' from the external input as the model name or model class.
            dataset (str): the dataset name, default is None, if it is None, config will search the parameter 'dataset'
            from the external input as the dataset name.
            config_file_list (list of str): the external config file, it allows multiple config files, default is None.
            config_dict (dict): the external parameter dictionaries, default is None.
        """
        # load dataset config file yaml
        config_dict['model'] = model
        config_dict['dataset'] = dataset
        # model type
        config_dict['model_type'] = get_model_type(model)
        self.final_config_dict = self._load_dataset_model_config(config_dict)
        # config in cmd and main.py are latest
        self.final_config_dict.update(config_dict)
        self._set_default_parameters()
        self._init_device()

    def _load_dataset_model_config(self, config_dict):
        file_config_dict = dict()
        file_list = []
        # get dataset and model files
        cur_dir = os.getcwd()
        cur_dir = os.path.join(cur_dir, 'configs')
        file_list.append(os.path.join(cur_dir, "overall.yaml"))
        file_list.append(os.path.join(cur_dir, "dataset", "{}.yaml".format(config_dict['dataset'])))
        file_list.append(os.path.join(cur_dir, "model", "{}.yaml".format(config_dict['model'])))

        for file in file_list:
            if os.path.isfile(file):
                with open(file, 'r', encoding='utf-8') as f:
                    file_config_dict.update(yaml.load(f.read(), Loader=self._build_yaml_loader()))
        return file_config_dict

    def _build_yaml_loader(self):
        loader = yaml.FullLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))
        return loader

    def _set_default_parameters(self):
        smaller_metric = ['rmse', 'mae', 'logloss']
        valid_metric = self.final_config_dict['valid_metric'].split('@')[0]
        self.final_config_dict['valid_metric_bigger'] = False if valid_metric in smaller_metric else True

    def _init_device(self):
        use_gpu = self.final_config_dict['use_gpu']
        if use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.final_config_dict['gpu_id'])
        self.final_config_dict['device'] = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.final_config_dict[key] = value

    def __getitem__(self, item):
        if item in self.final_config_dict:
            return self.final_config_dict[item]
        else:
            return None

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.final_config_dict

    def __str__(self):
        args_info = '\n'
        args_info += '\n'.join(["{}={}".format(arg, value) for arg, value in self.final_config_dict.items()])
        args_info += '\n\n'
        return args_info

    def __repr__(self):
        return self.__str__()
