import importlib


def parse_config(config_name):
    module = importlib.import_module(config_name, "configs")
    return module.Config