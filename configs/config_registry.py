config_hub = {}


def register_config(config_class):
    config_hub[config_class.__name__.lower()] = config_class
    return config_class
