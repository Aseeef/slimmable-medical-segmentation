import argparse

import configs
from core import SegTrainer
from models import model_hub, get_model, list_available_models #import model registry

import warnings
warnings.filterwarnings("ignore")

# TODO:
if __name__ == '__main__':
    # Get available configurations
    available_configs = configs.list_available_configs()

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Run segmentation training or testing")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=available_configs,
        help=f"Choose a config from the available options: {available_configs}"
    )

    args = parser.parse_args()

    # Fetch the selected configuration


    selected_config = args.config
    config = configs.get_config(selected_config)()
    config.init_dependent_config()

    trainer = SegTrainer(config)

    if config.is_testing:
        trainer.predict(config)
    else:    
        trainer.run(config)
    