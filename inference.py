import argparse
import os

import configs
from core import get_trainer

import warnings
warnings.filterwarnings("ignore")

# This file was used to measure the inference speed of our network.
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
    config = configs.get_config(selected_config.lower())()
    config.init_dependent_config()
    config.is_testing = True
    config.load_ckpt_path =  f'{config.save_dir}/best.pth'
    if config.test_data_folder is None:
        config.test_data_folder = os.path.join(config.data_root, 'test', 'images')

    trainer = get_trainer(config)
    trainer.predict(config, 1)  # warmup
    for w in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        print(f'Running width {w}')
        trainer.predict(config, w)
    