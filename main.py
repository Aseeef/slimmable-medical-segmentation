import argparse

import configs
from core import SegTrainer
from core.slimmable_seg_trainer import SlimmableSegTrainer

import warnings
warnings.filterwarnings("ignore")


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

    # Create the correct trainer. Slimmable or not
    if getattr(config, 'is_slimmable', False):
        trainer = SlimmableSegTrainer(config)
    else:
        trainer = SegTrainer(config)

    if config.is_testing:
        trainer.predict(config)
    else:    
        trainer.run(config)