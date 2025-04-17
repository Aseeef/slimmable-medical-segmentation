import argparse

import configs
from core import get_trainer

# import inference runner
from core import SlimmableSegInferenceRunner

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    # Get available configurations
    available_configs = configs.list_available_configs()
    print(available_configs)

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

    # instantiate trainer
    # instantiate inference runner
    inference_runner = SlimmableSegInferenceRunner(config)
    trainer = get_trainer(config)

    # run scripts
    if config.is_testing:
        trainer.predict(config)
    # added for inference runner
    else:    
        inference_runner.test(config)