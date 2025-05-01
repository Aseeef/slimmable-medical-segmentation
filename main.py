import argparse
import configs
from core import get_trainer

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    # Get available configurations
    available_configs = configs.list_available_configs()
    print(f"avaliable configs are {available_configs}")

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
    print('initializing config class')
    selected_config = args.config
    config = configs.get_config(selected_config)()
    config.init_dependent_config()

    # instantiate trainer
    # instantiate inference runner
    print('initializing trainer class by passing in configuration class')
    trainer = get_trainer(config)


    #run scripts
    # print('running scripts')
    # if config.is_testing:
    #     print('.predict')
    #     trainer.predict(config)
    #     print('finished predicitng')
    # # added for inference runner
    # else:    
    #     trainer.test(config)


    print('running scripts to run at different widths')
    widths = [0.25, 0.5, 0.75, 1.0]  
    results = []

    for width in widths:
        print(f'\nSetting network width to {width}')
        trainer.set_width(config, width)
        
        if config.is_testing:
            print('running .predict')
            outputs = trainer.predict(config)
            print('finished predicting')
            avg_time = outputs['avg_inference_time_sec']
        else:
            outputs = trainer.test(config)
        
        # Assume trainer.test or trainer.predict returns accuracy somehow
        # Adjust if your function returns differently
