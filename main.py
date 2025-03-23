from core import SegTrainer
from configs import DucknetConfig, load_parser

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    config = DucknetConfig()

    config.init_dependent_config()

    # If you want to use command-line arguments, please uncomment the following line
    # config = load_parser(config)

    trainer = SegTrainer(config)

    if config.is_testing:
        trainer.predict(config)
    else:    
        trainer.run(config)