trainer_hub = {}


def register_trainer(trainer_class):
    trainer_hub[trainer_class.__name__.lower()] = trainer_class
    return trainer_class
