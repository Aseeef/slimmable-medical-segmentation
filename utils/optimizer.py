from torch.optim import SGD, Adam, AdamW, RMSprop


def get_optimizer(config, model):    
    optimizer_hub = {'sgd':SGD, 'adam':Adam, 'adamw':AdamW, 'rmsprop': RMSprop}
    params = model.parameters()

    if config.optimizer_type == 'sgd':
        config.lr = config.base_lr * config.gpu_num
        optimizer = optimizer_hub[config.optimizer_type](params=params, lr=config.lr, 
                                                    momentum=config.momentum, 
                                                    weight_decay=config.weight_decay)

    elif config.optimizer_type in ['adam', 'adamw']:
        config.lr = 0.1 * config.base_lr * config.gpu_num
        optimizer = optimizer_hub[config.optimizer_type](params=params, lr=config.lr)
        
    elif config.optimizer_type == 'rmsprop':
        config.lr = 0.1 * config.base_lr * config.gpu_num
        optimizer = optimizer_hub[config.optimizer_type](
            params=params, lr=config.lr, 
            momentum=config.momentum, weight_decay=config.weight_decay
        )

    else:
        raise NotImplementedError(f'Unsupported optimizer type: {config.optimizer_type}')

    return optimizer