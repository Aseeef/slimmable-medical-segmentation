from torchmetrics import JaccardIndex, Dice


def get_seg_metrics(config, metric_name):
    if metric_name == 'iou':
        metrics = JaccardIndex(task='multiclass', num_classes=config.num_class, 
                                ignore_index=config.ignore_index, average='none',)
    elif metric_name == 'dice':
        metrics = Dice(num_classes=config.num_class, average='macro')
    else:
        raise ValueError(f'Unsupported metric: {metric_name}.\n')

    return metrics