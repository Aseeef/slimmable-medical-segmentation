from torchmetrics import JaccardIndex
from torchmetrics import F1Score


def get_seg_metrics(config, metric_name):
    if metric_name == 'iou':
        metrics = JaccardIndex(task='multiclass', num_classes=config.num_class,
                               ignore_index=config.ignore_index, average='none', )
    #elif metric_name == 'dice':
    #    metrics = Dice(num_classes=config.num_class, average='macro')  #TODO THIS ALREADY HAS DICE???
    # F1Score and Dice are mathematically equivalent so torchmetrics
    # removed Dice in recent versions
    elif metric_name == 'dice' or metric_name == 'f1':
        metrics = F1Score(task='multiclass', num_classes=config.num_class, ignore_index=config.ignore_index,
                          average='macro')
    else:
        raise ValueError(f'Unsupported metric: {metric_name}.\n')

    return metrics
