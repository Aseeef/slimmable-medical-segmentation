from torchmetrics import JaccardIndex
from torchmetrics import F1Score


def get_seg_metrics(config, metric_name):
    if metric_name == 'iou':
        metrics = JaccardIndex(task='multiclass', num_classes=config.num_class,
                               ignore_index=config.ignore_index, average='none', )
    elif metric_name == 'dice' or metric_name == 'f1':
        metrics = F1Score(task='multiclass', num_classes=config.num_class, ignore_index=config.ignore_index,
                          average='macro')
    else:
        raise ValueError(f'Unsupported metric: {metric_name}.\n')

    return metrics
