from .env import benchmark

from .dataset import config as _dataset_config
from .labels import config as _labels_config
from .optical_flow import config as _optical_flow_config
from .object_bbox import config as _object_bbox_config

dataset_config = _dataset_config[benchmark]
labels_config = _labels_config[benchmark]
optical_flow_config = _optical_flow_config[benchmark]
object_bbox_config = _object_bbox_config[benchmark]

__all__ = ['benchmark', 'dataset_config', 'labels_config', 'optical_flow_config', 'object_bbox_config']