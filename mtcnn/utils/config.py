import importlib
import os

from mtcnn.utils.filesystem import check_and_reset


def get_config(config_file):
    temp_config_name = os.path.basename(config_file)
    temp_module_name = os.path.splitext(temp_config_name)[0]
    config = importlib.import_module("configs.default")
    cfg = config.config
    config = importlib.import_module("configs.%s" % temp_module_name)
    job_cfg = config.config
    cfg.update(job_cfg)
    if cfg.output is None:
        cfg.output = os.path.join("work_dirs", temp_module_name)

    check_and_reset(cfg.output)
    return cfg
