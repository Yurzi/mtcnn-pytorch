from easydict import EasyDict as edict

config = edict()

config.net_name = "pnet"

# path
config.output = "work_dirs"
config.dataset_dir = "dataset"
config.weight_dir = "weights"

# dataset
config.dataset_split = (0.8, 0.1, 0.1) # train test eval

config.iou_threshold = (0.3, 0.7)
config.negative_num = 25
config.part_num = 25
config.positive_num = 75

config.img_size = (12, 12)

# train
# this property also control the dataset generation
config.cascade_train = True

config.fp16 = False
config.cuda = True

# worker num of dataloader
config.num_workers = 4
config.batch_size = 64
config.shuffle = True

config.optimizer_type="adam"    # "adam" | "sgd"
config.optimizer_params = {
    'lr': 1e-3,
    'betas': (0.9, 0.999),
    'eps': 1e-8,
    'weight_decay': 0
}
config.scheduler_type="steplr"    # "steplr" | "cosine"(unimplement)
config.scheduler_params = {
    'step_size': 100,
    'gamma' : 0.5
}

config.epoch = 20
config.save_epoch = 1

config.task_weight = (1, 0.5, 0.5)
config.ohem_rate = 0.7

# used for resume, test, eval and inference
config.weight = "last.pt"
config.save_all = True

# eval and test
config.acc_iou_threshold = 0.7
config.acc_ldmk_threshold = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

# log
config.use_wandb = False
config.wandb_key = None
config.wandb_project = "mtcnn"
config.wandb_enity = None
