from easydict import EasyDict as edict

config = edict()

# path
config.output = "work_dirs"
config.dataset_dir = "dataset"

# dataset
config.dataset_split = (0.8, 0.1, 0.1)
config.negative_num = 25
config.iou_threshold_1 = 0.3
config.part_num = 25
config.iou_threshold_2 = 0.7
config.positive_num = 75

config.img_size_pnet = (12, 12)
config.img_size_rnet = (24, 24)
config.img_size_onet = (48, 48)


# train
# this property also control the dataset generation
config.cascade_train = True


# log
config.use_wandb = False
config.wandb_key = None
config.wandb_project = "mtcnn"
config.wandb_enity = None
