from configs.data.base import cfg

# Training and Validation sources
cfg.DATASET.TRAINVAL_DATA_SOURCE = "StereoLunar"
cfg.DATASET.TRAIN_DATA_ROOT = "data/stereo_lunar/scenes"
cfg.DATASET.TRAIN_NPZ_ROOT = "data/stereo_lunar"
cfg.DATASET.TRAIN_LIST_PATH = "data/stereo_lunar/train_list.txt"
cfg.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.0

# Testing sources
cfg.DATASET.TEST_DATA_SOURCE = "StereoLunar"
cfg.DATASET.VAL_DATA_ROOT = cfg.DATASET.TEST_DATA_ROOT = "data/stereo_lunar/scenes"
cfg.DATASET.VAL_NPZ_ROOT = cfg.DATASET.TEST_NPZ_ROOT = "data/stereo_lunar"
cfg.DATASET.VAL_LIST_PATH = cfg.DATASET.TEST_LIST_PATH = "data/stereo_lunar/val_list.txt"
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0

# Sampling & Image dimensions
cfg.TRAINER.N_SAMPLES_PER_SUBSET = 50
cfg.DATASET.MGDPT_IMG_RESIZE = 512     # StereoLunar native resolution
cfg.DATASET.MGDPT_IMG_PAD = False      # Uniform square images, no zero-padding needed
cfg.DATASET.MGDPT_DEPTH_PAD = False    # Uniform square depth, no zero-padding needed
cfg.DATASET.NPE_NAME = 'stereo_lunar'
