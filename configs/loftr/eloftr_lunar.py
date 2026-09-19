from src.config.default import _CN as cfg

# === Fine-tuning config for StereoLunar (optimized for 6GB laptop GPU) ===

# Trainer config
cfg.TRAINER.CANONICAL_LR = 1e-4        # Fine-tuning rate (lower than 8e-3 training from scratch)
cfg.TRAINER.CANONICAL_BS = 1           # Single GPU laptop batch size
cfg.TRAINER.WARMUP_STEP = 100          # Short warmup
cfg.TRAINER.WARMUP_RATIO = 0.01
cfg.TRAINER.MSLR_MILESTONES = [5, 10, 15]
cfg.TRAINER.OPTIMIZER = "adamw"
cfg.TRAINER.ADAMW_DECAY = 0.01
cfg.TRAINER.GRADIENT_CLIPPING = 1.0
cfg.TRAINER.EPI_ERR_THR = 5e-4
cfg.TRAINER.POSE_GEO_MODEL = 'E'
cfg.TRAINER.RANSAC_PIXEL_THR = 0.5

# Loss config
cfg.LOFTR.LOSS.FINE_TYPE = 'l2'
cfg.LOFTR.LOSS.COARSE_OVERLAP_WEIGHT = True
cfg.LOFTR.LOSS.FINE_OVERLAP_WEIGHT = True
cfg.LOFTR.LOSS.LOCAL_WEIGHT = 0.25

# Matching config
cfg.LOFTR.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.3
cfg.LOFTR.MATCH_COARSE.SPARSE_SPVS = True
cfg.LOFTR.MATCH_COARSE.THR = 0.2
cfg.LOFTR.MATCH_COARSE.FP16MATMUL = False
cfg.LOFTR.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 50   # Reduced for batch_size=1 / 512px

cfg.LOFTR.MATCH_FINE.LOCAL_REGRESS_TEMPERATURE = 10.0
cfg.LOFTR.MATCH_FINE.LOCAL_REGRESS_SLICEDIM = 8

# Model architecture settings
cfg.LOFTR.RESOLUTION = (8, 1)
cfg.LOFTR.FINE_WINDOW_SIZE = 8
cfg.LOFTR.ALIGN_CORNER = False
cfg.LOFTR.MP = True                    # Mixed precision for VRAM savings
cfg.LOFTR.REPLACE_NAN = True
cfg.LOFTR.EVAL_TIMES = 1               # Fast validation
cfg.LOFTR.COARSE.NO_FLASH = False
cfg.LOFTR.COARSE.NPE = [512, 512, 512, 512]  # Match StereoLunar 512x512 resolution

# Dataset precision
cfg.DATASET.FP16 = False
