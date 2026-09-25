import os.path as osp
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from loguru import logger


_MANIFEST_CACHE = {}


def _get_manifest(split_file):
    if split_file not in _MANIFEST_CACHE:
        _MANIFEST_CACHE[split_file] = np.load(split_file, allow_pickle=True)
    return _MANIFEST_CACHE[split_file]


class StereoLunarDataset(Dataset):
    """
    Loads StereoLunar stereo pairs with depth maps and camera poses
    formatted for the EfficientLoFTR training pipeline.

    Enforces strict scene-level partitioning to guarantee zero terrain /
    illumination leakage between train, validation, and test splits:
    Scenes(train) ∩ Scenes(val) ∩ Scenes(test) = ∅.
    """

    def __init__(self,
                 root_dir,
                 split_file,
                 scene_name=None,
                 split='train',
                 img_resize=512,
                 img_padding=False,
                 depth_padding=False,
                 augment_fn=None,
                 fp16=False,
                 val_ratio=0.15,
                 test_ratio=0.15,
                 seed=42,
                 **kwargs):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.img_resize = img_resize
        self.img_padding = img_padding
        self.fp16 = fp16
        self.augment_fn = augment_fn if split == 'train' else None

        # Load manifest (cached in memory)
        data = _get_manifest(split_file)
        self.all_scenes = data['scenes']      # Array of scene subdirectories
        self.all_images = data['images']      # Array of image stems
        all_pairs = data['pairs']             # [scene_id, im1_id, im2_id, score]

        # Partition at SCENE / TERRAIN level to prevent data leakage:
        if 'split' in data:
            split_mask = (data['split'] == split)
            self.pairs = all_pairs[split_mask]
        else:
            unique_scene_ids = np.unique(all_pairs[:, 0])
            n_scenes = len(unique_scene_ids)

            rng = np.random.RandomState(seed)
            shuffled_scenes = rng.permutation(unique_scene_ids)

            n_val = max(1, int(n_scenes * val_ratio))
            n_test = max(1, int(n_scenes * test_ratio)) if n_scenes > 3 else 1

            val_scene_ids = set(shuffled_scenes[:n_val])
            test_scene_ids = set(shuffled_scenes[n_val:n_val + n_test])
            train_scene_ids = set(shuffled_scenes[n_val + n_test:])

            if split == 'train':
                target_scenes = train_scene_ids
            elif split == 'val':
                target_scenes = val_scene_ids
            elif split == 'test':
                target_scenes = test_scene_ids
            else:
                target_scenes = set(unique_scene_ids)

            self.pairs = np.array([p for p in all_pairs if p[0] in target_scenes])

        self.pairs = np.array(self.pairs)

        # Filter by specific scene if requested
        if scene_name is not None and len(self.pairs) > 0:
            target_matches = np.where(self.all_scenes == scene_name)[0]
            if len(target_matches) > 0:
                scene_sid = target_matches[0]
                self.pairs = self.pairs[self.pairs[:, 0] == scene_sid]
            else:
                self.pairs = np.empty((0, 4), dtype=np.float32)

        assigned_scene_ids = np.unique(self.pairs[:, 0]) if len(self.pairs) > 0 else np.array([])
        self.assigned_scenes = [str(self.all_scenes[int(sid)]) for sid in assigned_scene_ids]
        if scene_name is None:
            logger.info(
                f"[StereoLunarDataset] Loaded {len(self.pairs)} pair(s) across {len(self.assigned_scenes)} "
                f"assigned scene(s) for split='{split}'"
            )

    def __len__(self):
        return len(self.pairs)

    def _read_gray(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to read image at: {path}")
        return img

    def _read_depth(self, path):
        try:
            import OpenEXR
            import Imath
            file = OpenEXR.InputFile(str(path))
            dw = file.header()['dataWindow']
            w = dw.max.x - dw.min.x + 1
            h = dw.max.y - dw.min.y + 1
            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            channels = list(file.header()['channels'].keys())
            raw = file.channel(channels[0], pt)
            depth = np.frombuffer(raw, dtype=np.float32).reshape(h, w).copy()
        except Exception as e:
            raise RuntimeError(f"Failed to read EXR depth at {path}: {e}")

        # Sanitize non-finite and non-positive depth (e.g. sky/space background)
        depth[~np.isfinite(depth)] = 0.0
        depth[depth <= 0] = 0.0

        return depth

    def __getitem__(self, idx):
        scene_id, im1_id, im2_id, score = self.pairs[idx]
        scene = str(self.all_scenes[int(scene_id)])
        seq_path = osp.join(self.root_dir, scene)

        img_name0 = str(self.all_images[int(im1_id)])
        img_name1 = str(self.all_images[int(im2_id)])

        try:
            image0 = self._read_gray(osp.join(seq_path, f"{img_name0}.jpg"))
            image1 = self._read_gray(osp.join(seq_path, f"{img_name1}.jpg"))

            depth0 = self._read_depth(osp.join(seq_path, f"{img_name0}.exr"))
            depth1 = self._read_depth(osp.join(seq_path, f"{img_name1}.exr"))

            params0 = np.load(osp.join(seq_path, f"{img_name0}.npz"))
            params1 = np.load(osp.join(seq_path, f"{img_name1}.npz"))

            K0 = torch.tensor(params0['intrinsics'].copy(), dtype=torch.float32)
            K1 = torch.tensor(params1['intrinsics'].copy(), dtype=torch.float32)

            cam2world0 = params0['cam2world'].astype(np.float64)
            cam2world1 = params1['cam2world'].astype(np.float64)

            # Relative camera poses: T_0to1 = inv(cam2world1) @ cam2world0
            world2cam1 = np.linalg.inv(cam2world1)
            T_0to1 = torch.tensor(world2cam1 @ cam2world0, dtype=torch.float32)[:4, :4]
            T_1to0 = T_0to1.inverse()
        except Exception as e:
            logger.warning(f"Error loading pair {scene}/{img_name0}-{img_name1}: {e}. Retrying next pair.")
            return self.__getitem__((idx + 1) % len(self))

        h_orig, w_orig = image0.shape[:2]
        h_new = w_new = self.img_resize

        if h_new != h_orig or w_new != w_orig:
            image0 = cv2.resize(image0, (w_new, h_new))
            image1 = cv2.resize(image1, (w_new, h_new))
            # Depth maps and camera intrinsics K0, K1 are preserved at native resolution (512x512).
            # EfficientLoFTR supervision maps feature-level coordinates to original resolution
            # via scale0 / scale1, ensuring mutually consistent unprojection and reprojection.

        scale0 = torch.tensor([w_orig / w_new, h_orig / h_new], dtype=torch.float32)
        scale1 = torch.tensor([w_orig / w_new, h_orig / h_new], dtype=torch.float32)

        image0 = torch.from_numpy(image0).float()[None] / 255.0
        image1 = torch.from_numpy(image1).float()[None] / 255.0
        depth0 = torch.from_numpy(depth0).float()
        depth1 = torch.from_numpy(depth1).float()

        if self.fp16:
            image0, image1 = image0.half(), image1.half()
            depth0, depth1 = depth0.half(), depth1.half()
            scale0, scale1 = scale0.half(), scale1.half()

        return {
            'image0': image0,
            'depth0': depth0,
            'image1': image1,
            'depth1': depth1,
            'T_0to1': T_0to1,
            'T_1to0': T_1to0,
            'K0': K0,
            'K1': K1,
            'scale0': scale0,
            'scale1': scale1,
            'dataset_name': 'StereoLunar',
            'scene_id': scene,
            'pair_id': idx,
            'score': float(score),
            'pair_names': (f"{scene}/{img_name0}", f"{scene}/{img_name1}"),
        }
