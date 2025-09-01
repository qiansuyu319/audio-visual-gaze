import torch
import json
import os
import copy
from PIL import Image
import numpy as np

import gazelle.utils as utils

def load_data_vat(file, sample_rate):
    sequences = json.load(open(file, "r"))
    data = []
    for i in range(len(sequences)):
        for j in range(0, len(sequences[i]['frames']), sample_rate):
            data.append(sequences[i]['frames'][j])
    return data


def load_data_gazefollow(file):
    data = json.load(open(file, "r"))
    return data


class GazeDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset_name, path, split, transform, in_frame_only=True, sample_rate=1,
                 audio_root=None, audio_dim=856, audio_roots=None, audio_dims=None,
                 missing_audio="zeros", audio_map_json=None, image_root=None):
        self.dataset_name = dataset_name
        self.path = path
        self.split = split
        self.aug = self.split == "train"
        self.transform = transform
        self.in_frame_only = in_frame_only
        self.sample_rate = sample_rate
        self.image_root = image_root if image_root is not None else os.getenv('GAZELLE_IMAGE_ROOT', None)


        self.audio_root = audio_root
        self.audio_dim = audio_dim
        self.audio_roots = None
        self.audio_dims = None
        if audio_roots and audio_dims:
            assert isinstance(audio_roots, (list, tuple)) and isinstance(audio_dims, (list, tuple))
            assert len(audio_roots) == len(audio_dims) and len(audio_roots) > 0
            self.audio_roots = list(audio_roots)
            self.audio_dims = [int(d) for d in audio_dims]
        elif audio_root and audio_dim:
            self.audio_roots = [audio_root]
            self.audio_dims = [int(audio_dim)]
        self.total_audio_dim = int(sum(self.audio_dims)) if self.audio_dims else 0
        self.missing_audio = missing_audio
        self.audio_map = None
        if audio_map_json and os.path.exists(audio_map_json):
            with open(audio_map_json, "r") as f:
                self.audio_map = json.load(f)

        if dataset_name == "gazefollow":
            self.data = load_data_gazefollow(os.path.join(self.path, f"{split}_preprocessed.json"))
        elif dataset_name == "videoattentiontarget":
            self.data = load_data_vat(os.path.join(self.path, f"{split}_preprocessed.json"), sample_rate)
        else:
            raise ValueError(f"Invalid dataset: {dataset_name}")

        self.data_idxs = []
        for i in range(len(self.data)):
            for j in range(len(self.data[i]['heads'])):
                if not self.in_frame_only or self.data[i]['heads'][j]['inout'] == 1:
                    self.data_idxs.append((i, j))

    def __getitem__(self, idx):
        img_idx, head_idx = self.data_idxs[idx]
        img_data = self.data[img_idx]
        head_data = copy.deepcopy(img_data['heads'][head_idx])
        bbox_norm = head_data['bbox_norm']
        gazex_norm = head_data['gazex_norm']
        gazey_norm = head_data['gazey_norm']
        inout = head_data['inout']

        rel_img_path = img_data.get('path') or img_data.get('image_path')
        candidate_img_paths = []
        if rel_img_path:
            if self.image_root:
                candidate_img_paths.append(os.path.join(self.image_root, rel_img_path))
                candidate_img_paths.append(os.path.join(self.image_root, 'frames', rel_img_path))
            candidate_img_paths.append(os.path.join(self.path, rel_img_path))
            candidate_img_paths.append(os.path.join(self.path, 'frames', rel_img_path))
            parent_dir = os.path.dirname(os.path.abspath(self.path.rstrip('/')))
            candidate_img_paths.append(os.path.join(parent_dir, 'frames', rel_img_path))

        img_path = None
        for cand in candidate_img_paths:
            if os.path.exists(cand):
                img_path = cand
                break
        if img_path is None:
            raise FileNotFoundError(f"Image not found for rel='{rel_img_path}'. Tried: {candidate_img_paths}")

        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        if self.aug:
            bbox = head_data['bbox']
            gazex = head_data['gazex']
            gazey = head_data['gazey']

            if np.random.sample() <= 0.5:
                img, bbox, gazex, gazey = utils.random_crop(img, bbox, gazex, gazey, inout)
            if np.random.sample() <= 0.5:
                img, bbox, gazex, gazey = utils.horiz_flip(img, bbox, gazex, gazey, inout)
            if np.random.sample() <= 0.5:
                bbox = utils.random_bbox_jitter(img, bbox)

            width, height = img.size
            bbox_norm = [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]
            gazex_norm = [x / float(width) for x in gazex]
            gazey_norm = [y / float(height) for y in gazey]

        img = self.transform(img)

        if self.split == "train":
            heatmap = utils.get_heatmap(gazex_norm[0], gazey_norm[0], 64, 64)
            return img, bbox_norm, gazex_norm, gazey_norm, torch.tensor(inout), height, width, heatmap
        else:
            return img, bbox_norm, gazex_norm, gazey_norm, torch.tensor(inout), height, width

    def __len__(self):
        return len(self.data_idxs)


def collate_fn(batch):
    transposed = list(zip(*batch))
    return tuple(
        torch.stack(items) if isinstance(items[0], torch.Tensor) else list(items)
        for items in transposed
    )
