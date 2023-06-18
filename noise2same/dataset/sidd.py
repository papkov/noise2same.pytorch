from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Dict
import os
import cv2
import lmdb

import numpy as np
from noise2same.dataset.abc import AbstractNoiseDataset


def paired_paths_from_lmdb(folders, keys) -> List[Dict[str, str]]:
    """Generate paired paths from lmdb files.
    Contents of lmdb. Taking the `lq.lmdb` for example, the file structure is:
    lq.lmdb
    ├── data.mdb
    ├── lock.mdb
    ├── meta_info.txt
    The data.mdb and lock.mdb are standard lmdb files and you can refer to
    https://lmdb.readthedocs.io/en/release/ for more details.
    The meta_info.txt is a specified txt file to record the meta information
    of our datasets. It will be automatically created when preparing
    datasets by our provided dataset tools.
    Each line in the txt file records
    1)image name (with extension),
    2)image shape,
    3)compression level, separated by a white space.
    Example: `baboon.png (120,125,3) 1`
    We use the image name without extension as the lmdb key.
    Note that we use the same key for the corresponding lq and gt images.
    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
            Note that this key is different from lmdb keys.
    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 2, (
        'The len of keys should be 2 with [input_key, gt_key]. '
        f'But got {len(keys)}')
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    if not (input_folder.endswith('.lmdb') and gt_folder.endswith('.lmdb')):
        raise ValueError(
            f'{input_key} folder and {gt_key} folder should both in lmdb '
            f'formats. But received {input_key}: {input_folder}; '
            f'{gt_key}: {gt_folder}')
    # ensure that the two meta_info files are the same
    with open(os.path.join(input_folder, 'meta_info.txt')) as fin:
        input_lmdb_keys = [line.split('.')[0] for line in fin]
    with open(os.path.join(gt_folder, 'meta_info.txt')) as fin:
        gt_lmdb_keys = [line.split('.')[0] for line in fin]
    if set(input_lmdb_keys) != set(gt_lmdb_keys):
        raise ValueError(
            f'Keys in {input_key}_folder and {gt_key}_folder are different.')
    else:
        paths = []
        for lmdb_key in sorted(input_lmdb_keys):
            paths.append(
                dict([(f'{input_key}_path', lmdb_key),
                      (f'{gt_key}_path', lmdb_key)]))
        return paths


@dataclass
class SIDDDataset(AbstractNoiseDataset):
    path: Union[Path, str] = Path("data/SIDD-NAFNet")
    mode: str = "train"
    standardize_by_channel: bool = True
    n_channels: int = 3

    def _validate(self) -> None:
        assert self.mode in ("train", "val", "test")

    def _create_image_index(self) -> Dict[str, Union[List[str], np.ndarray]]:
        data_path = self.path / 'train' if self.mode == 'train' else self.path / 'val'
        input_path, gt_path = str(data_path / 'input_crops.lmdb'), str(data_path / 'gt_crops.lmdb')
        paired_paths = paired_paths_from_lmdb([input_path, gt_path], ['lq', 'gt'])
        input_db = lmdb.open(
            input_path,
            readonly=True,
            lock=False,
            readahead=False,
            map_size=8 * 1024 * 10485760,
        )
        gt_db = lmdb.open(
            gt_path,
            readonly=True,
            lock=False,
            readahead=False,
            map_size=8 * 1024 * 10485760,
        )
        input_data, gt_data = [], []
        for paired_path in paired_paths[:100]:
            with input_db.begin(write=False) as txn:
                value_buf = txn.get(paired_path['lq_path'].encode('ascii'))
            input_data.append(np.expand_dims(cv2.imdecode(np.frombuffer(value_buf, np.uint8), cv2.IMREAD_COLOR), 0))
            with gt_db.begin(write=False) as txn:
                value_buf = txn.get(paired_path['gt_path'].encode('ascii'))
            gt_data.append(np.expand_dims(cv2.imdecode(np.frombuffer(value_buf, np.uint8), cv2.IMREAD_COLOR), 0))
        return {
            'noisy_input': np.concatenate(input_data),
            'ground_truth': np.concatenate(gt_data)
        }

    def _get_image(self, i: int) -> Dict[str, np.ndarray]:
        return {'image': self.image_index['noisy_input'][i], 'ground_truth': self.image_index['ground_truth'][i]}
