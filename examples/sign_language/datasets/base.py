from typing import Tuple, List, Dict
from pathlib import Path

import numpy as np
import pandas as pd
from torch import Tensor, LongTensor
from torch.utils.data import Dataset

from pose_format.pose import Pose


class BaseDataset(Dataset):
    MANIFEST_COLUMNS = ['id', 'signs_file', 'signs_offset', 'signs_length',
                        'signs_type', 'signs_lang', 'translation',
                        'translation_lang', 'glosses', 'topic', 'signer_id']

    def __init__(self, tsv_path: str) -> None:
        self.tsv: str = tsv_path
        self.data: pd.DataFrame = pd.read_csv(
            self.tsv,
            sep='\t',
            na_filter=False,
        )
        self.indices: List[int] = list(range(len(self.data)))
        self.sizes: List[int] = [1 for _ in self.indices]

    def __getitem__(self, index: int) -> Dict:
        idx = self.indices[index]
        row = self.data.iloc[idx]

        # This check is needed for some test sets,
        # where target is not provided.
        if not pd.isna(row['signs_file']):
            signs_file = row.pop('signs_file')
            offset = row.pop('signs_offset')
            length = row.pop('signs_length')
            signs_file = Path(signs_file)

            if not signs_file.is_absolute():
                # Assuming that the path is relative to the tsv_path directory.
                base_dir = Path(self.tsv).parent
                signs_file = base_dir / signs_file
            
            if not signs_file.exists():
                raise FileNotFoundError(f"File not found: {signs_file}")

            with open(signs_file, 'rb') as f:
                p = Pose.read(f.read())
            p.body = p.body.select_frames(list(range(offset, offset+length)))
            row['signs'] = p.torch()

        sample = row.to_dict()
        return sample

    def __len__(self) -> int:        
        return len(self.indices)

    def filter_by_length(self, min_n_frames: int, max_n_frames: int) -> None:
        pre_len = len(self.data)
        self.data = self.data[self.data['signs_length'].between(min_n_frames, max_n_frames)]
        post_len = len(self.data)
        return pre_len - post_len
