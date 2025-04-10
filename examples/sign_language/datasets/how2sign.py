import re
from pathlib import Path

import pandas as pd

from . import SignLanguageDataset, register_dataset, Pose


@register_dataset('how2sign')
class How2Sign(SignLanguageDataset):
    """
    Create a Dataset for How2Sign. Each item is a dictionary with:
        id, file, offset, length, type, signs_lang, translation, translation_lang, topic
        base dir should be:
    """

    TRANSLATION_LANGS = ['en']  # TODO: add 'pt'
    SPLITS = ['train', 'val', 'test']
    SIGNS_TYPES = ['mediapipe', 'i3d']  # TODO: Add 'i3d'
    SIGNS_LANGS = ['asl']

    def __init__(self, tsv_file: Path):
        self.tsv_file = tsv_file
        self.data = pd.read_csv(self.tsv_file, sep='\t')
        self.indices = list(self.data.index)
        self.sizes = list(self.data['signs_length'])

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        return {
            "id": data_row["id"],
            "signs_file": data_row["signs_file"],
            "signs_offset": data_row["signs_offset"],
            "signs_length": data_row["signs_length"],
            "signs_type": data_row["signs_type"],
            "signs_lang": data_row["signs_lang"],
            "translation": data_row["translation"],
            "translation_lang": data_row["translation_lang"],
            "topic": data_row["topic"],
        }

    def __len__(self):
        return len(self.indices)

    def _get_data_by_sign_type(self, data, split, signs_type):
        if signs_type == 'mediapipe':

            find_groups = r'(^.{11})(.*)-([0-9]+)(.*)'  # noqa
            print_groups = r'\1-\3\4'  # backlash is not allowed inside f-string

            get_signs_file = lambda x: (
                self.base_dir / "How2Sign/video_level" / \
                split / "rgb_front/features" / signs_type / f"{re.sub(find_groups, print_groups, x['id'])}.pose"
            ).as_posix()
            
        self.data['signs_file'] = self.data.parallel_apply(get_signs_file, axis=1)
        
    #    def get_pose_length(pose_file: str) -> int:
    #        with open(pose_file, "rb") as f:
    #            p = Pose.read(f.read())
    #        return p.body.data.shape[0]

        self.data['signs_offset'] = data['START_FRAME']
        self.data['signs_length'] = data['END_FRAME'] - data['START_FRAME'] + 1

        self.data['translation'] = data['SENTENCE']
        self.data['topic'] = data['TOPIC_ID']

        return self.data

