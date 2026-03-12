import os

import numpy as np


class CVMPT_offline:

    def __init__(
            self,
            path=r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\data",
            ext='npy',
    ):
        self.path = path
        self.ext = ext

        self.all_items = [
            os.path.join(root, file)
            for root, _, files in os.walk(self.path)
            for file in files if file.endswith(self.ext)
        ]

    def __len__(self):
        return len(self.all_items)
    def __getitem__(self, idx):

        try:
            item_file = self.all_items[idx]
            item = np.load(item_file, allow_pickle=True).item()
        except Exception as e:
            item_file = self.all_items[idx-1]
            item = np.load(item_file, allow_pickle=True).item()

        return item #['audio'], item['mel'], item["audio_noise"], item["mel_noise"]


if __name__ == '__main__':
    cls = CVMPT_offline()
    print(cls.__getitem__(0)[1])
