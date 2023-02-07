import numpy as np
from torch.utils.data import Dataset
import glob
import re
from PIL import Image

class ASLDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None):
        self.image_dirs = glob.glob(data_dir+'/**/*.jpg',recursive=True)
        self.labels = [re.search('[a-zA-Z]*',image.split('/')[-1]).group()\
            for image in self.image_dirs]
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_size = len(self.labels)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        image = Image.open(self.image_dirs[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def TransformLetterToNumber(Letter: str)->int:
    letter_to_int_dict = {  'A': 0,'B': 1,'C': 2,'D': 3,
                            'E': 4,'F': 5,'G': 6,'H': 7,
                            'I': 8,'J': 9,'K': 10,'L': 11,
                            'M': 12,'N': 13,'O': 14,'P': 15,
                            'Q': 16,'R': 17,'S': 18,'T': 19,
                            'U': 20,'V': 21,'W': 22,'X': 23,
                            'Y': 24,'Z': 25,'space':26,'del':27,
                            'nothing':28}
    return letter_to_int_dict[Letter]
