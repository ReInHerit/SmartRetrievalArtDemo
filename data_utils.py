from pathlib import Path
from typing import List, Optional

import PIL.Image
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

server_base_path = Path(__file__).absolute().parent.absolute()
data_path = Path(__file__).absolute().parent.absolute() / 'data'


def is_image_file(filename: str) -> bool:
    '''
    Check whether a file has an image extension
    :param filename: name of file
    '''
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class TargetPad:
    """
    Pad the image if its aspect ratio is above a target ratio.
    Pad it to match such target ratio
    """

    def __init__(self, target_ratio: float, size: int, pad_value: Optional[int] = 0):
        """
        :param target_ratio: target ratio
        :param size: preprocessing output dimension
        :param pad_value: padding value, 0 is black-pad (zero-pad), 255 is white-pad
        """
        self.size = size
        self.target_ratio = target_ratio
        self.pad_value = pad_value

    def __call__(self, image):
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
            return image
        scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, self.pad_value, 'constant')


def targetpad_resize(target_ratio: float, dim: int, pad_value: int):
    """
    Yield a torchvision transform which resize and center crop an image using TargetPad pad
    :param target_ratio: target ratio for TargetPad
    :param dim: image output dimension
    :param pad_value: padding value, 0 is black-pad (zero-pad), 255 is white-pad
    :return: torchvision Compose transform
    """
    return Compose([
        TargetPad(target_ratio, dim, pad_value),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
    ])


class NoisyArtDataset(Dataset):
    def __init__(self, preprocess: callable, split: str):
        super().__init__()
        self.split = split
        self.preprocess = preprocess

        self.dataset_root = server_base_path / 'noisyart_dataset'
        self.class_names = set([class_path.name for class_path in (self.dataset_root / 'trainval_3120').iterdir()])

        if split in ['val', 'train', 'trainval']:
            split_file = self.dataset_root / 'noisyart' / 'splits' / 'trainval_3120' / f"{split}.txt"
            with open(split_file) as f:
                split_images_path = set(f.read().splitlines())

            self.image_paths = [image_path for image_path in (self.dataset_root / 'trainval_3120').glob('**/*') if
                                is_image_file(str(image_path)) and str(image_path.relative_to(
                                    self.dataset_root / 'trainval_3120')) in split_images_path]
        else:
            self.image_paths = [image_path for image_path in (self.dataset_root / 'test_200').glob('**/*') if
                                is_image_file(str(image_path))]
        self.class_names = sorted(list(self.class_names))

    def __getitem__(self, index):
        image_path = str(self.image_paths[index])
        image_class = self.image_paths[index].parent.name
        label = self.class_names.index(image_class)
        try:
            image = self.preprocess(PIL.Image.open(self.image_paths[index]))
        except Exception as e:
            print(f"Exception occured: {e}")
            return None
        return image, "/".join(image_path.split('/')[-2:]), label

    def __len__(self):
        return len(self.image_paths)

# ---------------------------------------------------------
