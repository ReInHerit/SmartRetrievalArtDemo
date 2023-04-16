import pickle
import clip
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import data_path, NoisyArtDataset
from utils import collate_fn

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def extract_and_save_index_features(dataset: NoisyArtDataset, clip_model: nn.Module, file_name: str):
    feature_dim = clip_model.visual.output_dim
    val_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=8, collate_fn=collate_fn)
    index_features = torch.empty((0, feature_dim)).to(device)
    index_names = []

    print(f"extracting NoisyArt index features")
    # iterate over the dataset object
    for batch in tqdm(val_loader):
        images, paths = batch[:2]
        images = images.to(device)

        # extract and concatenate features and names
        with torch.no_grad():
            batch_features = clip_model.encode_image(images)
            index_features = torch.vstack((index_features, batch_features))
            index_names.extend(paths)

    # Eliminate duplicates
    index_features, u_index = np.unique(index_features.cpu(), axis=0, return_index=True)
    index_names = np.array(index_names)[u_index].tolist()
    index_features = torch.tensor(index_features)

    # save the extracted features
    data_path.mkdir(exist_ok=True, parents=True)
    torch.save(index_features, data_path / f"{file_name}_index_features.pt")
    with open(data_path / f'{file_name}_index_names.pkl', 'wb+') as f:
        pickle.dump(index_names, f)


def main():
    clip_model, clip_preprocess = clip.load("RN50x4")
    clip_model.eval().float()

    trainval_noisyart_dataset = NoisyArtDataset(clip_preprocess, 'trainval')
    extract_and_save_index_features(trainval_noisyart_dataset, clip_model, 'trainval')

    trainval_noisyart_dataset = NoisyArtDataset(clip_preprocess, 'test')
    extract_and_save_index_features(trainval_noisyart_dataset, clip_model, 'test')


if __name__ == '__main__':
    main()
