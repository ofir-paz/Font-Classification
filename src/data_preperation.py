"""data_preperation.py

:author: Ofir Paz
:version: 03.02.2024

This file is responsible for the data preperation for the project.
"""


################################ Import section ################################

import os
import cv2 as cv
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import numpy as np
import h5py
from tqdm import tqdm

############################# End of import section ############################


################################ Globals section ###############################

# Set the path of the current directory.
PARENT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '\\'

# Set the encoding and decoding dictionaries.
FONT_ENCODER = {'Flower Rose Brush': 0, 'Skylark': 1, 'Sweet Puppy': 2, 'Ubuntu Mono': 3, 
                'VertigoFLF': 4, 'Wanted M54': 5, 'always forever': 6}
FONT_DECODER = np.array(list(FONT_ENCODER.keys()))

# Set the number of the classes
NUM_CLASSES = 7

# Set the letter size.
LETTER_SIZE = (20, 30)  # This is the calculated average size of the letters in the images.

# Set the mean and std of the resized cropped letter images (they were calculated).
MEAN = [0.46775997, 0.48115298, 0.48016804]
STD = [0.25156727, 0.24406362, 0.27375552]

# Set the batch size.
BATCH_SIZE = 256

# Set the precentage of each dataset.
TRAIN_SIZE = 0.9
VALID_SIZE = 0.1 

# Set epsilon for numerical stability.
EPSILON = 1e-10

# Set the transform for the letter images.
TRANSFORM = transforms.Compose([
    lambda image: cv.resize(image, LETTER_SIZE).transpose(1, 0, 2),
    transforms.ToTensor(),
    # transforms.Normalize(mean=MEAN, std=STD, inplace=True)
    # I found that this normalization technique works better.
    # I ended up using it instead of the one described in the project's report.
    lambda t: (t - t.mean()) / (t.std() + EPSILON)
])

DATASET_NAMES = ['train', 'valid', 'test']

############################ End of globals section ############################


################################ Classes section ###############################

class FontDataset(Dataset):
    """CostumDataset class.

    This class is responsible for creating a custom dataset for the project.
    """
    def __init__(self, images, charBBs, fonts):
        
        # Calculate the length of the dataset.
        self.len = sum(len(sublist) for sublist in charBBs)
        
        self.letter_images = torch.empty(self.len, 3, *LETTER_SIZE)
        idx = 0

        # Process the letter images.
        for image, charBBs_in_image in tqdm(zip(images, charBBs), total=len(images)):
            for charBB in charBBs_in_image:
                self.letter_images[idx] = TRANSFORM(crop_letter(image, charBB))
                idx += 1
        
        # Process the labels.
        if fonts is not None and fonts != []:
            self.labels = torch.cat([torch.from_numpy(font).to(torch.long) for font in fonts])
        else:
            self.labels = None

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        
        if self.labels is None:
            return self.letter_images[idx]
        
        return self.letter_images[idx], self.labels[idx]
    
############################ End of classes section ############################
    

############################### Functions section ##############################

def encode_fonts(fonts: list[str]) -> np.ndarray:
    """
    Encode the fonts.

    :param fonts: The fonts to encode.
    :return: The encoded fonts.
    """

    # Encode the fonts.
    return np.array([FONT_ENCODER[font] for font in fonts])


def get_data_from_h5(train: bool = True) -> dict:
    """
    Get data from the h5 file.

    :param train: If true, get the training data. If false, get the test data.
    :return: The data.
    """

    # Choose the set to get the data of.
    set_name = 'train' if train else 'test'

    # Get the data from the h5 file.
    with h5py.File(PARENT_PATH + f'data\\{set_name}.h5', 'r') as db:

        # Get the data from the h5 file.
        data = db['data']

        # Get the image names from the data.
        im_names = list(data.keys())  # type: ignore

        # Extract the number of images
        num_images = len(im_names)

        images = []
        fonts = []
        txts = []
        charBBs = []
        wordBBs = []
        
        for i, im_name in tqdm(enumerate(im_names), total=num_images):
            
            # Get the data point.
            data_point = data[im_name]  # type: ignore

            # Get the image and turn it to numpy array.
            images.append(np.array(data_point[:], dtype=np.float32))  # type: ignore

            # Get all other data related to the image and store it.
            if train:
                fonts.append(encode_fonts(data_point.attrs['font']))  # type: ignore
                
            txts.append(list(map(lambda word: word.decode('utf-8'),  # type: ignore
                                  data_point.attrs['txt'])))  # type: ignore

            charBB = data_point.attrs['charBB'].transpose(2, 1, 0)  # type: ignore
            charBB = charBB[..., [0, 1]]  # Revert swap of the last dimension  # type: ignore
            charBBs.append(charBB)

            wordBB = data_point.attrs['wordBB'].transpose(2, 1, 0)  # type: ignore
            wordBB = wordBB[..., [0, 1]]  # Revert swap of the last dimension  # type: ignore
            wordBBs.append(wordBB)

    return {'images': images, 'fonts': fonts, 'txts': txts, 'charBBs': charBBs,
             'wordBBs': wordBBs}


def crop_letter(image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """
    Crop a region from a larger image based on a bounding box.

    :param image: The larger image.
    :param bbox: The bounding box vertices (x, y) as a list of tuples.
    :return: The cropped image.
    """

    # Calculate the width and height of the bounding box
    width = int(np.linalg.norm(bbox[1] - bbox[0])) or 1
    height = int(np.linalg.norm(bbox[3] - bbox[0])) or 1

    # Create a transformation matrix for the perspective transformation
    src_pts = bbox.astype(np.float32)
    dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    M = cv.getPerspectiveTransform(src_pts, dst_pts)

    # Apply the perspective transformation to the image
    cropped_image = cv.warpPerspective(image / 255, M, (width, height))

    return cropped_image


def split_data(data: dict) -> tuple[dict, dict]:
    """
    Split the data to training and validation sets.

    :param data: The data to split.
    :return: The training and validation sets.
    """

    # Set the random seed.
    torch.manual_seed(0)

    # Get the indices for the training and validation sets.
    train_idx, valid_idx = random_split(torch.arange(len(data['images'])),  # type: ignore
                                         [TRAIN_SIZE, VALID_SIZE])

    train_dict = {'images': [data['images'][idx] for idx in train_idx],
                  'fonts': [data['fonts'][idx] for idx in train_idx],
                  'txts': [data['txts'][idx] for idx in train_idx],
                  'charBBs': [data['charBBs'][idx] for idx in train_idx],
                  'wordBBs': [data['wordBBs'][idx] for idx in train_idx]}
    
    valid_dict = {'images': [data['images'][idx] for idx in valid_idx],
                    'fonts': [data['fonts'][idx] for idx in valid_idx],
                    'txts': [data['txts'][idx] for idx in valid_idx],
                    'charBBs': [data['charBBs'][idx] for idx in valid_idx],
                    'wordBBs': [data['wordBBs'][idx] for idx in valid_idx]}
    
    return train_dict, valid_dict


def create_datasets(datas: list[dict]) -> list[FontDataset]:
    """
    Creates font datasets.

    :param datas: List of datas to create the datasets from.
    :return: The datasets.
    """

    return [FontDataset(data['images'], data['charBBs'], data['fonts']) for data in datas]


def save_created_datasets_to_disk(datasets: list[FontDataset], names: list[str]) -> None:
    """
    Save the created datasets to the disk.

    :param datasets: The datasets to save.
    :param names: The names of the datasets.
    :return: None.
    """

    for dataset, name in zip(datasets, names):

        # Save the datasets to the disk.
        torch.save(dataset, PARENT_PATH + f'data\\{name}_dataset.pt')


def load_created_datasets_from_disk(names: list[str] = DATASET_NAMES) -> list[FontDataset]:
    """
    Load created datasets from the disk.

    :param names: The names of the datasets.
    :return: The datasets.
    """

    return [torch.load(PARENT_PATH + f'data\\{name}_dataset.pt') for name in names]


def check_for_saved_datasets(names: list[str] = DATASET_NAMES) -> bool:
    """
    Check if all the datasets are saved.

    :param names: The names of the datasets.
    :return: True if all the datasets are saved, False otherwise.
    """

    return all([os.path.exists(PARENT_PATH + f'data\\{name}_dataset.pt') for name in names])


def delete_saved_datasets(names: list[str] = DATASET_NAMES) -> None:
    """
    Delete the saved datasets from the disk.

    :param names: The names of the datasets.
    :return: None.
    """

    for name in names:
        os.remove(PARENT_PATH + f'data\\{name}_dataset.pt')


def get_datasets() -> list[FontDataset]:
    """
    Get the datasets.
    If the datasets are saved, load them from the disk. Otherwise, create them.

    :return: The datasets.
    """

    # Check if the datasets are saved.
    if check_for_saved_datasets():

        # Load the datasets from the disk.
        datasets = load_created_datasets_from_disk()

    else:

        # Get the data from the h5 file.
        labeled_data = get_data_from_h5(train=True)
        unlabeled_data = get_data_from_h5(train=False)

        # Split the labeled data to training and validation.
        train_dict, valid_dict = split_data(labeled_data)

        # Create the datasets.
        datasets = create_datasets([train_dict, valid_dict, unlabeled_data])

        # Save the datasets to the disk.
        save_created_datasets_to_disk(datasets, DATASET_NAMES)

    return datasets


def get_data_loaders(shuffle_train: bool = True) -> tuple:
    """
    Get the data loaders.
    If the datasets are saved, load them from the disk. Otherwise, create them.

    :return: The data loaders.
    """

    # Get the datasets.
    train_dataset, valid_dataset, test_dataset = get_datasets()

    # Create the data loaders.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=shuffle_train)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, valid_loader, test_loader

############################# End of functions section ############################