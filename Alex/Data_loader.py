import random
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# define a training image loader that specifies transforms on images. See documentation for more details.
def train_transformer_list(params):
    train_transformer = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
        transforms.RandomVerticalFlip(),  # randomly flip image vertically
        transforms.RandomRotation(180), # randomly rotate image by 180 degrees
        transforms.Grayscale(num_output_channels=params.num_input_channels), # convert RGB image to greyscale (optional, 1 vs. 3 channels)
        transforms.ToTensor()])  # transform it into a torch tensor
    return train_transformer

# loader for evaluation, no horizontal flip
def eval_transformer_list(params):
    eval_transformer = transforms.Compose([
        transforms.Grayscale(num_output_channels=params.num_input_channels), 
        # transforms.Resize([177, 128]),  # resize the image to 177x128 (remove if images are already 64x64)
        transforms.ToTensor()])  # transform it into a torch tensor
    return eval_transformer

class FundusDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_dir, transform):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.filenames = os.listdir(data_dir)
        self.filenames = [os.path.join(data_dir, f) for f in self.filenames if f.endswith('.png')]

        # self.labels = [int(os.path.split(filename)[-1][0]) for filename in self.filenames]
        self.labels = []
        for filename in self.filenames:
            imagename = os.path.split(filename)[-1]
            ind = imagename.find('(')
            label = imagename[(ind+1):(ind+3)]
            finallabel = np.array(list(map(float, list(label))))
            self.labels.append(finallabel)
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        image = Image.open(self.filenames[idx])  # PIL image
        image = self.transform(image)
        return image, self.labels[idx], self.filenames[idx]


def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(data_dir, "{}_data".format(split))

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(FundusDataset(path, train_transformer_list(params)), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)
            else:
                dl = DataLoader(FundusDataset(path, eval_transformer_list(params)), batch_size=params.batch_size, shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders
