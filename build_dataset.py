"""Split our Fundus dataset into train/val/test and resize images to (192, 128).

Our Fundus dataset comes into the following format:
    diabetes/
        GlaucomaVSDiabetes_1_0 (1).jpg
        GlaucomaVSDiabetes_1_0 (2).jpg
        ...
    glaucoma/
        GlaucomaVSDiabetes_0_1 (1).jpg
        GlaucomaVSDiabetes_0_1 (2).jpg
        ...

Original images have various sizes including (2743, 1936), (2376, 1584), etc.
Resizing to (192, 128) reduces the dataset size from 3.3 MB to 6-60 KB, and loading smaller images
makes training faster.
"""

import argparse
import random
import os

from PIL import Image
from tqdm import tqdm

SIZE1, SIZE2 = 192, 128

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/GlaucomaVSDiabetes', help="Directory with the raw fundus dataset")
parser.add_argument('--output_dir', default='data/SplitData', help="Where to write the new data")


def resize_and_save(filename, output_dir, size1=SIZE1, size2=SIZE2):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((size1, size2), Image.BILINEAR)
    filename_base=os.path.basename(filename)
    image.save(os.path.join(output_dir, filename_base.split('\\')[-1])) # split('/') if using Mac OS


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    for disease in ['diabetes', 'glaucoma']:
        # Define the data directories
        data_directory = os.path.join(args.data_dir, disease)

        # Get the filenames in each directory (train and test)
        filenames = os.listdir(data_directory)
        filenames = [os.path.join(data_directory, f) for f in filenames if f.endswith('.jpg') or f.endswith('.tif')]

        # Split the images into 80% train, 10% val, 10% test.
        # Make sure to always shuffle with a fixed seed so that the split is reproducible
        random.seed(230)
        filenames.sort()
        random.shuffle(filenames)

        num_images = len(filenames)
        split1 = int(0.8 * num_images)
        split2 = int(0.9 * num_images)
        train_filenames = filenames[:split1]
        val_filenames = filenames[split1:split2]
        test_filenames = filenames[split2:]
        
        output_directory = os.path.join(args.output_dir, disease)
        
        filenames = {'train': train_filenames,
                     'val': val_filenames,
                     'test': test_filenames}

        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
        else:
            print("Warning: output dir {} already exists".format(output_directory))

        # Preprocess train, val and test
        for split in ['train', 'val', 'test']:
            output_dir_split = os.path.join(output_directory, '{}_data'.format(split))
            if not os.path.exists(output_dir_split):
                os.mkdir(output_dir_split)
            else:
                print("Warning: dir {} already exists".format(output_dir_split))

            print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
            for filename in tqdm(filenames[split]):
                resize_and_save(filename, output_dir_split, size1=SIZE1, size2=SIZE2)

    print("Done building dataset")
