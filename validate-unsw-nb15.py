# Copyright (c) 2020 Xilinx, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of Xilinx nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import numpy as np
from driver import io_shape_dict
from driver_base import FINNExampleOverlay

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate top-1 accuracy for FINN-generated accelerator"
    )
    parser.add_argument(
        "--batchsize", help="number of samples for inference", type=int, default=1000
    )
    parser.add_argument(
        "--platform", help="Target platform: zynq-iodma alveo", default="zynq-iodma"
    )
    parser.add_argument(
        "--bitfile",
        help='name of bitfile (i.e. "resizer.bit")',
        default="../bitfile/finn-accel.bit",
    )
    parser.add_argument("--dataset_root", help="dataset root dir for download/reuse", default=".")
    # parse arguments
    args = parser.parse_args()
    bsize = args.batchsize
    bitfile = args.bitfile
    platform = args.platform
    dataset_root = args.dataset_root

    print("Loading dataset...")
    (test_imgs, test_labels) = make_unsw_nb15_test_batches(bsize, dataset_root)

    ok = 0
    nok = 0
    n_batches = test_imgs.shape[0]
    total = n_batches * bsize

    print("Initializing driver, flashing bitfile...")

    driver = FINNExampleOverlay(
        bitfile_name=bitfile,
        platform=platform,
        io_shape_dict=io_shape_dict,
        batch_size=bsize,
    )

    n_batches = int(total / bsize)

    print("Starting...")

    for i in range(n_batches):
        inp = np.pad(test_imgs[i].astype(np.float32), [(0, 0), (0, 7)], mode="constant")
        exp = test_labels[i].astype(np.float32)
        inp = 2 * inp - 1
        exp = 2 * exp - 1
        out = driver.execute(inp)
        matches = np.count_nonzero(out.flatten() == exp.flatten())
        nok += bsize - matches
        ok += matches
        print("batch %d / %d : total OK %d NOK %d" % (i + 1, n_batches, ok, nok))

    acc = 100.0 * ok / (total)
    print("Final accuracy: %f" % acc)


    input_size = (3, 64, 64)

    def inference_with_finn_onnx(current_inp):
        current_inp = current_inp.reshape(input_size)
        out = driver.execute(current_inp)
        return finn_output


    data_path = dataset_root + "/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"

    # Prepare dataset
    def load_dataset(data_path):
        images = []
        labels = []
        for subfolder in os.listdir(data_path):
            subfolder_path = os.path.join(data_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            for image_filename in os.listdir(subfolder_path):
                image_path = os.path.join(subfolder_path, image_filename)
                images.append(image_path)
                labels.append(subfolder)
        return pd.DataFrame({'image': images, 'label': labels})

    data = load_dataset(data_path)
    train_df, dummy_df = train_test_split(data, train_size=0.01, shuffle=True, stratify=data['label'], random_state=123)
    valid_df, dummy_df = train_test_split(dummy_df, train_size=0.01, shuffle=True, stratify=dummy_df['label'], random_state=123)
    test_df, dummy_df = train_test_split(dummy_df, train_size=0.01, shuffle=True, stratify=dummy_df['label'], random_state=123)

    # Define Custom Dataset class
    class CustomDataset(Dataset):
        def __init__(self, dataframe, transform=None, class_indices=None):
            self.dataframe = dataframe
            self.transform = transform
            self.class_indices = class_indices

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            img_path = self.dataframe.iloc[idx]['image']
            image = Image.open(img_path).convert('RGB')
            label = self.class_indices[self.dataframe.iloc[idx]['label']]

            if self.transform:
                image = self.transform(image)

            return image, label


    transform = transforms.Compose([
        transforms.Resize(input_size[1:3]),
        #transforms.RandomHorizontalFlip(),
        #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets and loaders
    class_indices = {label: idx for idx, label in enumerate(test_df['label'].unique())}
    test_dataset = CustomDataset(test_df, transform=transform, class_indices=class_indices)


    for images, labels in test_dataset:
        # run in Brevitas with PyTorch tensor
        # print(images.shape)
        current_inp = images.reshape((1, input_size[0], input_size[1], input_size[2]))
        finn_output = inference_with_finn_onnx(current_inp)
        print(finn_output)
