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
from PIL import Image
import os

import cv2
from pynq import PL


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
    parser.add_argument("--dataset_root", help="dataset root dir for download/reuse", default="./")
    # parse arguments
    args = parser.parse_args()
    bsize = args.batchsize
    bitfile = args.bitfile
    platform = args.platform
    dataset_root = args.dataset_root

    print("Reset PL (clear cache)...")
    PL.reset()
    
    print("Loading dataset...")
    #(test_imgs, test_labels) = make_unsw_nb15_test_batches(bsize, dataset_root)

    input_size = (3, 64, 64)
    
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
    train_df, dummy_df = train_test_split(data, train_size=0.8, shuffle=True, stratify=data['label'], random_state=123)
    valid_df, test_df = train_test_split(dummy_df, train_size=0.5, shuffle=True, stratify=dummy_df['label'], random_state=123)

    # Define Custom Dataset class
    #class CustomDataset(Dataset):
    #    def __init__(self, dataframe, transform=None, class_indices=None):
    #        self.dataframe = dataframe
    #        self.transform = transform
    #        self.class_indices = class_indices

    #    def __len__(self):
    #        return len(self.dataframe)

    #    def __getitem__(self, idx):
    #        img_path = self.dataframe.iloc[idx]['image']
    #        image = Image.open(img_path).convert('RGB')
    #        label = self.class_indices[self.dataframe.iloc[idx]['label']]

    #        if self.transform:
    #            image = self.transform(image)

    #        return image, label

    #transform = transforms.Compose([
    #    transforms.Resize(input_size[1:3]),
    #    #transforms.RandomHorizontalFlip(),
    #    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    #    transforms.ToTensor(),
    #    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #])

    # Create datasets and loaders
    class_indices = {label: idx for idx, label in enumerate(test_df['label'].unique())}

    ok = 0
    nok = 0
    #n_batches = test_df.shape[0]
    #total = n_batches * bsize

    #n_batches = int(total / bsize)
    
    print("Initializing driver, flashing bitfile...")

    driver = FINNExampleOverlay(
        bitfile_name=bitfile,
        platform=platform,
        io_shape_dict=io_shape_dict,
        batch_size=1,
    )

    def inference_with_finn_onnx(current_inp):
        #current_inp = current_inp.reshape(input_size)
        out = driver.execute(current_inp)
        return finn_output

    print("Starting...")

    for idx in range(len(test_df)):
        
        img_path = test_df.iloc[idx]['image']
        #image = Image.open(img_path).convert('RGB')
        image = cv2.imread(img_path)
        print(image.shape)
        image = cv2.resize(image, (input_size[1], input_size[2]))
        print(image.shape)
        image = image/255.0
        label = class_indices[test_df.iloc[idx]['label']]
    
        #image = data["image"]
        #label = data["label"]
        # run in Brevitas with PyTorch tensor
        # print(images.shape)
        current_inp = image.reshape((input_size[0], input_size[1], input_size[2]))
        print(current_inp.shape)
        finn_output = inference_with_finn_onnx(current_inp)
        
        print(finn_output)
        
        matches = np.count_nonzero(finn_output.flatten() == labels.flatten())
        nok += bsize - matches
        ok += matches
        #print("batch %d / %d : total OK %d NOK %d" % (i + 1, n_batches, ok, nok))


    #acc = 100.0 * ok / (total)
    #print("Final accuracy: %f" % acc)


