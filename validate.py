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
import os
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from driver import io_shape_dict
from driver_base import FINNExampleOverlay

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate top-1 accuracy for FINN-generated accelerator"
    )
    parser.add_argument(
        "--batchsize", help="number of samples for inference", type=int, default=1
    )
    parser.add_argument(
        "--bitfile", help='name of bitfile (i.e. "resizer.bit")', default="resizer.bit"
    )
    parser.add_argument(
        "--dataset_root", help="dataset root dir for download/reuse", default="/tmp"
    )
    # parse arguments
    args = parser.parse_args()
    bsize = args.batchsize
    dataset = args.dataset
    bitfile = args.bitfile
    platform = args.platform
    dataset_root = args.dataset_root

    if dataset == "mnist":
        from dataset_loading import mnist

        trainx, trainy, testx, testy, valx, valy = mnist.load_mnist_data(
            dataset_root, download=True, one_hot=False
        )
    elif dataset == "cifar10":
        from dataset_loading import cifar

        trainx, trainy, testx, testy, valx, valy = cifar.load_cifar_data(
            dataset_root, download=True, one_hot=False
        )
    else:
        raise Exception("Unrecognized dataset")

    data_path = dataset_root + "/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"

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


    test_imgs = testx
    test_labels = testy

    ok = 0
    nok = 0
    total = test_imgs.shape[0]

    driver = FINNExampleOverlay(
        bitfile_name=bitfile,
        platform=platform,
        io_shape_dict=io_shape_dict,
        batch_size=bsize,
        runtime_weight_dir="runtime_weights/",
    )

    n_batches = int(total / bsize)

    #test_imgs = test_imgs.reshape(n_batches, bsize, -1)
    #test_labels = test_labels.reshape(n_batches, bsize)

    input_size = (3, 224, 244)
    class_indices = {label: idx for idx, label in enumerate(test_df['label'].unique())}
        
    for i in range(n_batches):
        
        images = []
        labels = []
        
        for j in range(bsize):
        
            img_path = test_df.iloc[j + i*bsize]['image']
            #image = Image.open(img_path).convert('RGB')
            image = cv2.imread(img_path)
            print(image.shape)
            image = cv2.resize(image, (input_size[1], input_size[2]))
            print(image.shape)
            image = image
            label = class_indices[test_df.iloc[j + i*bsize]['label']]
            
            images.append(image)
            labels.append(labels)
            
        test_imgs = np.array(images)
        test_labels = np.array(labels)
        
        #ibuf_normal = test_imgs[i].reshape(driver.ibuf_packed_device[0].shape)
        #exp = test_labels[i]
        
        ibuf_normal = test_imgs.reshape(driver.ibuf_packed_device[0].shape)
        exp = test_labels
        
        driver.copy_input_data_to_device(ibuf_normal)
        driver.execute_on_buffers()
        obuf_normal = np.empty_like(driver.obuf_packed_device[0])
        driver.copy_output_data_from_device(obuf_normal)
        ret = np.bincount(obuf_normal.flatten() == exp.flatten())
        nok += ret[0]
        ok += ret[1]
        print("batch %d / %d : total OK %d NOK %d" % (i + 1, n_batches, ok, nok))

    acc = 100.0 * ok / (total)
    print("Final accuracy: %f" % acc)