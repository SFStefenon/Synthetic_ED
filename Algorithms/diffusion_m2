import torch
from torch import nn, optim, autograd
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from dataclasses import dataclass
import time
import sys
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision.utils import make_grid
import torchvision.transforms as T
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)

model = Unet1D(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 32
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = 32,
    timesteps = 5000,
    objective = 'pred_v'
)

storage = './saved_results/'

img_size = 32
batchsize = 16
train_data_path = '32by32_rfi_train_perfect.csv'
size = '32by32'

for classes in range(0,133):
    print(f'Running: {classes}')
    try:
        if any(pd.read_csv(train_data_path).label.values==int(classes)): # Check if the class exists
            considered_label = int(classes)
            class_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'AD', 'AU', 'C00', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C30', 'C31', 'C32', 'C33', 'C34', 'C35', 'C36', 'C37', 'C38', 'C39', 'C40', 'C41', 'C42', 'C43', 'C44', 'C45', 'C46', 'C47', 'C48', 'C49', 'C50', 'C51', 'C52', 'C53', 'C54', 'C55', 'C56']
            class_list = class_list[considered_label]

            class RFI(Dataset):
                def __init__(self, path, img_size, c_label=considered_label, transform=None):
                    self.transform = transform
                    rfi_df = pd.read_csv(path)
                    ########################################################################
                    images = rfi_df.iloc[:, 1:].values.astype('uint8').reshape(-1, img_size, img_size)
                    self.images = images[np.where(rfi_df.label.values==c_label)]
                    self.labels = rfi_df.label.values[np.where(rfi_df.label.values==c_label)]
                    ########################################################################
                    print('Image size:', self.images.shape)
                    print('--- Label ---')
                    print(rfi_df.label.value_counts())
                def __len__(self):
                    return len(self.images)
                def __getitem__(self, idx):
                    label = self.labels[idx]
                    img = self.images[idx]
                    img = Image.fromarray(self.images[idx])
                    if self.transform:
                        img = self.transform(img)
                    return img, label
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
            dataset = RFI(train_data_path, img_size, considered_label, transform=transform)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True, drop_last=True)

            for data, labels in dataloader:
                fig, ax = plt.subplots(figsize=(12,8))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(make_grid(data, nrow=16).permute(1,2,0))
                figure_name = str(storage) + 'Results_DIF_class_' + str(classes) + '_'+size+'_image_original.pdf'
                plt.savefig(figure_name)
                break

            training_seq = data.reshape(batchsize, img_size, img_size)
            dataset = Dataset1D(training_seq)
            loss = diffusion(training_seq)
            loss.backward()

            trainer = Trainer1D(
                diffusion,
                dataset = dataset,
                train_batch_size = 32,
                train_lr = 8e-5,
                train_num_steps = 5000,         # total training steps
                gradient_accumulate_every = 2,    # gradient accumulation steps
                ema_decay = 0.995,                # exponential moving average decay
                amp = True,                       # turn on mixed precision
            )

            trainer.train()

            images = diffusion.sample(batch_size = batchsize).cpu()
            print(images.shape)
            name = str(storage) + 'Results_DIF_class_' + str(classes) + '_' + size + 'sampled_seq.pt'
            torch.save(images, name)

    except Exception as e:
       # By this way we can know about the type of error occurring
        print("The error is: ", e)
        pass

print('Finished')
