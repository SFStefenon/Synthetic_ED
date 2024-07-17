import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import time
import os
import torchvision
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = 32,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250    # number of sampling timesteps
)

batchsize = 16

for classes in range(0,133):
    print(f'Running: {classes}')

    data = '../inputdata/original/' + str(classes)
    storage = './saved_results/'

    try:
        start = time.time()
        trainer = Trainer(
            diffusion,
            data,
            train_batch_size = 16,
            train_lr = 8e-5,
            train_num_steps = 1000,           # total training steps
            results_folder = '',
            gradient_accumulate_every = 2,    # gradient accumulation steps
            ema_decay = 0.995,                # exponential moving average decay
            amp = True,                       # turn on mixed precision
            calculate_fid = True              # whether to calculate fid during training
        )

        trainer.train()

        images = diffusion.sample(batch_size = batchsize).cpu()
        # print(images.shape)
        name = str(storage) + 'Results_DIF_class_' + str(classes) + '_sampled_seq.pt'
        torch.save(images, name)

        for i in images:
            fig, ax = plt.subplots(figsize=(12,8))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(make_grid(images, nrow=4).permute(1,2,0))
            break
        figure_name = str(storage) + 'Results_DIF_class_' + str(classes) + '_image_generated.jpg'
        plt.savefig(figure_name)
        end = time.time()
        time_s = end - start
        print(f'{time_s:.2f}s')
        os.remove('model-1.pt')

    except Exception as e:
        pass
        print("The error is: ", e)

print('Finished')
