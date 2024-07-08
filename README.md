# Synthetic Engineering Drawings

This repository presents a rule and statistical-based model for creating synthetic engineering drawings (EDs). The application of the proposed method is based on the generation of synthetic objects using generative artificial intelligence (AI) models. In particular, generative adversarial networks (GANs), conditional GANs (cGANs), diffusion, and conditional diffusion (c_diffusion) models are considered.

The generated objects using generative AI are based on the annotations of relay-based railway interlocking systems (RRIS) EDs from the rail network. Regarding the annotations, symbols, labels, specifiers, and electrical connections are considered. The algorithms to generate the 28-pixel objects are available at:
[GANs](https://colab.research.google.com/github/SFStefenon/synthetic_ED/blob/main/gans.ipynb), 
[cGANs](https://colab.research.google.com/github/SFStefenon/synthetic_ED/blob/main/c_gans.ipynb), 
[diffusion](https://colab.research.google.com/github/SFStefenon/synthetic_ED/blob/main/difussion.ipynb), or 
[c_diffusion](https://colab.research.google.com/github/SFStefenon/synthetic_ED/blob/main/c_difussion.ipynb). Obs: to compute the standard diffusion is necessary to define the path.

Considering the PyTorch framework, the saved files are converted to be used to create the synthetic EDs. The synthetic data from the generative AI models is organized as follows:
```
inputdata/original/0 ... 133
inputdata/synthetic_GAN/0 ... 133
inputdata/synthetic_cGAN/0 ... 133
inputdata/synthetic_diffusion/0 ... 133
inputdata/synthetic_cdiffusion/0 ... 133
```

The objects are used considering probabilities of their appearance based on the model's performance using original data. This approach aims to improve the model's ability to identify rare objects, considering that using unbalanced data is one of the major challenges in applying deep learning models.





![image](https://github.com/SFStefenon/synthetic_ED/assets/88292916/1f6741c8-7800-454d-b95f-a80d514180a4)
