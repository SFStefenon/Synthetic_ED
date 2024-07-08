# Synthetic Engineering Drawings

This repository presents a rule and statistical-based model for creating synthetic engineering drawings (EDs). The application of the proposed method is based on the generation of synthetic objects using generative artificial intelligence (AI) models. In particular, generative adversarial networks (GANs), conditional GANs (cGANs), diffusion, and conditional diffusion (c_diffusion) models are considered.

The generated objects using generative AI are based on the annotations of relay-based railway interlocking systems (RRIS) EDs from the rail network. Regarding the annotations, symbols, labels, specifiers, and electrical connections are considered. The algorithms to generate the 28-pixel objects are available at:
[GANs](https://colab.research.google.com/github/SFStefenon/synthetic_ED/blob/main/GANs.ipynb), 
[cGANs](https://colab.research.google.com/github/SFStefenon/synthetic_ED/blob/main/cGANs.ipynb), 
[diffusion](https://colab.research.google.com/github/SFStefenon/synthetic_ED/blob/main/difussion.ipynb), or 
[c_diffusion](https://colab.research.google.com/github/SFStefenon/synthetic_ED/blob/main/difussion.ipynb).

The objects are used considering probabilities of their appearance based on the model's performance using original data. This approach aims to improve the model's ability to identify rare objects, considering that using unbalanced data is one of the major challenges in applying deep learning models.


![image](https://github.com/SFStefenon/synthetic_ED/assets/88292916/1f6741c8-7800-454d-b95f-a80d514180a4)
