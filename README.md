# Synthetic Engineering Drawings

This repository presents a rule and statistical-based model for creating synthetic engineering drawings (EDs). The application of the proposed method is based on the generation of synthetic objects using generative artificial intelligence (AI) models. In particular, generative adversarial networks (GANs), conditional GANs (cGANs), diffusion, and conditional diffusion (c_diffusion) models are considered. The original annotations are converted to a CSV file to automate their loading to generative AI models (using this [Algorithm 1](https://colab.research.google.com/github/SFStefenon/synthetic_ED/blob/main/save_CSV_file_from_images_annotated.ipynb)).

The generated objects using generative AI are based on the annotations of relay-based railway interlocking systems (RRIS) EDs from the rail network. Regarding the annotations, symbols, labels, specifiers, and electrical connections are considered. The algorithms to generate the 28-pixel objects are available at:
[GANs](https://colab.research.google.com/github/SFStefenon/synthetic_ED/blob/main/gans.ipynb), 
[cGANs](https://colab.research.google.com/github/SFStefenon/synthetic_ED/blob/main/c_gans.ipynb), 
[diffusion](https://colab.research.google.com/github/SFStefenon/synthetic_ED/blob/main/difussion.ipynb), or 
[c_diffusion](https://colab.research.google.com/github/SFStefenon/synthetic_ED/blob/main/c_difussion.ipynb). Obs: to compute the standard diffusion is necessary to define the path.

Considering the PyTorch framework, the saved files are converted to be used to create the synthetic EDs. The organization of the original annotations as well as those generated synthetically are saved according to their class. There are labels (upper and lower case letters), specifiers [arrows down or up], and symbols [57 variations]. The saved torch files are converted to JPG files by [Algorithm 2](https://colab.research.google.com/github/SFStefenon/synthetic_ED/blob/main/c_gan_convert_to_jpg.ipynb) (for GANs) and [Algorithm 3](https://colab.research.google.com/github/SFStefenon/synthetic_ED/blob/main/c_diffusion_convert_to_jpg.ipynb) for (for diffusion models). The synthetic data from the generative AI models is organized as follows:
```
inputdata/original/0 ... 132
inputdata/synthetic_GAN/0 ... 132
inputdata/synthetic_cGAN/0 ... 132
inputdata/synthetic_diffusion/0 ... 132
inputdata/synthetic_cdiffusion/0 ... 132
```

The objects are used considering probabilities of their appearance based on the model's performance using original data. This approach aims to improve the model's ability to identify rare objects, considering that using unbalanced data is one of the major challenges in applying deep learning models. The rule and statistical-based model to create the synthetic EDs is the [Algorithm 4](https://colab.research.google.com/github/SFStefenon/synthetic_ED/blob/main/synthetic_EDs.ipynb), and the flowchart is as follows:



![image](https://github.com/SFStefenon/synthetic_ED/assets/88292916/1f6741c8-7800-454d-b95f-a80d514180a4)

If you need to use synthetic EDs larger than 640 pixels, you will need to convert the coordinates to you only look once (YOLO) coordinate system. [Algorithm 5](https://colab.research.google.com/github/SFStefenon/synthetic_ED/blob/main/save_640_640_BBs_for_YOLO.ipynb) converts the coordinates of the bounding boxes to YOLO coordinate system and saves the names as needed to load this data. [Algorithm 6](https://colab.research.google.com/github/SFStefenon/synthetic_ED/blob/main/save_640_640_slide_window_for_YOLO.ipynb) crops 640 pixels and saves them to be used directly in conjunction with the annotations.

**Algorithms 1, 2, 3, 4, 5, and 6 were written from scratch by the author**, the generative AI models were modified from online available models.

Thank you.

Dr. **Stefano Frizzo Stefenon**.

Trento, Italy, July 10, 2024.
