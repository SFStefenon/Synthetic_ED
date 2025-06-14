# Synthetic Engineering Drawings

This repository presents a probabilistic ruled-based model for creating synthetic engineering drawings (EDs). 

---

## Synthetic Objects Using Generative AI

The application of the proposed method is based on the generation of synthetic objects using generative artificial intelligence (AI) models. In particular, generative adversarial networks (GANs), conditional GANs (cGANs), diffusion, and conditional diffusion (c_diffusion) models are considered. The original annotations are converted to a CSV file to automate their loading to apply generative AI models (using [Algorithm 1](https://github.com/SFStefenon/synthetic_ED/blob/main/Colabs/save_CSV_file_from_images_annotated.ipynb)).

The generated objects using generative AI are based on the annotations of relay-based railway interlocking systems (RRIS) EDs from the rail network. Regarding the annotations,  symbols, labels, specifiers, and electrical connections are considered. The algorithms to generate the 28-pixel objects are available at:
[GANs](https://github.com/SFStefenon/synthetic_ED/blob/main/Colabs/gans.ipynb), 
[cGANs](https://github.com/SFStefenon/synthetic_ED/blob/main/Colabs/c_gans.ipynb), 
[diffusion](https://github.com/SFStefenon/synthetic_ED/blob/main/Colabs/diffusion_m2.ipynb), or 
[c_diffusion](https://github.com/SFStefenon/synthetic_ED/blob/main/Colabs/c_difussion.ipynb). Obs: To compute these models is necessary to define the storage path.

---

## Synthetic Drawings Considering a Probabilistic Ruled-based Model

Considering the PyTorch framework, the saved files are converted to be used to create the synthetic EDs. The organization of the original annotations as well as those generated synthetically are saved according to their class. There are labels [numbers, upper and lower case letters], specifiers [arrows up or down], and symbols [57 variations]. The saved torch files are converted to JPG files by [Algorithm 2](https://github.com/SFStefenon/synthetic_ED/blob/main/Colabs/c_gan_convert_to_jpg.ipynb) (for GANs) and [Algorithm 3](https://github.com/SFStefenon/synthetic_ED/blob/main/Colabs/c_diffusion_convert_to_jpg.ipynb) (for diffusion models). The synthetic data from the generative AI models is organized as follows:
```
inputdata/original/0 ... 132
inputdata/synthetic_GAN/0 ... 132
inputdata/synthetic_cGAN/0 ... 132
inputdata/synthetic_diffusion/0 ... 132
inputdata/synthetic_cdiffusion/0 ... 132
```

The objects are used considering the probabilities of their appearance based on the model's performance using original data. This approach aims to improve the model's ability to identify rare objects, considering that using unbalanced data is one of the major challenges in applying deep learning models. The rule and statistical-based model to create the synthetic EDs is the [Algorithm 4](https://github.com/SFStefenon/synthetic_ED/blob/main/Colabs/synthetic_EDs.ipynb), and the flowchart is as follows:

![image](https://github.com/SFStefenon/synthetic_ED/assets/88292916/6377cc69-2606-4940-8b45-8cc3494283b9)

If you need to use synthetic EDs larger than 640 pixels, you will need to convert the coordinates to you only look once (YOLO) coordinate system. [Algorithm 5](https://github.com/SFStefenon/synthetic_ED/blob/main/Colabs/save_640_640_BBs_for_YOLO.ipynb) converts the coordinates of the bounding boxes to YOLO coordinate system and saves the names as needed to load this data. [Algorithm 6](https://github.com/SFStefenon/synthetic_ED/blob/main/Colabs/save_640_640_slide_window_for_YOLO.ipynb) crops out 640-pixel images and saves them to be used directly in along with the annotations.

---

## Annotations and Object Detection Method

The object detection was performed by the YOLOv8, to compute the experiments we followed the official developer [Ultralytics](https://github.com/ultralytics/ultralytics).

To create a custom dataset of RRIS an image labeling software was used. For this project, the [labelImg](https://github.com/heartexlabs/labelImg) based on Python was considered. An example of the annotations is presented as follows:

![image](https://github.com/SFStefenon/synthetic_ED/assets/88292916/5484c24f-65e2-4b41-b052-450772162d01)


---

**Algorithms [1](https://github.com/SFStefenon/synthetic_ED/blob/main/Colabs/save_CSV_file_from_images_annotated.ipynb), [2](https://github.com/SFStefenon/synthetic_ED/blob/main/Colabs/c_gan_convert_to_jpg.ipynb), [3](https://github.com/SFStefenon/synthetic_ED/blob/main/Colabs/c_diffusion_convert_to_jpg.ipynb), [4](https://github.com/SFStefenon/synthetic_ED/blob/main/Colabs/synthetic_EDs.ipynb), [5](https://github.com/SFStefenon/synthetic_ED/blob/main/Colabs/save_640_640_BBs_for_YOLO.ipynb), and [6](https://github.com/SFStefenon/synthetic_ED/blob/main/Colabs/save_640_640_slide_window_for_YOLO.ipynb) were written from scratch by the author**, the generative AI models were modified from online available models.

---

Wrote by Dr. **Stefano Frizzo Stefenon**

Fondazione Bruno Kessler

Trento, Italy, 2025
