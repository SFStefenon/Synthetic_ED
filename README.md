# Synthetic Engineering Drawings

This repository presents a rule and statistical-based model for creating synthetic engineering drawings (EDs). The application of the proposed method is based on the generation of synthetic objects using generative artificial intelligence (AI) models. In particular, generative adversarial networks (GANs), conditional GANs (cGANs), diffusion models, and conditional diffusion (c_diffusion) models are considered.

The generated objects using generative AI are based on the annotations of relay-based railway interlocking systems (RRIS) EDs from the rail network. Regarding the annotations, symbols, labels, specifiers, and electrical connections are considered. The algorithms to generate the objects are available at:
[GANs](https://colab.research.google.com/github/SFStefenon/synthetic_ED/blob/main/GANs.ipynb), 
[cGANs](), 
[diffusion](), 
[c_diffusion]().

    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyO7TQOs5808jhNJYQt4PLhq",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/SFStefenon/synthetic_ED/blob/main/cGANs.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },

The objects are used considering probabilities of their appearance based on the model's performance using original data. This approach aims to improve the model's ability to identify rare objects, considering that using unbalanced data is one of the major challenges in applying deep learning models.


![image](https://github.com/SFStefenon/synthetic_ED/assets/88292916/1f6741c8-7800-454d-b95f-a80d514180a4)
