# Normalization-Equivariant Neural Networks with Application to Image Denoising
Sébastien Herbreteau, Emmanuel Moebel and Charles Kervrann

## Requirements

Here is the list of libraries you need to install to execute the code:
* Python 3.8
* Pytorch 2.0
* Torchvision 0.15.1
* Numpy 1.24.2
* Pillow (PIL Fork) 9.5.0
* Matplotlib 3.7.1

## Install

To install in an environment using pip:

```
python -m venv .nenn_env
source .nenn_env/bin/activate
pip install /path/to/normalization_equivariant_nn
```

## Pre-trained models

The pre-trained models for the three variants (ordinary, scale-equivariant and normalization-equivariant) of the popular networks DRUNet [K. Zhang et al., IEEE Trans PAMI 2021] and FDnCNN [K. Zhang et al., IEEE Trans IP 2017] (see [`models`](models/)) are available at: 
https://drive.google.com/drive/u/0/folders/1qQV5AzhwlZBhjBPMG2IfQecUWVwcJKwY

## Demo

We provide a Python Jupyter Notebook with example code to reproduce the experiments of the paper: [`demo_jupyter_notebook.ipynb`](demo_jupyter_notebook.ipynb). Make sure to download the pre-trained models first before using it.

## SortPool and AffineConv2d

Channel-wise sort pooling and affine-constrained convolutional layers are implemented in Pytorch in the file [`basicblocks.py`](models/basicblocks.py).

## Acknowledgements

This work was supported by Bpifrance agency (funding) through the LiChIE contract. Computations  were performed on the Inria Rennes computing grid facilities partly funded by France-BioImaging infrastructure (French National Research Agency - ANR-10-INBS-04-07, “Investments for the future”).
