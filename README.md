# Requirements

Install [Miniconda](https://conda.io/miniconda.html)

# Installation

```
conda env create -f environment.yml
jupyter notebook
```

Download YOLO v3 weights and convert them to keras model using to instructions from: https://github.com/qqwweee/keras-yolo3

After you're done you should have file `yolo.h5` in your model_data directory.