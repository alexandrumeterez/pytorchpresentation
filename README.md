# PyTorch - a quick presentation

Introductory course on PyTorch for my college. Included is a demo on transfer learning and data collecting.

### Prerequisites

What things you need to install the software and how to install them:

```
    -anaconda3: download from https://www.anaconda.com/download/
    -create a new enviroment: conda create --name demoenv
    -activate the enviroment: source activate demoenv
    -follow the instructions here to install pytorch: https://pytorch.org/
    -install opencv: conda install -c conda-forge opencv 
    -install matplotlib: conda install matplotlib
```

## Running

To collect data, simply run data_collect.py.
With 0, 1 and 2 you select what class you are currently taking photos for. With r you start recording. With n you delete every photo you collected.
Use q to quit and p to start predicting(Warning: Choose an existing model in the code, saved in the models directory).

To train/test a model, run any of the 3 scripts: traditional, transfered_frozen, transfered_finetune.

## Authors

* **Alexandru Meterez** - (https://github.com/alexandrumeterez)

