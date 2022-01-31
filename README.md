 ---

<div align="center">    
 
# Olfactory bulb segmentation    

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.journals.elsevier.com/neural-networks)

![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)


<!--  
Conference   
-->   
</div>
 
## Description   
To address the challenging problem of fully automated segmentation of human olfactory bulb (OB) neuronal layers, we propose a new pipeline for tomographic data processing. Convolutional neural networks (CNN) was used to segment X-ray phase-contrast tomographic (XPCT) image of native unstained OB. Virtual segmentation of the whole OB and an accurate delineation of each morphological layer of OB in a healthy non-demented person is mandatory as the first step for assessing OB morphological changes in smell impairment research. Despite many studies, the human OB's impairment accompanying olfactory dysfunction is still a hotly-debated topic. In this framework, we proposed an effective tool that could help to shed light on OB layer-specific degeneration in patients with olfactory disturbance.   

## Data
```
data/
  ├── binary_data/
  |     ├── train
  |     |   └── imgs/
  |     |   |    └── raw_1.tif
  |     |   |    └── raw_2.tif
  |     |   └── masks/
  |     |        └── raw_1.png
  |     |        └── raw_2.png
  |     |   
  |     └── val
  |         └── imgs/
  |         |    └── raw_1.tif
  |         |    └── raw_2.tif
  |         └── masks/
  |              └── raw_1.png
  |              └── raw_2.png
  |     
  └── multiclass_data/
        ├── train
        |   └── imgs/
        |   |    └── raw_1.tif
        |   |    └── raw_2.tif
        |   └── masks/
        |        └── raw_1.png
        |        └── raw_2.png
        |   
        └── val
            └── imgs/
            |    └── raw_1.tif
            |    └── raw_2.tif
            └── masks/
                 └── raw_1.png
                 └── raw_2.png
```

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/Stevel705/TomoPCSegmentation.git

# install dependencies  
pip install -r requirements.txt
 ```   

Next, navigate to any file and run it.   
```bash
# run
# for binary segmenation   
python main.py --dataset data/binary_data/ --n_channels 1 --n_classes 1
# for multiclass segmenation
python main.py --dataset data/multiclass_data/ --n_channels 1 --n_classes 6
```

## Train
Description or del:
```python

train = OlfactoryBulbDataset(hparams.dataset, 'train', is_transform=True)
train = DataLoader(train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

val = OlfactoryBulbDataset(hparams.dataset, 'val', is_transform=True)
val = DataLoader(val, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

test = OlfactoryBulbDataset(hparams.dataset, 'test', is_transform=False)
test = DataLoader(test, batch_size=BATCH_SIZE)

# init model
model = LightningUnet(hparams.n_channels, hparams.n_classes, hparams.learning_rate)

# Initialize a trainer
logger = TensorBoardLogger(hparams.log_dir, name="my_model")
trainer = pl.Trainer(gpus=1, max_epochs=20, progress_bar_refresh_rate=20, logger=logger)

# Train the model ⚡
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```

[Check demo](https://ec2-52-73-12-215.compute-1.amazonaws.com/)

### Citation   
```
@article{A. Meshkov, A. Khafizov, A. Buzmakov, I. Bukeeva, O. Junemann, M. Fratini, A. Cedola, M. Chukalina, A. Yamaev, G. Gigli, F. Wilde, E. Longo V. Asadchikov S. Saveliev, D. Nikolaev},
  title={Your Title},
  author={Your team},
  journal={Location},
  year={2022}
}
```   