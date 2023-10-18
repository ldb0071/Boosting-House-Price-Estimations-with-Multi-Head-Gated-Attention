# ASI
Attention-Based Spatial Interpolation for House Price Prediction 

### Requirements

* python 3.8.10
* tensorflow (>=2.5.0)

## Data

* Data: a numpy saved file (.npz) containing:
  * data['dist_eucli']
  * data['dist_geo']
  * data['idx_eucli']
  * data['idx_geo']
  * data['X_train']
  * data['X_test']
  * data['y_train']
  * data['y_test']

## Installation
To install as a module:
```
$ conda create -n asi python=3.8.10
$ conda activate asi
$ git clone https://github.com/darniton/ASI
$ cd ASI
$ pip install -r requirements.txt
$ jupyter notebook
```

## Replicating experiments
To replicate the experimental results:

* ./notebooks: One notebook for each dataset
    