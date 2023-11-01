# Attention-Based Spatial Interpolation for House Price Prediction 

This project is an implementation based on the original code found at [darniton/ASI](https://github.com/darniton/ASI) 

## Project Overview
This project aims to leverage Multi-head gated Attention-Based Spatial Interpolation to improve the accuracy of house price predictions. It introduces key concepts and utilizes advanced machine learning techniques for spatial data analysis.

## Requirements

- python 3.8.10
- tensorflow (>=2.5.0)

## Data

Data is a numpy saved file (.npz) containing:
- `data['dist_eucli']`
- `data['dist_geo']`
- `data['idx_eucli']`
- `data['idx_geo']`
- `data['X_train']`
- `data['X_test']`
- `data['y_train']`
- `data['y_test']`

## Data Preparation
Explain the data preparation steps, including:
- How to download or generate the data
- Data pre-processing steps (if any)
- How to save the data in `.npz` format

## Installation

To install as a module:
```bash
$ conda create -n asi python=3.8.10
$ conda activate asi
$ git clone https://github.com/<Your_GitHub_Repo>/ASI
$ cd ASI
$ pip install -r requirements.txt
$ jupyter notebook
