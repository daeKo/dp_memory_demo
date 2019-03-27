# dp_demo
This is a short demo for training a simple two-layer LSTM character-level language model using differentially private SGD and regular one. It is based on the language model example given in https://github.com/tensorflow/privacy/blob/master/tutorials/lm_dpsgd_tutorial.py

## Installation

First, install matplotlib, then you just have to follow the instructions in https://github.com/tensorflow/privacy, since it covers all remaining dependencies.

You also should copy a version of the Pennchar Dataset to ./data/.

## Usage

For training a new model without DP-SGD:

python lm_dpsgd_tutorial.py --epochs 1 --data_dir ../data/pennchar_augmented --model_dir ../model_dir --batch_size 16 --microbatches 16 --nodpsgd --noload_model


For training a model with DP-SGD:

python lm_dpsgd_tutorial.py --epochs 1 --data_dir ../data/pennchar_augmented --model_dir ../model_dir --batch_size 16 --microbatches 16 --dpsgd --noload_model

Note: Training with DP-SGD requires significantly more memory than without. E.g. the test machine with 8GB could not handle to increase the batch_size and microbatches above 16.


For loading a model change --noload_model to --load_model. Note that the model will be trained for the specified number of epochs. So if you just want to load it, you should use --epochs 0.
