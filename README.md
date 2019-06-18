# DP Memory Example
This is a short example to show the memorization occurring in neural networks and how to prevent them with differentially private
optimization. It is based on the work of [Carlini et al.][1], which elaborates more on the topic.

In this example we train a toy character-level language model using the Penn Treebank data set and show that, when 
trained with a regular optimizer, the secret can be inferred. Using a differentially private optimizer from 
[TensorFlow Privacy][2] this is not possible.

## Installation

Usually, the easiest way to run the notebook is by running `setup.sh`.
This will create a virtual environment, install the requirements and register it as ipykernel for the jupyter notebook.
It will also print the name of the kernel you have to select.

If you are having trouble setting everything up, you can try to manually install everything.
This essentially means recreating the steps of the setup script.

`Note: Since TensorFlow Privacy is still under activ development, there might be some changes to the API.
So we have fixed the git commit.`

## Usage

The notebook should guide you through everything.
So, assuming everything installed correctly, you should just need to follow the instructions there.

[1]: https://arxiv.org/abs/1802.08232
[2]: https://github.com/tensorflow/privacy
