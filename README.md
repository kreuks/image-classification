# n00b tensorflow
This is just example usage of tensorflow and for research purpose

# Getting Started
* Make sure you have nvidia graphics card, run `lspci -nnk | grep -i nvidia`
* Install anaconda
* Run `sudo apt-get update` and `sudo apt-get install build-essential`
* Download nvidia driver [here](http://www.nvidia.com/Download/index.aspx?lang=en-us) and follow their instructions
* Make sure you can run `nvidia-smi`
* Install CUDA 8.0 [here](https://developer.nvidia.com/cuda-downloads) and follow their instructions. I'm using deb package
* Set env variables for CUDA
    ```
    echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
    echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64' >> ~/.bashrc
    source ~/.bashrc
    ```
* Verify cuda installation by running nvcc --version
* Install cuDNN
> The NVIDIA CUDA® Deep Neural Network library (cuDNN) is a GPU-accelerated library of primitives
> for deep neural networks. cuDNN provides highly tuned implementations for standard routines such as
> forward and backward convolution, pooling, normalization, and activation layers.
> cuDNN is part of the NVIDIA Deep Learning SDK.
* You need to register at [NVIDIA Developer program](https://developer.nvidia.com/developer-program) in order to download cuDNN
* Download cuDNN version 5.1 for CUDA 8.0 (tar version)
* scp to your machine and run
```
tar xvzf [filename].tgz

sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
```
* Install tensorflow [here](https://www.tensorflow.org/install/). Make sure you choose tensorflow with GPU
* Install keras [here](https://keras.io)

# Data Training

Placed your data training in `images` folder with `training_data` folder containing your training image and `testing_data`
containing your testing data
example:

```
+-- images
|   +-- catdog
|       +-- training_data
|           +-- cats
|               +-- cat00001.jpg
|               +-- cat00002.jpg
|               +-- ...
|           +-- dogs
|               +-- dog10001.jpg
|               +-- dog10002.jpg
|               +-- ...
|       +-- testing_data
|           +-- cats
|               +-- cat10001.jpg
|               +-- cat10002.jpg
|               +-- ...
|           +-- dogs
|               +-- dog10001.jpg
|               +-- dog10002.jpg
|               +-- ...
```