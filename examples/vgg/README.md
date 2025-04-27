# README

Everything should be run from this directory.

## getting the model

Run `python3 dump_weights.py` to download the model and prep it for Chapel.

## Processing an image

Run `python3 process_img.py <path to image>` to process an image. The image will be resized to 224x224 and normalized to the imagenet mean and standard deviation. The output will be saved in a convenient format for ChAI in `imgs/`

## Running the model

Compile the Chapel code with `chpl --fast test.chpl -M ../../src -M ../../lib -o vgg` and run it with `./vgg <path to image>.chdata`. The output will be the top 5 classes and their probabilities.

## Running the python model

Run `python3 test.py <path to image>` to run the model in python. The output will be the top 5 classes and their probabilities.

#### Compare with `python3 run_vgg.py imgs/frog.jpg` and `./vgg imgs/frog.chdata`


```bash
$ chpl --fast -o vgg test.chpl -M ../../lib /Users/iainmoncrief/Documents/Github/ChAI/bridge/include/bridge.h /Users/iainmoncrief/Documents/Github/ChAI/build/CMakeFiles/bridge.dir/bridge/lib/bridge.cpp.o -L /Users/iainmoncrief/Documents/Github/ChAI/libtorch/lib -ltorch -ltorch_cpu -lc10 -ltorch_global_deps --ldflags "-Wl,-rpath,/Users/iainmoncrief/Documents/Github/ChAI/libtorch/lib"
```