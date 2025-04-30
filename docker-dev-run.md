


To build the image, run the following command in the root directory of the project:
```bash
$ docker build -t chai-build-env . --platform="linux/amd64" --progress=plain
```

To make a volume for the container, use the following command:
```bash
$ docker volume create chai-vol
```
This volume part might not be necessary. 


To run the container, use the following command:
```bash
$ docker run -it --platform="linux/amd64" --name chai-vm --hostname chai-vm --privileged --memory 2g --cpus 2 -v chai-vol chai-build-env
```

Then you may run the following:
```bash
root@chai-vm:/app# cd demos/torchtest
root@chai-vm:/app/demos/torchtest# mkdir build
root@chai-vm:/app/demos/torchtest# cd build
root@chai-vm:/app/demos/torchtest/build# cmake ..
root@chai-vm:/app/demos/torchtest/build# make
root@chai-vm:/app/demos/torchtest/build# ./MyProject
Input: [1, 3, 10, 10]
Output: [1, 3, 10, 10]
```
OR
```bash
root@chai-vm:/app# cd build
root@chai-vm:/app/build# cmake -DCMAKE_BUILD_TYPE=Release ..
root@chai-vm:/app/build# make -j2
root@chai-vm:/app/build# ./MyExample
...
root@chai-vm:/app/build# ./examples/torch_model_loading/TorchLoad images/dog.jpg
Read file: /app/build/images/dog.jpg with ext: jpg
Resized image: (3, 224, 224)
Read file: /app/build/images/dog.jpg with ext: jpg
Batched image: (1, 3, 734, 1100)
Batched image resized: (1, 3, 734, 1100)
Squeezed image: (3, 734, 1100)
root@chai-vm:/app/build# ./vgg images/frog.jpg
 # no pytorch installed, and no weights preprocessed. 
```



Locally, you can do
```bash
$ export Torch_DIR=<ChAI-root>/libtorch
$ cd demos/torchtest
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=<ChAI-root>/libtorch/share/cmake ..
$ make
$ ./MyProject
Input: [1, 3, 10, 10]
Output: [1, 3, 10, 10]
```

