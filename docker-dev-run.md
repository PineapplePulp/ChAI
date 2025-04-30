


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

