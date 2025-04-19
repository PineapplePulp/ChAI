git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
pip install cmake ninja
pip install -r requirements.txt
pip install pkg-config libuv
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build --cmake-only


python3 setup.py develop
