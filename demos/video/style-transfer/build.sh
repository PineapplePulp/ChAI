# /usr/bin/clang++ -std=c++20 -c -fPIC mirror.cpp -o mirror.o -I ../../../libtorch/include -I ../../../libtorch/include/torch/csrc/api/include $(pkg-config --cflags --libs opencv4) -L ../../../libtorch/lib -ltorch -ltorch_cpu -lc10 -ltorch_global_deps
# /usr/bin/clang++ -shared -o libmirror.dylib mirror.o -I ../../../libtorch/include -I ../../../libtorch/include/torch/csrc/api/include $(pkg-config --cflags --libs opencv4) -L ../../../libtorch/lib -ltorch -ltorch_cpu -lc10 -ltorch_global_deps

# /usr/bin/clang++ -std=c++20 -c -fPIC mirror.cpp -o mirror.o $(pkg-config --cflags --libs opencv4)

g++ -std=c++20 -c -fPIC mirror.cpp -o mirror.o $(pkg-config --cflags opencv4)
chpl mirror.h mirror.o mirror.chpl --fast --print-commands --ldflags $(pkg-config --cflags --libs opencv4) -lstdc++
