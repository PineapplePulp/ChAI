

Docker linux
```bash
$ /usr/bin/c++ -DUSE_C10D_GLOO -DUSE_DISTRIBUTED -DUSE_RPC -DUSE_TENSORPIPE -isystem /app/libtorch/include -isystem /app/libtorch/include/torch/csrc/api/include -std=c++17 -lm -ldl -std=gnu++17 -D_GLIBCXX_USE_CXX11_ABI=1 -MD -MT CMakeFiles/MyProject.dir/torch_test.cpp.o -MF CMakeFiles/MyProject.dir/torch_test.cpp.o.d -o CMakeFiles/MyProject.dir/torch_test.cpp.o -c /app/demos/torchtest/torch_test.cpp

$ /usr/bin/c++  -std=c++17 -lm -ldl CMakeFiles/MyProject.dir/torch_test.cpp.o -o MyProject   -L/lib/intel64  -L/lib/intel64_win  -L/lib/win-x64  -Wl,-rpath,/lib/intel64:/lib/intel64_win:/lib/win-x64:/app/libtorch/lib /app/libtorch/lib/libtorch.so /app/libtorch/lib/libc10.so /app/libtorch/lib/libkineto.a -Wl,--no-as-needed,"/app/libtorch/lib/libtorch_cpu.so" -Wl,--as-needed /app/libtorch/lib/libc10.so -Wl,--no-as-needed,"/app/libtorch/lib/libtorch.so" -Wl,--as-needed 
```

Mac
```bash
$ /usr/bin/c++ -DUSE_C10D_GLOO -DUSE_DISTRIBUTED -DUSE_RPC -DUSE_TENSORPIPE -isystem /Users/iainmoncrief/Documents/Github/ChAI/libtorch/include -isystem /Users/iainmoncrief/Documents/Github/ChAI/libtorch/include/torch/csrc/api/include -std=c++17 -lm -ldl -std=gnu++17 -arch arm64 -MD -MT CMakeFiles/MyProject.dir/torch_test.cpp.o -MF CMakeFiles/MyProject.dir/torch_test.cpp.o.d -o CMakeFiles/MyProject.dir/torch_test.cpp.o -c /Users/iainmoncrief/Documents/Github/ChAI/demos/torchtest/torch_test.cpp

$ /usr/bin/c++  -std=c++17 -lm -ldl -arch arm64 -Wl,-search_paths_first -Wl,-headerpad_max_install_names -L/opt/homebrew/opt/ruby/lib CMakeFiles/MyProject.dir/torch_test.cpp.o -o MyProject  -Wl,-rpath,/Users/iainmoncrief/Documents/Github/ChAI/libtorch/lib /Users/iainmoncrief/Documents/Github/ChAI/libtorch/lib/libc10.dylib /Users/iainmoncrief/Documents/Github/ChAI/libtorch/lib/libkineto.a /Users/iainmoncrief/Documents/Github/ChAI/libtorch/lib/libtorch.dylib /Users/iainmoncrief/Documents/Github/ChAI/libtorch/lib/libtorch_cpu.dylib /Users/iainmoncrief/Documents/Github/ChAI/libtorch/lib/libc10.dylib
```