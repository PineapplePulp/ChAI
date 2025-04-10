

bridge.o: lib/bridge.cpp include/bridge.h 
	g++ -c lib/bridge.cpp -I include -o bridge.o -I /Users/iainmoncrief/Documents/Github/ChAI/bridge/libtorch/include/torch/csrc/api/include -I /Users/iainmoncrief/Documents/Github/ChAI/bridge/libtorch/include --std=c++17

Bridge: bridge.o lib/Bridge.chpl
	chpl lib/Bridge.chpl include/bridge.h bridge.o -L /Users/iainmoncrief/Documents/Github/ChAI/bridge/libtorch/lib -ltorch -ltorch_cpu -lc10 --ldflags "-Wl,-rpath,/Users/iainmoncrief/Documents/Github/ChAI/bridge/libtorch/lib" --print-commands

clean:
	rm -f bridge.o
	rm -f Bridge



