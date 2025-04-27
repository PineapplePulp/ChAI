
CXX = /usr/bin/clang++
PWD = $(shell pwd)
LIBTORCH = $(PWD)/libtorch

all: bridge.o Bridge

bridge.o: lib/bridge.cpp include/bridge.h 
	$(CXX) -c lib/bridge.cpp -I include -o bridge.o -I $(LIBTORCH)/include/torch/csrc/api/include -I $(LIBTORCH)/include --std=c++17

# g++ -c lib/bridge.cpp -I include -o bridge.o -I /Users/iainmoncrief/Documents/Github/ChAI/bridge/libtorch/include/torch/csrc/api/include -I /Users/iainmoncrief/Documents/Github/ChAI/bridge/libtorch/include --std=c++17

Bridge: bridge.o lib/Bridge.chpl
	chpl lib/Bridge.chpl include/bridge.h bridge.o -L $(LIBTORCH)/lib -ltorch -ltorch_cpu -lc10 --ldflags "-Wl,-rpath,$(LIBTORCH)/lib"

clean:
	rm -f bridge.o
	rm -f Bridge



