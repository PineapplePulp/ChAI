# make clean && make libsmol && make main && ./main
-include lib/Makefile.smol


all: main 

# chpl --library --library-makefile --library-cmakelists smol.chpl


CHAI_DIR := $(shell cd ../../.. && pwd)

# BRIDGE_CFLAGS = \
# 	-I$(CHAI_DIR)/bridge/include \
# 	$(CHAI_DIR)/build/libbridge_objs.a


BRIDGE_CFLAGS = \
	-I $(CHAI_DIR)/bridge/include

BRIDGE_LDFLAGS = \
	-L $(CHAI_DIR)/build \
	-L $(CHAI_DIR)/libtorch/lib \
	-Wl,-rpath,$(CHAI_DIR)/libtorch/lib

TORCH_LDFLAGS = \
	-L $(CHAI_DIR)/libtorch/lib \
	-Wl,-rpath,$(CHAI_DIR)/libtorch/lib

LIB_GEN_CMD = \
	chpl \
	smol.chpl \
	--library \
	--dynamic \
	--library-dir lib \
	--library-makefile \
	--library-cmakelists \
	--no-munge-with-ids \
	--fast \
	-M ../../../lib \
	--ccflags "$(BRIDGE_CFLAGS)" \
	--ldflags "$(BRIDGE_LDFLAGS)" 


# --ccflags "$(BRIDGE_CFLAGS)" \
# --ldflags "$(BRIDGE_LDFLAGS)" 
# --savec lib/savec \


pre-build:
	if [ -d lib ]; then \
		echo "lib directory already exists" \
	else \
		$(LIB_GEN_CMD) \
	fi
	


libsmol: smol.chpl
	echo "$(CHAI_DIR)"
	$(LIB_GEN_CMD)


OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LDFLAGS := $(shell pkg-config --cflags --libs opencv4)

main: main.cpp smol_wrapper.h lib/libsmol.so
	@echo $(OPENCV_CFLAGS)
	$(CHPL_LINKER) $(CHPL_CFLAGS) $(OPENCV_CFLAGS) $(BRIDGE_CFLAGS) -O2 -std=c++20 -fPIC main.cpp -o main $(CHPL_LDFLAGS) $(OPENCV_LDFLAGS) $(BRIDGE_LDFLAGS)

clean:
	rm -f maincpp maincpp.o main.o main

cleanall: clean
	rm -rf lib



cmain: main.c
	$(CHPL_COMPILER) $(CHPL_CFLAGS) -o main main.c $(CHPL_LDFLAGS)



CPP_CFLAGS = \
	-Ilib \
	-Wno-unused \
	-Wno-uninitialized \
	-Wno-pointer-sign \
	-Wno-incompatible-pointer-types \
	-Wno-tautological-compare \
	-I/opt/homebrew/Cellar/chapel/2.4.0_1/libexec/modules/internal \
	-I$(CHPL_RUNTIME_INCL)/localeModels/flat \
	-I$(CHPL_RUNTIME_INCL)/localeModels \
	-I$(CHPL_RUNTIME_INCL)/comm/none \
	-I$(CHPL_RUNTIME_INCL)/comm \
	-I$(CHPL_RUNTIME_INCL)/tasks/qthreads \
	-I$(CHPL_RUNTIME_INCL)/. \
	-I$(CHPL_RUNTIME_INCL)/./qio \
	-I$(CHPL_RUNTIME_INCL)/./atomics/cstdlib \
	-I$(CHPL_RUNTIME_INCL)/./mem/jemalloc \
	-I$(CHPL_THIRD_PARTY)/utf8-decoder \
	-I$(CHPL_THIRD_PARTY)/qthread/install/darwin-arm64-native-llvm-none-flat-jemalloc-system/include \
	-Wno-error=unused-variable \
	-I$(CHPL_THIRD_PARTY)/re2/install/darwin-arm64-native-llvm-none/include \
	-I. -I/opt/homebrew/Cellar/gmp/6.3.0/include \
	-I/opt/homebrew/Cellar/hwloc/2.12.0/include \
	-I/opt/homebrew/Cellar/jemalloc/5.3.0/include \
	-I/opt/homebrew/include

CPP_LDFLAGS = -Llib -lsmol -L$(CHPL_RUNTIME_LIB)/darwin/llvm/arm64/cpu-native/loc-flat/comm-none/tasks-qthreads/tmr-generic/unwind-none/mem-jemalloc/atomics-cstdlib/hwloc-system/re2-bundled/fs-none/lib_pic-none/san-none -lchpl -L$(CHPL_THIRD_PARTY)/qthread/install/darwin-arm64-native-llvm-none-flat-jemalloc-system/lib -Wl,-rpath,$(CHPL_THIRD_PARTY)/qthread/install/darwin-arm64-native-llvm-none-flat-jemalloc-system/lib -lqthread -L/opt/homebrew/Cellar/hwloc/2.12.0/lib -L$(CHPL_THIRD_PARTY)/re2/install/darwin-arm64-native-llvm-none/lib -lre2 -Wl,-rpath,$(CHPL_THIRD_PARTY)/re2/install/darwin-arm64-native-llvm-none/lib -lm -lpthread -L/opt/homebrew/Cellar/gmp/6.3.0/lib -lgmp -L/opt/homebrew/Cellar/hwloc/2.12.0/lib -Wl,-rpath,/opt/homebrew/Cellar/hwloc/2.12.0/lib -lhwloc -L/opt/homebrew/Cellar/jemalloc/5.3.0/lib -Wl,-rpath,/opt/homebrew/Cellar/jemalloc/5.3.0/lib -ljemalloc -L/opt/homebrew/lib

# $(CHPL_LINKER) $(CHPL_CFLAGS) -o main main.cpp $(CHPL_LDFLAGS)

maincpp.o: main.cpp smol_wrapper.h
	$(CHPL_COMPILER) $(CHPL_CFLAGS) -fPIC main.cpp -o maincpp.o $(CHPL_LDFLAGS)
	$(CHPL_COMPILER) $(CHPL_CFLAGS) -fPIC main.cpp -o maincpp $(CHPL_LDFLAGS)

# maincpp: maincpp.o
# 	$(CHPL_LINKER) -o maincpp maincpp.o
# #$(CHPL_LINKER) $(CHPL_CFLAGS) -o main main.o $(CHPL_LDFLAGS)

