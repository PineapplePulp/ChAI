set(CHPL_RUNTIME_LIB /opt/homebrew/Cellar/chapel/2.4.0_1/libexec/lib)

set(CHPL_RUNTIME_INCL /opt/homebrew/Cellar/chapel/2.4.0_1/libexec/runtime/include)

set(CHPL_THIRD_PARTY /opt/homebrew/Cellar/chapel/2.4.0_1/libexec/third-party)

set(CHPL_HOME /opt/homebrew/Cellar/chapel/2.4.0_1/libexec)

set(smol_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}  /opt/homebrew/Cellar/chapel/2.4.0_1/libexec/modules/internal /opt/homebrew/Cellar/chapel/2.4.0_1/libexec/modules/packages ../../../lib ${CHPL_RUNTIME_INCL}/localeModels/flat ${CHPL_RUNTIME_INCL}/localeModels ${CHPL_RUNTIME_INCL}/comm/none ${CHPL_RUNTIME_INCL}/comm ${CHPL_RUNTIME_INCL}/tasks/qthreads ${CHPL_RUNTIME_INCL}/. ${CHPL_RUNTIME_INCL}/./qio ${CHPL_RUNTIME_INCL}/./atomics/cstdlib ${CHPL_RUNTIME_INCL}/./mem/jemalloc ${CHPL_THIRD_PARTY}/utf8-decoder ${CHPL_THIRD_PARTY}/qthread/install/darwin-arm64-native-llvm-none-flat-jemalloc-system/include -Wno-error=unused-variable ${CHPL_THIRD_PARTY}/re2/install/darwin-arm64-native-llvm-none/include . /opt/homebrew/Cellar/gmp/6.3.0/include /opt/homebrew/Cellar/hwloc/2.12.0/include /opt/homebrew/Cellar/jemalloc/5.3.0/include /opt/homebrew/include)

set(smol_LINK_LIBS -L${CMAKE_CURRENT_LIST_DIR} -lsmol  -ltorch -ltorch_cpu -lc10 -ltorch_global_deps -lbridge_objs -L${CHPL_RUNTIME_LIB}/darwin/llvm/arm64/cpu-native/loc-flat/comm-none/tasks-qthreads/tmr-generic/unwind-none/mem-jemalloc/atomics-cstdlib/hwloc-system/re2-bundled/fs-none/lib_pic-none/san-none -lchpl -L${CHPL_THIRD_PARTY}/qthread/install/darwin-arm64-native-llvm-none-flat-jemalloc-system/lib -Wl,-rpath,${CHPL_THIRD_PARTY}/qthread/install/darwin-arm64-native-llvm-none-flat-jemalloc-system/lib -lqthread -L/opt/homebrew/Cellar/hwloc/2.12.0/lib -L${CHPL_THIRD_PARTY}/re2/install/darwin-arm64-native-llvm-none/lib -lre2 -Wl,-rpath,${CHPL_THIRD_PARTY}/re2/install/darwin-arm64-native-llvm-none/lib -lm -lpthread -L/opt/homebrew/Cellar/gmp/6.3.0/lib -lgmp -L/opt/homebrew/Cellar/hwloc/2.12.0/lib -Wl,-rpath,/opt/homebrew/Cellar/hwloc/2.12.0/lib -lhwloc -L/opt/homebrew/Cellar/jemalloc/5.3.0/lib -Wl,-rpath,/opt/homebrew/Cellar/jemalloc/5.3.0/lib -ljemalloc -L/opt/homebrew/lib -lsmol)

set(CHPL_COMPILER /opt/homebrew/Cellar/llvm@19/19.1.7/bin/clang)
set(CHPL_LINKER /opt/homebrew/Cellar/llvm@19/19.1.7/bin/clang++)
set(CHPL_LINKERSHARED /opt/homebrew/Cellar/llvm@19/19.1.7/bin/clang++ -shared)
