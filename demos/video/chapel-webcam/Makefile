
include lib/Makefile.smol

all: libsmol.a main

libsmol.a: smol.chpl
	chpl --library --library-makefile --library-cmakelists smol.chpl

main: main.c libsmol.a
	$(CHPL_COMPILER) $(CHPL_CFLAGS) -o main main.c $(CHPL_LDFLAGS)

clean:
	rm -f main lib/libsmol.a
