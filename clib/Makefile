all: myProgram

myProgram: test.chpl libmylib.a #libmylib.a is the dependency for the executable
		chpl -o myProgram test.chpl --ccflags mylib.h -L. -lmylib

# gcc -lm -o myProgram main.o -L. -lmylib
# chpl mylib.h -L. -lmylib -o myProgram test.chpl

mylib.o: mylib.c mylib.h
		gcc -O -c mylib.c mylib.h

libmylib.a: mylib.o
		ar rcs libmylib.a mylib.o

libs: libmylib.a

clean:
		rm -f myProgram *.o *.a *.gch #This way is cleaner than your clean