all: myProgram

myProgram: test.chpl libmylib.a #libmylib.a is the dependency for the executable
		chpl mylib.h -L. -lmylib -o myProgram test.chpl

# gcc -lm -o myProgram main.o -L. -lmylib

mylib.o: mylib.c mylib.h
		gcc -O -c mylib.c mylib.h

libmylib.a: mylib.o
		ar rcs libmylib.a mylib.o

libs: libmylib.a

clean:
		rm -f myProgram *.o *.a *.gch #This way is cleaner than your clean