INCLUDE =       -I. -I/usr/include -I./INCLUDE 
LIB     =       -L/usr/lib 
DEBUG   =
CC      =       gcc
##CFLAGS  =       $(DEBUG) -ggdb
CFLAGS  =       -O2
LDFLAGS =       -lm -std=c99



default: aa

clean:
	rm -f *.o

tagsfile:
	ctags *.c

zip:
	zip  GPU_cancer_$(shell date +%Y%m%d).tar *.c  *.cl  makefile Input/IC

aa: main.o
	$(CC) main.o $(CFLAGS) -lOpenCL -lm -o aa 

main.o: main.c
	$(CC) $(CFLAGS) -std=c99 -c main.c

