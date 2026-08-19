INCLUDE =       -I. -I/usr/include -I./INCLUDE 
LIB     =       -L/usr/lib 
DEBUG   =
CC      =       mpicc
##CFLAGS  =       $(DEBUG) -ggdb
CFLAGS  =       -O2
LDFLAGS =       -lm -std=c99



default: aa

clean:
	rm -f *.o rnn/*.o aa

tagsfile:
	ctags *.c

zip:
	zip  GPU_cancer_$(shell date +%Y%m%d).tar *.c  *.cl  makefile Input/IC

aa: main.o rnn/rnn_balancer.o
	$(CC) main.o rnn/rnn_balancer.o $(CFLAGS) -lOpenCL -lm -o aa 

main.o: main.c
	$(CC) $(CFLAGS) -std=c99 -c main.c

rnn/rnn_balancer.o: rnn/rnn_balancer.c rnn/rnn_balancer.h
	$(CC) $(CFLAGS) -std=c99 -c rnn/rnn_balancer.c -o rnn/rnn_balancer.o

