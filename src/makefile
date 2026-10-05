INCLUDE =       -I. -I/usr/include -I./INCLUDE 
LIB     =       -L/usr/lib 
DEBUG   =
CC      =       mpicc
##CFLAGS  =       $(DEBUG) -ggdb
CFLAGS  =       -O2 -DCL_TARGET_OPENCL_VERSION=300
LDFLAGS =       -lm -std=c99



default: aa

clean:
	rm -f *.o rnn/*.o aa

tagsfile:
	ctags *.c

zip:
	zip  GPU_cancer_$(shell date +%Y%m%d).tar *.c  *.h  *.cl  makefile Input/IC

aa: main.o sim_globals.o domain.o halo.o io.o biology.o sim.o ocl_setup.o rnn/rnn_balancer.o
	$(CC) $^ $(CFLAGS) -lOpenCL -lm -o aa

main.o: main.c sim_globals.h sim_types.h domain.h io.h biology.h mpi_assert.h sim.h rnn/rnn_balancer.h
	$(CC) $(CFLAGS) -std=c99 -c main.c

sim_globals.o: sim_globals.c sim_globals.h
	$(CC) $(CFLAGS) -std=c99 -c sim_globals.c

domain.o: domain.c domain.h sim_globals.h mpi_assert.h
	$(CC) $(CFLAGS) -std=c99 -c domain.c

halo.o: halo.c halo.h sim_globals.h sim_types.h mpi_assert.h
	$(CC) $(CFLAGS) -std=c99 -c halo.c

io.o: io.c io.h sim_globals.h mpi_assert.h
	$(CC) $(CFLAGS) -std=c99 -c io.c

biology.o: biology.c biology.h sim_globals.h
	$(CC) $(CFLAGS) -std=c99 -c biology.c

sim.o: sim.c sim.h sim_globals.h sim_types.h domain.h halo.h io.h mpi_assert.h ocl_setup.h rnn/rnn_balancer.h
	$(CC) $(CFLAGS) -std=c99 -c sim.c

ocl_setup.o: ocl_setup.c ocl_setup.h io.h sim_globals.h mpi_assert.h
	$(CC) $(CFLAGS) -std=c99 -c ocl_setup.c

rnn/rnn_balancer.o: rnn/rnn_balancer.c rnn/rnn_balancer.h
	$(CC) $(CFLAGS) -std=c99 -c rnn/rnn_balancer.c -o rnn/rnn_balancer.o

