#ifndef MPI_ASSERT_H
#define MPI_ASSERT_H

#include <stdio.h>
#include <mpi.h>

// an assert that tears down the whole job
#undef assert
#define assert(x) \
 do { if (!(x)) { \
 fprintf(stderr, "[MPI] Assertion failed: %s (%s:%d)\n", #x, __FILE__, __LINE__); \
 MPI_Abort(MPI_COMM_WORLD, 1); \
 } } while (0)

#endif // MPI_ASSERT_H
