#GSLROOT=/opt/apps/gsl/1.9
#GSLROOT=/usr
# use this if on 64-bit machine with 64-bit GSL libraries
# ARCH=x86_64
# use this if on 32-bit machine with 32-bit GSL libraries
# ARCH=i386
MPICC=mpicc
CC=gcc
CFLAGS=-Wall -std=c99 -I/usr/include
LDFLAGS=-L/usr/lib -lgsl -lgslcblas -lm

all: pGRockLasso

pGRockLasso: pGRockLasso.o mmio.o
	$(MPICC) $(CFLAGS) pGRockLasso.o mmio.o -o pGRockLasso $(LDFLAGS)

# gam: gam.o mmio.o
#	$(MPICC) $(CFLAGS) $(LDFLAGS) gam.o mmio.o -o gam

pGRockLasso.o: pGRockLasso.c mmio.o
	$(MPICC) $(CFLAGS)  -c pGRockLasso.c

# gam.o: gam.c mmio.o
#	$(MPICC) $(CFLAGS) -c gam.c

mmio.o: mmio.c
	$(CC) $(CFLAGS) -c mmio.c

clean:
	rm -vf *.o pGRockLasso
run:
	mpiexec -n 1 ./pGRockLasso 1