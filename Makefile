CC = mpicc
CCFLAGS = -03
CCGFLAGS = -g
LIBS = -lmpi -lm

BINS = code

all: $(BINS)

code: code.c
	$(CC) $(CCGFLAGS) -o $@ $^ $(LIBS)

clean:
	$(RM) $(BINS)
