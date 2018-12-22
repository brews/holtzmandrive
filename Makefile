CC = gcc
RM = rm -f
LIBS = -lm -fopenmp -lblas

default: all

all: ensrf

ensrf: ensrf.c
	$(CC) -o ensrf ensrf.c $(LIBS)

clean veryclean:
	$(RM) ensrf
