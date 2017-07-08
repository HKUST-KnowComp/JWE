ifeq ($(shell uname -s), Darwin)
	CC = clang
else
	CC = gcc
	CFLAGS += -pthread -lm
endif

CFLAGS += -O2 -std=c99 

all: jwe.c word_analogy.c
	${CC} jwe.c ${CFLAGS} -o jwe
	${CC} word_analogy.c ${CFLAGS} -o word_analogy
clean: 
	rm jwe word_analogy
