PROG = tests/main
CC = g++
CPPFLAGS = -Wall

INCS = $(wildcard include/*) src/Trainers

INC_DIRS = $(addprefix -I, $(INCS))

SRCS = $(foreach sdir, src/*, $(wildcard $(sdir)/*.cpp))

print-% : ; @echo $* = $($*)

main.o :
	$(CC) $(CPPFLAGS) $(PROG).cpp $(SRCS) -Iinclude $(INC_DIRS) -o $(PROG)
