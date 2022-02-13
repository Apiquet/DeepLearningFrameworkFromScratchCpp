PROG = tests/main
CC = g++
CPPFLAGS = -std=c++20

INCS = $(foreach sdir, include, $(wildcard $(sdir)/*)) src/Trainers

INC_DIRS = $(addprefix -I, $(INCS))

SRCS = $(foreach sdir, src, $(wildcard $(sdir)/*.cpp)) $(foreach sdir, src/*, $(wildcard $(sdir)/*.cpp))

print-% : ; @echo $* = $($*)

main.o :
	$(CC) $(CPPFLAGS) $(PROG).cpp $(SRCS) -Iinclude $(INC_DIRS) -o $(PROG)
