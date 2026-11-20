# Builds the reacting-flow solver, the pressure benchmark code and the MMS system generator.
#   make            -> bin/opposedflow  bin/opposedflow_rmta  bin/pbench  bin/pbench_rmta  bin/mms_gen
#   make CC=gcc-12  -> another compiler (GCC 12.3 was used on TRUBA)
CC       = gcc
CFLAGS  ?= -O3 -march=native -std=c11 -fopenmp
INC      = -Iinclude
LIBS     = -lm
BIN      = bin
HDRS     = include/rmt_impl.h include/rmta_impl.h

all: $(BIN)/opposedflow $(BIN)/opposedflow_rmta $(BIN)/pbench $(BIN)/pbench_rmta $(BIN)/mms_gen

$(BIN):
	mkdir -p $(BIN)

$(BIN)/opposedflow: solver/opposedflow.c $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) $(INC) $< $(LIBS) -o $@

$(BIN)/opposedflow_rmta: solver/opposedflow.c $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) $(INC) -DRMT_ANISO $< $(LIBS) -o $@

$(BIN)/pbench: pbench/pbench.c pbench/mms.h $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) $(INC) -Ipbench $< $(LIBS) -o $@

$(BIN)/pbench_rmta: pbench/pbench.c pbench/mms.h $(HDRS) | $(BIN)
	$(CC) $(CFLAGS) $(INC) -Ipbench -DRMT_ANISO $< $(LIBS) -o $@

$(BIN)/mms_gen: pbench/mms_gen.c pbench/mms.h | $(BIN)
	$(CC) -O2 -std=c11 -Ipbench $< $(LIBS) -o $@

clean:
	rm -rf $(BIN)

.PHONY: all clean
