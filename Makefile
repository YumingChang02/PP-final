CC    := g++
EXE    = hdr
FLAGS  = -std=c++11 -O2 -llapacke -llapack -lm -Wall
OPENCV = -lopencv_highgui -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

all  : hdr.cpp
	$(CC) -o $(EXE) $(EXE).cpp $(FLAGS) $(OPENCV) $(OBJs)

.PHONY: clean run

clean:
	rm -f $(EXE) $(wildcard *.hdr)
