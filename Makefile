CC    := g++
EXE    = hdr
FLAGS  = -O2 -llapacke -llapack -lblas -lm -Wall
OPENCV = -lopencv_highgui -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

all  : hdr.cpp
	$(CC) -o $(EXE) $(EXE).cpp $(FLAGS) $(OPENCV) $(OBJs)

.PHONY: clean run

clean:
	rm -f $(OBJs) $(EXE)

run:
	./hdr street street.hdr 11

