CXX = g++
CXXFLAGS = -O3 -std=c++11 -Wall

ENCODER = encoder.out
DECODER = decoder.out

SRC_EN = main.cpp encoder.cpp
SRC_DE = main2.cpp decoder.cpp encoder.cpp
SRC_PGM = pgm.cpp

OBJ_EN = $(SRC_EN:.cpp=.o)
OBJ_DE = $(SRC_DE:.cpp=.o)
OBJ_PGM = $(SRC_PGM:.cpp=.o)

all: $(DECODER) $(ENCODER)

$(ENCODER): $(OBJ_EN) $(OBJ_PGM)
	$(CXX) $(LDFLAGS) -o $@ $^

$(DECODER): $(OBJ_DE) $(OBJ_PGM)
	$(CXX) $(LDFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

encode: $(ENCODER)
	./$(ENCODER) 1080 720 list 20 results

decode: $(DECODER)
	./$(DECODER) 1080 720 list 20 results

clean:
	$(RM) $(OBJ_EN) $(OBJ_DE) $(OBJ_PGM) $(ENCODER) $(DECODER)