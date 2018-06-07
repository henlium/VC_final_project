CXX = g++
CXXFLAGS = -O3 -std=c++11 -Wall

TARGET = main.out
SRC = main.cpp pgm.cpp encoder.cpp
OBJ = $(SRC:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(LDFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

run: $(TARGET)
	./$(TARGET) 1080 720 list 20 results

clean:
	$(RM) $(OBJ) $(TARGET)