CXX = g++
CXXFLAGS = -std=c++17 -O2 `pkg-config --cflags opencv4`

LDFLAGS = `pkg-config --libs opencv4`

TARGET = program

SRCS = main.cpp algorithm.cpp metrics.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

run: $(TARGET)
	./$(TARGET)