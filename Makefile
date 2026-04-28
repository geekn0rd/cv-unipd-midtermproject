CXX = g++
CXXFLAGS = -std=c++17 -O2 `pkg-config --cflags opencv4` -Iinclude
LDFLAGS = `pkg-config --libs opencv4`

TARGET = build/program

SRCS = src/main.cpp \
       src/amirali.cpp \
       src/metrics.cpp \
       src/rilke.cpp

OBJS = $(SRCS:src/%.cpp=build/%.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS)

build/%.o: src/%.cpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf build

run: $(TARGET)
	./$(TARGET)