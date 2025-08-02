# Compiler: Use the MPI C++ wrapper
CXX = mpic++

# Compiler flags: Enable warnings and debugging symbols
CXXFLAGS = -g -Wall

# The name of the final executable
TARGET = random_walk

# All C++ source files
SRCS = random_walk.cpp

# Default target to build the executable
all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

# Command to clean up compiled files
clean:
	rm -f $(TARGET)