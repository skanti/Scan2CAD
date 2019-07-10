all : ExtendBox.so

ExtendBox.so : ExtendBox.cpp
	c++ -O3 -Wall -shared -std=c++14 -fPIC `python3 -m pybind11 --includes` $^ -o $@

clean:
	rm ExtendBox.so
