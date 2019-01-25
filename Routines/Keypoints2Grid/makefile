all : Keypoints2Grid.so

Keypoints2Grid.so : Keypoints2Grid.cpp
	c++ -I../Base -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` $^ -o $@

clean:
	rm Keypoints2Grid.so
