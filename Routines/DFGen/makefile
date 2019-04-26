all : main

main : main.cpp makelevelset3.cpp
	c++ -I../Base -std=c++14 -O3 -Wall $^ -o $@

clean:
	rm main
