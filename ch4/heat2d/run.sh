#!/bin/bash
g++ -O2 *.cpp -o heat2d
if [ $? -eq 0 ]; then
	./heat2d
fi
