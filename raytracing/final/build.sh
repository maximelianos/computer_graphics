cp -a ../letters .
cp -a ../*.bmp .
g++ ../base.cpp -std=c++11 -lpthread -O2 -o rt
