#!/bin/bash

echo "Making links to src in ./links/src"

if [ ! -d "./links/src" ]; then
  mkdir -p "./links/src"
fi

find $PWD/../rhic/ -name '*.cpp' -exec ln -f -s -t ./links/src/ {} \;
find $PWD/../rhic/ -name '*.cu' -exec ln -f -s -t ./links/src/ {} \;

echo "Making links to include in ./links/include"

if [ ! -d "./links/include" ]; then
  mkdir -p "./links/include"
fi

find $PWD/../rhic/ -name '*.h' -exec ln -f -s -t ./links/include/ {} \;
find $PWD/../rhic/ -name '*.cuh' -exec ln -f -s -t ./links/include/ {} \;
