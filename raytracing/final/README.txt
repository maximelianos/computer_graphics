В качестве шаблона использовались следующие источники:

https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-overview/light-transport-ray-tracing-whitted

https://github.com/RayTracing/


Before compiling, copy dir 'letters', files
panorama.bmp panorama2.bmp portrait.bmp
to this folder.

To compile:
mkdir build
cd build
cmake ..
make -j 4
./rt -out hellokitty.bmp -scene 1 -threads 2
