mkdir -p build
rsync -av --update assets build/
cd build
cmake .. && make
