# Install thirdparty dependencies
# Install OpenCV and its dependencies
# Install minimal prerequisites
sudo apt update && sudo apt install -y cmake g++ wget unzip
# Install deps for cvNamedWindow
sudo apt install -y libgtk2.0-dev pkg-config
# Download and unpack sources for onnxruntime and opencv
cd thirdparty
wget https://github.com/microsoft/onnxruntime/releases/download/v1.13.1/onnxruntime-linux-x64-1.13.1.tgz && \
tar -xvf onnxruntime-linux-x64-1.13.1.tgz && mv onnxruntime-linux-x64-1.13.1 onnxruntime && \
rm -rf onnxruntime-linux-x64-1.13.1.tgz
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
unzip opencv.zip
# Create build directory
mkdir -p build && cd build
# Configure
cmake  ../opencv-4.x
# Build
cmake --build .
# Install
sudo make install
# Install gcc-9 and g++-9
sudo apt install gcc-9 g++-9