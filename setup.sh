#!/bin/bash

# Exit on error
set -e

ONNX_VERSION="1.13.1"
OPENCV_VERSION="4.x"
# Not used; brew installs the latest version
#LLVM_VERSION="19.1.1"
GCC_VERSION="9"

ONNX_BASE_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}"
OPENCV_BASE_URL="https://github.com/opencv/opencv/archive"

install_linux_dependencies()
{	
	echo "Installing dependencies for Linux..."
	sudo apt update
	sudo apt install -y cmake wget unzip gcc-${GCC_VERSION} g++-${GCC_VERSION}
	
	# Install deps for cvNamedWindow
	sudo apt install -y libgtk2.0-dev pkg-config
	
	# Set compilers for building opencv from source
	export CC="/usr/bin/gcc-${GCC_VERSION}"
    export CXX="/usr/bin/g++-${GCC_VERSION}"
	
	OS="linux"
	ARCH="x64"
}

install_macos_dependencies()
{
	echo "Installing dependencies for macOS..."
	brew update
	brew install cmake wget llvm

	# Install deps for cvNamedWindow
	#brew install pkg-config # TODO: NOT sure if needed!
	
	# Dynamically set LLVM paths using brew --prefix
    LLVM_PATH=$(brew --prefix llvm)
	
	# Set compilers for building opencv from source
	export CC="${LLVM_PATH}/bin/clang"
    export CXX="${LLVM_PATH}/bin/clang++"
	export LDFLAGS="-L${LLVM_PATH}/lib"
    export CPPFLAGS="-I${LLVM_PATH}/include"

	OS="osx"
	ARCH="arm64"
}

# Download and extract ONNX runtime
# Arguments: OS, ARCH, ONNX_VERSION
download_onnxruntime()
{
	local OS=$1
	local ARCH=$2
	local VERSION=$3
	
	ONNX_RELEASE_FILE="onnxruntime-${OS}-${ARCH}-${VERSION}.tgz"
	
	# Check if ONNX Runtime directory already exists in thirdparty
	if [[ -d "onnxruntime" ]]; then
    	echo "ONNX Runtime already exists in thirdparty, skipping download."
    	return 0
	fi
	
	echo "Downloading ONNX Runtime for ${OS} ${ARCH}..."
	wget "${ONNX_BASE_URL}/${ONNX_RELEASE_FILE}" -O onnxruntime.tgz
	echo "Extracting ONNX Runtime..."
	tar -xvf onnxruntime.tgz
	mv onnxruntime-* onnxruntime
	rm -rf onnxruntime.tgz
}

# Download and build OpenCV from source
# Arguments: OPENCV_VERSION 
build_opencv()
{	
	# Check if opencv directory already exists in thirdparty
	if [[ -d "opencv-${OPENCV_VERSION}" ]]; then
    	echo "OpenCV already exists in thirdparty, skipping download."
	else		
		echo "Downloading OpenCV..."
		wget "${OPENCV_BASE_URL}/${OPENCV_VERSION}.zip" -O opencv.zip
		unzip opencv.zip
		rm -rf opencv.zip
	fi

	echo "Building OpenCV..."
	mkdir -p build && cd build
	# Cannot link for arm64 with gapi for MacOS; disable it as we dont use it
	cmake ../"opencv-${OPENCV_VERSION}" -D BUILD_opencv_gapi=OFF
	cmake --build .	
}

# Install opencv
install_opencv()
{
	echo "Installing OpenCV..."
	sudo make install
}

main()
{

	# Install dependencies per OS
	if [[ "$OSTYPE" == "linux-gnu"* ]]; then
		install_linux_dependencies
		
	elif [[ "$OSTYPE" == "darwin"* ]]; then
		install_macos_dependencies
	else
		echo "Please check dependencies listed in README.md and consult the appropriate documentation for your OS"
		exit 1
	fi


	# Install thirdparty dependencies
	mkdir -p thirdparty
	cd thirdparty

	download_onnxruntime "$OS" "$ARCH" "$ONNX_VERSION"

	build_opencv "$OPENCV_VERSION"
	install_opencv

	echo "Installation complete!"
}


#################################### Main entry point ####################################
# Call specific function based on the argument passed to the script
if [[ "$1" == "install_linux_dependencies" ]]; then
    install_linux_dependencies
elif [[ "$1" == "install_macos_dependencies" ]]; then
    install_macos_dependencies
elif [[ "$1" == "download_onnxruntime" ]]; then
	cd thirdparty
    download_onnxruntime "$OS" "$ARCH" "$ONNX_VERSION"
elif [[ "$1" == "build_opencv" ]]; then
	cd thirdparty
    build_opencv
elif [[ "$1" == "install_opencv" ]]; then
	cd thirdparty/build
    install_opencv
else
    # Default to calling the main function if no specific argument is passed
    main
fi
