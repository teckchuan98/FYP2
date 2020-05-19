FIT3162_Team5A_CodeDeliverables

# **Person Identification (Video Tagging)**

Explain aim of project

### **Prerequisites:**

1. Python 3.5.2 64-bit
2. Cmake 3.17.0 from https://cmake.org/
3. CUDA supported GPU 
4. CUDA 9.0 toolkit from https://developer.nvidia.com/cuda-90-download-archive?
5. cuDNN 7.1.4 from https://developer.nvidia.com/cudnn
6. Visual Studio 2019 with VisualC++ Tools for Cmake

### **Required python libraries:**

1. onnxruntime 1.2.0	
2. onnx 1.6.0
3. face-recognition 1.2.3	
4. sklearn 0.0
5. imutils 0.5.3	
6. numpy 1.18.1
7. dlib	19.19.99
8. opencv-python 4.2.0.32
9. opencv-contrib-python 4.2.0.32


### **How to install:**

Open command prompt and install the libraries like shown below:

onnxruntime: 'pip install onnxruntime'
onnx: 'pip install onnx'
sklearn: 'pip install sklearn'
imutils : 'pip install imutils '
numpy: 'pip install numpy'
opencv-python: 'pip install opencv-python'
opencv-contrib-python: 'pip install opencv-contrib-python'

Intall the dlib libraries like shown below for GPU support:

1. git clone https://github.com/davisking/dlib.git
2. cd dlib
3. mkdir build
4. cd build
5. cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
6. cmake --build .
7. cd ..
8. python setup.py install --yes USE_AVX_INSTRUCTIONS --yes DLIB_USE_CUDA

Since face_recognition is dependent on the dlib library, it should be installed after dlib hs been installed.



