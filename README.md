FIT3162_Team5A_CodeDeliverables

# **Person Identification (Video Tagging)**

Explain aim of project

### **Prerequisites:**

1. Install Python 3.5.2 64-bit (https://www.python.org/downloads/release/python-352/)
2. Install Cmake 3.17.0 (https://cmake.org/)
3. Have a CUDA supported GPU
4. Download and install CUDA 9.0 toolkit (https://developer.nvidia.com/cuda-90-download-archive)
5. Download and install cuDNN 7.1.4  (https://developer.nvidia.com/cudnn)
6. Visual Studio 2019 with Visual C++ Tools for Cmake


### **Required python libraries:**

1. onnxruntime 1.2.0
2. onnx 1.6.0
3. sklearn 0.0
4. imutils 0.5.3
5. numpy 1.18.1
6. opencv-python 4.2.0.32
7. opencv-contrib-python 4.2.0.32
8. multiprocessing.dummy
 


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

### **How to run the program:**

Run Main_app.py by typing the following command in cmd:

"Python Main_app.py"


