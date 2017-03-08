QT += core
QT -= gui
CONFIG += c++11
CONFIG += warn_off
TARGET = a.out
CONFIG += console
CONFIG -= app_bundle
TEMPLATE = app

SOURCES += main.cpp \
#SOURCES -= vecMean.cu

HEADERS += \
    HOST_GPU.h
# project build directories
DESTDIR     = $$system(pwd)
OBJECTS_DIR = $$DESTDIR/build

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CUDA NVCC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%% http://stackoverflow.com/questions/31261135/set-up-cuda-v7-0-in-qtcreator-vs-2010  %%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Cuda sources
CUDA_SOURCES += vecMean.cu
#%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Path to cuda toolkit install
CUDA_DIR      = /usr/local/cuda-8.0
# Path to header and libs files
INCLUDEPATH  += $$CUDA_DIR/include 

QMAKE_LIBDIR += $$CUDA_DIR/lib64     # Note I'm using a 64 bits Operating system
# libs used in your code
LIBS += -lcuda -lcudart #note that cudart should be linked via command: sudo ldconfig /usr/local/cuda-8.0/lib64/
# GPU architecture
CUDA_ARCH     = sm_35               
NVCCFLAGS     = -std=c++11
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')
cuda.commands = $$CUDA_DIR/bin/nvcc -Wno-deprecated-gpu-targets -m64 -O3 -arch=$$CUDA_ARCH \
                -c $$NVCCFLAGS \
                $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} \
                2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2

cuda.dependency_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -Wno-deprecated-gpu-targets -O3 -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}

cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o
# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_COMPILERS += cuda
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
