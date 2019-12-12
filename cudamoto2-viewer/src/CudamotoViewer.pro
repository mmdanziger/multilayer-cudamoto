#-------------------------------------------------
#
# Project created by QtCreator 2016-02-28T18:45:16
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = CudamotoViewer
TEMPLATE = app
CONFIG += c++11

QMAKE_CC = gcc-6
QMAKE_CXX = g++-6

QMAKE_CXXFLAGS += -std=c++11

SOURCES += main.cpp\
        cudamotoviewer.cpp
INCLUDEPATH += /usr/local/cuda/include/\
/usr/local/cuda/samples/common/inc
# C++ flags
QMAKE_CXXFLAGS_RELEASE =-O3


HEADERS  += cudamotoviewer.h

#OBJECTS += cudamoto2.o

FORMS    += cudamotoviewer.ui

unix:!macx: LIBS += -L$$PWD/ -lCudamoto2

INCLUDEPATH += $$PWD/
DEPENDPATH += $$PWD/

unix:!macx: PRE_TARGETDEPS += $$PWD/libCudamoto2.so

unix:!macx: LIBS += -L/usr/local/cuda/lib64 -lcudart
