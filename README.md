# multilayer-cudamoto
GPU-accelerated Kuramoto model simulation on multilayer networks. With interactive simulation viewer.

This code includes two components:
* `cudamoto2` - CUDA accelerated simulation for two-layer multilayer networks. Including interdependent and competitive interactions as described in [1].
* `cudamoto-viewer` - an interactive viewer written in Qt5 to display real-time simulation of `cudamoto2` and the ability to mess with the parameters and see the effects.
