# multilayer-cudamoto
GPU-accelerated Kuramoto model simulation on multilayer networks. With interactive simulation viewer.

This code includes two components:
* `cudamoto2` - CUDA accelerated simulation for two-layer multilayer networks. Including interdependent and competitive interactions as described in [1].
* `cudamoto2-viewer` - an interactive viewer written in Qt5 to display real-time simulation of `cudamoto2` and the ability to mess with the parameters and see the effects.


Requirements:  
NVIDIA graphics card with CUDA support (tested on GTX 970, GTX 1050, GTX1080, K40)  
CUDA (tested on CUDA 7.5 though CUDA 10.2)  
Qt5 (for the viewer)  


Build instructions:  
`mkdir cudamoto2/build`  
`mkdir cudamoto2-viewer/build`  
`bash build_all.sh`

And then run:  
`./cudamoto2-viewer/build/CudamotoViewer`  


With the viewer you can run simulations like this, where you mix and match interactions, forced syncrhonizations, noise inputs, and coupling between the layers. For details on the theory behind these simulations, see the paper [1].

![Example](http://www.mmdanziger.com/files/cudamoto2-viewer-spatial-competitive.png)

If you use this code, please cite the paper:

[1]
M.M. Danziger, I. Bonamassa, S. Boccaletti and S. Havlin  
Dynamic interdependence and competition in multilayer networks. 
Nature Physics **15**, 178–185 (2019) [doi:10.1038/s41567-018-0343-1](https://doi.org/10.1038/s41567-018-0343-1)  
