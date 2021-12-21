# Position Based Fluids
This repo implementes both CPU and CUDA version of the Position Based Fluids of <a href="https://mmacklin.com/pbf_sig_preprint.pdf">position=based-fluid.pdf</a>.

## Build and Run the Project
For CPU implementation, clone this repo with,

```
git clone --recursive https://github.com/Willqie/CSC417Project.git
```
Then build and run the project with,
```
cd CSC417Project
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
./pbf
```

CUDA implementation is on branch `gpu`,
```
git checkout gpu
```
Then build and run the project with,
```
mkdir build
cd build
cmake ..
make
./pbf
```
The CUDA implementation was only tested on the following configuration:
```
Cuda compilation tools, release 10.1, V10.1.243
Driver Version: 470.86, CUDA version: 11.4
```

## Video
TODO

## Report
See <a href="https://github.com/Willqie/CSC417Project/blob/master/report/report.pdf">paper.pdf</a>
