To use this code you will need to have a NVIDIA graphics card and CUDA installed.

To compile and run you will need to install libconfig and gtest files.
To compile simply type make. To run create an output directory and type
./gpu-vh --config rhic-conf -o output_directory_you_created -h
The directory rhic-conf is where all of the input files are located.
All of the source files are located in the rhic/ directory.

To run in ideal hydro mode commment out the macros PIMUNU and PI in DynamicalVariables.cuh.
To perform the Riemann problems, set the code to run in Cartesian coordinated by uncommenting the macro flag in SourceTerms.cu.
The configuration files for the different test problems are located in rhic/rhic-trunk/src/test/resources.
There is a flag in EquationOfState.cuh that allows you to switch between an ideal and QCD EoS.
The flux limiter parameter can be changed based on smooth or fluctuationg initial conditions and is set in FluxLimiter.cu.
