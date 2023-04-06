Name: raytrace

Description

Raytracing is a technique that generates a visually realistic image by tracing
the path of light through a scene. It leverages the physical property that the
path of light is always reversible to reduce the computational requirements by
following the light rays from the eye point through the image plane to the
source of the light. This way only light rays that contribute to the image are
considered.

The raytrace benchmark program uses a variety of the raytracing method that
would typically be employed for real-time animations such as computer games.
It is optimized for speed rather than realism. The computational complexity
of the algorithm depends on the resolution of the output image and the scene.

=======================================
Input/Output:

The input for raytrace is a data file describing a scene that is composed of
a single, complex object. The program automatically rotates the camera around
the object to simulate movement. The output is a video stream that is displayed
in a video. For the benchmark version output has been disabled.


This program generates a 2D image of a sphere using ray tracing. 
The image is stored as a 2D array of doubles, 
where each element represents the distance from the camera to the point of intersection between the ray and the sphere. 
The program uses the OpenMP parallel for directive to parallelize the loops that iterate over the image pixels.
The schedule(dynamic) clause tells OpenMP to assign iterations to threads dynamically as they become available, 
which can help balance the workload across threads.
