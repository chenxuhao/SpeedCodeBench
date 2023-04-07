In this program, the merge() function merges two sorted subarrays,
and the mergeSort() function recursively divides the array into two halves until it reaches the base case of a single element, then merge the two halves.

The #pragma omp parallel sections directive is used to create parallel sections that can be executed concurrently. 
This directive splits the parallel region into multiple sections, and each section is executed by a separate thread.

The two recursive calls to mergeSort() are made inside the #pragma omp section directives, which ensures that each call runs in a separate thread.

Note that this program assumes that the input array is of type int. You can modify it to work with other data types if needed.
