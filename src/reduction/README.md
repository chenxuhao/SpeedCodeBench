In this program, we first allocate an array of integers of size SIZE, and initialize it with random values. 
Then, we use OpenMP to parallelize the sum operation over the array, using the #pragma omp parallel for reduction(+: sum) directive. 
This directive instructs OpenMP to distribute the iterations of the loop across multiple threads, 
and to accumulate the results of each thread into the variable sum, using the + operator.

For maximum finding, inside the loop, 
we use an if statement to check if the current number is greater than the current maximum value. 
If it is, we update the maximum value. 
Finally, we output the maximum value.
Note that we use the reduction clause with the max operator to combine the individual maximum values found by each thread into a single maximum value.
This ensures that the final maximum value is correct, even when multiple threads are working on the problem concurrently.

Finally, we print the result of the reduction operation, and free the memory allocated for the array.

