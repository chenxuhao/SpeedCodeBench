[Prime](view-source:https://people.sc.fsu.edu/~jburkardt/c_src/prime_openmp/prime_openmp.html)
counts the number of primes between 1 and N, using OpenMP to carry out the calculation in parallel.

The algorithm is completely naive. For each integer I, it simply checks whether any smaller J evenly divides it.
The total amount of work for a given N is thus roughly proportional to 1/2*N^2.

This program is mainly a starting point for investigations into parallelization.

Here are the counts of the number of primes for some selected values of N:

<p>
<table border="1" align="center">
<tr>
<th>N</th><th>Pi(N), Number of Primes</th>
</tr>
<tr><td>         1</td><td>         0</td></tr>
<tr><td>         2</td><td>         1</td></tr>
<tr><td>         4 </td><td>        2</td></tr>
<tr><td>         8</td><td>         4</td></tr>
<tr><td>        16</td><td>         6</td></tr>
<tr><td>        32</td><td>        11</td></tr>
<tr><td>        64</td><td>        18</td></tr>
<tr><td>       128</td><td>        31</td></tr>
<tr><td>       256</td><td>        54</td></tr>
<tr><td>       512</td><td>        97</td></tr>
<tr><td>      1024</td><td>       172</td></tr>
<tr><td>      2048</td><td>       309</td></tr>
<tr><td>      4096</td><td>       564</td></tr>
<tr><td>      8192</td><td>      1028</td></tr>
<tr><td>     16384</td><td>      1900</td></tr>
<tr><td>     32768</td><td>      3512 </td></tr>
<tr><td>     65536</td><td>      6542</td></tr>
<tr><td>    131072</td><td>     12251</td></tr>

</table> 
</p>
