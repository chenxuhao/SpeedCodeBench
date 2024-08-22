# SSSP

Single-source Shortest Paths (SSSP)

Returns array of distances for all vertices from a given source vertex `v` in a graph `G=(V,E)`

# Single Source Shortest Path

## Description

The Single-Source Shortest Path (SSSP) problem consists of finding the shortest paths between a given vertex v and all other vertices in the graph. 
Algorithms such as Breadth-First-Search (BFS) for unweighted graphs or Dijkstra solve this problem.

For a weighted directed graph, the shortest path problem finds the path with the lowest total weight between two vertices.

## Input

* The input graph `G=(V,E)` is a weighted directed graph.

* The sourse vertex id `v`.

## Output

* The distances between `v` and all the vertices in the graph.

## Example

<p align="center">
  <img src="sssp.jpg" />
</p>

## Constraints

* Your code must be written in C/C++.

## Explanation 

* You are given the input graph as an object of the class `Graph g`, 
  here is the API to use this class:

  - `g.V()` returns the number of nodes in the graph
  - `g.E()` returns the number of edges in the graph
  - `g.N(v)` returns the neighbor set (in the type of `VertexSet`) of node `v` in the graph
  - `vidType` is the data type of the node ID in the graph

* You can use the public member functions of the `VertexSet` class:

  - `size()` returns the size of the `VertexSet`, i.e., the number of nodes in the `VertexSet`.
  - `operator[]`: `[i]` returns a reference to the element at position `i`.
  - `begin()` returns an iterator pointing to the first element in the `VertexSet`.
  - `end()` returns an iterator referring to the past-the-end element in the `VertexSet` container.
  - `data()` returns a direct pointer to the memory array used internally by the `VertexSet` to store its owned elements.

* Please contact the author for further explanation.

* Author: Xuhao Chen <cxh@mit.edu>

## Hints

* Parallel algorithm Bellman-Ford

* Use delta stepping instead of Bellman-Ford

## More Details

Two parallel algorithms are often used for SSSP: 
(1) Bellman Ford;
(2) Delta Stepping [1].

This SSSP implementation makes use of the δ-stepping algorithm [1].
The type used for weights and distances (WeightT) is typedefined in benchmark.h. 
The delta parameter (-d) should be set for each input graph.

The bins of width delta are actually all thread-local and of type std::vector
so they can grow but are otherwise capacity-proportional. Each iteration is
done in two phases separated by barriers. In the first phase, the current
shared bin is processed by all threads. As they find vertices whose distance
they are able to improve, they add them to their thread-local bins. During this
phase, each thread also votes on what the next bin should be (smallest
non-empty bin). In the next phase, each thread copies their selected
thread-local bin into the shared bin.

Once a vertex is added to a bin, it is not removed, even if its distance is
later updated and it now appears in a lower bin. We find ignoring vertices if
their current distance is less than the min distance for the bin to remove enough 
redundant work that this is faster than removing the vertex from older bins.

[1] Ulrich Meyer and Peter Sanders. "δ-stepping: a parallelizable shortest
	path algorithm." Journal of Algorithms, 49(1):114--152, 2003.

[2] A. Davidson, S. Baxter, M. Garland, and J. D. Owens, "Work-efficient
	parallel gpu methods for single-source shortest paths", in Proceedings
	of the IEEE 28th International Parallel and Distributed Processing
	Symposium (IPDPS), pp. 349--359, May 2014

## Reference Solutions

* sssp_omp_base: OpenMP implementation using Bellman-Ford algorithm, one thread per vertex
* sssp_omp_dstep: OpenMP implementation using delta-stepping algorithm, one thread per vertex
* sssp_cilk_base: OpenCilk implementation using Bellman-Ford algorithm, one thread per vertex
* sssp_cilk_dstep: OpenCilk implementation using delta-stepping algorithm, one thread per vertex
* sssp_gpu_base: data-driven GPU implementation using Bellman-Ford algorithm, one thread per vertex
* sssp_gpu_dstep: data-driven GPU using delta stepping algorithm, one thread per edge
