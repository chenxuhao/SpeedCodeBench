#pragma once

#include <stdint.h>
#include "VertexSet.h"

typedef int64_t eidType;   // edge ID type

class BaseGraph {
 protected:
  eidType *vertices;            // row pointers of CSR format
  vidType *edges;               // column indices of CSR format
  vidType n_vertices;           // number of vertices
  eidType n_edges;              // number of edges

  bool is_directed_;            // is it a directed graph?
  int vid_size, eid_size;       // number of bytes for vid, eid
  vidType max_degree;           // maximun degree
  vidType *reverse_edges;       // reverse column indices of CSR format
  eidType *reverse_vertices;    // reverse row pointers of CSR format
 
  void load(std::string prefix) {
    std::ifstream f_meta((prefix + ".meta.txt").c_str());
    assert(f_meta);
    f_meta >> n_vertices >> n_edges >> vid_size >> eid_size;
    f_meta.close();
    assert(sizeof(vidType) == vid_size);
    assert(sizeof(eidType) == eid_size);
    assert(n_vertices > 0 && n_edges > 0);
    if (vid_size == 4) assert(n_vertices < 4294967295);
    std::cout << "Reading graph: |V| " << n_vertices << " |E| " << n_edges << "\n";
    // read row pointers
    read_file(prefix + ".vertex.bin", vertices, n_vertices+1);
    // read column indices
    read_file(prefix + ".edge.bin", edges, n_edges);
  }

 public:
  BaseGraph(eidType* rowptr, vidType* colidx, vidType nv, eidType ne, bool directed) :
    vertices(rowptr), edges(colidx), n_vertices(nv), n_edges(ne), 
    is_directed_(directed), vid_size(4), eid_size(4), max_degree(0),
    reverse_edges(NULL), reverse_vertices(NULL) {}
  BaseGraph(std::string prefix) : BaseGraph(NULL, NULL, 0, 0, 0) { load(prefix); }
  ~BaseGraph() {
    if (edges != NULL) {
      custom_free(edges, n_edges);
      edges = NULL;
    }
    if (vertices != NULL) {
      custom_free(vertices, n_vertices+1);
      vertices = NULL;
    }
  }

  vidType V() const { return n_vertices; }
  eidType E() const { return n_edges; }
  vidType num_vertices() const { return n_vertices; }
  eidType num_edges() const { return n_edges; }
  vidType get_max_degree() const { return max_degree; }
  bool is_directed() const { return is_directed_; }
  bool has_reverse_graph() const { return (reverse_edges != NULL && reverse_vertices != NULL); }
  vidType get_degree(vidType v) const { return vertices[v+1] - vertices[v]; }
  eidType edge_begin(vidType v) const { return vertices[v]; }
  eidType edge_end(vidType v) const { return vertices[v+1]; }
  vidType* adj_ptr(vidType v) const { return &edges[vertices[v]]; }
  vidType* get_adj(vidType v) const { return &edges[vertices[v]]; }
  eidType* rowptr() { return vertices; }             // get row pointers array
  vidType* colidx() { return edges; }                // get column indices array
  const eidType* rowptr() const { return vertices; } // get row pointers array
  const vidType* colidx() const { return edges; }    // get column indices array
  eidType* in_rowptr() { return reverse_vertices; }  // get incoming row pointers array
  vidType* in_colidx() { return reverse_edges; }     // get incoming column indices array
  void orientation(std::string outfile = "");        // edge orientation: convert the graph from undirected to directed
  VertexSet N(vidType v) const;                      // get the neighbor list of vertex v
  VertexSet in_neigh(vidType v) const;               // get the ingoing neighbor list of vertex v
  VertexSet out_neigh(vidType v, vidType off = 0) const; // get the outgoing neighbor list of vertex v
  void build_reverse_graph();
};

template <typename T=vidType>
T set_intersect(T nA, T nB, const T* A, const T* B) {
    T count = 0;
    auto A_ptr = &A[0];
    auto B_ptr = &B[0];
    auto A_end = A_ptr + nA;
    auto B_end = B_ptr + nB;
    while (A_ptr < A_end && B_ptr < B_end) {
        if (*A_ptr < *B_ptr) A_ptr++;
        else if (*A_ptr > *B_ptr) B_ptr++;
        else { count ++; A_ptr++; B_ptr++; }
    }
    return count;
}

