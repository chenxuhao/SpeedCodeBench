#pragma once
#include "BaseGraph.hh"
#include "timer.h"

using namespace std;

template <typename T_>
class RangeIter {
  T_ x_;
 public:
  explicit RangeIter(T_ x) : x_(x) {}
  bool operator!=(RangeIter const& other) const { return x_ != other.x_; }
  T_ const& operator*() const { return x_; }
  RangeIter& operator++() {
    ++x_;
    return *this;
  }
};

template <typename T_>
class Range{
  T_ from_;
  T_ to_;
 public:
  explicit Range(T_ to) : from_(0), to_(to) {}
  Range(T_ from, T_ to) : from_(from), to_(to) {}
  RangeIter<T_> begin() const { return RangeIter<T_>(from_); }
  RangeIter<T_> end() const { return RangeIter<T_>(to_); }
};

class Graph : public BaseGraph {
protected:
  bool is_bipartite_;           // is it a bipartite graph?
  vidType n_vert0;              // number of type 0 vertices for bipartite graph
  vidType n_vert1;              // number of type 1 vertices for bipartite graph
  int vlabel_size, elabel_size; // number of bytes for vlabel, elabel
  int feat_len;                 // vertex feature vector length: '0' means no features
  int num_vertex_classes;       // number of distinct vertex labels: '0' means no vertex labels
  int num_edge_classes;         // number of distinct edge labels: '0' means no edge labels
 
  std::string name_;            // name of the graph
  std::string inputfile_path;   // input file path of the graph
  std::string inputfile_prefix; // input file prefix of the graph
  bool has_reverse;             // has reverse/incoming edges maintained
  eidType nnz;                  // number of edges in COO format (may be halved due to orientation)
  vidType max_label_frequency_; // maximum label frequency
  int max_label;                // maximum label
  int core_length_;

  vlabel_t *vlabels;            // vertex labels
  elabel_t *elabels;            // edge labels
  feat_t *features;             // vertex features; one feature vector per vertex
  vidType *src_list, *dst_list; // source and destination vertices of COO format
  std::vector<int> core_table;  // coreness for each vertex
  VertexList labels_frequency_; // vertex count of each label
  VertexList sizes;             // neighbor count of each source vertex in the edgelist
  VertexList reverse_index_;    // indices to vertices grouped by vertex label
  std::vector<eidType> reverse_index_offsets_; // pointers to each vertex group
  std::vector<vidType> degrees;

public:
  Graph(std::string prefix,
         bool use_dag = false,
         bool use_vlabel = false, bool use_elabel = false, 
         bool need_reverse = false, bool bipartite = false, bool partitioned = false);
  Graph(bool directed, bool bipartite, vidType nv, eidType ne) :
         BaseGraph(NULL, NULL, nv, ne, directed),
         is_bipartite_(bipartite),
         n_vert0(0), n_vert1(0),
         vlabel_size(0), elabel_size(0),
         feat_len(0), 
         num_vertex_classes(0),
         num_edge_classes(0),
         name_(""),
         inputfile_path(""),
         inputfile_prefix(""),
         has_reverse(0),
         nnz(0), 
         max_label_frequency_(0),
         max_label(0),
         core_length_(0),
         reverse_edges(NULL),
         reverse_vertices(NULL),
         vlabels(NULL),
         elabels(NULL),
         features(NULL),
         src_list(NULL), dst_list(NULL) { }
  Graph(vidType nv, eidType ne) : Graph(0, 0, nv, ne) { allocateFrom(nv, ne); }
  Graph() : Graph(0, 0, 0, 0) { }
  ~Graph();
  Graph(const Graph &)=delete;
  Graph& operator=(const Graph &)=delete;

  void load_graph(std::string prefix, 
                  bool use_dag = false,
                  bool use_vlabel = false, 
                  bool use_elabel = false,
                  bool need_reverse = false, 
                  bool partitioned = false);
  void load_graph_data(std::string prefix, 
                       bool use_dag = false,
                       bool use_vlabel = false, 
                       bool use_elabel = false,
                       bool need_reverse = false);
  void deallocate();
  void load_row_pointers(std::string prefix);
  void load_edge_labels(std::string prefix);

  // get methods for graph meta information
  vidType VB(int type) const { if (type == 0) return n_vert0; else return n_vert1; }
  Range<vidType> Vertices() const { return Range<vidType>(num_vertices()); }
  eidType get_num_tasks() const { return nnz; }
  vidType num_vertices() const { return n_vertices; }
  eidType num_edges() const { return n_edges; }
  std::string get_name() const { return name_; }
  std::string get_inputfile_path() const { return inputfile_path; }
  std::string get_inputfile_prefix() const { return inputfile_prefix; }
  bool is_bipartite() const { return is_bipartite_; }
  bool has_reverse_graph() const { return has_reverse; }
  vidType get_max_degree() const { return max_degree; }

  // get methods for graph topology information
  vidType out_degree(vidType v) const { return vertices[v+1] - vertices[v]; }
  vidType N(vidType v, vidType n) const { return edges[vertices[v]+n];} // get the n-th neighbor of v
  eidType get_eid(vidType v, vidType n) const { return vertices[v]+n;}  // get the edge id of the n-th edge of v
  eidType* out_rowptr() { return vertices; }         // get row pointers array
  vidType* out_colidx() { return edges; }            // get column indices array
  eidType* in_rowptr() { return reverse_vertices; }  // get incoming row pointers array
  vidType* in_colidx() { return reverse_edges; }     // get incoming column indices array
  bool is_neighbor(vidType v, vidType u) const;      // is vertex u an out-going neighbor of vertex v
  bool is_connected(vidType v, vidType u) const;     // is vertex v and u connected by an edge
  bool is_connected(std::vector<vidType> sg) const;  // is the subgraph sg a connected one
  VertexSet out_neigh(vidType v, vidType off = 0) const; // get the outgoing neighbor list of vertex v

  // Galois compatible APIs
  vidType size() const { return n_vertices; }
  eidType sizeEdges() const { return n_edges; }
  vidType getEdgeDst(eidType e) const { return edges[e]; } // get target vertex of the edge e
  vlabel_t getData(vidType v) const { return vlabels[v]; }
  vlabel_t getVertexData(vidType v) const { return vlabels[v]; }
  elabel_t getEdgeData(eidType e) const { return elabels[e]; }
  void fixEndEdge(vidType vid, eidType row_end) { vertices[vid + 1] = row_end; }
  void allocateFrom(vidType nv, eidType ne);
  void constructEdge(eidType eid, vidType dst) { edges[eid] = dst; }

  // get methods for labels and coreness
  vlabel_t get_vlabel(vidType v) const { return vlabels[v]; }
  elabel_t get_elabel(eidType e) const { return elabels[e]; }
  elabel_t get_elabel(vidType v, vidType n) const { return elabels[vertices[v]+n]; } // get the label of the n-th edge of v
  int get_vertex_classes() { return num_vertex_classes; } // number of distinct vertex labels
  int get_edge_classes() { return num_edge_classes; } // number of distinct edge labels
  int get_frequent_labels(int threshold);
  int get_max_label() { return max_label; }
  vlabel_t* getVlabelPtr() { return vlabels; }
  elabel_t* getElabelPtr() { return elabels; }
  vlabel_t* get_vlabel_ptr() { return vlabels; }
  elabel_t* get_elabel_ptr() { return elabels; }
  bool has_label() { return vlabels != NULL || elabels != NULL; }
  bool has_vlabel() { return vlabels != NULL; }
  bool has_elabel() { return elabels != NULL; }

  // edgelist or COO
  vidType* get_src_ptr() { return &src_list[0]; }
  vidType* get_dst_ptr() { return &dst_list[0]; }
  vidType get_src(eidType eid) { return src_list[eid]; }
  vidType get_dst(eidType eid) { return dst_list[eid]; }
  std::vector<vidType> get_sizes() const { return sizes; }
  eidType init_edgelist(bool sym_break = false, bool ascend = false);

  void sort_neighbors(); // sort the neighbor lists
  void sort_and_clean_neighbors(std::string outfile = ""); // sort the neighbor lists and remove selfloops and redundant edges
  void symmetrize(); // symmetrize a directed graph
  void write_to_file(std::string outfilename, bool v=1, bool e=1, bool vl=0, bool el=0);

  void compute_max_degree();
  void degree_histogram(int bin_width = 100, std::string outfile = ""); // compute the degree distribution

  // print graph information
  void print_meta_data() const;
  void print_graph() const;
  void print_neighbors(vidType v) const;

 protected:
  void read_meta_info(std::string prefix);
  bool binary_search(vidType key, eidType begin, eidType end) const;
};

