#include <set>
#include <map>
#include "graph.h"

Graph::Graph(std::string prefix, bool use_dag, bool directed, 
             bool use_vlabel, bool use_elabel, bool bipartite, bool partitioned) :
    Graph(directed, bipartite, 0, 0) {
  // parse file name
  inputfile_prefix = prefix;
  size_t i = prefix.rfind('/', prefix.length());
  if (i != string::npos) inputfile_path = prefix.substr(0, i);
  i = inputfile_path.rfind('/', inputfile_path.length());
  if (i != string::npos) name_ = inputfile_path.substr(i+1);
  //std::cout << "input file prefix: " << inputfile_prefix << ", graph name: " << name_ << "\n";
  if (bipartite) std::cout << "This is a Bipartite graph\n";
  load_graph(prefix, use_dag, use_vlabel, use_elabel, partitioned);
}

void Graph::load_graph(std::string prefix,
                      bool use_dag, bool use_vlabel, bool use_elabel,
                      bool partitioned) {
  VertexSet::release_buffers();

  BaseGraph::load(prefix);

  std::ifstream f_meta((prefix + ".meta.txt").c_str());
  assert(f_meta);
  uint64_t nv = 0;
  if (is_bipartite_) {
    f_meta >> n_vert0 >> n_vert1;
    n_vertices = int64_t(n_vert0) + int64_t(n_vert1);
  } else {
    f_meta >> nv;
  }
  f_meta >> n_edges >> vid_size >> eid_size;
  f_meta >> vlabel_size >> elabel_size
    >> max_degree >> feat_len 
    >> num_vertex_classes >> num_edge_classes;
  assert(sizeof(vidType) == vid_size);
  assert(sizeof(eidType) == eid_size);
  assert(sizeof(vlabel_t) == vlabel_size);
  //assert(sizeof(elabel_t) == elabel_size);
  f_meta.close();
  assert(nv > 0 && n_edges > 0);
  if (vid_size == 4) assert(nv < 4294967295);
  n_vertices = nv;
 
  f_meta.close();

  // load graph data
  if (partitioned) std::cout << "This graph is partitioned, not loading the full graph\n";
  else
    load_graph_data(prefix, use_dag, use_vlabel, use_elabel);

  // Orientation: convert the undirected graph into directed. 
  // An optimization used for k-cliques. This would likely decrease max_degree.
  if (use_dag) {
    assert(!is_directed_); // must be undirected before orientation
    this->orientation();
  }
}

void Graph::load_graph_data(std::string prefix, bool use_dag, bool use_vlabel, bool use_elabel) {
  // compute maximum degree
  if (max_degree == 0) compute_max_degree();
  //else std::cout << "max_degree: " << max_degree << "\n";
  assert(max_degree > 0 && max_degree < n_vertices);

  // read vertex labels
  if (use_vlabel) {
    assert (num_vertex_classes > 0);
    assert (num_vertex_classes < 255); // we use 8-bit vertex label dtype
    std::string vlabel_filename = prefix + ".vlabel.bin";
    ifstream f_vlabel(vlabel_filename.c_str());
    if (f_vlabel.good()) {
      read_file(vlabel_filename, vlabels, n_vertices);
      std::set<vlabel_t> labels;
      for (vidType v = 0; v < n_vertices; v++)
        labels.insert(vlabels[v]);
      //for (int i = 0; i < n_vertices; i++) std::cout << unsigned(vlabels[i]) << "\n";
      std::cout << "# distinct vertex labels: " << labels.size() << "\n";
      assert(size_t(num_vertex_classes) == labels.size());
    } else {
      //std::cout << "WARNING: vertex label file not exist; generating random labels\n";
      vlabels = new vlabel_t[n_vertices];
      for (vidType v = 0; v < n_vertices; v++) {
        vlabels[v] = rand() % num_vertex_classes + 1;
      }
    }
    auto max_vlabel = unsigned(*(std::max_element(vlabels, vlabels+n_vertices)));
    std::cout << "maximum vertex label: " << max_vlabel << "\n";
  }
  // read edge labels
  if (use_elabel) load_edge_labels(prefix);
  VertexSet::MAX_DEGREE = std::max(max_degree, VertexSet::MAX_DEGREE);
}

void Graph::load_edge_labels(std::string prefix) {
  std::string elabel_filename = prefix + ".elabel.bin";
  ifstream f_elabel(elabel_filename.c_str());
  if (f_elabel.good()) {
    assert (num_edge_classes > 0);
    read_file(elabel_filename, elabels, n_edges);
    std::set<elabel_t> labels;
    for (eidType e = 0; e < n_edges; e++)
      labels.insert(elabels[e]);
    std::cout << "# distinct edge labels: " << labels.size() << "\n";
    assert(size_t(num_edge_classes) >= labels.size());
  } else {
    //std::cout << "WARNING: edge label file not exist; generating random labels\n";
    elabels = new elabel_t[n_edges];
    if (num_edge_classes < 1) {
      num_edge_classes = 1;
      for (eidType e = 0; e < n_edges; e++) {
        elabels[e] = 1;
      }
    } else {
      for (eidType e = 0; e < n_edges; e++) {
        elabels[e] = rand() % num_edge_classes + 1;
      }
    }
  }
  //auto max_elabel = unsigned(*(std::max_element(elabels, elabels+n_edges)));
  //std::cout << "maximum edge label: " << max_elabel << "\n";
}

Graph::~Graph() {
  deallocate();
}
 
void Graph::deallocate() {
  if (dst_list != NULL && dst_list != edges) {
    delete [] dst_list;
    dst_list = NULL;
  }
  if (src_list != NULL) {
    delete [] src_list;
    src_list = NULL;
  }
  if (edges != NULL) {
    custom_free(edges, n_edges);
    edges = NULL;
  }
  if (vertices != NULL) {
    custom_free(vertices, n_vertices+1);
    vertices = NULL;
  }
  if (vlabels != NULL) {
    delete [] vlabels;
    vlabels = NULL;
  }
  if (elabels != NULL) {
    delete [] elabels;
    elabels = NULL;
  }
  if (features != NULL) {
    delete [] features;
    features = NULL;
  }
}

void Graph::compute_max_degree() {
  for (vidType v = 0; v < n_vertices; v++) {
    auto deg = this->get_degree(v);
    if (deg > max_degree) max_degree = deg;
  }
}

void Graph::sort_neighbors() {
  std::cout << "Sorting the neighbor lists (used for pattern mining)\n";
  for (vidType v = 0; v < n_vertices; v++) {
    auto begin = edge_begin(v);
    auto end = edge_end(v);
    std::sort(edges+begin, edges+end);
  }
}

VertexSet Graph::out_neigh(vidType vid, vidType offset) const {
  assert(vid >= 0);
  assert(vid < n_vertices);
  auto begin = vertices[vid];
  auto end = vertices[vid+1];
  if (begin > end) {
    fprintf(stderr, "vertex %u bounds error: [%lu, %lu)\n", vid, begin, end);
    exit(1);
  }
  assert(end <= n_edges);
  return VertexSet(edges + begin + offset, end - begin, vid);
}
 
void Graph::allocateFrom(vidType nv, eidType ne) {
  n_vertices = nv;
  n_edges    = ne;
  vertices = new eidType[nv+1];
  edges = new vidType[ne];
  vertices[0] = 0;
}

bool Graph::binary_search(vidType key, eidType begin, eidType end) const {
  auto l = begin;
  auto r = end-1;
  while (r >= l) { 
    auto mid = l + (r - l) / 2;
    auto value = getEdgeDst(mid);
    if (value == key) return true;
    if (value < key) l = mid + 1; 
    else r = mid - 1; 
  } 
  return false;
}

// if u is an outgoing neighbor of v
bool Graph::is_neighbor(vidType v, vidType u) const {
  return binary_search(u, edge_begin(v), edge_end(v));
}

bool Graph::is_connected(vidType v, vidType u) const {
  auto v_deg = this->get_degree(v);
  auto u_deg = this->get_degree(u);
  bool found;
  if (v_deg < u_deg) {
    found = binary_search(u, edge_begin(v), edge_end(v));
  } else {
    found = binary_search(v, edge_begin(u), edge_end(u));
  }
  return found;
}

void Graph::print_meta_data() const {
  std::cout << "|V|: " << n_vertices << ", |E|: " << n_edges << ", Max Degree: " << max_degree << "\n";
  if (num_vertex_classes > 0) {
    std::cout << "vertex-|\u03A3|: " << num_vertex_classes;
    std::cout << "\n";
  } else {
    //std::cout  << "This graph does not have vertex labels\n";
  }
  if (num_edge_classes > 0) {
    std::cout << "edge-|\u03A3|: " << num_edge_classes << "\n";
  } else {
    //std::cout  << "This graph does not have edge labels\n";
  }
  if (feat_len > 0) {
    std::cout << "Vertex feature vector length: " << feat_len << "\n";
  } else {
    //std::cout  << "This graph has no input vertex features\n";
  }
}

