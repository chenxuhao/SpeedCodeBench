#include "graph.h"
#include "scan.h"
#include "platform_atomics.h"
std::map<OPS,double> timers;

#pragma omp declare reduction(vec_int_plus : std::vector<int> : \
    std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<int>())) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

#pragma omp declare reduction(vec_uint_plus : std::vector<uint32_t> : \
    std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<uint32_t>())) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

template<bool map_vertices, bool map_edges>
GraphT<map_vertices, map_edges>::GraphT(std::string prefix, bool use_dag, bool directed, 
             bool use_vlabel, bool use_elabel, bool need_reverse, bool bipartite, bool partitioned) :
    GraphT<map_vertices, map_edges>(directed, bipartite) {
  // parse file name
  inputfile_prefix = prefix;
  size_t i = prefix.rfind('/', prefix.length());
  if (i != string::npos) inputfile_path = prefix.substr(0, i);
  i = inputfile_path.rfind('/', inputfile_path.length());
  if (i != string::npos) name_ = inputfile_path.substr(i+1);
  std::cout << "input file prefix: " << inputfile_prefix << ", graph name: " << name_ << "\n";
  VertexSet::release_buffers();
  load_graph(prefix, use_dag, use_vlabel, use_elabel, need_reverse, partitioned);
}

template<bool map_vertices, bool map_edges>
void GraphT<map_vertices, map_edges>::load_graph(std::string prefix,
                                                 bool use_dag, bool use_vlabel, bool use_elabel,
                                                 bool need_reverse, bool partitioned) {
  // read meta information
  read_meta_info(prefix);

  // load graph data
  if (partitioned) std::cout << "This graph is partitioned, not loading the full graph\n";
  else
    load_graph_data(prefix, use_dag, use_vlabel, use_elabel, need_reverse);
}

template<bool map_vertices, bool map_edges>
void GraphT<map_vertices, map_edges>::load_graph_data(std::string prefix, 
    bool use_dag, bool use_vlabel, bool use_elabel, bool need_reverse) {
  // read row pointers
  load_row_pointers(prefix);

  // read column indices
  if constexpr (map_edges) {
    std::cout << "mmap edges\n";
    map_file(prefix + ".edge.bin", edges, n_edges);
  } else {
    std::cout << "In-memory edges\n";
    read_file(prefix + ".edge.bin", edges, n_edges);
    //if (n_vertices > 1500000000) std::cout << "Update: edge loaded\n";
  }

  if (is_directed_) {
    std::cout << "This is a directed graph\n";
    if (need_reverse) {
      build_reverse_graph();
      std::cout << "This graph maintains both incomming and outgoing edge-list\n";
      has_reverse = true;
    }
  } else {
    has_reverse = true;
    reverse_vertices = vertices;
    reverse_edges = edges;
  }

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
      if constexpr (map_vertices)
        map_file(vlabel_filename, vlabels, n_vertices);
      else read_file(vlabel_filename, vlabels, n_vertices);
      std::set<vlabel_t> labels;
      for (vidType v = 0; v < n_vertices; v++)
        labels.insert(vlabels[v]);
      //for (int i = 0; i < n_vertices; i++) std::cout << unsigned(vlabels[i]) << "\n";
      std::cout << "# distinct vertex labels: " << labels.size() << "\n";
      assert(size_t(num_vertex_classes) == labels.size());
    } else {
      std::cout << "WARNING: vertex label file not exist; generating random labels\n";
      vlabels = new vlabel_t[n_vertices];
      for (vidType v = 0; v < n_vertices; v++) {
        vlabels[v] = rand() % num_vertex_classes + 1;
      }
    }
    auto max_vlabel = unsigned(*(std::max_element(vlabels, vlabels+n_vertices)));
    std::cout << "maximum vertex label: " << max_vlabel << "\n";
  }
  // read edge labels
  if (use_elabel) {
    std::string elabel_filename = prefix + ".elabel.bin";
    ifstream f_elabel(elabel_filename.c_str());
    if (f_elabel.good()) {
      assert (num_edge_classes > 0);
      if constexpr (map_edges)
        map_file(elabel_filename, elabels, n_edges);
      else read_file(elabel_filename, elabels, n_edges);
      std::set<elabel_t> labels;
      for (eidType e = 0; e < n_edges; e++)
        labels.insert(elabels[e]);
      //for (int i = 0; i < n_edges; i++) {
      //  if (elabels[i] > 5 || elabels[i] < 1)
      //    std::cout << "elabels[" << i << "]=" << elabels[i] << "\n";
      //}
      //for (int i = 0; i < 10; i++) std::cout << elabels[i] << "\n";
      std::cout << "# distinct edge labels: " << labels.size() << "\n";
      //for (auto l : labels) std::cout << l << "\n";
      assert(size_t(num_edge_classes) >= labels.size());
    } else {
      std::cout << "WARNING: edge label file not exist; generating random labels\n";
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
    auto max_elabel = unsigned(*(std::max_element(elabels, elabels+n_edges)));
    std::cout << "maximum edge label: " << max_elabel << "\n";
  }
  // Orientation: convert the undirected graph into directed. 
  // An optimization used for k-cliques. This would likely decrease max_degree.
  if (use_dag) {
    assert(!is_directed_); // must be undirected before orientation
    this->orientation();
  }
  VertexSet::MAX_DEGREE = std::max(max_degree, VertexSet::MAX_DEGREE);
  labels_frequency_.clear();
}

template<bool map_vertices, bool map_edges>
GraphT<map_vertices, map_edges>::~GraphT() {
  deallocate();
}
 
template<bool map_vertices, bool map_edges>
void GraphT<map_vertices, map_edges>::deallocate() {
  if (dst_list != NULL && dst_list != edges) {
    delete [] dst_list;
    dst_list = NULL;
  }
  if (src_list != NULL) {
    delete [] src_list;
    src_list = NULL;
  }
  if (edges != NULL) {
    if constexpr (map_edges) munmap(edges, n_edges*sizeof(vidType));
    else custom_free(edges, n_edges);
    edges = NULL;
  }
  if (vertices != NULL) {
    if constexpr (map_vertices) munmap(vertices, (n_vertices+1)*sizeof(eidType));
    else custom_free(vertices, n_vertices+1);
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

template<bool map_vertices, bool map_edges>
void GraphT<map_vertices, map_edges>::read_meta_info(std::string prefix) {
  std::ifstream f_meta((prefix + ".meta.txt").c_str());
  assert(f_meta);
  int64_t nv = 0;
  if (is_bipartite_) {
    f_meta >> n_vert0 >> n_vert1;
    nv = int64_t(n_vert0) + int64_t(n_vert1);
  } else f_meta >> nv;
  f_meta >> n_edges >> vid_size >> eid_size >> vlabel_size >> elabel_size
         >> max_degree >> feat_len >> num_vertex_classes >> num_edge_classes;
  assert(sizeof(vidType) == vid_size);
  assert(sizeof(eidType) == eid_size);
  assert(sizeof(vlabel_t) == vlabel_size);
  //assert(sizeof(elabel_t) == elabel_size);
  f_meta.close();
  assert(nv > 0 && n_edges > 0);
  if (vid_size == 4) assert(nv < 4294967295);
  n_vertices = nv;
  std::cout << "Reading graph: |V| " << nv << " |E| " << n_edges << "\n";
}

template<bool map_vertices, bool map_edges>
void GraphT<map_vertices, map_edges>::load_row_pointers(std::string prefix) {
  if constexpr (map_vertices) {
    std::cout << "mmap vertices\n";
    map_file(prefix + ".vertex.bin", vertices, n_vertices+1);
  } else {
    std::cout << "In-memory vertices\n";
    read_file(prefix + ".vertex.bin", vertices, n_vertices+1);
    //if (n_vertices > 1500000000) std::cout << "Update: vertex loaded\n";
  }
}
 
template<bool map_vertices, bool map_edges>
VertexSet GraphT<map_vertices, map_edges>::N(vidType vid) const {
  assert(vid >= 0);
  assert(vid < n_vertices);
  eidType begin = vertices[vid], end = vertices[vid+1];
  if (begin > end || end > n_edges) {
    fprintf(stderr, "vertex %u bounds error: [%lu, %lu)\n", vid, begin, end);
    exit(1);
  }
  assert(end <= n_edges);
  return VertexSet(edges + begin, end - begin, vid);
}

template<bool map_vertices, bool map_edges>
void GraphT<map_vertices,map_edges>::sort_neighbors() {
  std::cout << "Sorting the neighbor lists (used for pattern mining)\n";
  #pragma omp parallel for
  for (vidType v = 0; v < n_vertices; v++) {
    auto begin = edge_begin(v);
    auto end = edge_end(v);
    std::sort(edges+begin, edges+end);
  }
}

template<bool map_vertices, bool map_edges>
void GraphT<map_vertices,map_edges>::sort_and_clean_neighbors(std::string outfile_prefix) {
  std::cout << "Sorting the neighbor lists and remove selfloops and redundent edges (used for pattern mining)\n";
  degrees.resize(n_vertices);
  std::fill(degrees.begin(), degrees.end(), 0);
  eidType num_selfloops = 0;
  eidType num_redundants = 0;
  using T = typename std::conditional_t<map_edges,VertexList,VertexSet>;
  #pragma omp parallel for reduction(+:num_selfloops,num_redundants)
  for (vidType v = 0; v < n_vertices; v++) {
    auto begin = edge_begin(v);
    auto end = edge_end(v);
    T adj_list;
    if constexpr (map_edges) {
      adj_list.resize(end-begin);
      std::copy(edges+begin, edges+end, adj_list.begin());
      std::sort(adj_list.begin(), adj_list.end());
    } else {
      adj_list.duplicate(N(v));
      std::sort(edges+begin, edges+end);
    }
    eidType i = 0;
    vidType n_selfloops = 0;
    vidType n_redundants = 0;
    for (auto u : adj_list) {
      if (u == v) {
        n_selfloops += 1;
        i++;
        continue;
      }
      if (i>0 && u == adj_list[i-1]) {
        i++;
        n_redundants += 1;
        continue;
      }
      assert(u < n_vertices);
      degrees[v] ++;
      i++;
    }
    num_selfloops += n_selfloops;
    num_redundants += n_redundants;
    assert(get_degree(v) == (n_selfloops+n_redundants+degrees[v]));
  }
  std::cout << "Number of self loops: " << num_selfloops << "\n";
  std::cout << "Number of redundant edges: " << num_redundants << "\n";
  eidType *new_vertices = custom_alloc_global<eidType>(n_vertices+1);
  parallel_prefix_sum<vidType,eidType>(degrees, new_vertices);
  auto num_edges = new_vertices[n_vertices];
  std::cout << "|E| after clean: " << num_edges << "\n";
  assert(n_edges == (num_edges+num_selfloops+num_redundants));
  degrees.clear();
  degrees.shrink_to_fit(); // release memory
  std::string vertex_file_path = outfile_prefix + ".vertex.bin";
  std::string edge_file_path = outfile_prefix + ".edge.bin";

  vidType *new_edges;
  int fd = 0;
  size_t num_bytes = 0;
  void *map_ptr = NULL;
  if (outfile_prefix == "") {
    std::cout << "Generating the new graph in memory\n";
    new_edges = custom_alloc_global<vidType>(num_edges);
  } else {
    std::cout << "generating the new graph in disk\n";
    std::ofstream outfile(vertex_file_path.c_str(), std::ios::binary);
    if (!outfile) {
      std::cout << "File not available\n";
      throw 1;
    }
    outfile.write(reinterpret_cast<const char*>(new_vertices), (n_vertices+1)*sizeof(eidType));
    outfile.close();

    fd = open(edge_file_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
    if (fd == -1) {
      perror("Error opening file for writing");
      exit(EXIT_FAILURE);
    }
    num_bytes = num_edges * sizeof(vidType)+1;
    // Stretch the file size to the size of the (mmapped) bytes
    if (lseek(fd, num_bytes-1, SEEK_SET) == -1) {
      close(fd);
      perror("Error calling lseek() to 'stretch' the file");
      exit(EXIT_FAILURE);
    }
    // Something needs to be written at the end of the file to make the file actually have the new size.
    if (write(fd, "", 1) == -1) {
      close(fd);
      perror("Error writing last byte of the file");
      exit(EXIT_FAILURE);
    }
    // Now the file is ready to be mmapped
    map_ptr = mmap(0, num_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    new_edges = (vidType*)map_ptr;
  }

  #pragma omp parallel for
  for (vidType v = 0; v < n_vertices; v ++) {
    auto begin = new_vertices[v];
    T adj_list;
    if constexpr (map_edges) {
      adj_list.resize(get_degree(v));
      std::copy(edges+edge_begin(v), edges+edge_end(v), adj_list.begin());
      std::sort(adj_list.begin(), adj_list.end());
    } else {
      adj_list.duplicate(N(v));
    }
    eidType i = 0;
    eidType j = 0;
    for (auto u : adj_list) {
      if (u == v) {
        i++;
        continue;
      }
      if (i>0 && u == adj_list[i-1]) {
        i++;
        continue;
      }
      new_edges[begin+j] = u;
      j++;
      i++;
    }
  }
  std::cout << "deleting old graph\n";
  if constexpr (map_vertices) {
  } else {
    delete [] vertices;
  }
  if constexpr (map_edges) {
  } else {
    delete [] edges;
  }
  n_edges = num_edges;
  vertices = new_vertices;
  edges = new_edges;
  if (outfile_prefix != "") {
    // Write it now to disk
    if (msync(map_ptr, num_bytes, MS_SYNC) == -1)
      perror("Could not sync the file to disk");
    // Don't forget to free the mmapped memory
    if (munmap(map_ptr, num_bytes) == -1) {
      close(fd);
      perror("Error un-mmapping the file");
      exit(EXIT_FAILURE);
    }
    // Un-mmaping doesn't close the file, so we still need to do that.
    close(fd);
  }
}

template<bool map_vertices, bool map_edges>
void GraphT<map_vertices,map_edges>::symmetrize() {
  degrees.resize(n_vertices);
  std::fill(degrees.begin(), degrees.end(), 0);
  std::cout << "Computing degrees\n";
  #pragma omp parallel for
  for (vidType v = 0; v < n_vertices; v++) {
    degrees[v] = get_degree(v);
  }
  std::cout << "Computing new degrees\n";
  eidType num_new_edges = 0;
  #pragma omp parallel for reduction(+:num_new_edges)
  for (vidType v = 0; v < n_vertices; v++) {
    eidType i = 0;
    //std::sort(edges+edge_begin(v), edges+edge_end(v));
    for (auto u : N(v)) {
      assert(u < n_vertices);
      assert(u != v); // assuming self-loops are removed
      assert(i==0 || u != N(v,i-1)); // assuming redundant edges are removed
      if (binary_search(v, edge_begin(u), edge_end(u)))
        continue;
      fetch_and_add(degrees[u], 1);
      //degrees[u] ++;
      num_new_edges += 1;
    }
  }
  std::cout << "Adding " << num_new_edges << " new edges\n";
  eidType *new_vertices = custom_alloc_global<eidType>(n_vertices+1);
  parallel_prefix_sum<vidType,eidType>(degrees, new_vertices);
  degrees.clear();
  auto num_edges = new_vertices[n_vertices];
  std::cout << "|E| after symmetrization: " << num_edges << "\n";
  assert(num_edges <= 2*n_edges);
  vidType *new_edges = custom_alloc_global<vidType>(num_edges);
  std::cout << "Copying existing edges\n";
  #pragma omp parallel for
  for (vidType v = 0; v < n_vertices; v ++) {
    auto begin = new_vertices[v];
    std::copy(edges+edge_begin(v), edges+edge_end(v), &new_edges[begin]);
  }
  std::vector<vidType> offsets(n_vertices, 0);
  #pragma omp parallel for
  for (vidType v = 0; v < n_vertices; v++) {
    offsets[v] = get_degree(v);
  }
  std::cout << "Computing new column indices\n";
  #pragma omp parallel for
  for (vidType v = 0; v < n_vertices; v ++) {
    for (auto u : N(v)) {
      if (binary_search(v, edge_begin(u), edge_end(u)))
        continue;
      auto begin = new_vertices[u];
      auto offset = fetch_and_add(offsets[u], 1);
      new_edges[begin+offset] = v;
    }
  }
  if constexpr (map_vertices) {
  } else {
    delete [] vertices;
  }
  if constexpr (map_edges) {
  } else {
    delete [] edges;
  }
  vertices = new_vertices;
  edges = new_edges;
  n_edges = num_edges;
  sort_neighbors();
}
/*
template<bool map_vertices, bool map_edges>
void GraphT<map_vertices,map_edges>::symmetrize() {
  std::cout << "Symmetrizing the neighbor lists (used for pattern mining)\n";
  std::vector<std::set<vidType>> neighbor_lists(n_vertices);
  #pragma omp parallel for
  for (vidType v = 0; v < n_vertices; v++) {
    auto deg = get_degree(v);
    assert (deg <= max_degree);
    neighbor_lists[v].insert(edges+edge_begin(v), edges+edge_end(v));
  }
  std::cout << "Inserting reverse edges\n";
  for (vidType v = 0; v < n_vertices; v++) {
    for (auto u : N(v)) {
      if (u == v) continue;
      neighbor_lists[u].insert(v);
    }
  }

  std::cout << "Computing degrees\n";
  std::vector<vidType> degrees(n_vertices, 0);
  for (vidType v = 0; v < n_vertices; v++) {
    degrees[v] = neighbor_lists[v].size();
  }
  std::cout << "Computing indices by prefix sum\n";
  eidType *new_vertices = custom_alloc_global<eidType>(n_vertices+1);
  parallel_prefix_sum<vidType,eidType>(degrees, new_vertices);
  auto num_edges = new_vertices[n_vertices];
  std::cout << "|E| after symmetrization: " << num_edges << "\n";
  assert(num_edges <= 2*n_edges);
  vidType *new_edges = custom_alloc_global<vidType>(num_edges);
  #pragma omp parallel for
  for (vidType v = 0; v < n_vertices; v ++) {
    auto begin = new_vertices[v];
    std::copy(neighbor_lists[v].begin(), neighbor_lists[v].end(), &new_edges[begin]);
  }
  delete [] vertices;
  delete [] edges;
  vertices = new_vertices;
  edges = new_edges;
  n_edges = num_edges;
}
*/
template<bool map_vertices, bool map_edges>
void GraphT<map_vertices,map_edges>::write_to_file(std::string outfilename, bool v, bool e, bool vl, bool el) {
  std::cout << "Writing graph to file\n";
  if (v) {
    std::ofstream outfile((outfilename+".vertex.bin").c_str(), std::ios::binary);
    if (!outfile) {
      std::cout << "File not available\n";
      throw 1;
    }
    outfile.write(reinterpret_cast<const char*>(vertices), (n_vertices+1)*sizeof(eidType));
    outfile.close();
  }

  if (e) {
    std::ofstream outfile1((outfilename+".edge.bin").c_str(), std::ios::binary);
    if (!outfile1) {
      std::cout << "File not available\n";
      throw 1;
    }
    outfile1.write(reinterpret_cast<const char*>(edges), n_edges*sizeof(vidType));
    outfile1.close();
  }

  if (vl && vlabels) {
    std::ofstream outfile((outfilename+".vlabel.bin").c_str(), std::ios::binary);
    if (!outfile) {
      std::cout << "File not available\n";
      throw 1;
    }
    outfile.write(reinterpret_cast<const char*>(&vlabels[0]), n_vertices*sizeof(vlabel_t));
    outfile.close();
  }

  if (el && elabels) {
    std::ofstream outfile((outfilename+".elabel.bin").c_str(), std::ios::binary);
    if (!outfile) {
      std::cout << "File not available\n";
      throw 1;
    }
    outfile.write(reinterpret_cast<const char*>(&elabels[0]), n_edges*sizeof(elabel_t));
    outfile.close();
  }
}
 
template<bool map_vertices, bool map_edges>
void GraphT<map_vertices, map_edges>::build_reverse_graph() {
  std::vector<VertexList> reverse_adj_lists(n_vertices);
  for (vidType v = 0; v < n_vertices; v++) {
    for (auto u : N(v)) {
      reverse_adj_lists[u].push_back(v);
    }
  }
  reverse_vertices = custom_alloc_global<eidType>(n_vertices+1);
  reverse_vertices[0] = 0;
  for (vidType i = 1; i < n_vertices+1; i++) {
    auto degree = reverse_adj_lists[i-1].size();
    reverse_vertices[i] = reverse_vertices[i-1] + degree;
  }
  reverse_edges = custom_alloc_global<vidType>(n_edges);
  //#pragma omp parallel for
  for (vidType i = 0; i < n_vertices; i++) {
    auto begin = reverse_vertices[i];
    std::copy(reverse_adj_lists[i].begin(), 
        reverse_adj_lists[i].end(), &reverse_edges[begin]);
  }
  for (auto adjlist : reverse_adj_lists) adjlist.clear();
  reverse_adj_lists.clear();
}

template<> VertexSet GraphT<>::out_neigh(vidType vid, vidType offset) const {
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

// TODO: fix for directed graph
template<> VertexSet GraphT<>::in_neigh(vidType vid) const {
  assert(vid >= 0);
  assert(vid < n_vertices);
  auto begin = reverse_vertices[vid];
  auto end = reverse_vertices[vid+1];
  if (begin > end) {
    fprintf(stderr, "vertex %u bounds error: [%lu, %lu)\n", vid, begin, end);
    exit(1);
  }
  assert(end <= n_edges);
  return VertexSet(reverse_edges + begin, end - begin, vid);
}
 
template<bool map_vertices, bool map_edges>
void GraphT<map_vertices, map_edges>::allocateFrom(vidType nv, eidType ne) {
  n_vertices = nv;
  n_edges    = ne;
  vertices = new eidType[nv+1];
  edges = new vidType[ne];
  vertices[0] = 0;
}
 
template<bool map_vertices, bool map_edges>
void GraphT<map_vertices, map_edges>::compute_max_degree() {
  std::cout << "computing the maximum degree\n";
  Timer t;
  t.Start();
  #pragma omp parallel for reduction(max:max_degree)
  for (vidType v = 0; v < n_vertices; v++) {
    auto deg = this->get_degree(v);
    if (deg > max_degree) max_degree = deg;
  }
  t.Stop();
  std::cout << "maximum degree: " << max_degree << "\n";
  std::cout << "Time computing the maximum degree: " << t.Seconds() << " sec\n";
}

template<bool map_vertices, bool map_edges>
void GraphT<map_vertices, map_edges>::degree_histogram(int bin_width, std::string outfile_prefix) {
  std::cout << "Degree distribution\n";
  Timer t;
  t.Start();
  assert(bin_width > 0 && bin_width < int(max_degree));
  int num_bins = (max_degree-1)/bin_width + 1;
  std::vector<int> counts(num_bins, 0);
  #pragma omp parallel for reduction(vec_int_plus:counts)
  for (vidType v = 0; v < n_vertices; v++) {
    auto deg = this->get_degree(v);
    auto bin_id = deg/bin_width;
    counts[bin_id] ++;
  }
  for (int i = 0; i < std::min(11,num_bins); i++) {
    auto bin_id = num_bins - 1 - i;
    std::cout << "Number of vertices in range[" << bin_id*bin_width << "," 
      << (bin_id+1)*bin_width << "]: " << counts[bin_id] << "\n";
  }
  if (num_bins > 20) {
    for (int i = 0; i < std::min(11,num_bins); i++) {
      auto bin_id = i;
      std::cout << "Number of vertices in range[" << bin_id*bin_width << "," 
        << (bin_id+1)*bin_width << "]: " << counts[bin_id] << "\n";
    }
  }
}

template<bool map_vertices, bool map_edges>
void GraphT<map_vertices, map_edges>::orientation(std::string outfile_prefix) {
  std::cout << "Orientation enabled, generating DAG\n";
  Timer t;
  t.Start();
  degrees.resize(n_vertices);
  std::fill(degrees.begin(), degrees.end(), 0);
  #pragma omp parallel for
  for (vidType v = 0; v < n_vertices; v++) {
    degrees[v] = this->get_degree(v);
  }
  std::vector<vidType> new_degrees(n_vertices, 0);
  #pragma omp parallel for
  for (vidType src = 0; src < n_vertices; src ++) {
    for (auto dst : N(src)) {
      if (degrees[dst] > degrees[src] ||
          (degrees[dst] == degrees[src] && dst > src)) {
        new_degrees[src]++;
      }
    }
  }
  max_degree = *(std::max_element(new_degrees.begin(), new_degrees.end()));
  eidType *new_vertices = custom_alloc_global<eidType>(n_vertices+1);
  parallel_prefix_sum<vidType,eidType>(new_degrees, new_vertices);
  auto num_edges = new_vertices[n_vertices];

  std::cout << "|E| after clean: " << num_edges << "\n";
  assert(n_edges == num_edges*2);
  std::string vertex_file_path = outfile_prefix + ".vertex.bin";
  std::string edge_file_path = outfile_prefix + ".edge.bin";

  vidType *new_edges;
  int fd = 0;
  size_t num_bytes = 0;
  void *map_ptr = NULL;
  if (outfile_prefix == "") {
    std::cout << "Generating the new graph in memory\n";
    new_edges = custom_alloc_global<vidType>(num_edges);
  } else {
    std::cout << "generating the new graph in disk\n";
    std::ofstream outfile(vertex_file_path.c_str(), std::ios::binary);
    if (!outfile) {
      std::cout << "File not available\n";
      throw 1;
    }
    outfile.write(reinterpret_cast<const char*>(new_vertices), (n_vertices+1)*sizeof(eidType));
    outfile.close();

    fd = open(edge_file_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
    if (fd == -1) {
      perror("Error opening file for writing");
      exit(EXIT_FAILURE);
    }
    num_bytes = num_edges * sizeof(vidType)+1;
    // Stretch the file size to the size of the (mmapped) bytes
    if (lseek(fd, num_bytes-1, SEEK_SET) == -1) {
      close(fd);
      perror("Error calling lseek() to 'stretch' the file");
      exit(EXIT_FAILURE);
    }
    // Something needs to be written at the end of the file to make the file actually have the new size.
    if (write(fd, "", 1) == -1) {
      close(fd);
      perror("Error writing last byte of the file");
      exit(EXIT_FAILURE);
    }
    // Now the file is ready to be mmapped
    map_ptr = mmap(0, num_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    new_edges = (vidType*)map_ptr;
  }

  #pragma omp parallel for
  for (vidType src = 0; src < n_vertices; src ++) {
    auto begin = new_vertices[src];
    eidType offset = 0;
    for (auto dst : N(src)) {
      if (degrees[dst] > degrees[src] ||
          (degrees[dst] == degrees[src] && dst > src)) {
        new_edges[begin+offset] = dst;
        offset ++;
      }
    }
  }
  std::cout << "deleting old graph\n";
  if constexpr (map_vertices) {
  } else {
    delete [] vertices;
  }
  if constexpr (map_edges) {
  } else {
    delete [] edges;
  }
  n_edges = num_edges;
  vertices = new_vertices;
  edges = new_edges;
  if (outfile_prefix != "") {
    // Write it now to disk
    if (msync(map_ptr, num_bytes, MS_SYNC) == -1)
      perror("Could not sync the file to disk");
    // Don't forget to free the mmapped memory
    if (munmap(map_ptr, num_bytes) == -1) {
      close(fd);
      perror("Error un-mmapping the file");
      exit(EXIT_FAILURE);
    }
    // Un-mmaping doesn't close the file, so we still need to do that.
    close(fd);
  }
  t.Stop();
  std::cout << "Time on generating the DAG: " << t.Seconds() << " sec\n";
}

template<> void GraphT<>::print_neighbors(vidType v) const {
  eidType begin = vertices[v], end = vertices[v+1];
  std::cout << "[ ";
  for (auto e = begin; e != end; e++) {
    if (elabels != NULL)
      std::cout << "<";
    std::cout << getEdgeDst(e) << " ";
    if (elabels != NULL)
      std::cout << getEdgeData(e) << "> ";
  }
  std::cout << "]";
}

template<> void GraphT<>::print_graph() const {
  std::cout << "Printing the graph: \n";
  for (vidType n = 0; n < n_vertices; n++) {
    eidType begin = vertices[n], end = vertices[n+1];
    std::cout << "vertex " << n << ": degree = " << this->get_degree(n) 
      << " edge range: [" << begin << ", " << end << ")"
      << " edgelist = "; 
    print_neighbors(n);
    std::cout << "\n";
  }
}

template<> eidType GraphT<>::init_edgelist(bool sym_break, bool ascend) {
  Timer t;
  t.Start();
  if (nnz != 0) return nnz; // already initialized
  nnz = E();
  if (sym_break) nnz = nnz/2;
  sizes.resize(V());
  src_list = new vidType[nnz];
  if (sym_break) dst_list = new vidType[nnz];
  else dst_list = edges;
  size_t i = 0;
  for (vidType v = 0; v < V(); v ++) {
    for (auto u : N(v)) {
      if (u == v) continue; // no selfloops
      if (ascend) {
        if (sym_break && v > u) continue;  
      } else {
        if (sym_break && v < u) break;  
      }
      src_list[i] = v;
      if (sym_break) dst_list[i] = u;
      sizes[v] ++;
      i ++;
    }
  }
  //assert(i == nnz);
  t.Stop();
  std::cout << "Time on generating the edgelist: " << t.Seconds() << " sec\n";
  return nnz;
}

template<bool map_vertices, bool map_edges>
bool GraphT<map_vertices,map_edges>::binary_search(vidType key, eidType begin, eidType end) const {
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

template<> bool GraphT<>::is_connected(vidType v, vidType u) const {
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

template<> bool GraphT<>::is_connected(std::vector<vidType> sg) const {
  return false;
}

template<> vidType GraphT<>::intersect_num(vidType v, vidType u) {
  return N(v).get_intersect_num(N(u));
}

template<> vidType GraphT<>::intersect_num(vidType v, vidType u, vlabel_t label) {
  vidType num = 0;
  vidType idx_l = 0, idx_r = 0;
  vidType v_size = this->get_degree(v);
  vidType u_size = this->get_degree(u);
  vidType* v_ptr = &edges[vertices[v]];
  vidType* u_ptr = &edges[vertices[u]];
  while (idx_l < v_size && idx_r < u_size) {
    vidType a = v_ptr[idx_l];
    vidType b = u_ptr[idx_r];
    if (a <= b) idx_l++;
    if (b <= a) idx_r++;
    if (a == b && vlabels[a] == label) num++;
  }
  return num;
}

template<> vidType GraphT<>::intersect_num(VertexSet& vs, vidType u, vlabel_t label) {
  vidType num = 0;
  vidType idx_l = 0, idx_r = 0;
  vidType u_size = this->get_degree(u);
  vidType* u_ptr = &edges[vertices[u]];
  while (idx_l < vs.size() && idx_r < u_size) {
    vidType a = vs[idx_l];
    vidType b = u_ptr[idx_r];
    if (a <= b) idx_l++;
    if (b <= a) idx_r++;
    if (a == b && vlabels[a] == label) num++;
  }
  return num;
}

template<> vidType GraphT<>::intersect_set(vidType v, vidType u, vlabel_t label, VertexSet& result) {
  vidType num = 0;
  vidType idx_l = 0, idx_r = 0;
  vidType v_size = this->get_degree(v);
  vidType u_size = this->get_degree(u);
  vidType* v_ptr = &edges[vertices[v]];
  vidType* u_ptr = &edges[vertices[u]];
  while (idx_l < v_size && idx_r < u_size) {
    vidType a = v_ptr[idx_l];
    vidType b = u_ptr[idx_r];
    if (a <= b) idx_l++;
    if (b <= a) idx_r++;
    if (a == b && vlabels[a] == label) {
      result.add(a);
      num++;
    }
  }
  return num;
}

template<> vidType GraphT<>::intersect_set(VertexSet& vs, vidType u, vlabel_t label, VertexSet& result) {
  vidType num = 0;
  vidType idx_l = 0, idx_r = 0;
  vidType u_size = this->get_degree(u);
  vidType* u_ptr = &edges[vertices[u]];
  while (idx_l < vs.size() && idx_r < u_size) {
    vidType a = vs[idx_l];
    vidType b = u_ptr[idx_r];
    if (a <= b) idx_l++;
    if (b <= a) idx_r++;
    if (a == b && vlabels[a] == label) {
      result.add(a);
      num++;
    }
  }
  return num;
}

template<> vidType GraphT<>::difference_num_edgeinduced(vidType v, vidType u, vlabel_t label) {
  vidType num = 0;
  vidType* v_ptr = &edges[vertices[v]];
  for (vidType i = 0; i < this->get_degree(v); i ++) {
    auto w = v_ptr[i];
    if (w != u && vlabels[w] == label) num++;
  }
  return num;
}

template<> vidType GraphT<>::difference_num_edgeinduced(VertexSet& vs, vidType u, vlabel_t label) {
  vidType num = 0;
  for (auto w : vs)
    if (w != u && vlabels[w] == label) num++;
  return num;
}

template<> vidType GraphT<>::difference_set_edgeinduced(vidType v, vidType u, vlabel_t label, VertexSet& result) {
  vidType num = 0;
  vidType* v_ptr = &edges[vertices[v]];
  for (vidType i = 0; i < this->get_degree(v); i ++) {
    auto w = v_ptr[i];
    if (w != u && vlabels[w] == label) {
      result.add(w);
      num++;
    }
  }
  return num;
}

template<> vidType GraphT<>::difference_set_edgeinduced(VertexSet& vs, vidType u, vlabel_t label, VertexSet& result) {
  vidType num = 0;
  for (auto w : vs) {
    if (w != u && vlabels[w] == label) {
      result.add(w);
      num++;
    }
  }
  return num;
}

template<> vidType GraphT<>::difference_num(vidType v, vidType u, vlabel_t label) {
  vidType num = 0;
  vidType idx_l = 0, idx_r = 0;
  vidType v_size = this->get_degree(v);
  vidType u_size = this->get_degree(u);
  vidType* v_ptr = &edges[vertices[v]];
  vidType* u_ptr = &edges[vertices[u]];
  while (idx_l < v_size && idx_r < u_size) {
    auto a = v_ptr[idx_l];
    auto b = u_ptr[idx_r];
    if (a <= b) idx_l++;
    if (b <= a) idx_r++;
    if (a < b && a != u && vlabels[a] == label) num++;
  }
  while (idx_l < v_size) {
    auto a = v_ptr[idx_l];
    idx_l++;
    if (a != u && vlabels[a] == label)
      num ++;
  }
  return num;
}

template<> vidType GraphT<>::difference_num(VertexSet& vs, vidType u, vlabel_t label) {
  vidType num = 0;
  vidType idx_l = 0, idx_r = 0;
  vidType u_size = this->get_degree(u);
  vidType* u_ptr = &edges[vertices[u]];
  while (idx_l < vs.size() && idx_r < u_size) {
    auto a = vs[idx_l];
    auto b = u_ptr[idx_r];
    if (a <= b) idx_l++;
    if (b <= a) idx_r++;
    if (a < b && a != u && vlabels[a] == label) num++;
  }
  while (idx_l < vs.size()) {
    auto a = vs[idx_l];
    idx_l++;
    if (a != u && vlabels[a] == label)
      num ++;
  }
  return num;
}

template<> vidType GraphT<>::difference_set(vidType v, vidType u, vlabel_t label, VertexSet& result) {
  vidType num = 0;
  vidType idx_l = 0, idx_r = 0;
  vidType v_size = this->get_degree(v);
  vidType u_size = this->get_degree(u);
  vidType* v_ptr = &edges[vertices[v]];
  vidType* u_ptr = &edges[vertices[u]];
  while (idx_l < v_size && idx_r < u_size) {
    auto a = v_ptr[idx_l];
    auto b = u_ptr[idx_r];
    if (a <= b) idx_l++;
    if (b <= a) idx_r++;
    if (a < b && a != u && vlabels[a] == label) {
      result.add(a);
      num++;
    }
  }
  while (idx_l < v_size) {
    auto a = v_ptr[idx_l];
    idx_l++;
    if (a != u && vlabels[a] == label) {
      result.add(a);
      num ++;
    }
  }
  return num;
}

template<> vidType GraphT<>::difference_set(VertexSet& vs, vidType u, vlabel_t label, VertexSet& result) {
  vidType num = 0;
  vidType idx_l = 0, idx_r = 0;
  vidType u_size = this->get_degree(u);
  vidType* u_ptr = &edges[vertices[u]];
  while (idx_l < vs.size() && idx_r < u_size) {
    auto a = vs[idx_l];
    auto b = u_ptr[idx_r];
    if (a <= b) idx_l++;
    if (b <= a) idx_r++;
    if (a < b && a != u && vlabels[a] == label) {
      result.add(a);
      num++;
    }
  }
  while (idx_l < vs.size()) {
    auto a = vs[idx_l];
    idx_l++;
    if (a != u && vlabels[a] == label) {
      result.add(a);
      num ++;
    }
  }
  return num;
}

template<> void GraphT<>::computeLabelsFrequency() {
  labels_frequency_.resize(num_vertex_classes+1);
  std::fill(labels_frequency_.begin(), labels_frequency_.end(), 0);
  //max_label = int(*std::max_element(vlabels, vlabels+size()));
  #pragma omp parallel for reduction(max:max_label)
  for (vidType i = 0; i < size(); ++i) {
    max_label = max_label > vlabels[i] ? max_label : vlabels[i];
  }
  #pragma omp parallel for reduction(vec_uint_plus:labels_frequency_)
  for (vidType v = 0; v < size(); ++v) {
    int label = int(this->get_vlabel(v));
    assert(label <= num_vertex_classes);
    labels_frequency_[label] += 1;
  }
  max_label_frequency_ = vidType(*std::max_element(labels_frequency_.begin(), labels_frequency_.end()));
  //std::cout << "max_label = " << max_label << "\n";
  //std::cout << "max_label_frequency_ = " << max_label_frequency_ << "\n";
  //for (size_t i = 0; i < labels_frequency_.size(); ++i)
  //  std::cout << "label " << i << " vertex frequency: " << labels_frequency_[i] << "\n";
}

template<> void GraphT<>::BuildReverseIndex() {
  if (labels_frequency_.empty()) computeLabelsFrequency();
  int nl = num_vertex_classes;
  if (max_label == num_vertex_classes) nl += 1;
  reverse_index_.resize(size());
  reverse_index_offsets_.resize(nl+1);
  reverse_index_offsets_[0] = 0;
  vidType total = 0;
  for (int i = 0; i < nl; ++i) {
    total += labels_frequency_[i];
    reverse_index_offsets_[i+1] = total;
    //std::cout << "label " << i << " frequency: " << labels_frequency_[i] << "\n";
  }
  std::vector<eidType> start(nl);
  for (int i = 0; i < nl; ++i) {
    start[i] = reverse_index_offsets_[i];
    //std::cout << "label " << i << " start: " << start[i] << "\n";
  }
  for (vidType i = 0; i < size(); ++i) {
    auto vl = vlabels[i];
    reverse_index_[start[vl]++] = i;
  }
}

template<> int GraphT<>::get_frequent_labels(int threshold) {
  assert(threshold > 0);
  int num = 0;
  for (size_t i = 0; i < labels_frequency_.size(); ++i)
    if (labels_frequency_[i] > vidType(threshold))
      num++;
  return num;
}

template<> bool GraphT<>::is_freq_vertex(vidType v, int threshold) {
  assert(threshold > 0);
  assert(v >= 0 && v < size());
  auto label = int(vlabels[v]);
  assert(label <= num_vertex_classes);
  if (labels_frequency_[label] >= vidType(threshold)) return true;
  return false;
}

// NLF: neighborhood label frequency
template<> void GraphT<>::BuildNLF() {
  //std::cout << "Building NLF map for the data graph\n";
  nlf_.resize(size());
  #pragma omp parallel for
  for (vidType v = 0; v < size(); ++v) {
    for (auto u : N(v)) {
      auto vl = this->get_vlabel(u);
      if (nlf_[v].find(vl) == nlf_[v].end())
        nlf_[v][vl] = 0;
      nlf_[v][vl] += 1;
    }
  }
}

template<bool map_vertices, bool map_edges>
void GraphT<map_vertices,map_edges>::print_meta_data() const {
  std::cout << "|V|: " << n_vertices << ", |E|: " << n_edges << ", Max Degree: " << max_degree << "\n";
  if (num_vertex_classes > 0) {
    std::cout << "vertex-|\u03A3|: " << num_vertex_classes;
    if (!labels_frequency_.empty()) 
      std::cout << ", Max Label Frequency: " << max_label_frequency_;
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

template <> void GraphT<>::computeKCore() {
  int nv = size();
  int md = get_max_degree();
  std::vector<int> vertices(nv);          // Vertices sorted by degree.
  std::vector<int> position(nv);          // The position of vertices in vertices array.
  std::vector<int> degree_bin(md+1, 0);   // Degree from 0 to max_degree.
  std::vector<int> offset(md+1);          // The offset in vertices array according to degree.
  for (int i = 0; i < nv; ++i) {
    int degree = this->get_degree(i);
    core_table[i] = degree;
    degree_bin[degree] += 1;
  }
  int start = 0;
  for (int i = 0; i < md+1; ++i) {
    offset[i] = start;
    start += degree_bin[i];
  }
  for (int i = 0; i < nv; ++i) {
    int degree = this->get_degree(i);
    position[i] = offset[degree];
    vertices[position[i]] = i;
    offset[degree] += 1;
  }
  for (int i = md; i > 0; --i) {
    offset[i] = offset[i - 1];
  }
  offset[0] = 0;
  for (int i = 0; i < nv; ++i) {
    int v = vertices[i];
    for (vidType j = 0; j < this->get_degree(v); ++j) {
      int u = N(v, j);
      if (core_table[u] > core_table[v]) {
        // Get the position and vertex which is with the same degree
        // and at the start position of vertices array.
        int cur_degree_u = core_table[u];
        int position_u = position[u];
        int position_w = offset[cur_degree_u];
        int w = vertices[position_w];
        if (u != w) {
          // Swap u and w.
          position[u] = position_w;
          position[w] = position_u;
          vertices[position_u] = w;
          vertices[position_w] = u;
        }
        offset[cur_degree_u] += 1;
        core_table[u] -= 1;
      }
    }
  }
}

template<> void GraphT<>::buildCoreTable() {
  core_table.resize(size(), 0);
  computeKCore();
  for (vidType i = 0; i < size(); ++i) {
    if (core_table[i] > 1) {
      core_length_ += 1;
    }
  }
  //for (int v = 0; v < size(); v++)
  //  std::cout << "v_" << v << " core value: " << core_table[v] << "\n";
}

template class GraphT<false, false>;
template class GraphT<false, true>;
template class GraphT<true, true>;
