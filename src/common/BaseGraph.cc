#include "BaseGraph.hh"

void BaseGraph::orientation(std::string outfile_prefix) {
  std::cout << "Orientation enabled, generating DAG\n";
  if (is_directed_) return;
  //Timer t;
  //t.Start();
  std::vector<vidType> degrees(n_vertices);
  std::fill(degrees.begin(), degrees.end(), 0);
  for (vidType v = 0; v < n_vertices; v++) {
    degrees[v] = this->get_degree(v);
  }
  std::vector<vidType> new_degrees(n_vertices, 0);
  for (vidType src = 0; src < n_vertices; src ++) {
    //for (auto dst : N(src)) {
    auto adj = get_adj(src);
    for (vidType i = 0; i < degrees[src]; i++) {
      auto dst = adj[i];
      if (degrees[dst] > degrees[src] ||
          (degrees[dst] == degrees[src] && dst > src)) {
        new_degrees[src]++;
      }
    }
  }
  max_degree = *(std::max_element(new_degrees.begin(), new_degrees.end()));
  eidType *new_vertices = custom_alloc_global<eidType>(n_vertices+1);
  new_vertices[0] = 0;
  for (vidType i = 0; i < n_vertices; i++) {
    new_vertices[i+1] = new_vertices[i] + new_degrees[i];
  }
  auto num_edges = new_vertices[n_vertices];

  //std::cout << "|E| after clean: " << num_edges << "\n";
  assert(n_edges == num_edges*2);
  std::string vertex_file_path = outfile_prefix + ".vertex.bin";
  std::string edge_file_path = outfile_prefix + ".edge.bin";

  vidType *new_edges;
  int fd = 0;
  size_t num_bytes = 0;
  void *map_ptr = NULL;
  if (outfile_prefix == "") {
    //std::cout << "Generating the new graph in memory\n";
    new_edges = custom_alloc_global<vidType>(num_edges);
  } else {
    //std::cout << "generating the new graph in disk\n";
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

  for (vidType src = 0; src < n_vertices; src ++) {
    auto begin = new_vertices[src];
    eidType offset = 0;
    //for (auto dst : N(src)) {
    auto adj = get_adj(src);
    for (vidType i = 0; i < degrees[src]; i++) {
      auto dst = adj[i];
      if (degrees[dst] > degrees[src] ||
          (degrees[dst] == degrees[src] && dst > src)) {
        new_edges[begin+offset] = dst;
        offset ++;
      }
    }
  }
  delete [] vertices;
  delete [] edges;
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
  is_directed_ = true;
  //t.Stop();
  //std::cout << "Time on generating the DAG: " << t.Seconds() << " sec\n";
}

VertexSet BaseGraph::N(vidType vid) const {
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

// TODO: fix for directed graph
VertexSet BaseGraph::in_neigh(vidType vid) const {
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
 
void BaseGraph::build_reverse_graph() {
  if (!is_directed()) {
    reverse_vertices = vertices;
    reverse_edges = edges;
    return;
  }
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
  for (vidType i = 0; i < n_vertices; i++) {
    auto begin = reverse_vertices[i];
    std::copy(reverse_adj_lists[i].begin(), 
        reverse_adj_lists[i].end(), &reverse_edges[begin]);
  }
  for (auto adjlist : reverse_adj_lists) adjlist.clear();
  reverse_adj_lists.clear();
}

