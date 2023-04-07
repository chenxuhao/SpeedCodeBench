A [B+ tree](https://www.geeksforgeeks.org/introduction-of-b-tree/) is an m-ary tree with a variable but often large number of children per node. 
A B+ tree consists of a root, internal nodes and leaves.[1] The root may be either a leaf or a node with two or more children.

A B+ tree can be viewed as a B-tree in which each node contains only keys (not key–value pairs), 
and to which an additional level is added at the bottom with linked leaves.

The primary value of a B+ tree is in storing data for efficient retrieval in a block-oriented storage context — in particular, filesystems. 
This is primarily because unlike binary search trees, B+ trees have very high fanout (number of pointers to child nodes in a node,[1] 
typically on the order of 100 or more), which reduces the number of I/O operations required to find an element in the tree.

[1] Navathe, Ramez Elmasri, Shamkant B. (2010). Fundamentals of database systems (6th ed.). Upper Saddle River, N.J.: Pearson Education. pp. 652–660.
