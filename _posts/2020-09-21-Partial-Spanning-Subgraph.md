---
published: false
---
The Partial Spanning Subgraph (PSS) problem is:  Given a graph $G= (V; E)$ with non-negative edge weights and a subset of the vertices $T \in V$,  and a minimal weight connected subgraph that contains all of $T$ (and may or may not include other vertices from $V \setminus T$). 

Consider the following algorithm to find PSS: Build a $Complete \;Virtual \;graph$ $G'$ by adding an edge directly between each pair of vertices in $T$ and giving it weight equal to the minimum cost path between the pair in the original graph. An example of $G'$ for this above example is included below.  After making $G'$, find the minimum spanning tree of $T$ in $G'$ and return the edges in $G$ that corresponded to the virtual edge. Show that the cost of the algorithm is at most 2 times the cost of an optimal PSS.

## Solution:
