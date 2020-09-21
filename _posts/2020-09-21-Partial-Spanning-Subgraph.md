---
published: true
---
The Partial Spanning Subgraph (PSS) problem is:  Given a graph $G= (V; E)$ with non-negative edge weights and a subset of the vertices $T \in V$,  and a minimal weight connected subgraph that contains all of $T$ (and may or may not include other vertices from $V \setminus T$). 

Consider the following algorithm to find PSS: Build a $Complete \;Virtual \;graph$ $G'$ by adding an edge directly between each pair of vertices in $T$ and giving it weight equal to the minimum cost path between the pair in the original graph. An example of $G'$ for this above example is included below.  After making $G'$, find the minimum spanning tree of $T$ in $G'$ and return the edges in $G$ that corresponded to the virtual edge. Show that the cost of the algorithm is at most 2 times the cost of an optimal PSS.

## Solution:

Suppose we already know the optimal PSS solution of a graph as shown in the figure bellow (the blue nodes correspond to the $T$ set):

![figure]({{ site.baseurl }}/images/pss.jpg)

Now, we can create the route $R_G$: $1-a-2-3-4-3-a-1$ following the red arrows shown in the figure of the left. Note that this route may require, in general, to pass through some nodes of $V \setminus T$. Note that we also need to pass twice through each edge of our PSS. This means that the cost of this route is $2OPT$. We need to translate this route to graph $G'$. Given that we do not have the intermediate nodes (the white nodes) in graph $G'$, we directly connect each pair of nodes of $T$. For example, the first two red arrows of the original route are simplified to a single arrow that connects node 1 and 2. 

In addition, we consider the triangle inequality, similar to what we do with the Travelling Salesman Problem (metric TSP). We also take into account that the connected graph $G'$ was constructed taking the shortest distance between two nodes. Then: $cost(1,2) \leq cost(1,a) + cost(a,2)$. As a result, the new route $R_{G'}$ is as follows: $1-2-3-4-1$, as shown in the image of the right. Notice that the route $4-3-1$ was simplified to $4-1$. That is, in general, we avoid passing through nodes that have been already visited by directly connecting two nodes with the assumption that this new connection has a lower cost than the original route. Therefore, we conclude that the new route has a lower cost: $cost(R_{G'}) \leq cost(R_{G})$ or $cost(R_{G'}) \leq 2OPT$.

The route $R_{G'}$ is a Hamiltonian cycle because each node is visited exactly once. If we remove a single edge of this cycle, it becomes a spanning tree. The cost of this spanning tree is still lower than $2OPT$. This condition holds for any spanning tree in $G'$; therefore, the cost of the minimum spanning, $MST_{G'}$, tree obtained by the proposed algorithm \textbf{is at most 2 times the cost of an optimal PSS: $cost(MST_{G'})\leq 2OPT$}.


_Course: CSCI 532 - Algorithms. Instructor: Sean Yaw. Spring 2020._
