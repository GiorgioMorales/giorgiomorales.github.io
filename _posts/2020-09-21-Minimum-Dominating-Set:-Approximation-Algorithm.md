---
published: true
---
The Minimum Dominating Set problem is: Given a graph, find the smallest subset of the vertices such that each remaining vertex has a neighbor in the selected subset. Find an approximation algorithm for the Minimum Dominating Set problem.

## Solution:

Let $G=(V,E), k$ be the input of the Dominating Set (DS) problem. Then, we consider $U=V$ and the set $S$ such that each $S_i \in S$ is the set of nodes adjacent to vertex $v_i \in V$. The next [figure](2) shows an example of this translation. 


![figure]({{ site.baseurl }}/images/minDomSet.jpg)

Now, we need to prove:

$G$ has a $k$-Dominating Set $\iff$ $\bar{G}$ has a $(k)$ -Set Cover.
    
$\Rightarrow{}$ Suppose $G$ has a $k$-DS $Q$. It means the each vertex in $V$ has at least one neighbor in $Q$. Each node $q_i \in Q$ corresponds to one subset $s_i \in S$. The set of subsets corresponding to $Q$ in $S$ is called $C$ ($\|Q\| =\|C\| =k$). Then, the union of all the subsets $c_i \in C$ should be equal to the universe $U$; otherwise, it would mean that there is at least one node in $V$ that is not connected to at least one node in $Q$. Therefore, $C$ is a $k$-Set Cover.    
    
$\Leftarrow{}$ Suppose $U$ has a $(k)$-Set Cover $C$. It means that the union of all subsets $c_i \in C$ is equal to $U$. Each subset $s_i \in S$ corresponds to one node $q_i \in Q$ (in the [figure](2), a subset that starts with the number $x$ corresponds to the node $x$). Suppose there is a node $x$ in $V$ where neither neighbor is in $Q$; it would mean that none of the subsets in $C$ contains $x$ (because each subset in $S$ consists of neighbors) and there is another subset in $S$ that contains the missing node $x$. This contradicts the initial statement that $C$ is a set cover; therefore, such node $x$ cannot exist and $Q$ is a $k$-Dominating set.
    
Finally, the proposed approximation algorithm for the Minimum Dominating Set problem is as follows: Translate the problem into a Minimum Set Cover problem, apply the approximation algorithm we know for this problem ($\| approxSC \|$$ \leq ln(\| V \|)OPT_{SC}$, where $OPT_{SC}$ is the minimum SC). Then, we translate back the $approxSC$ solution into the set of nodes $approxDS$ which represents the approximation of the minimum dominating Set. Given that the size of the minimum dominating set is the same as that of the minimum set cover ($OPT_{DS} = OPT_{SC}$) we conclude that:

$$|approxDS| \leq ln(|V|)OPT_{DS}.$$

Course: CSCI 532 - Algorithms. Instructor: Sean Yaw
