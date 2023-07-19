## General Info
- notebook that at least loads the data and the vectors, so you can play around and come back to me if you have questions
    - you don't need all of the code

- take the noun forms without syncretism, as for them the clusters are much more clear
    - There is a syncretic, matrix that has a binary coding that functions like this: each of the 14 different forms (forms_full) corresponds to some power of 2 (from 2^0 to 2^13, or, in the binary representation, each from is a 1 in the respective series of 0's, e.g. nom sg is 2^0 and gen sg is 2^1). When a form is syncretic (e.g.the same form stands for both = nom sg and gen sg, then for both cases it is represented as 2^0 + 2^1, so in the end each unique form is coded in a way that you can get a set of paradigm cells it fills. For the non-syncretic forms it means that they fill just one cell and thus their representations are powers of 2/the binary representation has only one 1 and the rest are 0.

## How to apply the min cost flow problem to this data:

Gegeben sind zwei Clusterings:
- Das Zielclustering A_1,...,A_s, das sind die "wahren" Cluster
- "unser" Clustering B_1,...,B_t (Benennungen willkürlich)

- Dabei kann t > s sein, wir erlauben also, dass wir mehr Cluster finden als eigentlich da sind (bzw. dass wir eine feinere Untergliederung finden)
-  Wir modellieren nun ein min-kosten-flussproblem, das eine Zuordnung der B-Cluster zu A-Clustern ausrechnen soll (kein Matching, da t > s möglich ist)

Modellierung ist:
- vollständiger bipartiter Graph, links A, rechts B (oder andersrum, egal)
- auf die Kante (A_i,B_j) schreiben wir die Anzahl der Punkte im Schnitt von A_i und B_j. Also wie viele Punkte sowohl in A_i als auch in B_j sind.
- Allerdings negativ, damit kleinere Werte besser sind. Es steht also -|A_i \cup B_j| auf der Kante, für alle i aus {1,..,s} und j aus {1,..,t}. Das sind die Kosten.
- es kommt links noch eine Superquelle und rechts noch eine Supersenke dazu

Kapazitäten sind nun:
- 1 für jedes "unserer" Cluster (also alle Kanten B_j zur Supersenke)
- 1 für die Kanten in der Mitte, und die Kanten links muss man größer wählen (zum Beispiel unendlich, ist glaub ich egal, wichtig ist nur, dass man ja ein A_i mehreren B_j zuweisen können muss).

- Gesucht wird nun ein Fluss mit Flusswert t, der minimale Kosten hat. Idee ist, dass der dann die Anzahl der gematchten Punkte maximiert.

### In English
The problem is called minimum cost flow problem

https://en.wikipedia.org/wiki/Minimum-cost_flow_problem#endnote_GT89

and the algorithm is called *minimum mean cycle cancelling*. Look out for a library that has an implementation of it. Some libraries have restrictions on the allowed numbers on edges, so look out for that in particular. An alternative is the *capacity scaling algorithm* or *cost scaling algorithm*. I used the C++ Library for Efficient Modeling and Optimization in Netwoks (LEMON):

http://lemon.cs.elte.hu/trac/lemon

The modeling I described was this: Make a vertex for every cluster C_i in our clustering. Say this is k clusters. Make a vertex for every cluster D_j in the true clustering. Say that's t clusters. Now make a complete bipartite directed graph from C to D. On edge (C_i,D_j) write the cost "-|C_i \intersection D_j|" and a capacity of infinity.
Connect a source s to all C_i. Every edge (s,C_i) has a capacity of 1 and a cost of 0. Connect all D_j to a sink t. On all edges (D_j,t) put capacity infinity and cost 0. Now compute a minimum cost flow from s to t, i.e., a flow of maximum value of minimum cost. That should hopefully correspond to the cluster label matching that maximizes the number of correctly classified points. And it should hopefully be solvable by a min cost flow algorithm.