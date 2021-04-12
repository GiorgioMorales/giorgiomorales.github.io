---
published: false
---
Suppose we have $p$ machines and $n$ jobs. Each job has a time it takes to complete $t_j$ (i.e.  processing time).  Let $A_m$ denote the jobs assigned to machine $m$.  This means that machine $m$ will operate for: $T_m = \sum_{ j \in A_m}t_j$.  The goal is to assign all jobs to machines in such a way as to minimize: $max_mT_m$.
  
  Consider the following algorithm: For each job, assign it to the machine with the smallest current $T_m$.
  
_**1. Show that the algorithm is a 2-approximation algorithm.**_

## Solution:

Consider $ALG$ is the time of the longest running machine obtained by the algorithm, and $OPT$ the optimal time of the longest running machine. Now, consider that the total time consumed by all the jobs is $\sum_n t_n$, then, if we split all the jobs evenly among the $p$ machines, the optimal time has to be greater or equal than the time of each one of this machines: $\frac{1}{p} \sum_n t_n \leq OPT$.
  
  Now we consider the last job, $n_y$, scheduled on the machine $m_x$ responsible for $ALG$. The last job $n_y$ was assigned to $m_x$ because, before it was scheduled, $m_x$ had the smallest time of all the machines: $T_{m_x} - t_{n_y} \leq T_{m = \{1...p\}}$. Then, if $T_{m_x} - t_{n_y}$ is less or equal than any machine, it will be also less or equal than the average time of all the $p$ machines:
  
  $$ T_{m_x} - t_{n_y} \leq \frac{1}{p} \sum^m_{j=1} T_j$$
  
  However, the sum of times of all $p$ machines is equal to the sum of times of the $n$ jobs:
  
  $$ T_{m_x} - t_{n_y} \leq \frac{1}{p} \sum^m_{j=1} T_j = \frac{1}{p} \sum^n_{i=1} t_i \\
  T_{m_x} - t_{n_y} \leq \frac{1}{p} \sum^n_{i=1} t_i \leq OPT$$
  
  We also note that $t_{n_y} \leq OPT$ because the time of the longest running machine should be greater or equal than the time of any job. Therefore, after scheduling the last job $n_y$ we will have:
  
  $$T_{m_x} - t_{n_y} + t_{n_y}\leq OPT + OPT \\
  T_{m_x} \leq 2OPT \\
  ALG \leq 2OPT $$
      
      
      
_**2. Prove that the algorithm will be optimal if $n \leq p$.**_
 
## Solution:
 
 If there are more machines than jobs, then during each iteration the smallest current $T_m$ is zero, which means that each job will be assigned to one separate machine; therefore $ALG$ will be equal to the time of longest job $t_{max}$, which is also the optimal solution $OPT$.
 
 
 
 
_**3. Consider an instance with $p$ machines and $n=p(p-1) + 1$ jobs, where the first $n-1$ jobs require time $tj= 1$ and the last job requires $tn=p$. What is $ALG$? What is $OPT$? What can you conclude about the algorithm?**_
 
## Solution:
  
 ![figure]({{ site.baseurl }}/images/schedule.jpg)
 
 According to the proposed algorithm, each incoming job will be assigned to the machine with the smallest time. Given that the first $n-1$ jobs have a time of 1, they will be assigned to the $p$ machines as shown in the Figure of the left ($p$ machines with $p-1$ time each one). Then, the last job will be assigned to the first machine and $ALG=2p-1$. However, if we had received the last job before, as shown in the Figure of the right, we would have had $p$ machines with $p$ time each one ($OPT=p$). From this, we conclude that the proposed algorithm is sensitive to the incoming order, it would be better if we receive the heaviest jobs first.
 
 
 _Course: CSCI 532 - Algorithms. Instructor: Sean Yaw. Spring 2020._
