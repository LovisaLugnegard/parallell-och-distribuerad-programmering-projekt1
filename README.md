# parallell-och-distribuerad-programmering-projekt1

Implementing dense matrix multiplication for parallel
distributed-memory environments
1 Problem setting
Let two dense matrices be given, A = fai;jg and B = fai;jg, A;B 2 Rnn, i.e., i; j = 1; 2;    ; n. Let
C = AB. The elements of C are given by
ci;j =
Xn
k=1
ai;kbk;j ; i; j = 1; 2;    ; n:
The task is to design a matrix-matrix multiplication procedure for performing the multiplication in a
distributed memory parallel computer environment and implement the algorithm using C/C++/Fortran
and MPI. The implementation must be independent of the size of the matrices n and the number of
processors p to be used, thus, n and p must be input parameters.
2 Data generation and partitioning strategies
Create your matrices with entries initialized as random double precision numbers on a single processor
and distribute the elements according to the given partitioning strategy, see below.
Ap1-1,p2-1
A0,1
...
...
...
...
... ... ...
A1,1
A0,0
A1,0 A
A
p1-1,1 Ap1-1,0 A
1,p2-1
0,p2-1
Figure 1: Two dimensional data partitioning.
As shown in Figure 1 the processors are assumed to form a 2D Cartesian grid where p = p1  p2. It
is also assumed that n is divisible by both p1 and p2, i.e., r1 = n=p1 and r2 = n=p2. For simplicity
1
you can also assume or restrict to the case p1 = p2 =
p
p. Assume that the matrices A, B and C are
partitioned in blocks, as shown in Figure 1. Then, the blocks A(s;r);B(s;r);C(s;r) are local for processor
P(s;r), s = 0; 1;    ; p1 􀀀 1, r = 0; 1;    ; p2 􀀀 1.
3 Fox’s algorithm
Processor P(s;r) must compute, in block form,
C(s;r) =
p
Xp􀀀1
k=0
A(s;k)B(k;r)
where A(s;k)B(k;r) is a matrix-matrix multiplication of two blocks. Processor P(s;r) needs to have all
blocks A(s;:) (in row s) and all blocks B(:;r) (in column r). This requires interprocessor communication.
The communication and computations can be organized as follows (Fox’s algorithm).
For each step k (k = 0; :::;
p
p 􀀀 1):
1. Broadcast block A(i;m) within each block-row i (i = 0; :::;
p
p 􀀀 1) where m = (i + k)modp
p.
2. Multiply the broadcasted block with the B-block in each processor, C(s;r) = C(s;r)+A(s;m)B(m;r).
3. Shift the blocks of B one step cyclicly upwards.
4 MPI implementation
To get an efficient MPI implementation you are required to use a cartesian 2D processor topology and
create separate communicators for the rows and the columns. In phase 1, utilize the row-communicators
by using collective communication, i.e. MPI_Bcast. Then, overlap the communication in phase 3 with
the computations in phase 2 by using non-blocking send and receive within the column-communicators.
(Start up the communication of the B-blocks before phase 2 and then in phase 3 wait for these communication
operations to finish.)
5 Numerical experiments
Check the correctness of your algorithm by applying it to two simple matrices. Then, use your algorithm
to compute C = AB for matrices of different sizes. Go up in matrix size from 100  100 to 1440 
1440 with a few intervals. For each matrix size vary the number of processors and measure speedup.
Pay attention to your results and note any abnormalities. Make several runs for each problem size and
processor configuration and report the lowest run-time.
6 Coaching session
You are welcome anytime to ask about the assignment but we will also like to meet all groups in coaching
sessions. The intention is to see that you are on the right track and to correct any missunderstandings
2
in an early stage. Please, contact us for booking a time for the coaching session. Before the coaching
session you should have started the implementation of the assignment but there is no requirement that the
program can run (or even compile correctly).
7 Writing a report on the results
The report can be written in Swedish or English but you should use a word processor or a type setting
system (e.g. LATEX, Word, LibreOffice). The report should cover the following issues:
Report requirements:
1. Problem description, presenting the task.
2. Solution method, i.e. a description of the parallel implementation.
3. Results, presenting plots of speedup. The plots can be done using Matlab, Maple or any other
graphical tool.
4. Discussion, with observations and comments on the results (can also be included in 3 above).
5. Conclusions, with explanations of the results and with ideas for possible optimizations or improvements.
6. Appendix, with a listing of the program code and tables of the numerical results.
The assignments are a part of the examination, this means that you should protect your source code from
others and you cannot copy the solutionof others. The last day to hand in the report is April 13, 2017.
3
