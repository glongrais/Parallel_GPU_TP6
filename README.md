# Simple Game of Life

<p align="center">
  <img alt="UR1" src="https://img.shields.io/badge/-UR1-orange?style=flat-square" />
</p>

Lab 6 for the PPAR Parallel Programming course at [Université de Rennes 1](https://www.univ-rennes1.fr/).

## Principle

The objective of this lab assignment is to implement a 2-player version of [Conway’s Game of Life](http://en.wikipedia.org/wiki/Conway's_Game_of_Life)

## Task

We consider a 2-dimensional domain composed of ````domain x```` × ````domain y```` cells. Each cell can be either red, blue, or empty. The domain has a torus shape: the right neighbor of the rightmost cell is the leftmost cell.  
Each cell has 8 neighbors in the adjacent cells. At each time step, all cells evolve according to the following rules:  
* a cell that has strictly less than 2 alive neighbors among the 8 adjacent cells dies,
* a cell that has strictly more than 3 alive neighbors dies too,
* a cell that has 2 or 3 alive neighbors survives,
* an empty cell that has exactly 3 neighbors becomes occupied. Its color will be selected from the majority of its neighbors (i.e. if 2 or more neighbor cells are blue, the new cell is blue, otherwise it is red).

All cells are updated synchronously.  

To avoid race conditions, we follow a ping-pong approach: we maintain two copies of the domain. At each time step, we read from one copy and write to the other one, then exchange the pointers to the read domain and written domain.  
1. Program the simulation without optimizing memory accesses. We use the ````read_cell```` function to access neighbors.  
2. How many read memory accesses to global memory does each thread perform? How many read accesses per thread block does this make?
We now want to reduce the number of global memory accesses by sharing data within each thread block.
3. Consider one thread block. How many global memory locations are read by a least one thread of the block? Unlike in the prior question, locations that are accessed by multiple threads of the block are only counted once.
Hint: make a drawing
4. Same question if blocks are 2-dimensional. Which block shape would minimize the number of unique loca- tions read?
5. Use shared memory to avoid the redundant reads to global memory.

## Author

* [Guillaume Longrais](https://github.com/glongrais)
