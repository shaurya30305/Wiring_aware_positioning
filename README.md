# Wiring Aware Gate Positioning

## Overview
This project implements an efficient solution for optimizing gate placement in integrated circuit design using simulated annealing. It generates a 2D layout of gates while minimizing the total wire length between connections, producing near-optimal results in polynomial time.

## Problem Description
The algorithm solves the following gate placement optimization problem:

### Input
- A set of rectangular logic gates (g₁, g₂, ..., gₙ)
- Dimensions (width and height) for each gate
- Pin locations (x, y coordinates) on the boundary of each gate
- Pin-to-pin connections between gates

### Constraints
- No overlapping gates allowed
- All connections must be maintained
- Gates must be placed within the specified grid

### Objective
Minimize the total estimated wire length across all connections, where wire lengths are calculated using Manhattan distance.

## Algorithm: Simulated Annealing
The project uses simulated annealing for several key advantages in circuit design:

Simulated annealing is a probabilistic method which explores an entire solution space, initially exploring solutions with worse optimisation with high probability to try to
ultimately unlock paths to the desired optimal solution. It is able to solve an otherwise NP-hard problem in poly(n) time.

## Installation

### Prerequisites
- Python 3.x


### Setup
1. Clone the repository or download the source files
2. Ensure you have both core files:
   - `src/src.py`: Core placement algorithm
   - `visualizer/visualization.py`: Layout visualization tool

## Usage

### Running the Program
```bash
python visualization.py    
```

Example:
```bash
python visualization.py output.txt test1.ini 200 200
```

### Input File Format
```ini
[gates]
# gate_name width height
g1 10 20
g2 15 25

[pins]
# gate_name pin_name x y
g1 p1 5 0
g1 p2 5 20

[connections]
# source_gate.pin target_gate.pin
g1.p1 g2.p2
```

## Test Cases
In addition to the test cases in the repository, I tried testing with various inputs, some of which is shown below


1.)
![image](https://github.com/user-attachments/assets/5f4fd831-461f-4cfe-8b4e-bc34acee4deb)
2.) 
![image](https://github.com/user-attachments/assets/a3c6e808-9088-41af-abd9-d48d2881ba06)
3.)
![image](https://github.com/user-attachments/assets/a27eb684-fb13-4c0a-a13f-c876a6921479)
