# Introduction

This repository concerns an assignment of a KTH course called "Artificial Intelligence DD2380". The objective is to pratice and code a search algorithm in game theory. In this assignment, I implemented the minimax algorithm, alpha-beta pruning, iterative deepening search, move ordering and repeated states checking.


# Objective
The objective of this assignment was to implement the Minimax search algorithm for the best next possible move in the KTH Fishing Derby game tree.

The solution should be efficient, with a time limit per cycle of 75e-3 seconds.

The solution is provided in the file `player.py`.


# Manual Installation and run

The code runs in Python 3.7 AMD64 or Python 3.6 AMD64.

## Installation

You should start with a clean virtual environment and install the requirements for the code to run. You may create a Python 3.6 or Python 3.7 environment and install the required packages (`requirements.txt` for UNIX or `requirements_win.txt` for Windows).
 
I used a Mac to do it. So, on Mac OS X:

1. Install **python 3.7** or **python 3.6**

   https://www.python.org/downloads/mac-osx/

2. Install **virtualenv** and if you want **virtualenvwrapper**

   * Install them with pip3.

   ```
   $ sudo pip3 install virtualenv
   $ sudo pip3 install virtualenvwrapper
   ```

3. Make your virtualenvwrapper, create your virtual environment 'fishingderby' and install the requirements :

   ```
   (fishingderby) $ pip3 install -r requirements.txt
   ```


# Graphical Interface
To visualize the agent at work and understand the rules of the game better, a graphical
interface was added. You can start with:

```
(fishingderby) $ python3 main.py settings.yml
```

To play yourself using the keyboard (left, right, up, down), change the variable "player_type" in "settings.yml" to the value "human".

Note that can change the scenario of the game! In order to do so change "observations_file" in settings.yml.

## Run the program

To run the program, just activate the virtual environment `fishingderby` and run the following command in the `<Skeleton Full Path>`:

```
(fishingderby) $ python main.py settings.yml
```

# Branches = versions

In the repository you can observe that there are 4 branches in addition of `master`. Each of the branches represents a version of the algorithm. The master branch represents the original state of the files : the state in which they were given by the instructors of DD2380 KTH course. Then there are 4 different versions :
- `v1-iterative-depening-only`
- `v2-repeated-states-checking`
- `v3-move-ordering-basic`
- `v4-move-ordering-sophisticated`

Each of the versions (= implementations) is gradually more complex than the previous one and implement one more gimmick than the previous one (except the last one which has the same number of gimmicks as the v3). The v1 implements minimax algorithm with alpha-beta pruning + iterative-depening. The v2 implements v1 + repeated state checking. While v3 and v4 implement v2 + move ordering.

You will observe that this assignment uses lists and for loops. In fact, since the number for fishes is quite small, lists and for loops are more efficient than numpy.arrays and vector calculus. That is why I chose such an architecture.
