# Cell automata

Moscow State University, faculty of Computational Mathematics and Cybernetics. Computer Graphics course, Prof. Vladimir Frolov, autumn 2020.

GUI developed with OpenGL in C++ from scratch.

Conway Game of Life

<img src=pictures/1_life.png width=300>

Generations or "sparks"

<img src=pictures/2_generations.png width=300>

Unicell

<img src=pictures/3_unicell.png width=300>

Turmites

<img src=pictures/4_turmite.png width=300>

## Dependencies and compilation 

Install OpenGL library (Ubuntu): 
```
sudo apt install libglfw3-dev
```

C++ standard 11 is required.

Compile and run: 
```
mkdir build; cd build; cmake .. && make && cp ../game_save.json . && ./main
```

## Simulation control

Mouse:
* Press left mouse button and drag - field movement
* Press right mouse button - change cell state (editing mode)
* Mouse wheel - change scale

Keyboard:
* ESCAPE - close
* ENTER - editing mode, stop simulation
* Up or down arrow - simulation speed
* Right arrow - one simulation step
* F1, F2, F3, F4 - switch automata mode (Conway Game of Life, Generations "Sparks", Cell Universe "Unicell", Turmites) (Жизнь, Поколения, Клеточная вселенная, Тьюрмиты)  
* n - change to next parameter preset, reset simulation  
* c - reset simulation

## Automata rules

Neighbours are counted including diagonal.
* Conway Game of Life: 
    * if cell is dead, it becomes alive only if 3 of its neighbours are alive
    * if cell is alive, it stays alive only if 2 or 3 of its neighbours are alive
* Generations:
    * Cell state: 0 - dead, 1 - alive, (n+1, n, ..., 2) - phases of dying
    * If cell is dying, the phase decreases by 1, and dies at the end
    * If cell is dead, and has {3, 4, 6, 8} alive neighbours (configurable parameter), it becomes alive
    * If cell is alive, and has {1, 2, 3} alive neighbours (configurable parameter), it stays alive, otherwise starts dying
* Unicell:
    * Cell state: phases 0, 1, ... n-1
    * If cell has state x, and a neighbour cell has state x+1 (states are taken by modulo n), then the neighbour cell degrades to state x
* Turmites
    * Turmite is like an ant, sitting in a cell and with a movement direction divisible by 90 degrees. On each step the turmite follows a set rules of form (cur_state, cur_cell, new_state, new_cell, turn) like a Turing machine, hence the name Turmite.

Parameters for automata are stored in game_save.json.

## Libraries
* JsonCpp (https://github.com/open-source-parsers/jsoncpp)  
* DearImgui (https://github.com/ocornut/imgui)  

## References
* [Some Cell Automata](https://habr.com/ru/articles/719324/)
