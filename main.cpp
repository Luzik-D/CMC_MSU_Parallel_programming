#include "mpi.h"
#include <cstdio>


/// This class containts information about process tree
/// cur_pid         -   current process id
/// ppid            -   parent process id for cur_pid
/// left_child_pid  -   left child process id for cur_pid
/// right_child_pid -   right child process id for cur_pid
class Node {
    int cur_pid;
    int ppid;
    int left_child_pid;
    int right_child_pid;

public:
    Node(int a, int b, int c, int d) : cur_pid(a), ppid(b), left_child_pid(c), right_child_pid(d) {}

    // for test
    void print() {
        printf("cur pid %d\nppid %d\nleft %d\n right %d\n", cur_pid, ppid, left_child_pid, right_child_pid);
    }
};


// global vars
int rank, num_workers, init;


int main(int argc, char **argv) {
    //init group of process
    init = MPI_Init(&argc, &argv);
    if(init)  { //error
        printf("Init error %d\n", init);
        MPI_Abort(MPI_COMM_WORLD, init);
        return init;
    }
    
    return 0;
}