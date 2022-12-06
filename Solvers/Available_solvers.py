# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide the following
# attribution:
# This CSCE-689 RL assignment codebase was developed at Texas A&M University.
# The core code base was developed by Guni Sharon (guni@tamu.edu).

solvers = ['dqn','ddqn']

def get_solver_class(name):
    if name == solvers[0]:
        from Solvers.DQN import DQN
        return DQN
    elif name == solvers[1]:
        from Solvers.DDQN import DDQN
        return DDQN
    else:
        assert False, "unknown solver name {}. solver must be from {}".format(name, str(solvers))
