# RL_Wireless_Scheduling

Final Project for CSCE 689 - Deep Reinforcement Learning

The RL Setup and Solvers are modified from the codebase provided in class, where we have used our implementation of DDQN. 
The wireless RL environment is created from scratch to apply aquired RL concepts to wireless communications.

How To Use:

Install the required libraries from requirements.txt

python run.py -s ddqn -t 2000  -e 500 -a 0.01 -g 0.95 -p 1.0 -P 0.01 -c 0.99 -m 2000 -N 20 -b 32 -o "DDQN2"

-t is the max steps
-e is the no of episodes
-a is learning rate
-g is gamma
-p , -P are initial and final epsilons
-c is epsilon decay
-N is number of steps after which target_model is updated
-b is batch size

Our simulations in report were obtained from the above shown parameters




