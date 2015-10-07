#!/bin/bash
# You can watch how the agent is doing by refreshing the image /tmp/x.png
python tictactoe.py --agent "type=lstm,save=lstm.np,a=0.01,g=0.8,e=0.0,states=standard" --agent "type=sarsa,load=sarsa.np,a=0.1,g=0.9,e=0.1,states=standard" --plot_reward /tmp/x.png --episode_modulo 1000