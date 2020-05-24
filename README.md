# Pair-Trading-with-Reinforcement-Learning

## Requirements
* python packages
  * pytorch>=0.4.0
  * torchvision>=0.2.1
  * tensorboardX
  
* PPO codes are borrowed from the course assignment https://github.com/cuhkrlcourse/ierg6130-assignment/tree/master/assignment4

### Usage
```shell
python train.py --algo PPO --log-dir data/PPO_gamma_1 --input-length 10
```

Tensorboard visualization: 
```shell
tensorboard --logdir=data/PPO_gamma_1/runs/ --port={port_num}
```
Note that all the experiments above will save the tensorboard log file in data/PPO_gamma_1/runs/ directory
