"""
This file implements the train scripts for both A2C and PPO

You need to implement all TODOs in this script.

Note that you may find this file is completely compatible for both A2C and PPO.

-----
2019-2020 2nd term, IERG 6130: Reinforcement Learning and Beyond. Department
of Information Engineering, The Chinese University of Hong Kong. Course
Instructor: Professor ZHOU Bolei. Assignment author: PENG Zhenghao.
"""
import argparse
from collections import deque

import gym
import numpy as np
import torch

import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib as mpl 
import pandas as pd 

# from competitive_pong import make_envs
from EV_Pairtrading_env2 import PairtradingEnv
from baseline import baselineConfig, baseline 

from core.a2c_trainer import A2CTrainer, a2c_config
from core.ppo_trainer import PPOTrainer, ppo_config
from core.utils import verify_log_dir, pretty_print, Timer, evaluate, \
    summary, save_progress, FrameStackTensor, step_envs

import time
from datetime import datetime
from tensorboardX import SummaryWriter

gym.logger.set_level(40)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--algo",
    default="",
    type=str,
    help="(Required) The algorithm you want to run. Must in [PPO, A2C]."
)
parser.add_argument(
    "--log-dir",
    default="/tmp/ierg6130_hw4/",
    type=str,
    help="The path of directory that you want to store the data to. "
         "Default: /tmp/ierg6130_hw4/ppo/"
)
parser.add_argument(
    "--num-envs",
    default=15,
    type=int,
    help="The number of parallel environments. Default: 15"
)
parser.add_argument(
    "--learning-rate", "-LR",
    default=5e-4,
    type=float,
    help="The learning rate. Default: 5e-4"
)
parser.add_argument(
    "--seed",
    default=100,
    type=int,
    help="The random seed. Default: 100"
)
parser.add_argument(
    "--input-length",
    default=4,
    type=int,
    help="The range of input length. Default: 4"
)
parser.add_argument(
    "--max-steps",
    "-N",
    default=1e7,
    type=float,
    help="The random seed. Default: 1e7"
)
parser.add_argument(
    "--env-name",
    default="PairTrading",
    type=str,
    help="The environment id, should be in ['PairTrading']. "
         "Default: PairTrading"
)
parser.add_argument(
    "--restore-log-dir",
    default="None",
    type=str,
    help="Restore log dir"
)
parser.add_argument(
    "--restore-suffix",
    default="None",
    type=str,
    help="Restore suffix"
)
args = parser.parse_args()


def train(args):
    # Verify algorithm and config
    algo = args.algo
    if algo == "PPO":
        config = ppo_config
    elif algo == "A2C":
        config = a2c_config
    else:
        raise ValueError("args.algo must in [PPO, A2C]")
    config.num_envs = args.num_envs

    # Seed the environments and setup torch
    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_num_threads(1)

    # Clean log directory
    log_dir = verify_log_dir(args.log_dir, algo)

    # Create vectorized environments
    num_envs = args.num_envs
    env_name = args.env_name

    # Prepare tensorboard file 
    args.save_log = 'Pairtrding-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
    generate_date = str(datetime.now().date())

    writer = SummaryWriter(args.log_dir+'/runs/' + generate_date + '/' + args.save_log)

    # download stock price data from yahoo finance
    stocklist = ['0700.hk', '2318.hk', '3988.hk', '0998.hk', '1398.hk', '3968.hk','0981.hk', '0005.hk']
    # 腾讯，平安，中银，中信，工商，招商，中芯国际，汇丰
    stocktickers = ' '.join(stocklist)

    data = yf.download(tickers = stocktickers, start="2010-01-01", end="2019-12-31")
    data = data['Close']
    columnchange = []
    for stock in data.columns:
        name = stock+'change'
        columnchange.append(name) 
        data[name] = data[stock] - data[stock].shift(1)

    CorrDict = {}
    for i in columnchange :
        for j in columnchange :
            if i != j  and (i,j) not in CorrDict:
                CorrDict[(i,j)] = data[i].corr(data[j])
    pair = list(max(CorrDict))
    pair.append(pair[0][:7])
    pair.append(pair[1][:7]) 
    dataremain = data[pair] 

    from sklearn import linear_model     
    import numpy as np
    model = linear_model.LinearRegression()
    model.fit(dataremain[pair[0]][1:-250].to_numpy().reshape(-1,1), y=dataremain[pair[1]][1:-250])  
    beta = model.coef_[0]


    dataremain['Spread'] = beta * data[pair[0]] - data[pair[1]] 
    Spreadmean = dataremain['Spread'].mean()
    Spreadstd = dataremain['Spread'].std()
    dataremain['Z-score'] = (dataremain['Spread'] - Spreadmean)/Spreadstd

    envs = PairtradingEnv(stock1 = dataremain[pair[2]][:-250], stock2 = dataremain[pair[3]][:-250])
    eval_envs = PairtradingEnv(stock1 = dataremain[pair[2]][-250:], stock2 = dataremain[pair[3]][-250:])

    baseline_config = baselineConfig(mean=Spreadmean, std=Spreadstd, beta=beta) 
    baseline_trainer = baseline(env = envs, config = baseline_config)

    baseline_eval = baseline(env = eval_envs, config = baseline_config)

    test = env_name == "CartPole-v0"
    frame_stack = args.input_length if not test else 1

    # Setup trainer
    if algo == "PPO":
        trainer = PPOTrainer(envs, config, frame_stack, _test=test)
    else:
        trainer = A2CTrainer(envs, config, frame_stack, _test=test)
    
        
    # Create a placeholder tensor to help stack frames in 2nd dimension
    # That is turn the observation from shape [num_envs, 1, 84, 84] to
    # [num_envs, 4, 84, 84].
    frame_stack_tensor = FrameStackTensor(
        num_envs, envs.observation_space.shape, frame_stack, config.device) # envs.observation_space.shape: 1,42,42

    # Setup some stats helpers
    episode_rewards = np.zeros([num_envs, 1], dtype=np.float)
    total_episodes = total_steps = iteration = 0
    reward_recorder = deque(maxlen=100)
    episode_length_recorder = deque(maxlen=100)
    episode_values = deque(maxlen=100)
    sample_timer = Timer()
    process_timer = Timer()
    update_timer = Timer()
    total_timer = Timer()
    progress = []
    evaluate_stat = {}

    # Start training
    print("Start training!")
    while True:  # Break when total_steps exceeds maximum value
        # ===== Sample Data =====
        # episode_values = []
        episode_rewards = np.zeros([num_envs, 1], dtype=np.float)
        for env_id in range(num_envs):
            obs = envs.reset() # obs.shape: 15,1,42,42
            frame_stack_tensor.update(obs, env_id)
            trainer.rollouts.observations[0, env_id].copy_(frame_stack_tensor.get(env_id)) #trainer.rollouts.observations.shape: torch.Size([201, 15, 4, 42, 42])

            with sample_timer:
                for index in range(config.num_steps):
                    # Get action
                    # [TODO] Get the action
                    # Hint:
                    #   1. Remember to disable gradient computing
                    #   2. trainer.rollouts is a storage containing all data
                    #   3. What observation is needed for trainer.compute_action?
                    with torch.no_grad():
                        values, actions_cash, action_log_prob_cash, actions_beta, action_log_prob_beta = trainer.compute_action(trainer.rollouts.observations[index, env_id])

                    act = baseline_trainer.compute_action(actions_cash.view(-1), actions_beta.view(-1))

                    cpu_actions = act

                    # Step the environment
                    # (Check step_envs function, you need to implement it)
                    obs, reward, done, masks, total_episodes, \
                    total_steps, episode_rewards, episode_values = step_envs(
                        cpu_actions, envs, env_id, episode_rewards, episode_values, frame_stack_tensor,
                        reward_recorder, episode_length_recorder, total_steps,
                        total_episodes, config.device, test)

                    rewards = torch.from_numpy(
                        np.array(reward).astype(np.float32)).view(-1).to(config.device)
                    # Store samples
                    trainer.rollouts.insert(
                        frame_stack_tensor.get(env_id), actions_cash.view(-1),
                        action_log_prob_cash.view(-1), actions_beta.view(-1), action_log_prob_beta.view(-1), values.view(-1), rewards, masks.view(-1), env_id)

        # ===== Process Samples =====
        with process_timer:
            with torch.no_grad():
                next_value = trainer.compute_values(
                    trainer.rollouts.observations[-1])
            trainer.rollouts.compute_returns(next_value, config.GAMMA)

        # ===== Update Policy =====
        with update_timer:
            policy_loss, value_loss, dist_entropy, total_loss = \
                trainer.update(trainer.rollouts)
            trainer.rollouts.after_update()

            # Add training statistics to tensorboard log file 
            writer.add_scalar('train_policy_loss', policy_loss, iteration)
            writer.add_scalar('train_value_loss', value_loss, iteration)
            writer.add_scalar('train_dist_entropy', dist_entropy, iteration)
            writer.add_scalar('train_total_loss', total_loss, iteration)
            writer.add_scalar('train_episode_rewards', np.mean(episode_rewards), iteration)
            writer.add_scalar('train_episode_values', np.array(episode_values).mean(), iteration)

        # ===== Evaluate Current Policy =====
        if iteration % config.eval_freq == 0:
            eval_timer = Timer()
            evaluate_rewards, evaluate_lengths, evaluate_values = evaluate( 
                trainer, eval_envs, baseline_eval, frame_stack, 5)
            evaluate_stat = summary(evaluate_rewards, "episode_reward")
            if evaluate_lengths:
                evaluate_stat.update(
                    summary(evaluate_lengths, "episode_length"))
            evaluate_stat.update(dict(
                win_rate=float(
                    sum(np.array(evaluate_rewards) >= 0) / len(
                        evaluate_rewards)),
                evaluate_time=eval_timer.now,
                evaluate_iteration=iteration,
                evaluate_values = float(np.array(evaluate_values).mean())
            ))

            # Add evaluation statistics to tensorboard log file 
            writer.add_scalar('eval_episode_rewards', np.array(evaluate_rewards).mean(), iteration//config.eval_freq)
            writer.add_scalar('eval_episode_values', np.array(evaluate_values).mean(), iteration//config.eval_freq)


        # ===== Log information =====
        if iteration % config.log_freq == 0:
            stats = dict(
                log_dir=log_dir,
                frame_per_second=int(total_steps / total_timer.now),
                training_episode_reward=summary(reward_recorder,
                                                "episode_reward"),
                training_episode_values=summary(episode_values,
                                                "episode_value"), 
                training_episode_length=summary(episode_length_recorder,
                                                "episode_length"),               
                evaluate_stats=evaluate_stat,
                learning_stats=dict(
                    policy_loss=policy_loss,
                    entropy=dist_entropy,
                    value_loss=value_loss,
                    total_loss=total_loss
                ),
                total_steps=total_steps,
                total_episodes=total_episodes,
                time_stats=dict(
                    sample_time=sample_timer.avg,
                    process_time=process_timer.avg,
                    update_time=update_timer.avg,
                    total_time=total_timer.now,
                    episode_time=sample_timer.avg + process_timer.avg +
                                 update_timer.avg
                ),
                iteration=iteration
            )

            progress.append(stats)
            pretty_print({
                "===== {} Training Iteration {} =====".format(
                    algo, iteration): stats
            })

        if iteration % config.save_freq == 0:
            trainer_path = trainer.save_w(log_dir, "iter{}".format(iteration))
            progress_path = save_progress(log_dir, progress)
            print("Saved trainer state at <{}>. Saved progress at <{}>.".format(
                trainer_path, progress_path
            ))

        if iteration >=  args.max_steps:
            break

        iteration += 1

    trainer.save_w(log_dir, "final")
    envs.close()


if __name__ == '__main__':
    train(args)
