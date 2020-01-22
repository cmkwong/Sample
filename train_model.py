import os

from gym import wrappers
import ptan
import numpy as np

import torch
import torch.optim as optim

from lib import environ, data, models, common, validation

from tensorboardX import SummaryWriter

BATCH_SIZE = 32
BARS_COUNT = 20
TARGET_NET_SYNC = 1000
TRAINING_DATA = ""
VAL_DATA = ""

GAMMA = 0.99

REPLAY_SIZE = 100000 # 100000
REPLAY_INITIAL = 10000 # 10000

REWARD_STEPS = 2

LEARNING_RATE = 0.0001

STATES_TO_EVALUATE = 1000
EVAL_EVERY_STEP = 1000

EPSILON_START = 1.0
EPSILON_STOP = 0.1
EPSILON_STEPS = 1000000

CHECKPOINT_EVERY_STEP = 50000
VALIDATION_EVERY_STEP = 10000 # 10000

load_net = True
load_fileName = "checkpoint-950000.data"
saves_path = "C:\\Users\\user\\python_jupyter\\book_Hands_On_Reinforcement_Learning_Pytorch\\cmk_chapter8\\2_LSTM\\checkpoint"

if __name__ == "__main__":

    device = torch.device("cuda")

    stock_data = data.read_csv(
        "C:\\Users\\user\\python_jupyter\\book_Hands_On_Reinforcement_Learning_Pytorch\\cmk_chapter8\\2_LSTM\\data\\0005.HK.csv")

    # create the training and val set
    train_set, val_set = data.split_data(stock_data, percentage=0.8)
    train_set = {"train": train_set}
    val_set = {"eval": val_set}

    env = environ.StocksEnv(train_set, bars_count=BARS_COUNT, reset_on_close=True, state_1d=False, volumes=True)
    env = wrappers.TimeLimit(env, max_episode_steps=1000)
    env_val = environ.StocksEnv(val_set, bars_count=BARS_COUNT, reset_on_close=True, state_1d=False, volumes=True)
    # env_val = wrappers.TimeLimit(env_val, max_episode_steps=1000)

    # create neural network
    net = models.SimpleLSTM(input_size=5, n_hidden=512, n_layers=2, drop_prob=0.5, actions_n=3,
                 train_on_gpu=True, batch_first=True).to(device)
    # load the network
    if load_net is True:
        with open(os.path.join(saves_path, load_fileName), "rb") as f:
            checkpoint = torch.load(f)
        net = models.SimpleLSTM(input_size=5, n_hidden=512, n_layers=2, drop_prob=0.5, actions_n=3,
                                train_on_gpu=True, batch_first=True).to(device)
        net.load_state_dict(checkpoint['state_dict'])

    tgt_net = ptan.agent.TargetNet(net)

    # create buffer
    selector = ptan.actions.EpsilonGreedyActionSelector(EPSILON_START)
    agent = ptan.agent.DQNAgent(net, selector, do_preprocess=False, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=REWARD_STEPS)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, REPLAY_SIZE)

    # create optimizer
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # main training loop
    if load_net:
        step_idx = common.find_stepidx(load_fileName, "-", "\.")
    else:
        step_idx = 0
    eval_states = None
    best_mean_val = None

    writer = SummaryWriter(comment="0005_testing")
    with common.RewardTracker(writer, np.inf, group_rewards=100) as reward_tracker:
        while True:
            step_idx += 1
            net.init_hidden(1)
            buffer.populate(1)
            selector.epsilon = max(EPSILON_STOP, EPSILON_START - step_idx / EPSILON_STEPS)

            new_rewards = exp_source.pop_rewards_steps()
            if new_rewards:
                reward_tracker.reward(new_rewards[0], step_idx, selector.epsilon)

            if len(buffer) < REPLAY_INITIAL:
                continue

            optimizer.zero_grad()
            batch = buffer.sample(BATCH_SIZE)

            # init the hidden both in network and tgt network
            net.train()
            net.zero_grad()
            net.init_hidden(BATCH_SIZE)
            tgt_net.target_model.init_hidden(BATCH_SIZE)
            loss_v = common.calc_loss(batch, net, tgt_net.target_model, GAMMA ** REWARD_STEPS, device=device)
            loss_v.backward()
            optimizer.step()

            if step_idx % TARGET_NET_SYNC == 0:
                tgt_net.sync()

            if step_idx % CHECKPOINT_EVERY_STEP == 0:
                # idx = step_idx // CHECKPOINT_EVERY_STEP
                checkpoint = {
                    "obs_space": env.observation_space.shape[0],
                    "action_n": env.action_space.n,
                    "state_dict": net.state_dict()
                }
                with open(os.path.join(saves_path,"checkpoint-%d.data" % step_idx), "wb") as f:
                    torch.save(checkpoint, f)

            if step_idx % VALIDATION_EVERY_STEP == 0:
                net.eval()
                net.init_hidden(1)
                res = validation.validation_run(env_val, net, device=device)
                for key, val in res.items():
                    writer.add_scalar(key + "_val", val, step_idx)