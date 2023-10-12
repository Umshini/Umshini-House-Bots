import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pytest
import torch
import umshini
from pettingzoo.classic import connect_four_v3, go_v5, texas_holdem_no_limit_v6
from torch import nn

house_keys = {
    "connect_four_v3": [
        {
            "path": "cleanrl_connect_four_1/cleanrl_connect_four.cleanrl_model",
            "bot_name": "cleanrl-dqn-connect-1",
            "user_key": "cleanrl-1",
        },
        {
            "path": "cleanrl_connect_four_2/cleanrl_connect_four.cleanrl_model",
            "bot_name": "cleanrl-dqn-connect-2",
            "user_key": "cleanrl-2",
        },
        {
            "path": "cleanrl_connect_four_3/cleanrl_connect_four.cleanrl_model",
            "bot_name": "cleanrl-dqn-connect-3",
            "user_key": "cleanrl-3",
        },
        {
            "path": "cleanrl_connect_four_4/cleanrl_connect_four.cleanrl_model",
            "bot_name": "cleanrl-dqn-connect-4",
            "user_key": "cleanrl-4",
        },
    ],
    "go_v5": [
        {
            "path": "cleanrl_go_1/cleanrl_go.cleanrl_model",
            "bot_name": "cleanrl-dqn-go-1",
            "user_key": "cleanrl-1",
        },
        {
            "path": "cleanrl_go_2/cleanrl_go.cleanrl_model",
            "bot_name": "cleanrl-dqn-go-2",
            "user_key": "cleanrl-2",
        },
        {
            "path": "cleanrl_go_3/cleanrl_go.cleanrl_model",
            "bot_name": "cleanrl-dqn-go-3",
            "user_key": "cleanrl-3",
        },
        {
            "path": "cleanrl_go_4/cleanrl_go.cleanrl_model",
            "bot_name": "cleanrl-dqn-go-4",
            "user_key": "cleanrl-4",
        },
    ],
    "texas_holdem_no_limit_v6": [
        {
            "path": "cleanrl_texas_holdem_1/cleanrl_texas_holdem.cleanrl_model",
            "bot_name": "cleanrl-dqn-texas-1",
            "user_key": "cleanrl-1",
        },
        {
            "path": "cleanrl_texas_holdem_2/cleanrl_texas_holdem.cleanrl_model",
            "bot_name": "cleanrl-dqn-texas-2",
            "user_key": "cleanrl-2",
        },
        {
            "path": "cleanrl_texas_holdem_3/cleanrl_texas_holdem.cleanrl_model",
            "bot_name": "cleanrl-dqn-texas-3",
            "user_key": "cleanrl-3",
        },
        {
            "path": "cleanrl_texas_holdem_4/cleanrl_texas_holdem.cleanrl_model",
            "bot_name": "cleanrl-dqn-texas-4",
            "user_key": "cleanrl-4",
        },
    ],
}


class ConnectQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(84, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, env.action_space("player_0").n),
        )

    def forward(self, x):
        return self.network(x / 255.0)


class GoQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6137, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, env.action_space("black_0").n),
        )

    def forward(self, x):
        return self.network(x / 255.0)


class TexasQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(54, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, env.action_space("player_0").n),
        )

    def forward(self, x):
        return self.network(x / 255.0)


# wrapping the policy within a global class to get around multiprocessing issues
class RLHousePolicy:
    def __init__(self, model):
        self.model = model

    def my_pol(self, obs, rew, term, trunc, info):
        if term or trunc:
            action = None
        else:
            obs_in = torch.unsqueeze(torch.Tensor(obs["observation"]), 0)
            q_values = self.model(obs_in)
            action_mask = torch.Tensor((obs["action_mask"] - 1) * 100)
            q_values = q_values + action_mask
            action = torch.squeeze(torch.argmax(q_values, dim=1)).cpu().numpy()
        return action


def test_house_bots(
    env_name: str = "connect_four_v3", num_players: int = 4, testing: bool = True
):
    num_players = int(num_players)
    testing = bool(testing)
    master_params = []
    print("Starting housebots...")
    for i in range(num_players):
        if env_name == "connect_four_v3":
            model = ConnectQNetwork(connect_four_v3.env())
        elif env_name == "go_v5":
            model = GoQNetwork(go_v5.env())
        else:
            model = TexasQNetwork(texas_holdem_no_limit_v6.env())

        print("Model initialized, loading parameters...")
        housebots_dir = Path(__file__).absolute().parent
        path = os.path.join(housebots_dir, house_keys[env_name][i]["path"])
        model.load_state_dict(torch.load(path))
        print("Loaded parameters")

        my_pol = RLHousePolicy(model).my_pol

        master_params.append(
            (
                env_name,
                house_keys[env_name][i]["bot_name"],
                house_keys[env_name][i]["user_key"],
                my_pol,
                testing,
            )
        )

    print("Loaded housebots")
    with Pool(num_players) as pool:
        pool.starmap(run_player, master_params)


def run_player(
    env_name: str, botname: str, userkey: str, policy: callable, testing: bool
):
    umshini.connect(
        env_name,
        botname,
        userkey,
        policy,
        debug=False,
        testing=testing,
    )


if __name__ == "__main__":
    env_name = sys.argv[1]
    num_players = int(sys.argv[2])
    testing = bool(sys.argv[3])

    test_house_bots(env_name, num_players, testing)
