from __future__ import annotations

import sys
import os
from pathlib import Path
base_path = Path(os.path.dirname(__file__)).parent
sys.path.append(str(base_path))

import argparse
import uuid

import hockey.hockey_env as h_env
import numpy as np


from comprl.client import Agent, launch_client

from utils.actors import Actor
from utils.factory import construct_actor



class HockeyNeuralAgent(Agent):
    def __init__(self, actor: Actor) -> None:
        super().__init__()

        self.actor = actor

    def get_step(self, observation: list[float]) -> list[float]:

        action = self.actor.act(np.array(observation)).tolist()
        return action

    def on_start_game(self, game_id) -> None:
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


# Function to initialize the agent.  This function is used with `launch_client` below,
# to lauch the client and connect to the server.
def initialize_agent(agent_args: list[str]) -> Agent:
    # Use argparse to parse the arguments given in `agent_args`.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        choices=["td3", "sac", "crq"],
        default="crq",
        help="Which agent to use.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to model's weight")
    args = parser.parse_args(agent_args)

    # Initialize the agent based on the arguments.
    
    actor = construct_actor(args.agent, args.model, h_env.HockeyEnv_BasicOpponent())
    agent: Agent = HockeyNeuralAgent(actor)

    # And finally return the agent.
    return agent


def main() -> None:
    launch_client(initialize_agent)


if __name__ == "__main__":
    main()
