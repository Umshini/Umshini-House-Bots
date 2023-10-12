import argparse
import sys

from umshini import LLM_GAMES

from umshini_server.house_bots.LLM.hf_endpoints_house_bots import (
    ENDPOINT_URLS,
    test_hf_endpoint,
)

MODEL_NAME = "llama2"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script with argparse")
    parser.add_argument(
        "env_name",
        type=str,
        default="debate",
        nargs="?",
        help="Name of the environment",
        choices=LLM_GAMES,
    )
    parser.add_argument(
        "num_players", type=int, default=1, nargs="?", help="Number of players"
    )
    parser.add_argument(
        "testing", type=bool, default=True, nargs="?", help="Enable local testing mode"
    )
    parser.add_argument(
        "mock_llm",
        type=bool,
        default=False,
        nargs="?",
        help="Use a mock LLM, for testing LangChain logic.",
    )

    args = parser.parse_args()

    env_name = args.env_name
    num_players = args.num_players
    testing = args.testing
    mock_llm = args.mock_llm

    test_hf_endpoint(
        env_name=env_name,
        num_players=num_players,
        testing=testing,
        MODEL_NAME=MODEL_NAME,
    )
