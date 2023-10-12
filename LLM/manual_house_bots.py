import sys
from multiprocessing import Pool

from umshini import connect

house_keys = {
    "debate": [
        {
            "bot_name": "llm_aligning_ant_bot",
            "user_key": "aligning_ant",
        },
        {
            "bot_name": "llm_abject_antelope_bot",
            "user_key": "abject_antelope",
        },
        {
            "bot_name": "llm_avid_aardvark_bot",
            "user_key": "avid_aardvark",
        },
        {
            "bot_name": "llm_humorous_bot",
            "user_key": "culturedstork",
        },
    ],
    "content_moderation": [
        {
            "bot_name": "llm_garrulous_goat_bot",
            "user_key": "garrulous_goat",
        },
        {
            "bot_name": "llm_gigantic_gecko_bot",
            "user_key": "gigantic_gecko",
        },
        {
            "bot_name": "llm_goading_gerbil_bot",
            "user_key": "goading_gerbil",
        },
    ],
    "deception": [
        {
            "bot_name": "llm_tentative_tapir_bot",
            "user_key": "tentative_tapir",
        },
        {
            "bot_name": "llm_top_hat_tuna_bot",
            "user_key": "top_hat_tuna",
        },
        {
            "bot_name": "llm_timid_tanuki_bot",
            "user_key": "timid_tanuki",
        },
    ],
}


def my_pol(observation, reward, termination, truncation, info):
    if termination or truncation:
        action = None
    else:
        response = "I AM THE WINNER"
        return response


def test_manual_house_bots(env_name: str, num_players: int, testing: bool):
    master_params = []

    for i in range(num_players):
        master_params.append(
            (
                env_name,
                house_keys[env_name][i]["bot_name"],
                house_keys[env_name][i]["user_key"],
                my_pol,
                False,
                testing,
            )
        )

    with Pool(num_players) as pool:
        pool.starmap(connect, master_params)


if __name__ == "__main__":
    env_name = sys.argv[1]
    num_players = int(sys.argv[2])
    testing = bool(sys.argv[3])
    test_manual_house_bots(env_name, num_players, testing)
