from umshini import LLM_GAMES
from umshini.envs import make_test_env

from umshini_server.house_bots.LLM.langchain_agents.content_moderation_agents import (
    ContentChatMultiRoleAgent,
    ContentCompletionMultiRoleAgent,
    RuleSimplificationChatContentAttacker,
    SimpleChatContentAttacker,
    SimpleChatContentDefender,
    SimpleCompletionContentAttacker,
    SimpleCompletionContentDefender,
)
from umshini_server.house_bots.LLM.langchain_agents.debate_agents import (
    SimpleChatDebateAgent,
    SimpleCompletionDebateAgent,
    StructuredChatDebateAgent,
    StructuredCompletionDebateAgent,
)
from umshini_server.house_bots.LLM.langchain_agents.deception_agents import (
    DeceptionMultiRoleAgent,
    PresidentDeceptionAttacker,
    SimpleDeceptionAttacker,
    SimpleDeceptionDefender,
)


class BaseHouseBotChat:
    def __init__(self, env_name, llm):
        env, is_turn_based = make_test_env(env_id=env_name)
        if env_name == "debate":
            self.langchain_agents = {
                env.possible_agents[0]: SimpleChatDebateAgent(llm=llm),
                env.possible_agents[1]: StructuredChatDebateAgent(llm=llm),
            }
        elif env_name == "content_moderation":
            self.langchain_agents = {
                env.possible_agents[0]: ContentChatMultiRoleAgent(
                    RuleSimplificationChatContentAttacker(llm=llm),
                    SimpleChatContentDefender(llm=llm),
                ),
                env.possible_agents[1]: ContentChatMultiRoleAgent(
                    SimpleChatContentAttacker(llm=llm),
                    SimpleChatContentDefender(llm=llm),
                ),
            }
        elif env_name == "deception":
            self.langchain_agents = {
                env.possible_agents[0]: DeceptionMultiRoleAgent(
                    PresidentDeceptionAttacker(llm=llm),
                    SimpleDeceptionDefender(llm=llm),
                ),
                env.possible_agents[1]: DeceptionMultiRoleAgent(
                    SimpleDeceptionAttacker(llm=llm), SimpleDeceptionDefender(llm=llm)
                ),
            }
        else:
            raise Exception(
                f"Environment name not found: {env_name}. Options: {LLM_GAMES}"
            )


# TODO: maybe remove all of this because it's simpler to just have the whole prompts and roles and such be handled directly in the policy
class BaseHouseBotCompletion:
    def __init__(self, env_name, llm):
        env, is_turn_based = make_test_env(env_id=env_name)
        if env_name == "debate":
            self.langchain_agents = {
                env.possible_agents[0]: SimpleCompletionDebateAgent(llm=llm),
                env.possible_agents[1]: StructuredCompletionDebateAgent(llm=llm),
            }
        # TODO: refactor content and deception Agent classes to have completion versions, current ones are hard coded to Chat models, System messages and such are not used by completion models AFAIK

        elif env_name == "content_moderation":
            self.langchain_agents = {
                env.possible_agents[0]: ContentCompletionMultiRoleAgent(
                    SimpleCompletionContentAttacker(llm=llm),
                    SimpleCompletionContentDefender(llm=llm),
                ),
                env.possible_agents[1]: ContentCompletionMultiRoleAgent(
                    SimpleCompletionContentAttacker(llm=llm),
                    SimpleCompletionContentDefender(llm=llm),
                ),
            }
        elif env_name == "deception":
            self.langchain_agents = {
                env.possible_agents[0]: DeceptionMultiRoleAgent(
                    PresidentDeceptionAttacker(llm=llm),
                    SimpleDeceptionDefender(llm=llm),
                ),
                env.possible_agents[1]: DeceptionMultiRoleAgent(
                    SimpleDeceptionAttacker(llm=llm), SimpleDeceptionDefender(llm=llm)
                ),
            }
        else:
            raise Exception(
                f"Environment name not found: {env_name}. Options: {LLM_GAMES}"
            )
