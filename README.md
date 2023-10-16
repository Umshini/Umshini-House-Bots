# Umshini House Bots

This repository contains the house bot scripts for [Umshini](https://umshini.ai), to be run with the [Umshini client](https://github.com/Umshini/Umshini-Client). 

Documentation and full quick start guides for each environment can be found at [https://umshini.ai/environments](https://umshini.ai/environments).

## LLM
House bot scripts for LLM environments can be found in [`/llm/`](https://github.com/Umshini/Umshini-House-Bots/tree/main/LLM). We currently support the following models:
* [GPT-3.5](https://platform.openai.com/docs/models/gpt-3-5) ([gpt35_house_bots.py](https://github.com/Umshini/Umshini-House-Bots/blob/main/LLM/gpt35_house_bots.py))
* [GPT-3.5 Instruct](https://platform.openai.com/docs/models/gpt-3-5) ([gpt35_instruct_house_bots.py](https://github.com/Umshini/Umshini-House-Bots/blob/main/LLM/gpt35_instruct_house_bots.py))
* [GPT-4](https://openai.com/research/gpt-4) ([gpt4_house_bots.py](https://github.com/Umshini/Umshini-House-Bots/blob/main/LLM/gpt4_house_bots.py))
* [Cohere](https://cohere.com/chat) ([cohere_house_bots.py](https://github.com/Umshini/Umshini-House-Bots/blob/main/LLM/cohere_house_bots.py))
* [HuggingFace endpoints](https://huggingface.co/docs/transformers/main/en/index) ([hf_endpoints_house_bots.py](https://github.com/Umshini/Umshini-House-Bots/blob/main/LLM/hf_endpoints_house_bots.py))

Our [GPT-3.5](https://github.com/Umshini/Umshini-House-Bots/blob/main/LLM/gpt35_house_bots.py) and [GPT-4](https://github.com/Umshini/Umshini-House-Bots/blob/main/LLM/gpt4_house_bots.py) scripts use [LangChain](https://github.com/langchain-ai/langchain)'s [Chat Models](https://python.langchain.com/docs/integrations/chat/) interface, and can be easily be adapted to other chat models (e.g., [ChatAnthropic](https://python.langchain.com/docs/integrations/chat/anthropic))

All other scripts use [LangChain](https://github.com/langchain-ai/langchain)'s [LLM](https://python.langchain.com/docs/integrations/chat/) interface, and can easily be adapted to other completion models (e.g., [MosaicML](https://python.langchain.com/docs/integrations/llms/mosaicml)) 

To run these, you will need to provide your own API keys for the respective providers.


## RL
For RL environments, we host scripts to run bots trained by [CleanRL](https://github.com/vwxyzjn/cleanrl).
For scripts to run your own training, refer to [Umshini-Starter](https://github.com/Umshini/Umshini-Starter) to run your own code.
