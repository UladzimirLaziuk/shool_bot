from langchain import LLMChain, PromptTemplate
from langchain.chat_models.base import BaseChatModel
from langchain.llms import BaseLLM

from babyhelpergpt.logger import time_logger
from babyhelpergpt.prompts import (HELPER_AGENT_INCEPTION_PROMPT,
                              STAGE_ANALYZER_INCEPTION_PROMPT, HELPER_AGENT_INCEPTION_PROMPT_NEW)


class StageAnalyzerChain(LLMChain):
    """Chain to analyze which conversation stage should the conversation move into."""

    @classmethod
    @time_logger
    def from_llm(cls, llm: BaseChatModel, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        stage_analyzer_inception_prompt_template = STAGE_ANALYZER_INCEPTION_PROMPT
        prompt = PromptTemplate(
            template=stage_analyzer_inception_prompt_template,
            input_variables=[
                "conversation_history",
                "conversation_stage_id",
                "conversation_stages",
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class HelperConversationChain(LLMChain):
    """Chain to generate the next utterance for the conversation."""

    @classmethod
    @time_logger
    def from_llm(
        cls,
        llm: BaseChatModel,
        verbose: bool = True,
        use_custom_prompt: bool = False,
        custom_prompt: str = "You are an AI agent, sell me this pencil",
    ) -> LLMChain:
        """Get the response parser."""
        if use_custom_prompt:
            helper_agent_inception_prompt = custom_prompt
            prompt = PromptTemplate(
                template=helper_agent_inception_prompt,
                input_variables=[
                    "helper_person_name",
                    "helper_person_role",
                    "company_name",
                    "company_business",
                    "company_values",
                    # "conversation_purpose",
                    "conversation_type",
                    "conversation_history",
                    "conversation_stage",
                ],
            )
        else:
            helper_agent_inception_prompt = HELPER_AGENT_INCEPTION_PROMPT_NEW
            prompt = PromptTemplate(
                template=helper_agent_inception_prompt,
                input_variables=[
                    "helper_person_name",
                    "helper_person_role",
                    "company_name",
                    "company_business",
                    "company_values",
                    # "conversation_purpose",
                    "conversation_type",
                    "conversation_history",
                    "conversation_stage",
                ],
            )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
