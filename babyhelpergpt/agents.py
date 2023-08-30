from copy import deepcopy
from typing import Any, Callable, Dict, List, Union, Optional

from langchain import LLMChain
from langchain.agents import AgentExecutor, LLMSingleActionAgent
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains import RetrievalQA
from langchain.chains.base import Chain
from langchain.chat_models.base import BaseChatModel
from langchain.llms import BaseLLM
from langchain.llms.base import create_base_retry_decorator
from pydantic import Field

from babyhelpergpt.chains import HelperConversationChain, StageAnalyzerChain
from babyhelpergpt.logger import time_logger
from babyhelpergpt.parsers import HelperConvoOutputParser
from babyhelpergpt.prompts import HELPER_AGENT_TOOLS_PROMPT
from babyhelpergpt.stages import CONVERSATION_STAGES
from babyhelpergpt.templates import CustomPromptTemplateForTools
from babyhelpergpt.tools import get_tools, setup_knowledge_base


def _create_retry_decorator(llm: Any) -> Callable[[Any], Any]:
    import openai

    errors = [
        openai.error.Timeout,
        openai.error.APIError,
        openai.error.APIConnectionError,
        openai.error.RateLimitError,
        openai.error.ServiceUnavailableError,
    ]
    return create_base_retry_decorator(error_types=errors, max_retries=llm.max_retries)


class BabyChatGPT(Chain):
    """Controller model for the Sales Agent."""

    conversation_history: List[str] = []
    conversation_stage_id: str = "1"
    current_conversation_stage: str = CONVERSATION_STAGES.get("1")
    stage_analyzer_chain: StageAnalyzerChain = Field(...)
    helper_agent_executor: Union[AgentExecutor, None] = Field(...)#sales_agent_executor
    knowledge_base: Union[RetrievalQA, None] = Field(...)
    helper_conversation_utterance_chain: HelperConversationChain = Field(...)#sales_conversation_utterance_chain
    conversation_stage_dict: Dict = CONVERSATION_STAGES

    use_tools: bool = False
    helper_person_name: str = "John"
    helper_person_role: str = "A kind cheerful helper for children"
    company_name: str = "Cloud Teacher"
    company_business: str = "Cloud Teacher - мы помогаем детям играть в различные игры, чтобы обучение проходило легко и весело. "
    company_values: str = "Наша миссия в Cloud Teacher - помочь детям лучше учиться, предлагая увлекательные игры для повышения эффективности обучения. Мы считаем, что веселые игры - это ключ к хорошей успеваемости и общему усвоению урока, и стремимся помочь детям достичь оптимального уровня обучения, составляя игры для выполнения домашних заданий."
    # conversation_purpose: str = "find out whether they are looking to achieve better sleep via buying a premier mattress."
    conversation_type: str = "write"

    def retrieve_conversation_stage(self, key):
        return self.conversation_stage_dict.get(key, "1")

    @property
    def input_keys(self) -> List[str]:
        return []

    @property
    def output_keys(self) -> List[str]:
        return []

    @time_logger
    def seed_agent(self):
        # Step 1: seed the conversation
        self.current_conversation_stage = self.retrieve_conversation_stage("1")
        self.conversation_history = []

    @time_logger
    def determine_conversation_stage(self):
        self.conversation_stage_id = self.stage_analyzer_chain.run(
            conversation_history="\n".join(self.conversation_history).rstrip("\n"),
            conversation_stage_id=self.conversation_stage_id,
            conversation_stages="\n".join(
                [
                    str(key) + ": " + str(value)
                    for key, value in CONVERSATION_STAGES.items()
                ]
            ),
        )

        print(f"Conversation Stage ID: {self.conversation_stage_id}")
        self.current_conversation_stage = self.retrieve_conversation_stage(
            self.conversation_stage_id
        )

        print(f"Conversation Stage: {self.current_conversation_stage}")

    def human_step(self, human_input):
        # process human input
        human_input = "Child: " + human_input + " <END_OF_TURN>"
        self.conversation_history.append(human_input)

    @time_logger
    def step(
        self, return_streaming_generator: bool = False, model_name="gpt-3.5-turbo-0613"
    ):
        """
        Args:
            return_streaming_generator (bool): whether or not return
            streaming generator object to manipulate streaming chunks in downstream applications.
        """
        if not return_streaming_generator:
            self._call(inputs={})
        else:
            return self._streaming_generator(model_name=model_name)

    @time_logger
    async def astep(
        self, return_streaming_generator: bool = False, model_name="gpt-3.5-turbo-0613"
    ):
        """
        Args:
            return_streaming_generator (bool): whether or not return
            streaming generator object to manipulate streaming chunks in downstream applications.
        """
        if not return_streaming_generator:
            await self._acall(inputs={})
        else:
            return self._astreaming_generator(model_name=model_name)

    @time_logger
    def acall(self, *args, **kwargs):
        raise NotImplementedError("This method has not been implemented yet.")

    @time_logger
    def _prep_messages(self):
        """
        Helper function to prepare messages to be passed to a streaming generator.
        """
        prompt = self.helper_conversation_utterance_chain.prep_prompts(
            [
                dict(
                    conversation_stage=self.current_conversation_stage,
                    conversation_history="\n".join(self.conversation_history),
                    helper_person_name=self.helper_person_name,
                    helper_person_role=self.helper_person_role,
                    company_name=self.company_name,
                    company_business=self.company_business,
                    company_values=self.company_values,
                    # conversation_purpose=self.conversation_purpose,
                    conversation_type=self.conversation_type,
                )
            ]
        )

        inception_messages = prompt[0][0].to_messages()

        message_dict = {"role": "system", "content": inception_messages[0].content}

        if self.helper_conversation_utterance_chain.verbose:
            print("\033[92m" + inception_messages[0].content + "\033[0m")
        return [message_dict]

    @time_logger
    def _streaming_generator(self, model_name="gpt-3.5-turbo-0613"):
        """
        Sometimes, the sales agent wants to take an action before the full LLM output is available.
        For instance, if we want to do text to speech on the partial LLM output.

        This function returns a streaming generator which can manipulate partial output from an LLM
        in-flight of the generation.

        Example:

        >> streaming_generator = self._streaming_generator()
        # Now I can loop through the output in chunks:
        >> for chunk in streaming_generator:
        Out: Chunk 1, Chunk 2, ... etc.
        See: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb
        """

        messages = self._prep_messages()

        return self.helper_conversation_utterance_chain.llm.completion_with_retry(
            messages=messages,
            stop="<END_OF_TURN>",
            stream=True,
            model=model_name,
        )

    async def acompletion_with_retry(self, llm: Any, **kwargs: Any) -> Any:
        """Use tenacity to retry the async completion call."""
        retry_decorator = _create_retry_decorator(llm)

        @retry_decorator
        async def _completion_with_retry(**kwargs: Any) -> Any:
            # Use OpenAI's async api https://github.com/openai/openai-python#async-api
            return await llm.client.acreate(**kwargs)

        return await _completion_with_retry(**kwargs)

    async def _astreaming_generator(self, model_name="gpt-3.5-turbo-0613"):
        """
        Asynchronous generator to reduce I/O blocking when dealing with multiple
        clients simultaneously.

        Sometimes, the sales agent wants to take an action before the full LLM output is available.
        For instance, if we want to do text to speech on the partial LLM output.

        This function returns a streaming generator which can manipulate partial output from an LLM
        in-flight of the generation.

        Example:

        >> streaming_generator = self._astreaming_generator()
        # Now I can loop through the output in chunks:
        >> async for chunk in streaming_generator:
            await chunk ...
        Out: Chunk 1, Chunk 2, ... etc.
        See: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb
        """

        messages = self._prep_messages()

        return await self.acompletion_with_retry(
            llm=self.helper_conversation_utterance_chain.llm,
            messages=messages,
            stop="<END_OF_TURN>",
            stream=True,
            model=model_name,
        )

    def _call(self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None   ) -> Dict[str, Any]:
        """Run one step of the sales agent."""

        # Generate agent's utterance
        # if use tools
        if self.use_tools:
            ai_message = self.helper_agent_executor.run(
                input="",
                conversation_stage=self.current_conversation_stage,
                conversation_history="\n".join(self.conversation_history),
                helper_person_name=self.helper_person_name,
                helper_person_role=self.helper_person_role,
                company_name=self.company_name,
                company_business=self.company_business,
                company_values=self.company_values,
                # conversation_purpose=self.conversation_purpose,
                conversation_type=self.conversation_type,
            )

        else:
            # else
            ai_message = self.helper_conversation_utterance_chain.run(
                conversation_stage=self.current_conversation_stage,
                conversation_history="\n".join(self.conversation_history),
                helper_person_name=self.helper_person_name,
                helper_person_role=self.helper_person_role,
                company_name=self.company_name,
                company_business=self.company_business,
                company_values=self.company_values,
                # conversation_purpose=self.conversation_purpose,
                conversation_type=self.conversation_type,
            )

        # Add agent's response to conversation history
        agent_name = self.helper_person_name
        ai_message = agent_name + ": " + ai_message
        if "<END_OF_TURN>" not in ai_message:
            ai_message += " <END_OF_TURN>"
        self.conversation_history.append(ai_message)
        print(ai_message.replace("<END_OF_TURN>", ""))
        return {}
    async def _acall(self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None   ) -> Dict[str, Any]:
        """Run one step of the sales agent."""

        # Generate agent's utterance
        # if use tools
        if self.use_tools:
            ai_message = self.helper_agent_executor.run(
                input="",
                conversation_stage=self.current_conversation_stage,
                conversation_history="\n".join(self.conversation_history),
                helper_person_name=self.helper_person_name,
                helper_person_role=self.helper_person_role,
                company_name=self.company_name,
                company_business=self.company_business,
                company_values=self.company_values,
                # conversation_purpose=self.conversation_purpose,
                conversation_type=self.conversation_type,
            )

        else:
            # else
            ai_message = await self.helper_conversation_utterance_chain.arun(
                conversation_stage=self.current_conversation_stage,
                conversation_history="\n".join(self.conversation_history),
                helper_person_name=self.helper_person_name,
                helper_person_role=self.helper_person_role,
                company_name=self.company_name,
                company_business=self.company_business,
                company_values=self.company_values,
                # conversation_purpose=self.conversation_purpose,
                conversation_type=self.conversation_type,
            )

        # Add agent's response to conversation history
        agent_name = self.helper_person_name
        ai_message = agent_name + ": " + ai_message
        if "<END_OF_TURN>" not in ai_message:
            ai_message += " <END_OF_TURN>"
        self.conversation_history.append(ai_message)
        print(ai_message.replace("<END_OF_TURN>", ""))
        return {}
    @classmethod
    @time_logger
    def from_llm(cls, llm: BaseLLM, verbose: bool = False, **kwargs) -> "BabyChatGPT":
        """Initialize the SalesGPT Controller."""
        stage_analyzer_chain = StageAnalyzerChain.from_llm(llm, verbose=verbose)
        if (
            "use_custom_prompt" in kwargs.keys()
            and kwargs["use_custom_prompt"] == "True"
        ):
            use_custom_prompt = deepcopy(kwargs["use_custom_prompt"])
            custom_prompt = deepcopy(kwargs["custom_prompt"])

            # clean up
            del kwargs["use_custom_prompt"]
            del kwargs["custom_prompt"]

            helper_conversation_utterance_chain = HelperConversationChain.from_llm(
                llm,
                verbose=verbose,
                use_custom_prompt=use_custom_prompt,
                custom_prompt=custom_prompt,
            )

        else:
            helper_conversation_utterance_chain = HelperConversationChain.from_llm(
                llm, verbose=verbose
            )

        if "use_tools" in kwargs.keys() and kwargs["use_tools"] == True:
            # set up agent with tools
            product_catalog = kwargs["product_catalog"]
            knowledge_base = setup_knowledge_base(product_catalog)
            tools = get_tools(knowledge_base)

            prompt = CustomPromptTemplateForTools(
                template=HELPER_AGENT_TOOLS_PROMPT,
                tools_getter=lambda x: tools,
                # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
                # This includes the `intermediate_steps` variable because that is needed
                input_variables=[
                    "input",
                    "intermediate_steps",
                    "helper_person_name",
                    "helper_person_role",
                    "company_name",
                    "company_business",
                    "company_values",
                    # "conversation_purpose",
                    "conversation_type",
                    "conversation_history",
                ],
            )
            llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)

            tool_names = [tool.name for tool in tools]

            # WARNING: this output parser is NOT reliable yet
            ## It makes assumptions about output from LLM which can break and throw an error
            output_parser = HelperConvoOutputParser(ai_prefix=kwargs["helper_person_name"])

            helper_agent_with_tools = LLMSingleActionAgent(
                llm_chain=llm_chain,
                output_parser=output_parser,
                stop=["\nObservation:"],
                allowed_tools=tool_names,
            )

            helper_agent_executor = AgentExecutor.from_agent_and_tools(
                agent=helper_agent_with_tools, tools=tools, verbose=verbose
            )
        else:
            helper_agent_executor = None
            knowledge_base = None

        return cls(
            stage_analyzer_chain=stage_analyzer_chain,
            helper_conversation_utterance_chain=helper_conversation_utterance_chain,
            helper_agent_executor=helper_agent_executor,
            knowledge_base=knowledge_base,
            verbose=verbose,
            **kwargs,
        )

    @classmethod
    @time_logger
    def from_llm_my(cls, llm: BaseChatModel, llm_analizer: BaseChatModel, verbose: bool = False, **kwargs) -> "BabyChatGPT":
        """Initialize the SalesGPT Controller."""
        stage_analyzer_chain = StageAnalyzerChain.from_llm(llm=llm_analizer, verbose=verbose)
        if (
                "use_custom_prompt" in kwargs.keys()
                and kwargs["use_custom_prompt"] == "True"
        ):
            use_custom_prompt = deepcopy(kwargs["use_custom_prompt"])
            custom_prompt = deepcopy(kwargs["custom_prompt"])

            # clean up
            del kwargs["use_custom_prompt"]
            del kwargs["custom_prompt"]

            helper_conversation_utterance_chain = HelperConversationChain.from_llm(
                llm,
                verbose=verbose,
                use_custom_prompt=use_custom_prompt,
                custom_prompt=custom_prompt,
            )

        else:
            helper_conversation_utterance_chain = HelperConversationChain.from_llm(
                llm, verbose=verbose
            )

        if "use_tools" in kwargs.keys() and kwargs["use_tools"] == True:
            # set up agent with tools
            product_catalog = kwargs["product_catalog"]
            knowledge_base = setup_knowledge_base(product_catalog)
            tools = get_tools(knowledge_base)

            prompt = CustomPromptTemplateForTools(
                template=HELPER_AGENT_TOOLS_PROMPT,
                tools_getter=lambda x: tools,
                # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
                # This includes the `intermediate_steps` variable because that is needed
                input_variables=[
                    "input",
                    "intermediate_steps",
                    "helper_person_name",
                    "helper_person_role",
                    "company_name",
                    "company_business",
                    "company_values",
                    # "conversation_purpose",
                    "conversation_type",
                    "conversation_history",
                ],
            )
            llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)

            tool_names = [tool.name for tool in tools]

            # WARNING: this output parser is NOT reliable yet
            ## It makes assumptions about output from LLM which can break and throw an error
            output_parser = HelperConvoOutputParser(ai_prefix=kwargs["helper_person_name"])

            helper_agent_with_tools = LLMSingleActionAgent(
                llm_chain=llm_chain,
                output_parser=output_parser,
                stop=["\nObservation:"],
                allowed_tools=tool_names,
            )

            helper_agent_executor = AgentExecutor.from_agent_and_tools(
                agent=helper_agent_with_tools, tools=tools, verbose=verbose
            )
        else:
            helper_agent_executor = None
            knowledge_base = None

        return cls(
            stage_analyzer_chain=stage_analyzer_chain,
            helper_conversation_utterance_chain=helper_conversation_utterance_chain,
            helper_agent_executor=helper_agent_executor,
            knowledge_base=knowledge_base,
            verbose=verbose,
            **kwargs,
        )
