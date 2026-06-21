'''
    aitomia.py: the top level class that deals with chat
'''
from aitomia_agents.logger import logger
from .agent_cards import agent_cards
from .settings import settings
from typing import List
from .agent_template import BaseAgent
import uuid
from .utils import FileManager, Analysis, GlobalFlag
import time
import asyncio
class Aitomia():

    # postgresSaver need to be async initialized
    @classmethod
    async def create(cls, agent="planner_agent", config=None, cmd=True, callback=None, checkpointer=None):
        Analysis.set_start_time(time.time())
        self = cls()
        self.agent_card = agent_cards[agent]
        if config is None: config = self.thread_id_gen()
        logger.debug(f"Aitomia.create() generated config: {config}")
        if "recursion_limit" not in config.keys():
            config["recursion_limit"] = settings.recursion_limit # to-do: it should not be hardcoded here ~P.O.D.
        # If checkpointer is provided (e.g., from server), use it directly
        if checkpointer is not None:
            logger.info("Using externally provided checkpointer (shared global instance).")
            self.checkpointer = checkpointer
        elif settings.memory_saver == "postgres":
            # Postgres persistence is optional; import lazily so the in-memory
            # path works without the langgraph-checkpoint-postgres extra installed.
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            # Fallback: create instance-specific checkpointer (for standalone usage)
            if settings.db_dsn != None:
                self.checkpointer_cm = AsyncPostgresSaver.from_conn_string(settings.db_dsn)
                checkpointer = await self.checkpointer_cm.__aenter__()
                logger.info("Using instance-specific PostgresSaver for persistence.")
                await checkpointer.setup()
                self.checkpointer = checkpointer
            else:
                raise ValueError("When persistence is set to True, db_dsn must be provided in settings.")
        else:
            from langgraph.checkpoint.memory import MemorySaver
            checkpointer = MemorySaver()
            self.checkpointer = checkpointer

        
        self.config = config
        self.cmd = cmd
        self.load_agent()
        return self

    def __str__(self):
        agent_str = self.agent_card.__str__()
        return agent_str

    def chat(self, 
        messages:List[str]=None, 
        messages_type:List[str]=None, 
        messages_role:List[str]=None,
        develop_mode=False) -> str:

        if not self.agent._compiled: self.agent.compile(self.checkpointer)

        response = self.agent.chat(
            messages=messages, messages_type=messages_type, messages_role=messages_role, config=self.config)
        
        if "__interrupt__" in self.agent.response:
            self.agent._interrupt = True
            message_for_user = self.agent.response["__interrupt__"][0].value["messages_to_user"]
            return message_for_user
        
        if develop_mode:
            result = {'msg': response, 'error': 'no'}
        else:
            result = response
        return result

    
    async def stream(self, 
        messages:List[str]=None, 
        messages_type:List[str]=None, 
        messages_role:List[str]=None):

        if not self.agent._compiled: self.agent.compile(self.checkpointer)

        response_generator = self.agent.stream(
            messages=messages, messages_type=messages_type, messages_role=messages_role, config=self.config)
        print('response generator:', response_generator)
        
        # yield "system:" + json.dumps(self.config)

        async for response in response_generator:
            yield response


    def chat_cmd(self):

        bye_messages = ['bye', 'exit']; user_input = ""
        aitomia_agents_name = "aitomia_agents"; user_name = "You"
        while user_input.lower() not in bye_messages: 
            greeting = "\033[32m" + f"{aitomia_agents_name:<8} > " + "How can I help with today?\n" + "\033[0m"
            user_input = input(greeting + "\033[33m" + f"{user_name:<8} > " + "\033[0m")

            if user_input.lower() in bye_messages:
                return 

            aitomia_msg = self.chat(messages=user_input)
            if "__interrupt__" in self.agent.response:
                self.agent._interrupt = True
                message_for_user = self.agent.response["__interrupt__"][0].value["messages"]
                user_input = input("\033[32m" + f"{aitomia_agents_name:<8} > " + message_for_user + "\n" + "\033[33m" + f"{user_name:<8} > " + "\033[0m")
                aitomia_msg = self.chat(messages=user_input)
            print("\033[32m"+ f"{aitomia_agents_name:<8} > " + aitomia_msg + "\033[0m"+"\n")
    
    def load_agent(self):
        import importlib
        agent_module = importlib.import_module(self.agent_card.entry, package="aitomia_agents")
        self.agent:"BaseAgent" = getattr(agent_module, self.agent_card.graph_name)
    
    def thread_id_gen(self):
        thread_id = str(uuid.uuid4())
        return {"configurable": {"thread_id": thread_id, "checkpoint_ns":""}}

    # reset memory saver or reset config
    async def reset(self, config=None):
        """Reset the agent with a new or existing config.
        
        Args:
            config: Optional config dict. If None, creates new thread_id.
                    If provided, switches to that thread's state.
        """
        # CRITICAL: Clear FileManager global state to prevent path leak between sessions
        # from .utils import FileManager

        FileManager.set_path(None)
        Analysis.clear_summary_and_query()
        Analysis.set_start_time(time.time())
        GlobalFlag.clear_flag()
        logger.info("Reset: Cleared FileManager global path state")
        
        # Track if we're using an external/shared checkpointer
        using_external_checkpointer = (
            hasattr(self, 'checkpointer') and 
            not hasattr(self, 'checkpointer_cm')
        )
        
        if settings.memory_saver == "postgres":
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            if using_external_checkpointer:
                # Using shared checkpointer - don't recreate, just update config
                logger.info("Using existing shared checkpointer for reset.")
                if config is not None:
                    self.config = config
                    logger.info(f"Switched to thread_id: {self.config['configurable']['thread_id']}")
                else:
                    self.config = self.thread_id_gen()
                    logger.info(f"Created new thread_id: {self.config['configurable']['thread_id']}")
                    logger.warning(f"[DEBUG] Aitomia.reset() generated new config: {self.config}")
                
                # CRITICAL FIX: Force recompile to clear any cached state
                # This ensures the graph starts fresh for the new thread
                logger.warning(f"[DEBUG] Aitomia.reset() forcing graph recompile")
                self.agent._compiled = False
                self.agent.reset(self.checkpointer)
            else:
                # Managing our own checkpointer instance
                if config is None:
                    # Create new thread with new checkpointer
                    if settings.db_dsn is None:
                        raise ValueError("db_dsn must be provided when memory_saver is 'postgres'.")
                    
                    try:
                        # Close old connection if exists
                        if hasattr(self, 'checkpointer_cm') and self.checkpointer_cm is not None:
                            await self.checkpointer_cm.__aexit__(None, None, None)
                        
                        # Create new connection
                        self.checkpointer_cm = AsyncPostgresSaver.from_conn_string(settings.db_dsn)
                        checkpointer = await self.checkpointer_cm.__aenter__()
                        await checkpointer.setup()
                        
                        self.checkpointer = checkpointer
                        self.config = self.thread_id_gen()
                        self.agent.reset(self.checkpointer)
                        
                        logger.info(f"Reset with new PostgreSQL thread: {self.config['configurable']['thread_id']}")
                    except Exception as e:
                        logger.error(f"Failed to reset PostgreSQL checkpointer: {e}")
                        raise
                else:
                    # Use existing thread_id - just update config, keep same checkpointer
                    self.config = config
                    logger.info(f"Switched to existing thread: {self.config['configurable']['thread_id']}")
        else:
            # MemorySaver mode
            if config is None:
                # Create new thread
                self.config = self.thread_id_gen()
                self.config["recursion_limit"] = settings.recursion_limit
                # Only create MemorySaver if it doesn't exist
                if not hasattr(self, 'checkpointer') or self.checkpointer is None:
                    logger.info("Creating new MemorySaver for new thread.")
                    from langgraph.checkpoint.memory import MemorySaver
                    self.checkpointer = MemorySaver()
                    self.agent.reset(self.checkpointer)
                logger.info(f"Created new thread_id: {self.config['configurable']['thread_id']}")
            else:
                # Switch to existing thread
                self.config = config
                logger.info(f"Switched to thread_id: {self.config['configurable']['thread_id']}")
        
        logger.info('Set a two-second reset buffer')
        await asyncio.sleep(2)
        return self.config

    
    @property
    def state(self):
        return self.agent.state(self.config)

    # Load chat history
    def load(self, data_type, filename):
        self.agent.load(config=self.config, data_type=data_type, filename=filename) 

    # Dump chat history
    def dump(self, data_type='chk', filename=None):
        self.agent.dump(config=self.config, data_type=data_type, filename=filename) 




    # expose llm for outside to handel analysis task
    def get_agent(self):
        return self.agent.create_agent()
