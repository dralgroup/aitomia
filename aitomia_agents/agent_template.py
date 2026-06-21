"""
    Agent base class
    The general class for each agent to inherit and some useful utilities

TODO

- [ ] finish dump and load in base agent
"""
from typing import List

from .utils import CustomCallbackHandler, TokenUsageCallback

from .logger import logger
from .utils import Analysis

#-------------------------------------------------
# Agent
#-------------------------------------------------

def create_llm(model_kwargs={'temperature':0.0}, tools:list=None,tool_kwargs:dict={}):

    from langchain_openai import ChatOpenAI 
    from .settings import settings
    llm =  ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base,
        **model_kwargs
        )
    if tools is not None:
        llm_with_tools = llm.bind_tools(tools, **tool_kwargs)
        return llm_with_tools
    else:
        return llm

def create_message(mm, mt, mr):
    if mt.lower() == "human":
        if mr is None: mr = 'user'
        from langchain_core.messages import HumanMessage
        mt = HumanMessage
    elif mt.lower() == "ai":
        if mr is None: mr = 'ai'
        from langchain_core.messages import AIMessage
        mt = AIMessage
    elif mt.lower() == "system":
        if mr is None: mr = 'system'
        from langchain_core.messages import SystemMessage
        mt = SystemMessage
    return mt(content=mm, role=mr)

def state2json(state):
    # will refine later. Now just explicitly convert messages
    messages = []; _messages = state["messages"]
    for mm in _messages:
        messages.append({"message": mm.content, "type":mm.type, "role":mm.role})
    state["messages"] = messages

    current_task_messages = []; _current_task_messages = state["current_task_messages"]
    for ctm in _current_task_messages:
        messages = []
        for mm in ctm:
            messages.append({"message": mm.content, "type":mm.type, "role":mm.role})
        current_task_messages.append(messages)
    state["current_task_messages"] = current_task_messages
    return state

def json2state(dd):
    messages = []; _messages = dd["messages"]
    for mm in _messages:
        messages.append(create_message(mm["message"], mm["type"], mm["role"]))
    dd["messages"] = messages 

    current_task_messages = []; _current_task_messages = dd["current_task_messages"]
    for ctm in _current_task_messages:
        messages = []
        for mm in ctm:
            messages.append(create_message(mm["message"], mm["type"], mm["role"]))
        current_task_messages.append(messages)
    dd["current_task_messages"] = current_task_messages
    return dd
  

class BaseAgent(): # the backbone of agent is graph
    
    def __init__(self, 
        name:str=None, description:str=None, # for agent cards - imp later
        graph=None): 

        self.name = name 
        self.description = description
        self.graph = graph
        self._interrupt = False
        self._compiled = False

    def chat(self, 
        messages:List[str]=None, 
        messages_type:List[str]=None, 
        messages_role:List[str]=None,
        config:dict=None) -> str:

        """Chat to the agent with message. Prompts, user message, and additional AI message can be invoked here.
    
        Args:
            messages (List[str]|str):
                The messages to be sent to the agent. 
            messages_type (List[str]|str, optional):
                The type of messages to be sent to the agent. Avalilable options are human, ai and system. Default: "human"
            messages_role (List[str]|str, optional):
                The role of the messages. Default "user" for HumanMessage, "ai" for AIMessage and "system" for SystemMessage.
            config (dict): 
                The configurations for message invoke
        
        Example:

            response = agent.chat(
                messages="You are a greeting bot",
                messages_type="system")

            response = agent.chat(
                messages=["You are a greeting bot", "Hi, I'm aitomia_agents!"],
                messages_type=["system", "human"],
                messages_role=["system", "user"])
        """
        
        if isinstance(messages, str): messages = [messages]
        if isinstance(messages_type, str): messages_type = [messages_type]
        if isinstance(messages_role, str): messages_role = [messages_role]
        if messages_type is None: messages_type = ["human"]*len(messages)
        if messages_role is None: messages_role = [None]*len(messages)
        if messages:
            Analysis.add_user_query(messages)

        assert messages is not None,"Please provide message"
        assert len(messages) == len(messages_type) and len(messages_type) == len(messages_role), "Different lengths of list of messages, messages_type and messages_role"

        hanlder = CustomCallbackHandler()
        if config is None:
            config = {}
        callbacks = list(config.get("callbacks", []))
        for cb in (hanlder, TokenUsageCallback()):
            if not any(isinstance(existing, cb.__class__) for existing in callbacks):
                callbacks.append(cb)
        config["callbacks"] = callbacks


        if not self._interrupt:
            input_messages = []
            for mm, mt, mr in zip(messages, messages_type, messages_role):
                input_messages += [create_message(mm, mt, mr)]

            self.response = self.compiled_graph.invoke({"messages": input_messages}, config=config)
             
        else:
            from langgraph.types import Command
            self.response = self.compiled_graph.invoke(Command(resume={"messages":create_message(messages[0], messages_type[0], messages_role[0])}), config=config)
            self._interrupt = False

        return self.response['messages'][-1].content
    
    async def stream(self, 
        messages:List[str]=None, 
        messages_type:List[str]=None, 
        messages_role:List[str]=None,
        config:dict=None):

        if isinstance(messages, str): messages = [messages]
        if isinstance(messages_type, str): messages_type = [messages_type]
        if isinstance(messages_role, str): messages_role = [messages_role]
        if messages_type is None: messages_type = ["human"]*len(messages)
        if messages_role is None: messages_role = [None]*len(messages)
        if messages:
            Analysis.add_user_query(messages)

        assert messages is not None,"Please provide message"
        assert len(messages) == len(messages_type) and len(messages_type) == len(messages_role), "Different lengths of list of messages, messages_type and messages_role"

        hanlder = CustomCallbackHandler()
        if config is None:
            config = {}
        callbacks = list(config.get("callbacks", []))
        for cb in (hanlder, TokenUsageCallback()):
            if not any(isinstance(existing, cb.__class__) for existing in callbacks):
                callbacks.append(cb)
        config["callbacks"] = callbacks
        
        if not self._interrupt:
            input_messages = []
            for mm, mt, mr in zip(messages, messages_type, messages_role):
                input_messages += [create_message(mm, mt, mr)]

            async for chunk in self.compiled_graph.astream(
                    {"messages": input_messages}, config=config, stream_mode=["updates"], subgraphs=True):
                response = self.chunk2message(chunk)
                if response is not None:
                    yield self.chunk2message(chunk)
             
        else:
            from langgraph.types import Command
            async for chunk in self.compiled_graph.astream(
                    Command(
                        resume={"messages":create_message(messages[0], messages_type[0], messages_role[0])}), 
                    config=config, stream_mode=["updates"], subgraphs=True):
                response = self.chunk2message(chunk)
                self._interrupt = False
                if response is not None:
                    yield self.chunk2message(chunk)
    
    # def chunk2message(self, chunk):
    #     subgraph, stream_type, subgraph_info = chunk
    #     messages = None
    #     if stream_type == 'updates':
    #         for node_name, state in subgraph_info.items():
    #             if node_name != "__interrupt__": # interrupt node
    #                 # if state is not None: # nodes with updates
    #                 #     if state.get("messages", None) is not None: # get only final AI reply
    #                 #         # Here is a bit tricky. The messages are not processed by reduced function in root graph. But those from other subgraph have been processed.
    #                 #         if isinstance(state["messages"], list):
    #                 #             if state["messages"][-1].type == 'ai': messages = state["messages"][-1].content
    #                 #         else: 
    #                 #             if state["messages"].type == "ai": messages = state["messages"].content
    #                 messages = None
    #             else: 
    #                 messages = state[0].value["messages"]
    #                 self._interrupt = True 
    #         if messages is not None:
    #             return {'messages':messages, 'status':'pending'}
    #     elif stream_type == 'custom':
    #         if subgraph_info.endswith(('Calculation completed', 'Answer completed')):
    #             return {'messages':subgraph_info, 'status':'completed'}#########计算结束后加一个Calculation completed
    #         else:
    #             return {'messages':subgraph_info, 'status':'running'}
    #         # return subgraph_info


    def chunk2message(self, chunk):
        subgraph, stream_type, subgraph_info = chunk
        messages = None
        task_title = None
        options = None
        if stream_type == 'updates':
            for node_name, state in subgraph_info.items():
                if node_name != "__interrupt__": # interrupt node
                    if state is not None: # nodes with updates
                        if state.get("messages_to_user", None) is not None: # get only final AI reply
                            # Here is a bit tricky. The messages are not processed by reduced function in root graph. But those from other subgraph have been processed.
                            if isinstance(state["messages_to_user"], list):
                                if state["messages_to_user"][-1].type == 'ai': messages = state["messages_to_user"][-1].content
                            else: 
                                if state["messages_to_user"].type == "ai": messages = state["messages_to_user"].content
                            if state.get("status", None) is not None: 
                                status = state['status']
                            else:
                                status = 'running'
                        # Extract task_title if available
                        if state.get("task_title", None) is not None:
                            task_title = state["task_title"]
                else: 
                    interrupt_value = state[0].value
                    messages = interrupt_value.get("messages_to_user")
                    options = interrupt_value.get("options")
                    logger.info(f"Interrupt received with messages: {messages} and options: {options}")
                    status = 'pending'
                    self._interrupt = True 
            if messages is not None:
                # return messages
                result = {'messages_to_user':messages, 'status': status}
                if task_title is not None:
                    result['task_title'] = task_title
                if options is not None:
                    result['options'] = options
                return result
            
        # elif stream_type == 'custom':
        #     if subgraph_info.endswith(('Calculation completed', 'Answer completed')):
        #         return {'messages':subgraph_info, 'status':'completed'}
        #     else:
        #         return {'messages':subgraph_info, 'status':'running'}
            # return subgraph_info



    def compile(self, checkpointer=None):
        if checkpointer is None:
            from langgraph.checkpoint.memory import MemorySaver
            self.checkpointer = MemorySaver()
        else: self.checkpointer = checkpointer

        self.compiled_graph = self.graph.compile(self.checkpointer)
        self._compiled = True

    def delete(self, config):
        self.checkpointer.delete_thread(config["configurable"]["thread_id"])

    def reset(self, checkpointer=None):
        """Reset the agent. Only recompile if checkpointer changed or not compiled yet."""
        self._interrupt = False
        
        # Only recompile if necessary
        if checkpointer is not None and (not self._compiled or checkpointer != self.checkpointer):
            # New checkpointer provided, need to recompile
            self.compile(checkpointer)
        elif not self._compiled:
            # Never compiled before, compile with existing or new checkpointer
            self.compile(checkpointer)

    def dump(self, config=None, data_type='chk', filename=None):

        assert config is not None, "Please provide config to dump graph"

        if data_type == 'chk': self.dump_chk(config=config, filename=filename)
        elif data_type == 'chat': self.dump_chat(config=config, filename=filename)
        else: raise ValueError("not supported dump type")

    def load(self, config=None, data_type='chk', filename=None):

        assert config is not None, "Please provide config to load graph"

        if data_type == 'chk': self.load_chk(config=config, filename=filename)
        elif data_type == 'chat': self.load_chat(config=config, filename=filename)
        else: raise ValueError("not supported load type") 

    def dump_chk(self, config=None, filename=None): # need to figure out what is the checkpoint

        if filename is None: filename = self.name.replace(" ", "_") + '.chk'

        import json
        chk = self.compiled_graph.checkpointer.get(config)
        current_state = chk["channel_values"]
        current_state = state2json(current_state)
        chk["channel_values"] = current_state

        sn = list(self.state_history(config))
        latest_metadata = sn[0].metadata
        chk['meta_data'] = latest_metadata
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(chk, f, ensure_ascii=False, indent=4)

    def load_chk(self, config=None, filename=None): 

        import json
        with open(filename, "r", encoding="utf-8") as f:
            chk = json.load(f)
        chk["channel_values"] = json2state(chk["channel_values"])
        if not self._compiled: self.reset(None)

        new_versions = chk["channel_versions"]
        meta_data = chk["meta_data"]
        self.compiled_graph.checkpointer.put(config=config, checkpoint=chk, new_versions=new_versions, metadata=meta_data)

    def dump_chat(self, config=None, filename=None): # dump the last snapshot
        if filename is None: filename = self.name+'.'+'chat'
        import json 
        messages = self.compiled_graph.checkpointer.get(config)['channel_values']["messages_to_user"]
        export = []
        for mm in messages:
            export.append({"message": mm.content,"type": mm.type,'role': mm.role})
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(export, f, ensure_ascii=False, indent=4)
        
    def load_chat(self, config=None, filename=None): 

        if filename is None: filename = self.name+'.'+'chat'
        import json 
        with open(filename, "r", encoding="utf-8") as f:
            _messages = json.load(f)
        messages = []
        for mm in _messages:
            messages.append(create_message(mm['message'], mm['type'], mm['role']))

        self.compiled_graph.update_state(config, {"messages_to_user":messages})

    def print_chat(self):
        """Print nice look chat history (you need to have history first :D)"""
        assert 'response' in self.__dict__
        for m in self.response['messages']:
            m.pretty_print()  
    
    def view(self, tool=None, tool_kwargs=None):
        """Visialization of the langgraph 
        Tips: Check offical documention of langgraph about visualization. The graph cannot be rendered if it's not well organized or defined :D 
    
        Args:
            graph: the graph to be visualized
            tool: str
                'ascii': print the graph with ascii code
                'mermaid': return the url that can visualize the graph with mermaid
        """
        import base64, sys

        assert tool.lower() in ['mermaid', 'ascii'], 'tool must be either mermaid or ascii'
        assert self.graph is not None, "please compile or provide graph first"

        if tool.lower() == 'mermaid':
            mermaid_syntax = self.graph.get_graph().draw_mermaid()
            mermaid_syntax_encoded = base64.b64encode(mermaid_syntax.encode("utf8")).decode("ascii")
            background_color=tool_kwargs.get('background_color', 'white')
            url = f"https://mermaid.ink/img/{mermaid_syntax_encoded}&bgColor=!{background_color}"
            print(f'Graph visualization url: \n{url}\n'); sys.stdout.flush()
            return url 
        
        if tool.lower() == 'ascii':
            graph_ascii = self.graph.get_graph().draw_ascii()
            print(f"Graph visualization with ascii\n{graph_ascii}")
            sys.stdout.flush()
            return graph_ascii

    def state_history(self, config):
        """Get the snapshots from the start to the current state"""
        assert self._compiled, "Please provide compiled graph"
        assert config is not None, "Please provide graph config to load history"
        return self.compiled_graph.get_state_history(config)

    def state(self, config):
        return self.compiled_graph.get_state(config).values  
    
    def create_agent(self):
        return create_llm

    

#-------------------------------------------------
# Utils
#-------------------------------------------------                  
                    
def import_attr(module, *attrs):
    """import attribute from module

    Example:

        init_func = import_attr("mlatom.irc", "irc", "__init__")
        print(init_func)

    """
    import importlib
    imported_attr = importlib.import_module(module)
    for attr in attrs:
        imported_attr = getattr(imported_attr, attr)
    return imported_attr

def clean_sphinx_doc(doc):
    """Clean up sphinx grammar in doc"""
    import re 
    inline_ref = re.compile(
    r":(?:py:)?(?:class|meth|func|attr|mod|obj|data|exc|const|paramref|ref):`([^`]+)`")
    symbols = re.compile(r"``([^`]*)``")
    doc = inline_ref.sub(r"\1", doc)
    doc = symbols.sub(r"`\1`", doc)
    return doc

def deep_update(dd_ori, dd_upd):
    for kk, vv in dd_upd.items():
        if isinstance(vv, dict) and isinstance(dd_ori.get(kk), dict):
            deep_update(dd_ori[kk], vv)
        else: dd_ori[kk] = vv 
    return dd_ori

def args_dict_from_func(func=None, doc=None):
    """Extract arguments, type of arguments, descriptions from function
    
    Args:
        func (Callable)
            The function to extract information
        doc (str)
            The __doc__ of a function 
    
    Return:
        The dict containing information of arguments:
        {
            "function": {"description": "..."},
            "arguments": {
                    "argument1": {
                        "type": ...,
                        "description": ...,
                        "default": ...
                    },
                    "argument2": {
                        "type": ...,
                        "description": ...,
                        "default": ...
                    }, ... }    
    """

    from collections import defaultdict
    import inspect, docstring_parser

    func_dict = defaultdict(dict)

    sig = inspect.signature(func).parameters.copy()
    if doc is None:
        doc = inspect.getdoc(func)
    
    doc = clean_sphinx_doc(doc)
    parsed_doc = docstring_parser.parse(doc) 
    func_dict['function'] = {"description":parsed_doc.short_description}

    # remove attr if they are not in the doc
    attrs = [attr.arg_name for attr in parsed_doc.params]
    updated_sig = {}
    for attr_name, attr_val in sig.items():
        if attr_name in attrs: updated_sig[attr_name] = attr_val
    
    for attr in parsed_doc.params:
        attr_dict = {}
        attr_dict["type"] = sig[attr.arg_name]._annotation
        attr_dict["description"] = attr.description
        attr_dict["default"] = attr.default
        func_dict["arguments"][attr.arg_name] = attr_dict
    
    return func_dict

def args_dict_from_mlatom(mod_path:str=None, func_path:str=None, doc_func_path:str=None, replace:dict=None, delete:list=None):
    func_path = func_path.split(".")
    imported_func = import_attr(mod_path, *func_path)
    if doc_func_path is not None: # func path and doc path are different
        doc_func_path = doc_func_path.split(".")
        imported_doc_func = import_attr(mod_path, *doc_func_path)
        imported_doc = imported_doc_func.__doc__
    args_dict = args_dict_from_func(imported_func, imported_doc)

    # update the entries that we want to replace and delete the None entry
    if replace is not None: args_dict = deep_update(args_dict, replace)
    if delete is not None:
        for args_kk in list(args_dict["arguments"].keys()):
            if args_kk in delete:
                del args_dict["arguments"][args_kk]
    return args_dict

def schema_from_args_dict(schema_name=None, schema_base=None, args_dict=None):
    """Build schema from arguments dict"""
    from pydantic import Field, create_model
    import inspect
    from typing import Any
    
    model_fields = {}
    for arg_name, arg_info in args_dict.items():
        # Get the type annotation
        arg_type = arg_info.get('type', Any)
        
        # Check if the type is inspect._empty and replace with Any
        if arg_type is inspect._empty or arg_type == inspect._empty:
            arg_type = Any
        
        # Get default value if it exists
        default = arg_info.get('default', ...)
        if default is inspect._empty:
            default = ...
        
        # Create field info
        if default is ...:
            model_fields[arg_name] = (arg_type, ...)
        else:
            model_fields[arg_name] = (arg_type, default)
    
    return create_model(schema_name, __base__=schema_base, **model_fields)

def schema_to_str(schema=None, base=None) -> str:
    if base is None: base = ""
    else: base = f"({base.__name__})"
    lines = [f"class {schema.__name__}{base}:"]
    for name, field in schema.model_fields.items():
        typ = field.annotation.__name__ if hasattr(field.annotation, "__name__") else str(field.annotation)
        default = f" = {repr(field.default)}" if field.default is not None else ""
        desc = f"  # {field.description}" if field.description else ""
        lines.append(f"    {name}: {typ}{default}{desc}")
    return "\n".join(lines) 


#-------------------------------------------------
# Tool functions
#-------------------------------------------------

def tool_from_mlatom(mod_path:str=None, func_path:str=None, doc_func_path:str=None, replace:dict=None, delete:list=None, tool_name:str=None):
    """Build tool from function in mlatom
    
    Args:
        mod_path (str): The path to the module of the function
        func_path (str): The path to the function under the module
        doc_func_path (str, optional): The path to the function that has doc (within the same module in current imp). Default: None
        replace (dict, optional): Update the argument information in args_dict. Default: None
        delete (list, optional): Delete the unwanted arguments in args_dict. Default: None
        tool_name (str): The name to the tool function which will be seen by the agent.

    Example:

        irc_tool = tool_from_mlatom(
                        mod_path = "mlatom.irc",
                        func_path = "irc.__init__",
                        doc_func_path = "irc",
                        replace = {"argument": {"molecule": {"type":str}}},
                        tool_name = "irc")
    
    """
    from langchain_core.tools import StructuredTool

    args_info = args_dict_from_mlatom(mod_path, func_path, doc_func_path, replace, delete)

    # create tool schema
    tool_schema = schema_from_args_dict(schema_name=tool_name+"Args", args_dict=args_info["arguments"])

    # create tool extractor that will return arguments of tool
    def _tool_args_extractor(**kwargs):
        args = tool_schema(**kwargs)
        return args.model_dump()
    
    tool_args_extractor = StructuredTool.from_function(
        name = tool_name,
        func = _tool_args_extractor,
        description = args_info['function']['description'],
        args_schema=tool_schema
    )

    return tool_args_extractor

#-------------------------------------------------
# Schema
#-------------------------------------------------

from .states import AitomiaState
    
def schema_from_mlatom(mod_path:str=None, func_path:str=None, doc_func_path:str=None, replace:dict=None, delete:list=None, schema_name:str=None):
    args_info = args_dict_from_mlatom(mod_path, func_path, doc_func_path, replace, delete)
    schema = schema_from_args_dict(schema_name, AitomiaState, args_dict=args_info["arguments"])
    return schema

def prompt_template():
    prompt = ""
    return prompt 
PROMPT_TEMPLATE = prompt_template


#-------------------------------------------------
# Prompts
#-------------------------------------------------