"""
    Method judge agent
"""
import json
from typing import Union, List, Literal#, Optional
import traceback
from langchain_core.messages import SystemMessage, AIMessage, AnyMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END 
from langgraph.types import Command

from .states import AitomiaState
from .logger import logger 
from .utils import create_agent, pretty_dict, Analysis

from langgraph.config import get_stream_writer

#-------------------------------------------------
# Schema
#-------------------------------------------------
class MethodState(AitomiaState):
    method: str = None
    program: Union[str,None] = None
    method_messages: List[AnyMessage] = []
    iters: int = 0

#-------------------------------------------------
# Prompts
#-------------------------------------------------
import mlatom as ml
supported_methods = {}
supported_methods['None'] = []
for method_class in ml.models.known_classes:
    if 'supported_methods' in method_class.__dict__:
        if '_methods' in method_class.__name__:
            supported_methods[method_class.__name__[:-8]] = method_class.supported_methods
            if 'pyscf' in method_class.__name__:
                supported_methods[method_class.__name__[:-8]] += ['Quantum mechanical methods']
        else:
            supported_methods['None'] += method_class.supported_methods
    else:
        if '_methods' in method_class.__name__:
            supported_methods[method_class.__name__[:-8]] = ['Quantum mechanical methods']
def prompt_available_methods():
    prompt = ""
    prompt += 'Below shows supported programs and the corresponding supported methods. The programs and methods are case insensitive. The methods are separated by ",".\n'
    for program,methods in supported_methods.items():
        prompt += f"    {program}: {', '.join(methods)}\n"
    prompt += 'Notation: Program here means Method program not calculation program, if it is not specified or none then the program is None\n'
    prompt += 'Note that if the program is None, the methods are implemented directly in MLatom. If the method is specified as "Quantum mechanical methods", the program supports quantum mechanical methods.\n'
    return prompt

PROMPT_AVAILABLE_METHODS = prompt_available_methods()

def prompt_mlatom_supported_methods():
    
    prompt = PROMPT_AVAILABLE_METHODS
    prompt += 'IMPORTANT: The calculation program has NO RELATIONSHIP with the method program. NEVER infer the method program from the calculation program.\n'
    prompt += 'Note that if the program is None, the methods are implemented directly in MLatom. If the method is specified as "Quantum mechanical methods", the program supports quantum mechanical methods. Please strictly follow the relationship specified above and choose one program only.\n'
    # If more than one programs found, use the first program.

    return prompt

PROMPT_MLATOM_SUPPORTED_METHODS = prompt_mlatom_supported_methods()


def prompt_method_supervisor():
    prompt = PROMPT_AVAILABLE_METHODS
    prompt += "You need to check if the given method and program follow the above rules. If they do not, please keep the method as it is and correct the program. If they do, just print out the current method and program, and say they do not need correction.\n"
    return prompt
PROMPT_METHOD_SUPERVISOR = prompt_method_supervisor()

#-------------------------------------------------
# Tool functions
#-------------------------------------------------
def get_method(method:str="",program:str|None=None):
    """
    This function is used to extract method from the user input or the current computation task.

    Args:
        method: The method provided by the user or the current computation task.
        program: The program where the method is implemented.
    """
    if program == "":
        program = None 
    elif program == "None":
        program = None
        
    return {"method":method,"program":program}

def regenerate(
    regenerate:bool=False,
):
    """
    Decide whether to re-extract the method and program.

    Args: 
        regenerate: Whether to re-extract the method and program. Re-extract if the program should be corrected.
    """

    if regenerate: return "get_method_node"
    else: return "method_correction_node"

#-------------------------------------------------
# Agent
#-------------------------------------------------
tools = [get_method]
method_agent = create_agent(tools=tools, tool_kwargs={"tool_choice":"any"})
method_tool = ToolNode(tools)
llm = create_agent()
tools = [regenerate]
method_regenerate_agent = create_agent(tools=tools, tool_kwargs={"tool_choice":"any"})
method_regenerate_tool = ToolNode(tools)
#-------------------------------------------------
# Graph
#-------------------------------------------------

def initialize_node(state:MethodState):

    return {
        "iters":-1,
    }

def get_method_node(state:MethodState):
    logger.info("Start getting method")

    try:
        logger.debug("Input state:"); pretty_dict(state.model_dump(), logger)

        method_avail_prompt = SystemMessage(content=PROMPT_MLATOM_SUPPORTED_METHODS)

        if len(state.method_messages) == 0:
            response = method_agent.invoke(state.current_task_messages[-1]+[method_avail_prompt])
        else:
            response = method_agent.invoke(state.method_messages)
                
        logger.debug("response from method agent:")
        logger.debug("\t"+response.content)

        output = method_tool.invoke({"messages":[response]})["messages"][-1].content

        logger.debug("response from method tool:")
        logger.debug("\t"+output)

        # output = ast.literal_eval(output)
        output = json.loads(output)
        output_state = {
            "method": output["method"],
            "program": output["program"],
            "messages_to_user": [AIMessage(content='Start getting method')],
            "method_messages": [],
            "iters": state.iters + 1,
        }

        logger.debug("Output state:"); pretty_dict(output_state, logger)
        return output_state
    
    except Exception as e:
        error_message = f"Error in get_method_node: {str(e)}"
        error_info = traceback.format_exc()
        logger.error(error_message + error_info)
        response = Analysis.error_analysis(error_message)
        error_analysis = error_message + '\n' + '\n' + response
        # writer = get_stream_writer()
        # writer(f"❌ {error_message}")
        return {
            "error": error_message,
            "has_error": True,
            "messages": [AIMessage(content=f"Failed to get method: {error_message}")],
            "messages_to_user": [AIMessage(content=f"Failed to get method: {error_analysis}")],
        }
    
def check_method_node(state:MethodState):
    logger.info("Check the method")
    logger.debug("Input State:"); pretty_dict(state.model_dump(),logger)

    supervisor_prompt = SystemMessage(content=PROMPT_METHOD_SUPERVISOR)
    current_method = f"Method: {state.method}; Program: {state.program}"
    current_method = AIMessage(content=current_method)
    response = llm.invoke([supervisor_prompt,current_method])

    logger.debug("Response from LLM:")
    logger.debug(response.content)

    return {"method_messages":[AIMessage(content=response.content)]}

def method_regenerate_node(state:MethodState) -> Command[Literal["get_method_node","method_correction_node"]]:
    logger.info("Check whether to re-extract method and program")
    logger.debug("Input state:"); pretty_dict(state.model_dump(),logger)

    response = method_regenerate_agent.invoke(state.method_messages)
    logger.debug("Response from the agent:")
    logger.debug("\t"+response.content)

    output = method_regenerate_tool.invoke({"messages":[response]})["messages"][-1].content 
    logger.debug(output)
    next_node = output
    logger.info(f"Next node: {next_node}")

    if next_node == "method_correction_node":
        if state.iters > 0:
            writer = get_stream_writer()
            writer(f"The method and program are corrected.\nCurrent method: {state.method}\nCurrent method program: {state.program}")
    else:
        if state.iters >= 2:
            next_node = "method_correction_node" # In case there are too many iterations

    return Command(
        goto = next_node,
    )

def method_correction_node(state:MethodState):
    logger.info("Check method and program in a hard coded way")
    method = state.method 
    program = state.program
    if not program is None:
        program = program.casefold()
    available_programs = []
    for each_program, each_methods in supported_methods.items():
        if method.casefold() in [each.casefold() for each in each_methods]:
            if each_program == "None":
                available_programs.append(None)
            else:
                available_programs.append(each_program.casefold())
    if len(available_programs) > 0:
        if not program in available_programs:
            program = available_programs[0]
            logger.debug(f"Change the program to {program}")

    if program == 'Gaussian'.casefold():
        if '/' in method:
            method_tmp = method.split('/')
            if 'cc' in method_tmp[-1].casefold() or 'g' in method_tmp[-1].casefold():
                pass 
            else:
                method_tmp[-1] = method_tmp[-1].replace("-","")
            method_tmp[0] = method_tmp[0].replace("-","")
            method = '/'.join(method_tmp)
        else:
            method = method.replace("-","")
    return {
        "program": program,
        "method": method,
    }
method_builder = StateGraph(MethodState)
method_builder.add_node("initialize_node",initialize_node)
method_builder.add_node("get_method_node",get_method_node)
method_builder.add_node("check_method_node",check_method_node)
method_builder.add_node("method_regenerate_node",method_regenerate_node)
method_builder.add_node("method_correction_node",method_correction_node)
method_builder.add_edge(START,"initialize_node")
method_builder.add_edge("initialize_node","get_method_node")
method_builder.add_edge("get_method_node","check_method_node")
method_builder.add_edge("check_method_node","method_regenerate_node")
method_builder.add_edge("method_correction_node",END)

method_graph = method_builder.compile()
