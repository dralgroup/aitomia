"""  
    Raman planner agent
"""

import os 
from pathlib import Path
import mlatom as ml
import numpy as np
import traceback

from typing import Literal, List
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END 
from langgraph.types import Command

from aitomia_agents.user_context import user_context

from .agent_cards import agent_cards
from .file_manager import get_current_result_files_prompt, get_folder_name_node
from .optimize_geometry import geomopt_builder 
from .raman_static import raman_static_builder 

from .states import AitomiaState 
from .logger import logger 
from .utils import create_agent, pretty_array

from langgraph.config import get_stream_writer

class RamanPlannerState(AitomiaState):
    raman_task_sequence: List[str] = None
    task_type: str = None 
    
def prompt_raman_planner():
    prompt = "" 
    prompt += "You need to design a workflow according to the given messages. For Raman spectrum calculation, you need to first optimize the molecule, then calculate its spectrum. The workflow is a list of tasks. Below shows all the available tasks: \n"
    # Add the description of each task to the prompt
    # Only geometry optimization and frequency calculation are needed
    for agent_name, agent_card in agent_cards.items():
        if agent_name in ["geomopt_agent","raman_static_agent"]:
            prompt += "\t" + agent_card.description_for_planner + "\n"
            
    prompt += "If the molecule is already optimized in a previous task (check the available result files), you can skip this task.\n"
    # Clarify the format of AI output
    prompt += "Below is the format of your output:\n"
    prompt += "\tEach task should be put in one line.\n"
    prompt += "\tEach line should also contain the information that is needed to perform the task, e.g., molecule, method, Method program (if specified, otherwise state not specified), Calculation program (primary recommended program), and other conditions, according to the user's input.\n"
    prompt += "\tNo additional explanations or comments are needed.\n"
    prompt += "\tDo not provide the working directory in each line.\n"
    
    return prompt 
    
PROMPT_RAMAN_PLANNER = prompt_raman_planner()

#-------------------------------------------------
# Tool functions
#-------------------------------------------------
def decide_task_type(
    optimize_geometry:bool=False,
    raman_static:bool=False,
):
    """
    Decide which task to perform. It is necessary that one and only one task is chosen to be True.

    Args:
        optimize_geometry: Whether to optimize geometry of a molecule.
        raman_static: Whether to calculate static Raman spectrum of a molecule.
    """

    argvals = list(locals().values())
    if sum(argvals) != 1:
        # return "task_type_node"
        # Selecting task_type_node means more than one task or no task is choosen. Raise a warning.   
        logger.warning("More than one task or no task is choosen")

    if optimize_geometry: return "geomopt_node"
    if raman_static: return "raman_static_node"
    
#-------------------------------------------------
# Agent
#-------------------------------------------------
llm = create_agent()
tools = [decide_task_type]
task_type_agent = create_agent(tools=tools,tool_kwargs={"tool_choice":"any"})
task_type_tool = ToolNode(tools)

#-------------------------------------------------
# Graph
#-------------------------------------------------
def raman_planner_node(state:RamanPlannerState):
    logger.info("Start making a plan of the Raman spectrum calculation")
    message = "Start making a plan of the Raman spectrum calculation"
    
    raman_planner_prompt = SystemMessage(content=PROMPT_RAMAN_PLANNER)
    
    # Create an empty list of message for sut-tasks 
    current_task_messages = state.current_task_messages
    current_task_messages.append([])
    
    # Now current_task_messages[-2] is the raman task message list, current_task_messages[-1] is the sub-task message list
    response = llm.invoke(current_task_messages[-2] + [raman_planner_prompt])

    logger.debug("Agent response:")
    logger.debug(response.content)

    sequence = [each.strip() for each in response.content.split("\n")]
    logger.debug(sequence) 

    return {
        "raman_task_sequence": sequence,
        "current_task_messages": current_task_messages,
        "messages_to_user": [AIMessage(content=message)]
    }
    

def workdir_manager_node(state:RamanPlannerState):
    logger.info("Start managing the working directory of a task")
    message = "Start managing the working directory of a task"
    # workdir_manager_prompt = SystemMessage(content=PROMPT_WORKDIR_MANAGER)
    task_sequence = state.raman_task_sequence
    task = task_sequence.pop(0)
    if state.result_files is None: state.result_files = []


    task_message = SystemMessage(content=task)
    result_files_message = get_current_result_files_prompt(state.result_files)

    task_messages = state.current_task_messages

    # Put the last message (summary) in the sub-task message list into the raman task message list
    if len(task_messages[-1]) > 0:
        task_messages[-2].append(task_messages[-1][-1])

    # Clear the sub-task message list and put the new task into it
    task_messages[-1] = [result_files_message,task_message]
    return {
        "raman_task_sequence":task_sequence,
        "current_task_messages":task_messages,
        "result_files": state.result_files,
        "messages_to_user": [AIMessage(content=message)],
    }
    
def task_type_node(state:RamanPlannerState) -> Command[Literal["geomopt_node","raman_static_node"]]:
    logger.info("Start choosing task type in Raman calculation")

    response = task_type_agent.invoke(state.current_task_messages[-1])

    logger.debug("Response from the task type agent:")
    logger.debug("\t"+response.content)

    output = task_type_tool.invoke({"messages":[response]})["messages"][-1].content 
    logger.debug(output)
    task_type = output 

    logger.info(f"Next node: {task_type}")

    return Command(
        goto = task_type,
        update = {
            "task_type": task_type.replace("_node","")
        }
    )
    
def raman_analysis_node(state:RamanPlannerState):
    
    try:
        # Put the last message (summary) in the sub-task message list into the raman task message list
        task_messages = state.current_task_messages
        if len(task_messages[-1]) > 0:
            task_messages[-2].append(task_messages[-1][-1])

        # Remove the messages of sub-tasks (geometry optimization or frequency calculation)
        task_messages.pop()
        
        # Deal with the working directory and the stack at the end of the task
        stack = state.working_directory_stack
        stack.pop()
        if len(stack) == 0:
            current_working_directory = user_context.home_dir or Path.home()
        else:
            current_working_directory = stack[-1]
                
        return {
            "current_task_messages": task_messages,
            "working_directory": current_working_directory,
            "working_directory_stack": stack,
        }
    except Exception as e:
        error_message = f"Error in raman_analysis_node: {str(e)}"
        error_info = traceback.format_exc()
        logger.error(error_message + error_info)
        # writer(f"❌ {error_message}")
        return {
            "error": error_message,
            "has_error": True,
            "messages": [AIMessage(content=f"Failed to analyze raman results: {error_message}")],
            "messages_to_user": [AIMessage(content=f"Failed to analyze raman results: {error_message}")],
        } 
        
def conditional_edge(state:RamanPlannerState):
    ntasks = len(state.raman_task_sequence)
    logger.info(f"Number of remaining tasks for Raman: {ntasks}")
    if ntasks == 0:
        return "raman_analysis_node"
    else:
        return "workdir_manager_node"
    
raman_builder = StateGraph(RamanPlannerState)

geomopt_graph = geomopt_builder.compile()
raman_static_graph = raman_static_builder.compile()

raman_builder.add_node("get_folder_name_node",get_folder_name_node)
raman_builder.add_node("raman_planner_node",raman_planner_node)
raman_builder.add_node("workdir_manager_node",workdir_manager_node)
raman_builder.add_node("task_type_node",task_type_node)
raman_builder.add_node("geomopt_node",geomopt_graph)
raman_builder.add_node("raman_static_node",raman_static_graph)
raman_builder.add_node("raman_analysis_node",raman_analysis_node)

raman_builder.add_edge(START,"get_folder_name_node")
raman_builder.add_edge("get_folder_name_node","raman_planner_node")
raman_builder.add_edge("raman_planner_node","workdir_manager_node")
raman_builder.add_edge("workdir_manager_node","task_type_node")
raman_builder.add_conditional_edges(
    "geomopt_node",
    conditional_edge,
    {
        "workdir_manager_node",
        "raman_analysis_node",
    }
)
raman_builder.add_conditional_edges(
    "raman_static_node",
    conditional_edge,
    {
        "workdir_manager_node",
        "raman_analysis_node",
    }
)
raman_builder.add_edge("raman_analysis_node",END)

raman_graph = raman_builder.compile() 
from .agent_template import BaseAgent
raman_agent = BaseAgent(
    name='raman_agent',
    description='Perform Raman intensity calculation of a molecular structure.',
    graph=raman_builder,
)