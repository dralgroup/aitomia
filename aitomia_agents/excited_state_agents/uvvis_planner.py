"""  
    UV-vis planner agent
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

from ..agent_cards import agent_cards
from ..file_manager import get_current_result_files_prompt, get_folder_name_node
from ..optimize_geometry import geomopt_builder
# from ..frequency import freq_builder
from .uvvis_spc import uvvis_spc_builder 

from ..states import AitomiaState 
from ..logger import logger 
from ..utils import create_agent, pretty_array

from langgraph.config import get_stream_writer

class UVVISPlannerState(AitomiaState):
    uvvis_task_sequence: List[str] = None
    task_type: str = None 
    
def prompt_uvvis_planner():
    prompt = "" 
    prompt += "You need to design a workflow according to the given messages. For UV-vis spectrum calculation, you need to first optimize the molecule, then calculate its spectrum. The workflow is a list of tasks. Below shows all the available tasks: \n"
    # Add the description of each task to the prompt
    # Only geometry optimization and frequency calculation are needed
    for agent_name, agent_card in agent_cards.items():
        if agent_name in ["geomopt_agent", "uvvis_spc_agent"]:
            prompt += "\t" + agent_card.description_for_planner + "\n"
    
    prompt += "In any case, you should perform geometry optimization before calculating the UV-vis spectrum, unless the user explicitly indicates that geometry optimization is not needed.\n"
    prompt += "You should be really careful that NEVER FALL INTO ground-state single-point calculation in any case, any time and anywhere in this task. The last step shoule always be the uvvis calculation (aka excited-state single-point calculation, but will NEVER BE ground-state single-point calculation.)\n"
    prompt += "If the molecule is already optimized in a previous task (check the available result files), you can skip the optimization and frequnecy calculation.\n"
    prompt += "If optimization task is included in this workflow, you should do uvvis spectrum calculation with the optimized geometry, but not the initial guess of optimization."
    # Method and program
    prompt += "Below is the principles for method and program:"
    prompt += "\tFor method: The method of geometry optimization and UV-vis calculation could be different. If the user requests OMNI-P2x method for UVvis, you need to confirm the method for optimization from the user.\n"
    prompt += "\tFor program: If the user requests geomopt, choose `geometric` for optimization program unless otherwise specified. The program for UVvis should be None\n"
    prompt += "\tBe really careful that `OMNI-P2x` and `OM2` are different methods - `OMNI-P2x` is pure machine learning method and `OM2` is semi-empirical method, Be careful to distinguish them, don't confuse them.\n"
    # Clarify the format of AI output
    prompt += "Below is the format of your output:\n"
    prompt += "\tEach task should be put in one line.\n"
    prompt += "\tEach line should also contain the information that is needed to perform the task, e.g., molecule, method, Method program (if specified, otherwise state not specified), Calculation program (primary recommended program), and other conditions, according to the user's input.\n"
    prompt += "\tNo additional explanations or comments are needed.\n"
    prompt += "\tDo not provide the working directory in each line.\n"
    
    return prompt 


def prompt_method_reconfirmer():
    prompt = ""
    prompt += f"If the user requests the uvvis calculation using omnip2x, OMNI_P2x, or OMNI-P2x (regardless of cases), you must tell the user that this method is not a proper method for geometry optimization and ask the user for another method for optimization."
    return prompt


PROMPT_UVVIS_PLANNER = prompt_uvvis_planner()
PROMPT_MULTITASK_PLANNER = prompt_method_reconfirmer()

#-------------------------------------------------
# Tool functions
#-------------------------------------------------
def decide_task_type(
    optimize_geometry:bool=False,
    uvvis_spc:bool=False,
):
    """
    Decide which task to perform. It is necessary that one and only one task is chosen to be True.

    Args:
        optimize_geometry: Whether to optimize geometry of a molecule.
        uvvis_spc: Whether to calculate single-point convolution uvvis spectrum of a molecule.
    """

    argvals = list(locals().values())
    if sum(argvals) != 1:
        # return "task_type_node"
        # Selecting task_type_node means more than one task or no task is choosen. Raise a warning.   
        logger.warning("More than one task or no task is choosen")

    if optimize_geometry: return "geomopt_node"
    if uvvis_spc: return "uvvis_spc_node"
    
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
def uvvis_planner_node(state:UVVISPlannerState):
    logger.info("Start making a plan of the UV-vis spectrum calculation")
    message = "Start making a plan of the UV-vis spectrum calculation"
    
    uvvis_planner_prompt = SystemMessage(content=PROMPT_UVVIS_PLANNER)
    
    # Create an empty list of message for sut-tasks 
    current_task_messages = state.current_task_messages
    current_task_messages.append([])
    
    # Now current_task_messages[-2] is the uvvis task message list, current_task_messages[-1] is the sub-task message list
    response = llm.invoke(current_task_messages[-2] + [uvvis_planner_prompt])

    logger.debug("Agent response:")
    logger.debug(response.content)

    sequence = [each.strip() for each in response.content.split("\n")]
    logger.debug(sequence)

    return {
        "uvvis_task_sequence": sequence,
        "current_task_messages": current_task_messages,
        "messages_to_user": [AIMessage(content=message)]
    }
    

def workdir_manager_node(state:UVVISPlannerState):
    logger.info("Start managing the working directory of a task")
    message = "Start managing the working directory of a task"
    # workdir_manager_prompt = SystemMessage(content=PROMPT_WORKDIR_MANAGER)
    task_sequence = state.uvvis_task_sequence
    task = task_sequence.pop(0)
    if state.result_files is None: state.result_files = []


    task_message = SystemMessage(content=task)
    result_files_message = get_current_result_files_prompt(state.result_files)

    task_messages = state.current_task_messages

    # Put the last message (summary) in the sub-task message list into the uvvis task message list
    if len(task_messages[-1]) > 0:
        task_messages[-2].append(task_messages[-1][-1])

    # Clear the sub-task message list and put the new task into it
    task_messages[-1] = [result_files_message,task_message]
    return {
        "uvvis_task_sequence":task_sequence,
        "current_task_messages":task_messages,
        "result_files": state.result_files,
        "messages_to_user": [AIMessage(content=message)],
    }
    
def task_type_node(state:UVVISPlannerState) -> Command[Literal["geomopt_node","uvvis_spc_node"]]:
    logger.info("Start choosing task type in UV-vis calculation")

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
    
def uvvis_analysis_node(state:UVVISPlannerState):
    
    try:
        # Put the last message (summary) in the sub-task message list into the reaction task message list
        task_messages = state.current_task_messages
        if len(task_messages[-1]) > 0:
            task_messages[-2].append(task_messages[-1][-1])

        # Remove the messages of sub-tasks (geometry optimization or frequency calculation)
        task_messages.pop()
        
        # Deal with the working directory and the stack at the end of the task
        stack = state.working_directory_stack
        stack.pop()
        if len(stack) == 0:
            i = 1
            while True:
                workdir = os.path.join(user_context.home_dir, f"{state.uvvis_task_sequence[-1]}_{i}")
                if os.path.exists(workdir):
                    i += 1
                else:
                    current_working_directory = workdir
                    break
        else:
            current_working_directory = stack[-1]
                
        return {
            "current_task_messages": task_messages,
            "working_directory": current_working_directory,
            "working_directory_stack": stack,
        }
    except Exception as e:
        error_message = f"Error in reaction_analysis_node: {str(e)}"
        error_info = traceback.format_exc()
        logger.error(error_message + error_info)
        # writer(f"❌ {error_message}")
        return {
            "error": error_message,
            "has_error": True,
            "messages": [AIMessage(content=f"Failed to analyze reaction results: {error_message}")],
            "messages_to_user": [AIMessage(content=f"Failed to analyze reaction results: {error_message}")],
        } 
        
def conditional_edge(state:UVVISPlannerState):
    ntasks = len(state.uvvis_task_sequence)
    logger.info(f"Number of remaining tasks for UV-vis: {ntasks}")
    if ntasks == 0:
        return "uvvis_analysis_node"
    else:
        return "workdir_manager_node"
    
uvvis_workflow_builder = StateGraph(UVVISPlannerState)

geomopt_graph = geomopt_builder.compile()
uvvis_spc_graph = uvvis_spc_builder.compile()

uvvis_workflow_builder.add_node("get_folder_name_node",get_folder_name_node)
uvvis_workflow_builder.add_node("uvvis_planner_node",uvvis_planner_node)
uvvis_workflow_builder.add_node("workdir_manager_node",workdir_manager_node)
uvvis_workflow_builder.add_node("task_type_node",task_type_node)
uvvis_workflow_builder.add_node("geomopt_node",geomopt_graph)
# uvvis_workflow_builder.add_node("freq_node", freq_builder)
uvvis_workflow_builder.add_node("uvvis_spc_node",uvvis_spc_graph)
uvvis_workflow_builder.add_node("uvvis_analysis_node",uvvis_analysis_node)

uvvis_workflow_builder.add_edge(START,"get_folder_name_node")
uvvis_workflow_builder.add_edge("get_folder_name_node","uvvis_planner_node")
uvvis_workflow_builder.add_edge("uvvis_planner_node","workdir_manager_node")
uvvis_workflow_builder.add_edge("workdir_manager_node","task_type_node")
uvvis_workflow_builder.add_conditional_edges(
    "geomopt_node",
    conditional_edge,
    {
        "workdir_manager_node",
        "uvvis_analysis_node",
    }
)
# uvvis_workflow_builder.add_conditional_edges(
#     "freq_node",
#     conditional_edge,
#     {
#         "workdir_manager_node",
#         "uvvis_analysis_node",
#     }
# )
uvvis_workflow_builder.add_conditional_edges(
    "uvvis_spc_node",
    conditional_edge,
    {
        "workdir_manager_node",
        "uvvis_analysis_node",
    }
)

uvvis_workflow_builder.add_edge("uvvis_analysis_node",END)

uvvis_graph = uvvis_workflow_builder.compile() 
from ..agent_template import BaseAgent
uvvis_planner_agent = BaseAgent(
    name='uvvis_planner_agent',
    description='Perform UV-vis spectrum calculation of a molecular structure.',
    graph=uvvis_workflow_builder,
)