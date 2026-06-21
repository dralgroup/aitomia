"""
    Task type agent
"""
import json
import ast
import os
from typing import Union, Optional, Literal
import traceback
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END 
from langgraph.types import Command
from langgraph.config import get_stream_writer

from .prepare_molecule import prepare_molecule_builder
from .chat import chat_builder
# from .method_confirm import method_confirm_builder
from .single_point import single_point_builder
from .optimize_geometry import geomopt_builder
from .transition_state import ts_builder
from .frequency import freq_builder
from .ir import ir_builder
from .irc import irc_builder
from .raman import raman_builder
from .reaction import reaction_builder
# from .excited_state_agents.uvvis_spc import uvvis_spc_builder
from .excited_state_agents.uvvis_planner import uvvis_workflow_builder

from .states import AitomiaState
from .logger import logger 
from .utils import create_agent, pretty_dict
from .settings import settings

#-------------------------------------------------
# Schema
#-------------------------------------------------
class TaskTypeState(AitomiaState):
    task_type: str = None

#-------------------------------------------------
# Prompts
#-------------------------------------------------


#-------------------------------------------------
# Tool functions
#-------------------------------------------------






    # Task hierarchy rule (STRICT):
    # - 'reaction' is a TOP-LEVEL task.
    # - If the task involves a chemical reaction (i.e., multiple molecules such as reactants/products,
    # reaction energies, reaction thermochemistry, or reaction pathways),
    # THEN 'reaction' MUST be selected as the ONLY True value.

    # CRITICAL NOTE:
    # - Geometry optimization, frequency analysis, single-point calculations, etc.,
    # are considered INTERNAL STEPS of a reaction workflow.
    # - Even if geometry optimization, frequency, or thermochemistry is explicitly mentioned
    # in the user request, they MUST NOT be selected as separate tasks when 'reaction' is True.
def decide_task_type(
    prepare_molecule:bool=False,
    chat:bool=False,
    single_point:bool=False,
    optimize_geometry:bool=False,
    ts:bool=False,
    frequency:bool=False,
    ir:bool=False,
    irc:bool=False,
    raman:bool=False,
    reaction:bool=False,
    # uvvis_spc:bool=False,
    uvvis_workflow:bool=False

):
    """
    Decide which task to perform. It is necessary that one and only one task is chosen to be True.
    IMPORTANT:
    Exactly ONE and ONLY ONE task flag must be set to True.

    Task decision is HIERARCHICAL, not flat.

    Level-A (Workflow-level tasks):
    - reaction

    Level-B (Single-molecule / method-level tasks):
    - prepare_molecule
    - single_point
    - optimize_geometry
    - ts
    - frequency
    - ir
    - irc
    - raman
    - excited_state(uvvis_workflow)

    STRICT DECISION PROCEDURE:
    1. First, determine whether the task belongs to Level-A.
    - If the task involves a chemical reaction (i.e., multiple molecules such as reactants/products,
        reaction energies, reaction thermochemistry, or reaction pathways),
        THEN select 'reaction' as the ONLY True value.
    - When a Level-A task is selected, ALL Level-B task flags MUST be False.

    2. Only if the task does NOT belong to Level-A,
    determine exactly ONE Level-B task to be True.




    Args:
        prepare_molecule: Whether to get the structure of the molecule only.
        single_point: Whether to perform single point calculation.
        optimize_geometry: Whether to optimize geometry of a molecule.
        ts: Whether to get the transition state of a molecule.
        frequency: Whether to calculate frequency or thermodynamic properties of a molecule.
        ir: Whether to calculate infrared (IR) spectrum of a molecule. 
        irc: Whether to perform intrinsic reaction coordinates (IRC) calculation of a molecule. 
        raman: Whether to calculate Raman spectrum of a molecule.
        reaction: Whether to perform reaction calculation,  Select this option whenever the task involves a chemical reaction (i.e., any task that is not a single-molecule calculation but requires reactants/products or reaction profiles). 
        uvvis_workflow: To decide the workflow for UV-vis spectrum calculation.
    """
    argvals = list(locals().values())
    if sum(argvals) != 1:
        logger.warning("More than one task or no task is choosen")
        logger.warning(f"Task flags: {argvals}")


    if prepare_molecule: return "prepare_molecule_node"
    if chat: return "chat_node"
    if single_point: return "single_point_node"
    if optimize_geometry: return "geomopt_node"
    if ts: return "ts_node"
    if frequency: return "freq_node"
    if ir: return "ir_node"
    if irc: return "irc_node"
    if raman: return "raman_node"
    if reaction: return "reaction_node"
    # if uvvis_spc: return "uvvis_spc_node"
    if uvvis_workflow: return "uvvis_workflow_node"

#-------------------------------------------------
# Agent
#-------------------------------------------------
tools = [decide_task_type]
task_type_agent = create_agent(tools=tools,tool_kwargs={"tool_choice":"decide_task_type"})
task_type_tool = ToolNode(tools)

#-------------------------------------------------
# Graph
#-------------------------------------------------

def task_type_node(state:TaskTypeState) -> Command[Literal["prepare_molecule_node","chat_node","single_point_node","geomopt_node","ts_node","freq_node","ir_node","irc_node","raman_node","reaction_node", "uvvis_workflow_node"]]:
    logger.info("Start choosing task type")

    try:
        logger.debug("Input state:"); pretty_dict(state.model_dump(), logger)
        prompt = [SystemMessage("You must use the decide_task_type function to decide the task type.")]
        response = task_type_agent.invoke(state.current_task_messages[-1] + prompt)
        # updated_messages = [response]

        logger.debug("Response from the task type agent:")
        logger.debug("\t"+response.content)

        output = task_type_tool.invoke({"messages":[response]})["messages"][-1].content 
        logger.debug(output)
        task_type = output
        # task_type = ast.literal_eval(output)

        logger.info(f"Next node: {task_type}")    

        return Command(
            goto = task_type,
            update = {
                # "messages": updated_messages,
                "task_type": task_type.replace("_node","")
            }
        )
    
    except Exception as e:
        error_message = f"Error in task_type_node: {str(e)}"
        error_info = traceback.format_exc()
        logger.error(error_message + error_info)
        # writer = get_stream_writer()
        # writer(f"❌ {error_message}")
        return Command(
            goto = "chat_node",
            update = {
                "error": error_message,
                "has_error": True,
                "messages": [AIMessage(content=f"Failed to determine task type: {error_message}")],
                "messages_to_user": [AIMessage(content=f"Failed to determine task type: {error_message}")],
            }
        )



task_type_builder = StateGraph(TaskTypeState)
prepare_molecule_graph = prepare_molecule_builder.compile()
chat_graph = chat_builder.compile()
single_point_graph = single_point_builder.compile()
geomopt_graph = geomopt_builder.compile()
ts_graph = ts_builder.compile()
freq_graph = freq_builder.compile()
ir_graph = ir_builder.compile()
irc_graph = irc_builder.compile()
raman_graph = raman_builder.compile()
reaction_graph = reaction_builder.compile()
# uvvis_spc_graph = uvvis_spc_builder.compile()
uvvis_workflow_graph = uvvis_workflow_builder.compile()




task_type_builder.add_node("task_type_node",task_type_node)
task_type_builder.add_node("prepare_molecule_node",prepare_molecule_graph)
task_type_builder.add_node("chat_node",chat_graph)
task_type_builder.add_node("single_point_node",single_point_graph)
task_type_builder.add_node("geomopt_node",geomopt_graph)
task_type_builder.add_node("ts_node",ts_graph)
task_type_builder.add_node("freq_node",freq_graph)
task_type_builder.add_node("ir_node",ir_graph)
task_type_builder.add_node("irc_node",irc_graph)
task_type_builder.add_node("raman_node",raman_graph)
task_type_builder.add_node("reaction_node",reaction_graph)
# task_type_builder.add_node("uvvis_spc_node",uvvis_spc_graph)
task_type_builder.add_node("uvvis_workflow_node", uvvis_workflow_graph)

task_type_builder.add_edge(START,"task_type_node")
task_type_builder.add_edge("prepare_molecule_node",END)
task_type_builder.add_edge("chat_node",END)
task_type_builder.add_edge("single_point_node",END)
task_type_builder.add_edge("geomopt_node",END)
task_type_builder.add_edge("ts_node",END)
task_type_builder.add_edge("freq_node",END)
task_type_builder.add_edge("ir_node",END)
task_type_builder.add_edge("irc_node",END)
task_type_builder.add_edge("raman_node",END)
task_type_builder.add_edge("reaction_node",END)
# task_type_builder.add_edge("uvvis_spc_node",END)
task_type_builder.add_edge("uvvis_workflow_node",END)
task_type_graph = task_type_builder.compile()

from .agent_template import BaseAgent
task_agent = BaseAgent(
    name='task_agent',
    description='handle different tasks',
    graph=task_type_builder,
)

