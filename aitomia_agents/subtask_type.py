"""
    Task type agent
"""
from typing import Literal
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
from .excited_state_sp import excited_state_builder

from .states import AitomiaState
from .logger import logger 
from .utils import create_agent, pretty_dict

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
    excited_state:bool=False
):
    """
    Decide which task to perform. It is necessary that one and only one task is chosen to be True.
    IMPORTANT:
    Exactly ONE and ONLY ONE task flag must be set to True.


    Args:
        prepare_molecule: Whether to get the structure of the molecule only.
        single_point: Whether to perform single point calculation.
        optimize_geometry: Whether to optimize geometry of a molecule.
        ts: Whether to get the transition state of a molecule.
        frequency: Whether to calculate frequency or thermodynamic properties of a molecule.
        ir: Whether to calculate infrared (IR) spectrum of a molecule. 
        irc: Whether to perform intrinsic reaction coordinates (IRC) calculation of a molecule. 
        raman: Whether to calculate Raman spectrum of a molecule.
        excited_state: Whether to perform excited state calculation.

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
    if excited_state: return "excited_state_node"

#-------------------------------------------------
# Agent
#-------------------------------------------------
tools = [decide_task_type]
task_type_agent = create_agent(tools=tools,tool_kwargs={"tool_choice":"decide_task_type"})
task_type_tool = ToolNode(tools)

#-------------------------------------------------
# Graph
#-------------------------------------------------

def task_type_node(state:TaskTypeState) -> Command[Literal["prepare_molecule_node","chat_node","single_point_node","geomopt_node","ts_node","freq_node","ir_node","irc_node","raman_node"]]:
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



subtask_type_builder = StateGraph(TaskTypeState)
prepare_molecule_graph = prepare_molecule_builder.compile()
chat_graph = chat_builder.compile()
single_point_graph = single_point_builder.compile()
geomopt_graph = geomopt_builder.compile()
ts_graph = ts_builder.compile()
freq_graph = freq_builder.compile()
ir_graph = ir_builder.compile()
irc_graph = irc_builder.compile()
raman_graph = raman_builder.compile()
excited_state_graph = excited_state_builder.compile()




subtask_type_builder.add_node("task_type_node",task_type_node)
subtask_type_builder.add_node("prepare_molecule_node",prepare_molecule_graph)
subtask_type_builder.add_node("chat_node",chat_graph)
subtask_type_builder.add_node("single_point_node",single_point_graph)
subtask_type_builder.add_node("geomopt_node",geomopt_graph)
subtask_type_builder.add_node("ts_node",ts_graph)
subtask_type_builder.add_node("freq_node",freq_graph)
subtask_type_builder.add_node("ir_node",ir_graph)
subtask_type_builder.add_node("irc_node",irc_graph)
subtask_type_builder.add_node("raman_node",raman_graph)
subtask_type_builder.add_node("excited_state_node",excited_state_graph)

subtask_type_builder.add_edge(START,"task_type_node")
subtask_type_builder.add_edge("prepare_molecule_node",END)
subtask_type_builder.add_edge("chat_node",END)
subtask_type_builder.add_edge("single_point_node",END)
subtask_type_builder.add_edge("geomopt_node",END)
subtask_type_builder.add_edge("ts_node",END)
subtask_type_builder.add_edge("freq_node",END)
subtask_type_builder.add_edge("ir_node",END)
subtask_type_builder.add_edge("irc_node",END)
subtask_type_builder.add_edge("raman_node",END)
subtask_type_builder.add_edge("excited_state_node",END)
subtask_type_graph = subtask_type_builder.compile()

from .agent_template import BaseAgent
task_agent = BaseAgent(
    name='subtask_agent',
    description='handle different sub tasks',
    graph=subtask_type_builder,
)

