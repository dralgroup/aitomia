"""
    Planner agent
"""
import json
import ast
import os
import time
# import mlatom as ml
from pathlib import Path
from typing import Union, Optional, Literal, List
import traceback
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END 
from langgraph.types import Command

from aitomia_agents.user_context import user_context

from .task_type import task_type_graph
from .agent_cards import agent_cards
from .file_manager import get_current_result_files_prompt, get_working_directory_agent, get_working_directory_tool

from .states import AitomiaState
from .logger import logger 
from .utils import create_agent, pretty_dict, Analysis

from langgraph.config import get_stream_writer

from .utils import create_agent, pretty_dict, FileManager
from .method_confirm import method_confirm_builder
from .judge_calculation import judge_calculation_builder


#-------------------------------------------------
# Schema
#-------------------------------------------------
class PlannerState(AitomiaState):
    task_sequence: List[str] = None
    judge_calculation: str = None

#-------------------------------------------------
# Prompts
#-------------------------------------------------
# for agent_name, agent_card in agent_cards.items():
#     if agent_name in ["sp_agent","uvvis_agent","geomopt_agent","ts_agent","freq_agent","ir_agent","raman_agent","reaction_agent","irc_agent"]:
def prompt_general_planner():
    prompt = ""
    prompt += "You need to design a workflow accoring to the user's input. The workflow is a list of tasks.\n"
    prompt += "The workflow should be user-oriented. You may add additional tasks only when they genuinely help answer the user's underlying question or achieve the user's explicit goal, but never simply because they are commonly paired steps in computational chemistry.\nDo not follow typical workflows rigidly. Instead, infer whether an additional calculation is logically useful for the user's intent. For example, a frequency calculation may be added after a geometry optimization only if it is necessary to confirm stability or relevant to what the user is trying to obtain. IR, Raman, IRC, or any other extra tasks should only appear when they meaningfully contribute to the user's purpose.\nAlways prioritize the user's intent, and ensure that any additional tasks remain tightly aligned with the user's question rather than standard procedural habits.\n"
    prompt += "Below shows all the available tasks:\n"

    # Add the description of each task to the prompt
    for agent_name, agent_card in agent_cards.items():
        if agent_name in ["sp_agent","uvvis_agent","geomopt_agent","ts_agent","freq_agent","ir_agent","raman_agent","reaction_agent"]:
            prompt += "\t" + agent_card.description_for_planner + "\n"
    # Clarify the format of AI output
    prompt += "Below is the format of your output:\n"
    prompt += "\tEach task should be put in one line.\n"
    prompt += "\tEach line should also contain the information that is needed to perform the task containing:\n \t\t- molecule (Provide the path to the molecule, or the name of the molecule (it can be common name), or information about the chemical system extracted from user's input. You can use molecule from calculation result of previous task. In this case, the name for brief summary of previous task should be explicitly specified. If none is observed, state not specified)\n\t\t- method(If the user specifies a DFT functional with a basis set, write both together in the method field, e.g., 'wB97X/Def2-SVP'. The basis set should not be listed separately or under 'other conditions')\n\t\t- Method program (if specified, otherwise state not specified)\n\t\t- Calculation program (primary recommended program)\n\t\t- other conditions, according to the user's input.\n"
    prompt += "please be very careful that method program has nothing to do with calculation program, do not infer calculation program from method program :\n"
    prompt += "    1) a **method program** (the program used to implement the method, e.g., PySCF for many QM methods), and 2) a **calculation program** (the program or tool used to perform the specific task such as geometry optimization or frequency calculation, e.g., Geometric, Gaussian, ASE, SciPy).- If the chosen method is ML-based (AIQM1, AIQM2, ANI-1ccx, UAIQM, etc.), state that no specific method program is required (or list typical ML toolkits if relevant), but still provide a recommended calculation program when applicable.\n"

    prompt += "\tIf the user's request involves a chemical reaction (e.g., reaction energy, reaction profile, barrier height, transition state search, thermochemistry of a reaction, comparing reactant and product, Atomization energy, etc.), you must design the workflow as a single reaction-level task rather than splitting it into multiple separate calculations for reactants, products, or intermediates. Do not decompose a reaction into individual single-point or geometry-optimization tasks unless the user explicitly requests such decomposition. We have dedicated reaction-handling mechanisms, so you must represent any reaction-related computation as one unified reaction task. Also, you must keep all the information from the reaction and do not simplify or summarize it."
    prompt += "If the user's request involves a chemical transformation or a comparison between two or more chemically distinct states of a system (e.g., reactant, product, intermediate, transition state), such that the requested property cannot be obtained from a single chemical species in a single state, you must design the workflow as a single reaction-level task rather than splitting it into multiple separate calculations. This includes both real and formal reactions, as well as reaction-induced changes in properties such as energies, structures, vibrational spectra (IR/Raman), electronic spectra, or charge distributions.\n"

    prompt += "\tIf answering the user's question requires calculations on multiple distinct chemical entities, and you decide NOT to use a reaction-level task,\n"
    prompt += "\tyou MUST explicitly include a separate task line for EACH required molecule, atom, radical, or ion.\n"
    prompt += "\tThis includes entities that are not explicitly mentioned by the user but are required by the definition of the requested property.\n"
    prompt += "\tFor example, calculating the proton affinity of NH3 requires separate task lines for NH3, NH4+, and H+.\n"



    prompt += "\tIf the molecule is a file, you should always provide the absolute path."
    prompt += "\tThe user might provide common names or chemical formula of molecules, like water, H2, O2, H2O, etc. They are molecules."
    prompt += "\tIf the user asks to do geometry optimization only, do not plan for frequency calculations. If the user asks to do frequency calculations only, do not plan to do geometry optimizations.\n"
    prompt += "\tIR and Raman spectrum calculation already includes geomtry optimization. Do not do geometry optization or frequency calculation before IR and Raman spectrum calculation.\n"
    prompt += "\tRules for Handling UV-Vis Calculation Requests:"
    prompt += "\t\tRule 1. If the user requests UV-vis calculation, go to the `uvvis_agent` directly and do not go to other tasks."
    prompt += f"\t\tRule 2. For any UV-Vis calculation request, do not add a separate geometry optimization step to the main workflow. Directly invoke the uvvis_agent, which will autonomously assess and manage the necessity of optimization within its sub-process.\n"
    prompt += f"\t\tRule 3. If the user requests a UV-Vis spectrum using the OMNI-P2x method, you must proactively ask the user to specify a different method for the geometry optimization stage, as OMNI-P2x is not suitable for that purpose.\n"
    prompt += "\tFor reactions, do not need to provide the stoichiometric coefficients."
    prompt += "\tNo additional explanations or comments are needed.\n"
    prompt += "\tDo not provide the working directory in each line.\n"
    prompt += "Notes:\n"
    prompt += "\tDesign the computational method and program based on the user-confirmed latest calculation parameters.\n"
    prompt += "\tCalculating energy means doing single point calculation.\n"
    prompt += "\tsingle-point calculation can only provide the electronic energy and, optionally, the energy gradient. Properties that require the Hessian, such as vibrational frequencies, zero-point energy, thermal corrections, enthalpy, entropy, or Gibbs free energy, must be obtained from a frequency calculation. Do not attempt to compute thermochemistry from a single-point calculation alone.\n"
    prompt += "\tNotice: For standard computational tasks, you must confirm the user's selected computational method before proceeding."


    return prompt
PROMPT_GENERAL_PLANNER = prompt_general_planner()



def prompt_workdir_manager():
    current_dir = Path.home()
    prompt = ""
    prompt += f"You need to manage the working directory of a given task. The user may provide a general working directory. If not, use {current_dir} as the general working directory. The working directory of the given task should be a sub-directory of the general working directory. The naming of the sub-directory should be consistent with the task.\n"
    prompt += "**Output format:**\n"
    prompt += "- The working directory of this task is **a sub-directory of the general working directory**\n"
    prompt += "**Examples:**\n"
    prompt += "- The working directory of this task is /general/working/directory/single_point"

    return prompt

PROMPT_WORKDIR_MANAGER = prompt_workdir_manager()
#-------------------------------------------------
# Tool functions
#-------------------------------------------------


#-------------------------------------------------
# Agent
#-------------------------------------------------
llm = create_agent()

#-------------------------------------------------
# Graph
#-------------------------------------------------
def judge_calculation_node(state:PlannerState):
    judge_calculation_graph = judge_calculation_builder.compile()
    judge_state = judge_calculation_graph.invoke(state)
    logger.debug(f'judge_state: {judge_state}')
    
    # Include a messages_to_user so that task_title can be captured in the stream
    task_title = judge_state.get('task_title', 'New Conversation')
    if judge_state['judge_calculation'] == 'calculation':
        return {
            "judge_calculation":judge_state['judge_calculation'],
            "task_title":task_title,
            "messages_to_user": [AIMessage(content=f"")] # Empty message to carry task_title
        }
    else:
        return {
            "messages": judge_state['messages'],
            "judge_calculation":judge_state['judge_calculation'],
            "task_title":task_title,
            "messages_to_user": [AIMessage(content=f"")] # Empty message to carry task_title
        }


def get_working_directory_node(state:PlannerState): 
    init_start_time = Analysis.get_start_time()
    if init_start_time == 0:
        current_time = time.time()
        Analysis.set_start_time(current_time)
    logger.info("Start getting working directory")
    try:
        prompt = SystemMessage(content=f"User's home directory is {Path.home()}")
        user_choose_path = SystemMessage(content=f"The chosen path is: {FileManager.read_path()}")
        logger.debug(f'user_choose_path: {user_choose_path}')
        logger.debug(f'input of working directory node: {state.messages+[user_choose_path]+[prompt]}')
        response = get_working_directory_agent.invoke(state.messages+[user_choose_path]+[prompt]) #user msg + user choose + sys path
        output = get_working_directory_tool.invoke({"messages":[response]})["messages"][-1].content
        logger.debug(f'output: {output}')

        output = json.loads(output)
        # FileManager.set_path(output["working_directory"])
        current_workdir = output["working_directory"]
        logger.debug(f"Extracted working directory: {current_workdir}")
        if not os.path.exists(current_workdir):
            os.makedirs(current_workdir)
        stack = [current_workdir]
        return {
            "working_directory":current_workdir,
            "working_directory_stack": stack,
        }

    
    except Exception as e:
        error_message = f"Error in get_working_directory_node: {str(e)}"
        error_info = traceback.format_exc()
        logger.error(error_message + error_info)
        response = Analysis.error_analysis(error_message)
        error_analysis = error_message + '\n' + '\n' + response
        # writer = get_stream_writer()
        # writer(f"❌ {error_message}")
        return {
            "error": error_message,
            "has_error": True,
            "messages": [AIMessage(content=f"Failed to get working directory: {error_message}")],
            "messages_to_user": [AIMessage(content=f"Failed to get working directory: {error_analysis}")]
        }

def method_confirm_node(state:PlannerState):
    method_confirm_graph = method_confirm_builder.compile()
    method_state = method_confirm_graph.invoke(state)
    return method_state

        
def general_planner_node(state:PlannerState):
    logger.info("Start making a plan of the required calculations")
    message = "Start making a plan of the required calculations"
    
    try:
        # logger.debug(PROMPT_GENERAL_PLANNER)
        general_planner_prompt = SystemMessage(content=PROMPT_GENERAL_PLANNER)
        response = llm.invoke(state.messages+[general_planner_prompt])
        sequence = [each.strip() for each in response.content.split('\n')]
        logger.debug(sequence) 
        out_msg = message + '\n' + "The calculation plan is:\n" + response.content
        logger.debug(f'work_flow_design {out_msg}')

        # writer = get_stream_writer()
        # writer("The calculation plan is:\n" + response.content)
        
        return {"task_sequence": sequence,
                "messages": [AIMessage(content="The calculation plan is:\n" + response.content)],
                "messages_to_user": [AIMessage(content=out_msg)]}
    
    except Exception as e:
        error_message = f"Error in general_planner_node: {str(e)}"
        error_info = traceback.format_exc()
        logger.error(error_message + error_info)
        # writer = get_stream_writer()
        # writer(f"❌ {error_message}")
        response = Analysis.error_analysis(error_message)
        error_analysis = error_message + '\n' + '\n' + response
        return {
            "error": error_message,
            "has_error": True,
            "messages": [AIMessage(content=f"Failed to make plan: {error_message}")],
            "messages_to_user": [AIMessage(content=f"Failed to make plan: {error_analysis}")],
        }

def workdir_manager_node(state:PlannerState):
    logger.info("Start managing the working directory of a task")
    
    try:
        # workdir_manager_prompt = SystemMessage(content=PROMPT_WORKDIR_MANAGER)
        task_sequence = state.task_sequence
        task = task_sequence.pop(0)
        if state.result_files is None: state.result_files = []

        task_message = SystemMessage(content=task) #不用systemessage,  就直接用content防止偏好
        #1.Perform the calculation of a chemical reaction. This task already contains the geometry optimization and frequency calculation of each reactant and product. molecule: ethanol to acetaldehyde oxidation reaction method: AIQM2 Method program: not specified Calculation program: Geometric for geometry optimization,PySCF for frequency calculation other conditions: generate IR spectra for both ethanol and acetaldehyde to compare O–H and C=O absorption bands
        result_files_message = get_current_result_files_prompt(state.result_files) 
        Analysis.add_summary({'current_task':task_message})
        return {
            "task_sequence":task_sequence,
            "current_task_messages":[[result_files_message,task_message]], 
            "result_files": state.result_files
        }
    
    except Exception as e:
        error_message = f"Error in workdir_manager_node: {str(e)}"
        error_info = traceback.format_exc()
        logger.error(error_message + error_info)
        response = Analysis.error_analysis(error_message)
        error_analysis = error_message + '\n' + '\n' + response
        # writer = get_stream_writer()
        # writer(f"❌ {error_message}")
        return {
            "error": error_message,
            "has_error": True,
            "messages": [AIMessage(content=f"Failed to manage working directory: {error_message}")],
            "messages_to_user": [AIMessage(content=f"Failed to manage working directory: {error_analysis}")],
        }

def conditional_edge(state:PlannerState):
    """Check for errors first, then check remaining tasks"""
    # If error occurred, stop the workflow
    if state.has_error:
        logger.warning("Error detected in task execution, stopping workflow")
        return END
    
    # Otherwise check remaining tasks
    ntasks = len(state.task_sequence)
    logger.info(f"Number of remaining tasks: {ntasks}")
    if ntasks == 0:
        return 'summary_node'
    else:
        return "workdir_manager_node"

def check_error(state:PlannerState):
    """Check if an error has occurred and route accordingly"""
    if state.has_error:
        logger.warning("Error detected, skipping remaining steps")
        return END
    return "continue"

def summary_prompt():
    prompt = """
    You are an assistant that summarizes completed quantum chemistry calculations for record-keeping. 

    Input: a detailed AIMessage containing:
    - The user's original question or request
    - The detailed calculation results, including energy, optimized geometry, and output files
    - Any commentary on computation or chemical insight

    Task: generate a **concise summary** that includes:
    1. User question/request
    2. Key results (final energy, optimized coordinates or bond lengths/angles, output file location)
    3. Status of the calculation (completed)

    Constraints:
    - Keep it brief and structured (no long paragraphs)
    - Do not include full detailed explanations or commentary
    - Output in a format suitable for logging or feeding into the next calculation step

    Example output:

    User question: Optimize H2O
    Key results:
    - Final Energy: -76.3838 Hartree
    - O-H bond lengths: ~0.960 Å
    - H-O-H angle: ~104.5°
    - Output file: /home/user/water_optimization_AIQM2_1/optmol.json
    Status: Completed
    """
    return prompt
SUMMARY_PROMPT = summary_prompt()



def summary_node(state:PlannerState):
    current_time = time.time()
    readable_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
    Analysis.set_end_time(current_time)
    time_consuming = Analysis.get_time_consuming()
    analysis_result = Analysis.result_analysis()
    Analysis.set_start_time(current_time)
    summary_content = f"""
    Calculation Summary
    -------------------
    Time: {readable_time}
    User question: {Analysis.get_user_query()}

    The Final summary for this calculation is:
    {analysis_result}

    Status: Completed
    """
    final_msg = AIMessage(content=summary_content)
    summary_msg_prompt = SystemMessage(content=SUMMARY_PROMPT)
    restore_msgs = llm.invoke([summary_msg_prompt] + [final_msg] ).content
    logger.info(f'Final answer is: [AIMessage(content=f"Time:{readable_time}. User question:{Analysis.get_user_query()}. The Final summary for this calculation is:{analysis_result}\n")]')
    final_msg = [AIMessage(content=restore_msgs)]
    
    return {"messages": final_msg,
            "messages_to_user": [AIMessage(content=f'Summary for the calculations:\nThe total time usage (including thinking time):{time_consuming}\n{analysis_result}\n')],
            "status":"completed"}


planner_builder = StateGraph(PlannerState)


planner_builder.add_node("get_working_directory_node",get_working_directory_node)
planner_builder.add_node("judge_calculation_node", judge_calculation_node)
planner_builder.add_node("method_confirm_node", method_confirm_node)
planner_builder.add_node("general_planner_node",general_planner_node)
planner_builder.add_node("workdir_manager_node",workdir_manager_node)
planner_builder.add_node("summary_node",summary_node)
planner_builder.add_node("task_node",task_type_graph)

planner_builder.add_edge(START,"get_working_directory_node")
planner_builder.add_conditional_edges("get_working_directory_node", check_error, {"continue": "judge_calculation_node", END: END})
planner_builder.add_conditional_edges(
    "judge_calculation_node",
    lambda state: state.judge_calculation,
    {
        "calculation":"method_confirm_node",
        "no_calculation":END
    }

)
planner_builder.add_edge("method_confirm_node", "general_planner_node")
planner_builder.add_conditional_edges("general_planner_node", check_error, {"continue": "workdir_manager_node", END: END})
planner_builder.add_conditional_edges("workdir_manager_node", check_error, {"continue": "task_node", END: END})
planner_builder.add_conditional_edges(
    "task_node",
    conditional_edge,
    {
        "workdir_manager_node",
        "summary_node",
    }
)
planner_builder.add_edge("summary_node", END)

planner_graph = planner_builder.compile()

from .agent_template import BaseAgent
planner_agent = BaseAgent(
    name='planner_agent',
    description='planner',
    graph=planner_builder,
)