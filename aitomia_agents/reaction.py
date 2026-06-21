"""
    Reaction agent
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
from .subtask_type import subtask_type_graph
from .agent_cards import agent_cards
from .file_manager import get_current_result_files_prompt, get_folder_name_node
from .optimize_geometry import geomopt_builder 
from .frequency import freq_builder 

from .states import AitomiaState 
from .logger import logger 
from .utils import create_agent, pretty_array, Analysis
from langgraph.config import get_stream_writer

#-------------------------------------------------
# Schema
#-------------------------------------------------
class ReactionState(AitomiaState):
    reactant_filenames: List[str] = None
    product_filenames: List[str] = None
    # optprog: str 
    # freqprog: str 
    reaction_task_sequence: List[str] = None
    task_type: str = None
    # stoichiometric_coefficients: List[List[int]] = None

#-------------------------------------------------
# Prompts
#-------------------------------------------------
def prompt_reactants_products():
    prompt = ""
    prompt += "You need to extract the reactants and products of the reaction from the given messages.\n"
    # Clarify the format of AI output
    prompt += "Below is the format of your output:\n"
    prompt += "\tThe output should contain two lines, the first line specifies all the reactants, and the second line specifies the products. In each line, the molecule should be seperated by `|`.\n"
    prompt += "\tEach molecule should be only either the molecule name or the absolute path to the molecule file, according to the given messages.\n"
    prompt += "\tNo additional explanations or comments are needed.\n"
    return prompt
PROMPT_REACTANTS_PRODUCTS = prompt_reactants_products()




def prompt_get_reaction_equation():
    prompt = ""
    prompt += "You are given a natural language description of a chemical reaction task.\n"
    prompt += "Your task is to write the chemical reaction equation that should be used for the calculation.\n\n"

    prompt += "Rules:\n"
    prompt += "- Infer standard reactants and products implied by the reaction type (e.g., oxidation, reduction, hydrolysis).\n"
    prompt += "- You may add common species such as O2, H2O, H2, etc., if they are typically involved.\n"
    prompt += "- You do NOT need to strictly balance stoichiometry.\n"
    prompt += "- Focus on listing the main reactants and products relevant for the calculation.\n"
    prompt += "- Use molecule names or simple chemical formulas.\n"
    prompt += "- If multiple reasonable equations exist, choose the most common textbook form.\n\n"

    prompt += "Output format:\n"
    prompt += "- Output ONLY a single reaction equation in the form:\n"
    prompt += "  reactant1 + reactant2 -> product1 + product2\n"
    prompt += "- Do NOT add explanations, comments, or extra text.\n"

    return prompt


PROMPT_GET_REACTION_EQUATION = prompt_get_reaction_equation()
def prompt_show_reactants_products(reactants,products):
    prompt = ""
    prompt += "The reactants of the reaction are:\n"
    for each in reactants:
        prompt += f"\t{each}\n"
    prompt += "The products of the reaction are:\n"
    for each in products:
        prompt += f"\t{each}\n"
    return prompt 


# for agent_name, agent_card in agent_cards.items():
#     if agent_name in ["sp_agent","excited_state_agent","geomopt_agent","ts_agent","freq_agent","ir_agent","raman_agent","irc_agent"]:
#         prompt += "\t" + agent_card.description_for_planner + "\n"
def prompt_reaction_planner():
    prompt = ""
    prompt += "You need to design a workflow according to the given messages. The workflow is a list of tasks."
    prompt += "The workflow should be user-oriented. You may add additional tasks only when they genuinely help answer the user's underlying question or achieve the user's explicit goal, but never simply because they are commonly paired steps in computational chemistry.\nDo not follow typical workflows rigidly. Instead, infer whether an additional calculation is logically useful for the user's intent. For example, a frequency calculation may be added after a geometry optimization only if it is necessary to confirm stability or relevant to what the user is trying to obtain. IR, Raman, IRC, or any other extra tasks should only appear when they meaningfully contribute to the user's purpose.\nAlways prioritize the user's intent, and ensure that any additional tasks remain tightly aligned with the user's question rather than standard procedural habits.\n"
    prompt += "All planned tasks must be physically and chemically well-defined for the specific system they are applied to.\n"
    prompt += "Tasks that are undefined or meaningless for a given species (e.g., frequency analysis for isolated atoms) must never be included.\n"
    prompt += "You should use your understanding of chemistry to decide which steps are necessary. Do not make common-sense errors; for example, do not compute  reaction enthalpies using only single-point energies without including vibrational frequency calculations to account for zero-point and thermal corrections.\n"
    prompt += "Below shows all the available tasks: "
    # Add the description of sub task to the prompt
    for agent_name, agent_card in agent_cards.items():
        if agent_name in ["sp_agent","excited_state_agent","geomopt_agent","ts_agent","freq_agent","ir_agent","raman_agent"]:
            prompt += "\t" + agent_card.description_for_planner + "\n"

    prompt += "Below is the format of your output:\n"
    prompt += "\tEach task should be put in one line.\n"
    prompt += "\tEach line should also contain the information that is needed to perform the task, e.g., molecule, method, Method program (if specified, otherwise state not specified), Calculation program (primary recommended program).\n"
    prompt += "\tNo additional explanations or comments are needed.\n"
    prompt += "\tDo not provide the working directory in each line.\n"
    return prompt 

PROMPT_REACTION_PLANNER = prompt_reaction_planner()

def prompt_extract_result_files():
    prompt = ""
    prompt += "You need to extract all the reactant and product result files with thermodynamic properties from the given messages.\n"
    prompt += "Below is the format of your output:\n"
    prompt += "\tThe output should contain two lines, the first line specifies the all the reactant result files, and the second line specifies all the product result files. In each line, the files should be seperated by `|`.\n"
    prompt += "\tEach file should be absolute path to the result file, according to the given messages.\n"
    prompt += "\t4. Do NOT include any XML tags like <Path> or any other surrounding markup; only output the plain file paths.\n"
    prompt += "\tNo additional explanations or comments are needed.\n"
    return prompt 

PROMPT_EXTRACT_RESULT_FILES = prompt_extract_result_files()

PROMPT_REACTION_ANALYSIS = "Below shows the result of the reaction calculation. It might contain multiple properties of the reaction, like Gibbs free energy change, enthalpy change, etc. Please give a brief summary of the resulf in a few sentences. If there is any file path, please use <Path>file_path</Path> format to indicate the file path. Note: This is a strict computational task. All calculations must be done with full numerical precision, and all reported results must exactly reflect the computed values. Do not perform any rounding or truncation of decimal points."

#-------------------------------------------------
# Tool functions
#-------------------------------------------------
def decide_task_type(
    optimize_geometry:bool=False,
    frequency:bool=False,
):
    """
    Decide which task to perform. It is necessary that one and only one task is chosen to be True.

    Args:
        optimize_geometry: Whether to optimize geometry of a molecule.
        frequency: Whether to calculate frequency or thermodynamic properties of a molecule.
    """

    argvals = list(locals().values())
    if sum(argvals) != 1:
        # return "task_type_node"
        # Selecting task_type_node means more than one task or no task is choosen. Raise a warning.   
        logger.warning("More than one task or no task is choosen")

    if optimize_geometry: return "geomopt_node"
    if frequency: return "freq_node"

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
def get_reactants_products_node(state:ReactionState):
    logger.info("Start getting reactants and products")
    message = "Start getting reactants and products"
    get_reaction_equation = SystemMessage(content=PROMPT_GET_REACTION_EQUATION)
    equation_output = llm.invoke(state.current_task_messages[-1]+[get_reaction_equation])
    equation_output_prompt = SystemMessage(content = 'The reaction currently under calculation is::' + equation_output.content + '\n')
    reactants_products_prompt = SystemMessage(content=PROMPT_REACTANTS_PRODUCTS)
    response = llm.invoke([equation_output_prompt]+[reactants_products_prompt])

    logger.debug("Response from llm:")
    logger.debug(response.content)

    reactants,products = response.content.split('\n')
    reactants = reactants.strip().split('|')
    products = products.strip().split('|')

    return {
        "reactant_filenames": reactants,
        "product_filenames": products,
        "messages_to_user": [equation_output_prompt] + [AIMessage(content=message)],
    }

def reaction_planner_node(state:ReactionState):
    logger.info("Start making a plan of the reaction calculation")
    message = "Start making a plan of the reaction calculation"
    show_reactants_products_prompt = SystemMessage(content=prompt_show_reactants_products(state.reactant_filenames,state.product_filenames))
    reaction_planner_prompt = SystemMessage(content=PROMPT_REACTION_PLANNER)

    # Create an empty list of message for sub-tasks (geometry optimization or frequency calculation)
    current_task_messages = state.current_task_messages
    current_task_messages.append([])

    response = llm.invoke(current_task_messages[-2] + [show_reactants_products_prompt,reaction_planner_prompt])
    message = message + f"\n{response.content}"  #内部的流程也打印一下

    logger.debug("Agent response:")
    logger.debug(response.content)

    sequence = [each.strip() for each in response.content.split("\n")]
    logger.debug(sequence) 

    return {
        "reaction_task_sequence": sequence,
        "current_task_messages": current_task_messages,
        "messages_to_user": [AIMessage(content=message)]
    }

def workdir_manager_node(state:ReactionState):
    logger.info("Start managing the working directory of a task")
    message = "Start managing the working directory of a task"
    reaction_task_sequence = state.reaction_task_sequence
    task = reaction_task_sequence.pop(0)
    if state.result_files is None: state.result_files = []


    task_message = SystemMessage(content=task)
    result_files_message = get_current_result_files_prompt(state.result_files) #其实就是往working_manager_node里面以及对应的summary里面加一个summary就行了

    task_messages = state.current_task_messages

    # Put the last message (summary) in the sub-task message list into the reaction task message list
    if len(task_messages[-1]) > 0:
        task_messages[-2].append(task_messages[-1][-1])

    # Clear the sub-task message list and put the new task into it
    task_messages[-1] = [result_files_message,task_message]

    Analysis.add_summary({'current_task':task_message})
    return {
        "reaction_task_sequence":reaction_task_sequence,
        "current_task_messages":task_messages,
        "result_files": state.result_files,
        "messages_to_user": [AIMessage(content=message)],
    }


def reaction_analysis_node(state:ReactionState): 
    logger.info("Start analyzing the result of reaction")
    message = "Start analyzing the result of reaction"

    try:
        show_reactants_products_prompt = SystemMessage(content=prompt_show_reactants_products(state.reactant_filenames,state.product_filenames))
        extract_result_files_prompt = SystemMessage(content=PROMPT_EXTRACT_RESULT_FILES)
        analysis_prompt = SystemMessage(content=PROMPT_REACTION_ANALYSIS)

        # Put the last message (summary) in the sub-task message list into the reaction task message list
        task_messages = state.current_task_messages
        if len(task_messages[-1]) > 0:
            task_messages[-2].append(task_messages[-1][-1])

        # Remove the messages of sub-tasks (geometry optimization or frequency calculation)
        task_messages.pop()

        # Analyze the result of the reaction
        response = llm.invoke(task_messages[-1] + [show_reactants_products_prompt,extract_result_files_prompt])
        logger.debug("Response of llm:")
        logger.debug(response.content)
        reactant_filenames, product_filenames = response.content.split("\n")
        reactant_filenames = reactant_filenames.split('|')
        product_filenames = product_filenames.split('|')
        reactants = [ml.data.molecule.load(each,format='json') for each in reactant_filenames]
        products = [ml.data.molecule.load(each,format='json') for each in product_filenames]
        
        stoichiometric_coefficients = get_stoichiometric_coefficients(reactants,products)
        if isinstance(stoichiometric_coefficients,str):
            stoichiometric_coefficients_message = stoichiometric_coefficients + "\n"
            stoichiometric_coefficients = np.ones(len(reactants+products))
        else:
            stoichiometric_coefficients_message = ""
        stoichiometric_coefficients_message += "Current reaction:\n"
        reactants_str = []
        products_str = []
        for ii in range(len(reactants)):
            reactants_str.append(f"{stoichiometric_coefficients[ii]:.2f} R{ii+1}")
        for ii in range(len(products)):
            products_str.append(f"{stoichiometric_coefficients[len(reactants)+ii]:.2f} P{ii+1}")
        stoichiometric_coefficients_message += " + ".join(reactants_str)
        stoichiometric_coefficients_message += " == " + " + ".join(products_str) + "\n"
        

        enthalpy_change = (
            np.sum([each.H * stoichiometric_coefficients[len(reactants) + ii]
                    for ii, each in enumerate(products)])
            - np.sum([each.H * stoichiometric_coefficients[ii]
                    for ii, each in enumerate(reactants)])
            if all(hasattr(each, "H") and each.H is not None
                for each in (*reactants, *products))
            else None
        )

        gibbs_free_energy_change = (
            np.sum([each.G * stoichiometric_coefficients[len(reactants)+ii]
                    for ii, each in enumerate(products)])
            - np.sum([each.G * stoichiometric_coefficients[ii]
                    for ii, each in enumerate(reactants)])
            if all(hasattr(each, "G") and each.G is not None
                for each in (*reactants, *products))
            else None
        )

        energy_change = (
            np.sum([each.energy * stoichiometric_coefficients[len(reactants)+ii]
                    for ii, each in enumerate(products)])
            - np.sum([each.energy * stoichiometric_coefficients[ii]
                    for ii, each in enumerate(reactants)])
            if all(hasattr(each, "energy") and each.energy is not None
                for each in (*reactants, *products))
            else None
        )

        # enthalpy_change = np.sum([each.H*stoichiometric_coefficients[len(reactants)+ii] for ii,each in enumerate(products)]) - np.sum([each.H*stoichiometric_coefficients[ii] for ii,each in enumerate(reactants)])
        # gibbs_free_energy_change = np.sum([each.G*stoichiometric_coefficients[len(reactants)+ii] for ii,each in enumerate(products)]) - np.sum([each.G*stoichiometric_coefficients[ii] for ii,each in enumerate(reactants)])
        # energy_change = np.sum([each.energy*stoichiometric_coefficients[len(reactants)+ii] for ii,each in enumerate(products)]) - np.sum([each.energy*stoichiometric_coefficients[ii] for ii,each in enumerate(reactants)])

        # Output log
        output_log = ""
        if len(state.reactant_filenames) == len(reactant_filenames):
            output_log += f"Reactants:\n"
            for ii in range(len(state.reactant_filenames)):
                output_log += f"Thermodynamic properties of {state.reactant_filenames[ii]} is saved in <Path>{reactant_filenames[ii]}</Path>\n"
                output_log += f"    Electronic energy: {reactants[ii].energy} Hartree\n"
                if getattr(reactants[ii], "H", None) is not None:
                    output_log += f"    Enthalpy at 298 Kelvin: {reactants[ii].H} Hartree\n"
                if getattr(reactants[ii], "G", None) is not None:
                    output_log += f"    Gibbs free energy at 298 Kelvin: {reactants[ii].G} Hartree\n"
        if len(state.product_filenames) == len(product_filenames):
            output_log += f"Products:\n"
            for ii in range(len(state.product_filenames)):
                output_log += f"Thermodynamic properties of {state.product_filenames[ii]} is saved in <Path>{product_filenames[ii]}</Path>\n"
                output_log += f"    Electronic energy: {products[ii].energy} Hartree\n"
                if getattr(products[ii], "H", None) is not None:
                    output_log += f"    Enthalpy at 298 Kelvin: {products[ii].H} Hartree\n"
                if getattr(products[ii], "G", None) is not None:
                    output_log += f"    Gibbs free energy at 298 Kelvin: {products[ii].G} Hartree\n"

        # Prepare formatted summary
        result_message_str = "Reaction calculation result: \n"
        if energy_change is not None:
            result_message_str += f"    Electronic energy change at 298 Kelvin: {energy_change} Hartree = {energy_change*ml.constants.Hartree2kcalpermol:.2f} kcal/mol\n"
        if enthalpy_change is not None:
            result_message_str += f"    Enthalpy change at 298 Kelvin: {enthalpy_change} Hartree = {enthalpy_change*ml.constants.Hartree2kcalpermol:.2f} kcal/mol\n"
        if gibbs_free_energy_change is not None:
            result_message_str += f"    Gibbs free energy change at 298 Kelvin: {gibbs_free_energy_change} Hartree = {gibbs_free_energy_change*ml.constants.Hartree2kcalpermol:.2f} kcal/mol\n"
        result_message = AIMessage(content=result_message_str)

        # Get analysis of formatted summary from LLM
        response = llm.invoke(task_messages[-1]+[analysis_prompt,result_message])
        logger.debug("Response of llm:")
        logger.debug(response.content)
        task_messages[-1].append(AIMessage(content=response.content))

        # Write formatted summary into summary.out
        with open(os.path.join(state.working_directory,"summary.out"),'w') as f:
            f.write(output_log + '\n' + result_message_str)
            Analysis.add_summary({'current_task_summary':output_log + '\n' + result_message_str})
        result_message_str += f"Detailed summary can be found in {os.path.join(state.working_directory,'summary.out')}"

        # Deal with the working directory and the stack at the end of the task
        stack = state.working_directory_stack
        stack.pop()
        if len(stack) == 0:
            current_working_directory = user_context.home_dir or Path.home()
        else:
            current_working_directory = stack[-1]

        all_messages = output_log + stoichiometric_coefficients_message + result_message_str + '\n' + response.content
        logger.debug(all_messages)
        # writer = get_stream_writer()
        # writer(all_messages + '\n' + "Calculation completed")
        return {
            "messages": [AIMessage(content=response.content)],
            "messages_to_user": [AIMessage(content=all_messages)],
            "current_task_messages":task_messages,
            "working_directory":current_working_directory,
            "working_directory_stack":stack,
            
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
        
def get_stoichiometric_coefficients(reactants,products):
    from collections import Counter
    from scipy.linalg import null_space
    logger.info("Start getting stoiciometric coefficient of the reaction")
    all_atomic_numbers = []
    for each in reactants + products:
        all_atomic_numbers.append(each.atomic_numbers)
    atomic_numbers = np.unique(np.concatenate(all_atomic_numbers))
    logger.debug("Unique atomic numbers: " + ", ".join(str(atomic_numbers)))
    
    # Prepare the matrix
    mat = np.zeros((len(atomic_numbers),len(reactants+products)))
    for imol,each in enumerate(reactants):
        dic = Counter(list(each.atomic_numbers))
        for ii,atomic_number in enumerate(atomic_numbers):
            mat[ii,imol] = dic[atomic_number]
    for imol,each in enumerate(products):
        dic = Counter(list(each.atomic_numbers))
        for ii,atomic_number in enumerate(atomic_numbers):
            mat[ii,imol+len(reactants)] = - dic[atomic_number]
            
    # Compute the null space of the matrix
    null_space_basis = null_space(mat).T
    logger.debug("Null space of the matrix:")
    logger.debug(pretty_array(null_space_basis))
    
    if len(null_space_basis) == 0:
        logger.debug("Cannot find a solution for the stoichiometric coefficients")
        return "Cannot find a solution for the stoichiometric coefficients"
    elif len(null_space_basis) == 1:
        vec = null_space_basis[0]
        vec = vec / np.min(vec)
        logger.debug("Find a solution for the stoichiometric coefficients")
        logger.debug(pretty_array(vec))
        return vec 
    elif len(null_space_basis) > 1:
        logger.debug("Find multiple solutions, cannot get a unique set of stoichiometric coefficients")
        return "Find multiple solutions, cannot get a unique set of stoichiometric coefficients"   

def conditional_edge(state:ReactionState):
    ntasks = len(state.reaction_task_sequence)
    logger.info(f"Number of remaining tasks for reaction: {ntasks}")
    if ntasks == 0:
        return "reaction_analysis_node"
    else:
        return "workdir_manager_node"

# def extract_result(state:ReactionState):
#     pass


reaction_builder = StateGraph(ReactionState)


geomopt_graph = geomopt_builder.compile()
freq_graph = freq_builder.compile()

reaction_builder.add_node("get_folder_name_node",get_folder_name_node)
reaction_builder.add_node("get_reactants_products_node",get_reactants_products_node)
reaction_builder.add_node("reaction_planner_node",reaction_planner_node)
reaction_builder.add_node("workdir_manager_node",workdir_manager_node)
reaction_builder.add_node("task_type_node",subtask_type_graph)
reaction_builder.add_node("reaction_analysis_node",reaction_analysis_node)

reaction_builder.add_edge(START,"get_folder_name_node")
reaction_builder.add_edge("get_folder_name_node","get_reactants_products_node")
reaction_builder.add_edge("get_reactants_products_node","reaction_planner_node")
reaction_builder.add_edge("reaction_planner_node","workdir_manager_node")
reaction_builder.add_edge("workdir_manager_node","task_type_node")
reaction_builder.add_conditional_edges(
    "task_type_node",
    conditional_edge,
    {
        "workdir_manager_node",
        "reaction_analysis_node",
    }
)
# reaction_builder.add_conditional_edges(
#     "freq_node",
#     conditional_edge,
#     {
#         "workdir_manager_node",
#         "reaction_analysis_node",
#     }
# )
reaction_builder.add_edge("reaction_analysis_node",END)

reaction_graph = reaction_builder.compile()

from .agent_template import BaseAgent
reaction_agent = BaseAgent(
    name='reaction_agent',
    description='Level1 task: Perform the calculation of a chemical reaction. This task is level1 task which already contains the geometry optimization and frequency calculation and other level2 tasks of each reactant and product, so if you choose this one , no need to do addtional calculations for other level2 tasks.',
    graph=reaction_builder,
)