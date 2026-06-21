"""
    Single point agent
"""
# import json
# import ast
import os
import mlatom as ml
import numpy as np
from typing import Union#, Optional
import traceback
from langchain_core.messages import SystemMessage, AIMessage
# from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END 

from .method_judge import method_builder
from .prepare_molecule import prepare_molecule_builder
from .file_manager import get_folder_name_node, get_result_file_node

from .states import AitomiaState
from .logger import logger 
from .utils import create_agent, pretty_array, Analysis

from langgraph.config import get_stream_writer

#-------------------------------------------------
# Schema
#-------------------------------------------------
class SinglePointState(AitomiaState):
    molecule_file_name: str = None
    method: str = None
    program: Union[str,None] = None
    single_point_result: Union[str,None] = None

#-------------------------------------------------
# Prompts
#-------------------------------------------------
PROMPT_SINGLE_POINT_ANALYSIS = "Below shows the result of the single point calculation. It might contain multiple properties of the molecule, e.g., energy, energy gradients, dipole moments, hessian, etc. You should give explicitly the absolute path of the result file for further calculation of analysis. Please give a brief summary of the result in a few sentences. `Nan` in the result means the corresponding properties are not calculated. If there is any file path, please use <Path>file_path</Path> format to indicate the file path."

#-------------------------------------------------
# Tool functions
#-------------------------------------------------

#-------------------------------------------------
# scripts
#-------------------------------------------------
# render script with jinja2 template
run_py = """#!/bin/env python

import mlatom as ml 
import os 

# 1. load molecule
molecule_path = {{molecule_file_name | pyrepr}}
{% if molecule_format == 'json' %}
mol = ml.molecule.load(molecule_path, format='json')
{% elif molecule_format == 'xyz' %}
mol = ml.molecule.from_xyz_file(molecule_path)
{% endif %}

# 2. define method
MODEL = ml.methods(method={{method | pyrepr}}, program={{program | pyrepr}})

# 3. Single point calculation
working_directory = {{working_directory | pyrepr}}
MODEL.working_directory = working_directory
MODEL.predict(molecule=mol,calculate_energy=True,calculate_energy_gradients=True)

# 4. Save results
mol.dump({{single_point_result | pyrepr}},format='json')

"""

#-------------------------------------------------
# Agent
#-------------------------------------------------
llm = create_agent()

#-------------------------------------------------
# Graph
#-------------------------------------------------

def single_point_node(state:SinglePointState):
    logger.info("Start single point calculation")
    message = "Start single point calculation"

    try:
        molecule_file_name = state.molecule_file_name
        if '.xyz' in molecule_file_name:
            mol = ml.data.molecule.from_xyz_file(molecule_file_name)
        elif '.json' in molecule_file_name:
            mol = ml.data.molecule.load(molecule_file_name,format='json')
        else:
            raise ValueError("Unknown molecule file format")
        
        method = ml.models.methods(method=state.method,program=state.program)
        method.predict(molecule=mol,calculate_energy=True,calculate_energy_gradients=True)
        # Result file name
        filename = "single_point.json"
        if os.path.exists(os.path.join(state.working_directory,filename)):
            ii = 1
            while True:
                if os.path.exists(os.path.join(state.working_directory,f"single_point_{ii}.json")):
                    ii += 1
                else:
                    break 
            filename = f"single_point_{ii}.json"
        result_file_name = os.path.join(state.working_directory,filename)
        mol.dump(result_file_name,format='json')

        return {
            "single_point_result":result_file_name,
            "messages_to_user": [AIMessage(content=message)]
        }

    except Exception as e:
        error_message = f"Error in single_point_node: {str(e)}"
        error_info = traceback.format_exc()
        logger.error(error_message + error_info)
        response = Analysis.error_analysis(error_message)
        error_analysis = error_message + '\n' + '\n' + response
        # writer = get_stream_writer()
        # writer(f"❌ {error_message}")
        return {
            "error": error_message,
            "has_error": True,
            "messages": [AIMessage(content=f"Single point calculation failed: {error_message}")],
            "messages_to_user": [AIMessage(content=f"Single point calculation failed: {error_analysis}")],
        }

def single_point_coder_node(state):
    from jinja2 import Environment 

    logger.info("Single point calculation coder")
    message = "Single point calculation coder"

    # Prepare result filename
    filename = "single_point.json"
    if os.path.exists(os.path.join(state.working_directory,filename)):
        ii = 1
        while True:
            if os.path.exists(os.path.join(state.working_directory,f"single_point_{ii}.json")):
                ii += 1
            else:
                break 
        filename = f"single_point_{ii}.json"
    result_file_name = os.path.join(state.working_directory,filename)
    state.single_point_result = result_file_name

    input_args = state.model_dump().copy()
    if state.molecule_file_name[-4:] == ".xyz": input_args['molecule_format'] = 'xyz'
    elif state.molecule_file_name[-5:] == ".json": input_args['molecule_format'] = 'json' 
    else: raise ValueError("Unhandled molecule format")

    env = Environment(autoescape=False)
    env.filters["pyrepr"] = repr
    code = env.from_string(run_py).render(**input_args)

    working_directory = state.working_directory_stack[-1]
    python_script = os.path.join(working_directory, "run_sp.py")

    with open(python_script, 'w') as f:
        f.write(code)

    logger.debug("Output state:"); logger.debug({"scripts":python_script})
    return {"scripts":python_script,
            "single_point_result": result_file_name,
            "messages_to_user": [AIMessage(content=message)]} 

def single_point_exec_node(state):
    logger.info("Start single point calculation")
    message = "Start single point calculation"
    # writer = get_stream_writer()
    # writer(message)
    try:
        python_script = state.scripts
        result_log = os.path.join(state.working_directory_stack[-1], "sp.log")
        result_err = os.path.join(state.working_directory_stack[-1], "sp.err")

        logger.debug("Log file:"); logger.debug("\t" + result_log)
        logger.debug("Err file:"); logger.debug("\t" + result_err)
        logger.debug("Exe file:"); logger.debug("\t" + python_script)
        
        # Execute and check return code
        exit_code = os.system(f"/bin/env python {python_script} > {result_log} 2>{result_err}")
        
        # Check if execution failed
        if exit_code != 0:
            # Read error output
            error_content = ""
            if os.path.exists(result_err):
                with open(result_err, 'r') as f:
                    error_content = f.read()
            
            error_message = f"Single point calculation script failed with exit code {exit_code}"
            if error_content:
                error_message += f"\nError output:\n{error_content}"
            
            logger.error(error_message)
            response = Analysis.error_analysis(error_message)
            error_analysis = error_message + '\n' + '\n' + response
            return {
                "error": error_message,
                "has_error": True,
                "messages": [AIMessage(content=error_message)],
                "messages_to_user": [AIMessage(content=error_analysis)],
            }

        return {"messages_to_user": [AIMessage(content=message)]}
    
    except Exception as e:
        error_message = f"Error in single_point_exec_node: {str(e)}"
        error_info = traceback.format_exc()
        logger.error(error_message + error_info)
        return {
            "error": error_message,
            "has_error": True,
            "messages": [AIMessage(content=f"Single point calculation failed: {error_message}")],
            "messages_to_user": [AIMessage(content=f"Single point calculation failed: {error_message}")]
        }


def single_point_analysis_node(state:SinglePointState):
    logger.info("Start analyzing the result of single point calculation")
    message = "Analyzing the result of single point calculation"
    
    try:
        mol = ml.data.molecule.load(state.single_point_result,format='json')
        energy = mol.energy
        energy_gradients = mol.energy_gradients 
        if 'dipole_moment' in mol.__dict__:
            dipole = mol.dipole_moment
        else:
            dipole = np.nan
        result_file_name = state.single_point_result
        single_point_prompt = SystemMessage(content=PROMPT_SINGLE_POINT_ANALYSIS)

        # Prepare formatted summary
        result_message_str = f"Single point calculation result: \n"
        result_message_str += f"    Energy: {energy} Hartree\n"
        result_message_str += f"    Energy gradients (Hartree/Angstrom): \n{pretty_array(energy_gradients)}\n"
        result_message_str += f"    Dipole moments (Debye): {pretty_array(dipole)}\n"
        result_message_str += f"    Result file: {result_file_name}\n"
        result_message = AIMessage(content=result_message_str)

        # Get analysis of formatted summary from LLM
        response = llm.invoke(state.current_task_messages[-1]+[single_point_prompt]+[result_message])
        task_messages = state.current_task_messages
        task_messages[-1].append(AIMessage(content=response.content))

        # Write formatted summary into summary.out
        with open(os.path.join(state.working_directory,"summary.out"),'w') as f:
            f.write(result_message_str)
            Analysis.add_summary({'current_task_summary':result_message_str})
        result_message_str += f"Detailed summary can be found in {os.path.join(state.working_directory,'summary.out')}"

        # writer = get_stream_writer()
        # writer(result_message_str + '\n' + response.content + '\n' + "Calculation completed")
 
        return {
            "messages": [AIMessage(content=result_message_str + '\n' + response.content)], # Put here formatted summary and LLM analysis
            "messages_to_user": [AIMessage(content=message + '\n' + result_message_str + '\n' + response.content)],
            "current_task_messages":task_messages,
        }
    
    except Exception as e:
        error_message = f"Error in single_point_analysis_node: {str(e)}"
        error_info = traceback.format_exc()
        logger.error(error_message + error_info)
        # writer = get_stream_writer()
        # writer(f"❌ {error_message}")
        return {
            "error": error_message,
            "has_error": True,
            "messages": [AIMessage(content=f"Failed to analyze single point results: {error_message}")],
            "messages_to_user": [AIMessage(content=f"Failed to analyze single point results: {error_message}")],
        }
    
#-------------------------------------------------
# Error checking function
#-------------------------------------------------
def check_error(state:SinglePointState):
    """Check if an error has occurred and route accordingly"""
    if state.has_error:
        logger.warning("Error detected, skipping remaining steps")
        return END
    return "continue"

single_point_builder = StateGraph(SinglePointState)

method_graph = method_builder.compile()
prepare_molecule_graph = prepare_molecule_builder.compile()

single_point_builder.add_node("get_folder_name_node",get_folder_name_node)
single_point_builder.add_node("method",method_graph)
single_point_builder.add_node("prepare_molecule",prepare_molecule_graph)
# single_point_builder.add_node("single_point_node",single_point_node)
single_point_builder.add_node("single_point_coder_node",single_point_coder_node)
single_point_builder.add_node("single_point_exec_node",single_point_exec_node)
single_point_builder.add_node("single_point_analysis_node",single_point_analysis_node)
single_point_builder.add_node("get_result_file_node",get_result_file_node)

single_point_builder.add_edge(START,"get_folder_name_node")
single_point_builder.add_conditional_edges("get_folder_name_node", check_error, {"continue": "method", END: END})
single_point_builder.add_conditional_edges("method", check_error, {"continue": "prepare_molecule", END: END})
# single_point_builder.add_conditional_edges("prepare_molecule", check_error, {"continue": "single_point_node", END: END})
# single_point_builder.add_conditional_edges("single_point_node", check_error, {"continue": "single_point_analysis_node", END: END})
single_point_builder.add_conditional_edges("prepare_molecule", check_error, {"continue": "single_point_coder_node", END: END})
single_point_builder.add_edge("single_point_coder_node","single_point_exec_node")
single_point_builder.add_conditional_edges("single_point_exec_node", check_error, {"continue": "single_point_analysis_node", END: END})
single_point_builder.add_conditional_edges("single_point_analysis_node", check_error, {"continue": "get_result_file_node", END: END})
single_point_builder.add_edge("get_result_file_node",END)


single_point_graph = single_point_builder.compile()

from .agent_template import BaseAgent
single_point_agent = BaseAgent(
    name='single_point_agent',
    description='Perform the single point calculation of a molecular structure.',
    graph=single_point_builder,
)
