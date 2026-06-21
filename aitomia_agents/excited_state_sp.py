"""
    Excited-state single-point agent
"""
# import json
# import ast
import os
import mlatom as ml
from typing import Union
import numpy as np
import glob
from langchain_core.messages import SystemMessage, AIMessage
# from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END 
from langgraph.types import interrupt, Command

from .method_judge import method_builder
from .prepare_molecule import prepare_molecule_builder
from .file_manager import get_folder_name_node, get_result_file_node

from .states import AitomiaState
from .logger import logger 
from .utils import create_agent, pretty_dict, pretty_array, Analysis

from langgraph.config import get_stream_writer


#-----------------------Schema-----------------------
class ExcitedStateSPState(AitomiaState):
    molecule_file_name:str=None
    method:str=None
    program:Union[str,None]=None
    nstates:int=11
    current_state:int=1
    ex_single_point_result:Union[str,None]=None
    calculate_energy_gradients:bool=False
    calculate_hessian:bool=False
    band_width:float=0.3

#-----------------------Prompt-----------------------
PROMPT_SP_ES_ANALYSIS = "Below shows the result of the excited-state calculation. It might contain multiple properties of the molecule, e.g., electronic states, excitation energies, oscillator strengths, energy gradients, hessian, etc. You should give explicitly the absolute path of the result file for further calculation of analysis. Please give a brief summary of the result in a few sentences. `Nan` in the result means the corresponding properties are not calculated.If there is any file path, please use <Path>file_path</Path> format to indicate the file path."

#-----------------------Agent-----------------------
llm = create_agent()

run_es_py = """#!/bin/env python

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

# 3. fill in settings of raman
working_directory = {{working_directory | pyrepr}}
MODEL.working_directory = working_directory
MODEL.predict(
            molecule=mol,
            calculate_energy=True,
            calculate_energy_gradients={{calculate_energy_gradients}},
            calculate_hessian={{calculate_hessian}},
            nstates={{nstates}},
            current_state={{current_state}}
            )

# 4. plot uvvis spectrum
ml.spectra.plot_uvvis(molecule=mol,
                      spc=True,
                      labels=[{{method | pyrepr}}],
                      filename="es_sp.png",
                      band_width={{band_width}})
# 5. Save results
mol.dump({{ex_single_point_result | pyrepr}},format='json')
"""
#working_directory=working_directory
#analyze_component=True, contribution_threshold=0.1,

def excited_state_coder_node(state:ExcitedStateSPState):
    from jinja2 import Environment

    logger.info("excited-state code node")
    message = "Start calculating excited-state and plot UV-vis spectrum."
    logger.debug("Input state: "); pretty_dict(state.model_dump(), logger)

    filename = "ex_single_point.json"
    if os.path.exists(os.path.join(state.working_directory,filename)):
        ii = 1
        while True:
            if os.path.exists(os.path.join(state.working_directory,f"ex_single_point_{ii}.json")):
                ii += 1
            else:
                break 
        filename = f"ex_single_point_{ii}.json"
    result_file_name = os.path.join(state.working_directory,filename)
    state.ex_single_point_result = result_file_name

    input_args = state.model_dump().copy()

    if state.molecule_file_name[-4:] == ".xyz": input_args['molecule_format'] = 'xyz'
    elif state.molecule_file_name[-5:] == ".json": input_args['molecule_format'] = 'json' 
    else: raise ValueError("Unhandled molecule format")

    env = Environment(autoescape=False)
    env.filters["pyrepr"] = repr
    es_code = env.from_string(run_es_py).render(**input_args)

    working_directory = state.working_directory_stack[-1]
    python_script = os.path.join(working_directory, "run_sp_es.py")

    with open(python_script, "w") as f:
        f.write(es_code)
    logger.debug("Output state:"); logger.debug({"scripts":python_script})
    return {"scripts":python_script,
            "ex_single_point_result": result_file_name,
            "messages_to_user": [AIMessage(content=message)]} 


#-----------------------Graph-----------------------
def excited_state_sp_exec_node(state:ExcitedStateSPState):
    message = "Start calculating excited-state single-point energies"
    logger.info(message)

    try:
        python_script = state.scripts
        result_log = os.path.join(state.working_directory_stack[-1], "es_sp.log")
        result_err = os.path.join(state.working_directory_stack[-1], "es_sp.err")
        # result_mol = os.path.join(state.working_directory_stack[-1], "es_mol.json")
        # result_spec = os.path.join(state.working_directory_stack[-1], "es_sp.png")

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
            
            error_message = f"Excited-state calculation script failed with exit code {exit_code}"
            if error_content:
                error_message += f"\nError output:\n{error_content}"
            
            logger.error(error_message)
            return {
                "error": error_message,
                "has_error": True,
                "messages": [AIMessage(content=error_message)],
                "messages_to_user": [AIMessage(content=error_message)],
            }

        return {"excited_state_result_files": [result_log, result_err],
                "messages_to_user": [AIMessage(content=message)]}
    
    except Exception as e:
        import traceback
        error_message = f"Error in excited_state_sp_exec_node: {str(e)}"
        error_info = traceback.format_exc()
        logger.error(error_message + error_info)
        response = Analysis.error_analysis(error_message)
        error_analysis = error_message + '\n' + '\n' + response

        return {
            "error": error_message,
            "has_error": True,
            "messages": [AIMessage(content=f"excited-state single-point calculation failed: {error_message}")],
            "messages_to_user": [AIMessage(content=f"excited-state single-point calculation failed: {error_analysis}")]
        }


def excited_state_analysis_node(state:ExcitedStateSPState):
    message = "Start analyzing the result of excited-state calculation"
    logger.info(message)

    try:
        molecule = ml.molecule.load(filename=state.ex_single_point_result, format="json")
        excitation_energies = molecule.excitation_energies
        if state.calculate_energy_gradients:
            energy_gradients = molecule.energy_gradients
        oscillator_strengths = molecule.oscillator_strengths

        result_file_name = state.ex_single_point_result
        excited_state_sp_prompt = SystemMessage(content=PROMPT_SP_ES_ANALYSIS)

        # Formatted summary
        result_message_str = f"Excited-state calculation: \n\n"
        result_message_str += f"\tExcitation\tExcitation_energies\tOscillator_strengths\n"
        for excitation, (ex, f) in enumerate(zip(excitation_energies, oscillator_strengths), start=1):
            result_message_str += f"\t{excitation}\t{ex:.2f}\t{f:.4f}\n"
        result_message_str += f"\n\tResult file: {result_file_name}\n"
        result_message = AIMessage(content=result_message_str)

        # Analysis from LLM
        response = llm.invoke(state.current_task_messages[-1] + [excited_state_sp_prompt] + [result_message])
        task_message = state.current_task_message
        task_message[-1].append(AIMessage(content=response.content))

        with open(os.path.join(state.working_directory, "summary.out"), "w") as f:
            f.write(result_message_str)
            Analysis.add_summary({'current_task_summary':result_message_str})
        result_message_str += f"Detailed summary can be found in {os.path.join(state.working_directory, 'summary.out')}"

        return {
            "messages": [AIMessage(content=result_message_str + "\n" + response.content)],
            "messages_to_user": [AIMessage(content=message + "\n" + result_message_str + "\n" + response.content)],
            "current_task_message": task_message
        }

    except Exception as e:
        error_message = f"Error in excited_state_analysis_node: {str(e)}"
        logger.error(error_message)
        writer = get_stream_writer()
        writer(f"❌ {error_message}")
        return {
            "error": error_message,
            "has_error": True,
            "messages": [AIMessage(content=f"Failed to analyze excited-state results: {error_message}")],
            "messages_to_user": [AIMessage(content=f"Failed to analyze excited-state results: {error_message}")]
        }

def check_error(state:ExcitedStateSPState):
    """Check if an error has occurred and route accordingly"""
    if state.has_error:
        logger.warning("Error detected, skipping remaining steps")
        return END
    return "continue"
        

excited_state_builder = StateGraph(ExcitedStateSPState)

method_graph = method_builder.compile()
prepare_molecule_graph = prepare_molecule_builder.compile()

excited_state_builder.add_node("get_folder_name_node",get_folder_name_node)
excited_state_builder.add_node("method",method_graph)
excited_state_builder.add_node("prepare_molecule",prepare_molecule_graph)
excited_state_builder.add_node("excited_state_coder_node",excited_state_coder_node)
excited_state_builder.add_node("excited_state_sp_exec_node",excited_state_sp_exec_node)
excited_state_builder.add_node("excited_state_analysis_node",excited_state_analysis_node)
excited_state_builder.add_node("get_result_file_node",get_result_file_node)

excited_state_builder.add_edge(START,"get_folder_name_node")
excited_state_builder.add_conditional_edges("get_folder_name_node", check_error, {"continue": "method", END: END})
excited_state_builder.add_conditional_edges("method", check_error, {"continue": "prepare_molecule", END: END})
excited_state_builder.add_conditional_edges("prepare_molecule", check_error, {"continue": "excited_state_coder_node", END: END})
excited_state_builder.add_edge("excited_state_coder_node","excited_state_sp_exec_node")
excited_state_builder.add_conditional_edges("excited_state_sp_exec_node", check_error, {"continue": "excited_state_analysis_node", END: END})
excited_state_builder.add_conditional_edges("excited_state_analysis_node", check_error, {"continue": "get_result_file_node", END: END})
excited_state_builder.add_edge("get_result_file_node",END)


excited_state_graph = excited_state_builder.compile()

from .agent_template import BaseAgent
excited_state_agent = BaseAgent(
    name='excited_state_agent',
    description=f'Calculate excitation energies and oscillator strengths of a molecular structure.',
    graph=excited_state_builder,
)
