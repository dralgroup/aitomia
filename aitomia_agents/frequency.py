"""
    Agent template
    The general class for each agent to inherit and some useful utilities
"""
# shema: can be inherit from mlatom or user provide
# tool: can inherit from mlatom or user provide
# prompts: user generated
# graph: user generated

import json
import ast
import os
import mlatom as ml
from typing import Union, Optional, Literal, List
import traceback
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END 
from langgraph.types import Command
import numpy as np

from .method_judge import method_builder
from .prepare_molecule import prepare_molecule_builder
from .file_manager import get_folder_name_node, get_result_file_node

from .states import AitomiaState
from .logger import logger 
from .utils import create_agent, pretty_dict, pretty_array, Analysis

#-------------------------------------------------
# Schema
#-------------------------------------------------
class FreqState(AitomiaState):
    freqprog: str = None
    molecule_file_name: str = None 
    method: str = None
    program: Union[str,None] = None
    freq_result: Union[str,None] = None


#-------------------------------------------------
# Prompts
#-------------------------------------------------
PROMPT_MLATOM_SUPPORTED_FREQPROG = "Available programs for frequency calculation in MLatom are pyscf, Gaussian, and ase. By default, choose pyscf, unless the program is specified."
PROMPT_FREQ_ANALYSIS = "Below shows the result of frequency calculation. It contains the frequencies and thermodynamic properties of the molecule. If the molecule is a local minimum, there should be no negative frequency. For large molecules with hundreds of atoms, it may be possible to have negative frequencies that are close to zero. If the molecule is a transition state, there must be one and only one negative frequency. You should give explicitly the absolute path of the result file for further calculation of analysis. Please show all the properties given as they are and give a brief summary of the result in a few sentences. If there is any file path, please use <Path>file_path</Path> format to indicate the file path."


#-------------------------------------------------
# Tool functions
#-------------------------------------------------
def get_freqprog(freqprog:str=""):
    """
    This function is used to get program for frequency calculation,  The program here means Calculation program. 

    Args:
        freqprog: The program for frequency calculation.
    """
    if freqprog == "":
        freqprog = "pyscf"

    return locals()

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

# 3. Frequency calculation
working_directory = {{working_directory | pyrepr}}
MODEL.working_directory = working_directory
freq = ml.thermochemistry(
    model=MODEL,
    molecule=mol,
    program={{freqprog | pyrepr}},
    working_directory=working_directory,
)

# 4. Save results
mol.dump({{freq_result | pyrepr}},format='json')

"""
#-------------------------------------------------
# Agent
#-------------------------------------------------
llm = create_agent()
tools = [get_freqprog]
get_freqprog_agent = create_agent(tools=tools)
get_freqprog_tool = ToolNode(tools)


#-------------------------------------------------
# Graph
#-------------------------------------------------
def get_freqprog_node(state:FreqState):
    logger.info("Starting getting frequency program")

    try:
        program_avail_prompt = SystemMessage(content=PROMPT_MLATOM_SUPPORTED_FREQPROG)
        response = get_freqprog_agent.invoke(state.current_task_messages[-1]+[program_avail_prompt])

        logger.debug("respose from the method agent")
        logger.debug("\t"+response.content)

        output = get_freqprog_tool.invoke({"messages":[response]})["messages"][-1].content
        logger.debug("response from get_freqprog tool:")
        logger.debug("\t"+output)
        output = json.loads(output)

        output_state = {
            "freqprog": output["freqprog"]
        }
        return output_state
    
    except Exception as e:
        error_message = f"Error in get_freqprog_node: {str(e)}"
        error_info = traceback.format_exc()
        logger.error(error_message + error_info)

        # from langgraph.config import get_stream_writer
        # writer = get_stream_writer()
        # writer(f"Failed to get freqprog, using default 'pyscf': {error_message}")

        return {
            "error": error_message,
            "freqprog": "pyscf",  # Use default
            "messages": [AIMessage(content=f"Failed to get freqprog, using default 'pyscf': {error_message}")],
            "messages_to_user": [AIMessage(content=f"Failed to get freqprog, using default 'pyscf': {error_message}")],
        }

def freq_node(state:FreqState):
    logger.info("Start frequency calculation")
    message = "Start frequency calculation"
    
    try:
        molecule_file_name = state.molecule_file_name
        if '.xyz' in molecule_file_name:
            mol = ml.data.molecule.from_xyz_file(molecule_file_name)
        elif '.json' in molecule_file_name:
            mol = ml.data.molecule.load(molecule_file_name,format='json')
        else:
            raise ValueError("Unknown molecule file format")
        
        method = ml.models.methods(method=state.method,program=state.program)
        freq = ml.thermochemistry(model=method,molecule=mol,program=state.freqprog,working_directory=state.working_directory)
        # Result file name
        filename = "freqmol.json"
        if os.path.exists(os.path.join(state.working_directory,filename)):
            ii = 1
            while True:
                if os.path.exists(os.path.join(state.working_directory,f"freqmol_{ii}.json")):
                    ii += 1
                else:
                    break 
            filename = f"freqmol_{ii}.json"
        result_file_name = os.path.join(state.working_directory,filename)

        mol.dump(result_file_name,format='json')

        return {"freq_result":result_file_name,
                "messages_to_user": [AIMessage(content=message)],}
    
    except Exception as e:
        error_message = f"Error in freq_node: {str(e)}"
        error_info = traceback.format_exc()
        logger.error(error_message + error_info)
        return {
            "error": error_message,
            "has_error": True,
            "messages": [AIMessage(content=f"Frequency calculation failed: {error_message}")],
            "messages_to_user": [AIMessage(content=f"Frequency calculation failed: {error_message}")],
        }
    
def freq_coder_node(state:FreqState):
    from jinja2 import Environment 

    logger.info("Frequency calculation coder")
    message = "Frequency point calculation coder"

    # Prepare result filename
    filename = "freqmol.json"
    if os.path.exists(os.path.join(state.working_directory,filename)):
        ii = 1
        while True:
            if os.path.exists(os.path.join(state.working_directory,f"freqmol_{ii}.json")):
                ii += 1
            else:
                break 
        filename = f"freqmol_{ii}.json"
    result_file_name = os.path.join(state.working_directory,filename)
    state.freq_result = result_file_name

    input_args = state.model_dump().copy()
    if state.molecule_file_name[-4:] == ".xyz": input_args['molecule_format'] = 'xyz'
    elif state.molecule_file_name[-5:] == ".json": input_args['molecule_format'] = 'json' 
    else: raise ValueError("Unhandled molecule format")

    env = Environment(autoescape=False)
    env.filters["pyrepr"] = repr
    code = env.from_string(run_py).render(**input_args)

    working_directory = state.working_directory_stack[-1]
    python_script = os.path.join(working_directory, "run_freq.py")

    with open(python_script, 'w') as f:
        f.write(code)

    logger.debug("Output state:"); logger.debug({"scripts":python_script})
    return {"scripts":python_script,
            "freq_result": result_file_name,
            "messages_to_user": [AIMessage(content=message)],} 

def freq_exec_node(state:FreqState):
    logger.info("Start frequency calculation")
    message = "Start frequency calculation"
    try:
        python_script = state.scripts
        result_log = os.path.join(state.working_directory_stack[-1], "freq.log")
        result_err = os.path.join(state.working_directory_stack[-1], "freq.err")

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
            
            error_message = f"Frequency calculation script failed with exit code {exit_code}"
            if error_content:
                error_message += f"\nError output:\n{error_content}"
            
            logger.error(error_message)
            return {
                "error": error_message,
                "has_error": True,
                "messages": [AIMessage(content=error_message)],
                "messages_to_user": [AIMessage(content=error_message)],
            }

        return {"messages_to_user": [AIMessage(content=message)],}
    
    except Exception as e:
        error_message = f"Error in freq_exec_node: {str(e)}"
        error_info = traceback.format_exc()
        logger.error(error_message + error_info)
        response = Analysis.error_analysis(error_message)
        error_analysis = error_message + '\n' + '\n' + response


        # from langgraph.config import get_stream_writer
        # writer = get_stream_writer()
        # writer(f"❌ {error_message}")

        return {
            "error": error_message,
            "has_error": True,
            "messages": [AIMessage(content=f"Frequency calculation failed: {error_message}")],
            "messages_to_user": [AIMessage(content=f"Frequency calculation failed: {error_analysis}")],
        }

def freq_analysis_node(state:FreqState):
    logger.info("Start analyzing the result of frequency calculation")
    message = "Analyzing the result of frequency calculation"
    
    try:
        mol = ml.data.molecule.load(state.freq_result,format='json')
        analysis_prompt = SystemMessage(content=PROMPT_FREQ_ANALYSIS)

        # Prepare formatted summary
        result_message_str = f"Frequency calculation result: \n"
        if 'frequencies' in mol.__dict__:
            result_message_str += f"    Frequencies: {pretty_array(mol.frequencies)} cm^-1\n"
            result_message_str += f"    Number of imaginary frequencies: {np.sum(mol.frequencies<0)}\n"
        if 'ZPE' in mol.__dict__:
            result_message_str += f"    Zero-point energy: {mol.ZPE} Hartree\n"
        if 'H0' in mol.__dict__:
            result_message_str += f"    Internal energy at 0 Kelvin: {mol.H0} Hartree\n"
        if 'H0' in mol.__dict__:
            result_message_str += f"    Enthalpy at 0 Kelvin: {mol.H0} Hartree\n"
        if 'H' in mol.__dict__:
            result_message_str += f"    Enthalpy at 298 Kelvin: {mol.H} Hartree\n"
        if 'G' in mol.__dict__:
            result_message_str += f"    Gibbs free energy at 298 Kelvin: {mol.G} Hartree\n"
        if 'DeltaHf298' in mol.__dict__:
            result_message_str += f"    Heat of formation at 298 Kelvin: {mol.DeltaHf298} Hartree\n"
        result_file_name = os.path.join(state.working_directory,"freqmol.json")
        result_message_str += f"    Frequencies and thermodynamic properties are saved in <Path>{result_file_name}</Path>"
        result_message = AIMessage(content=result_message_str)

        # Get analysis of formatted summary from LLM
        response = llm.invoke(state.current_task_messages[-1]+[analysis_prompt,result_message])
        task_messages = state.current_task_messages
        task_messages[-1].append(AIMessage(content=response.content))

        # Write formatted summary into summary.out
        with open(os.path.join(state.working_directory,"summary.out"),'w') as f:
            f.write(result_message_str)
            Analysis.add_summary({'current_task_summary':result_message_str})
        result_message_str += f"Detailed summary can be found in {os.path.join(state.working_directory,'summary.out')}"

        # from langgraph.config import get_stream_writer
        # writer = get_stream_writer()
        # writer(result_message_str + '\n' + response.content + '\n' + "Calculation completed")
        return {
            "messages": [AIMessage(content=result_message_str + '\n' + response.content)],
            "messages_to_user": [AIMessage(content=message + '\n' + result_message_str + '\n' + response.content)],
            "current_task_messages":task_messages
        }
    
    except Exception as e:
        error_message = f"Error in freq_analysis_node: {str(e)}"
        error_info = traceback.format_exc()
        logger.error(error_message + error_info)

        # from langgraph.config import get_stream_writer
        # writer = get_stream_writer()
        # writer(f"❌ {error_message}")
        return {
            "error": error_message,
            "has_error": True,
            "messages": [AIMessage(content=f"Failed to analyze frequency results: {error_message}")],
            "messages_to_user": [AIMessage(content=f"Failed to analyze frequency results: {error_message}")],
        }

#-------------------------------------------------
# Error checking function
#-------------------------------------------------
def check_error(state:FreqState):
    """Check if an error has occurred and route accordingly"""
    if state.has_error:
        logger.warning("Error detected, skipping remaining steps")
        return END
    return "continue"

freq_builder = StateGraph(FreqState)
prepare_molecule_graph = prepare_molecule_builder.compile()
method_graph = method_builder.compile()
freq_builder.add_node("get_folder_name_node",get_folder_name_node)
freq_builder.add_node("method",method_graph)
freq_builder.add_node("prepare_molecule",prepare_molecule_graph)
freq_builder.add_node("get_freqprog_node",get_freqprog_node)
# freq_builder.add_node("freq_node",freq_node)
freq_builder.add_node("freq_coder_node",freq_coder_node)
freq_builder.add_node("freq_exec_node",freq_exec_node)
freq_builder.add_node("freq_analysis_node",freq_analysis_node)
freq_builder.add_node("get_result_file_node",get_result_file_node)

freq_builder.add_edge(START,"get_folder_name_node")
freq_builder.add_conditional_edges("get_folder_name_node", check_error, {"continue": "method", END: END})
freq_builder.add_conditional_edges("method", check_error, {"continue": "prepare_molecule", END: END})
freq_builder.add_conditional_edges("prepare_molecule", check_error, {"continue": "get_freqprog_node", END: END})
# freq_builder.add_conditional_edges("get_freqprog_node", check_error, {"continue": "freq_node", END: END})
# freq_builder.add_conditional_edges("freq_node", check_error, {"continue": "freq_analysis_node", END: END})
freq_builder.add_conditional_edges("get_freqprog_node",check_error, {"continue": "freq_coder_node", END: END})
freq_builder.add_edge("freq_coder_node","freq_exec_node")
freq_builder.add_conditional_edges("freq_exec_node", check_error, {"continue": "freq_analysis_node", END: END})
freq_builder.add_conditional_edges("freq_analysis_node", check_error, {"continue": "get_result_file_node", END: END})
freq_builder.add_edge("get_result_file_node",END)

freq_graph = freq_builder.compile()


from .agent_template import BaseAgent
freq_agent = BaseAgent(
    name='freq_agent',
    description='Perform frequency calculation of a molecular structure, usually an optimized molecule.',
    graph=freq_builder,
)