"""
    Raman agent
"""

from .agent_template import BaseAgent, tool_from_mlatom, create_llm, schema_from_mlatom
import json, os
import traceback
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END 
from langgraph.types import interrupt, Command
from langchain_core.messages import SystemMessage, AIMessage
import mlatom as ml
import numpy as np

# load other graphs
from .method_judge import method_builder
from .prepare_molecule import prepare_molecule_builder
from .file_manager import get_folder_name_node, get_result_file_node

from .logger import logger 
from .utils import create_agent, pretty_dict, pretty_array, Analysis

from typing import Optional, Literal, List

from langgraph.config import get_stream_writer

mod_path = "mlatom.simulations"
func_path = "freq.__init__"
doc_func_path = "freq"
replace = {
    "arguments":{
        "raman_program":{
            "type":Optional[str], "default":None, "description":"The program that provide algorithm to calculate frequencies and Raman intensities instead of the program that provides energy, energy gradients, and Hessian. Only pyscf, gaussian, and None are available. By default pyscf should be used."
        },
        "raman_program_kwargs":{
            "type":Optional[str], "default":None,
            "description":"Control the behavior of the algorithm used in frequencies and Raman intensities calculations. Do not provide keywords that are not related to the algorithm."
        },
        "scaling_factor":{
            "type":Optional[str], "default":None,
            "description":"The global scaling factor of the calculated frequencies. For AIQM2 method, the scaling factor is 0.962."
        }
    }
}
delete = ["molecule", "model", "model_predict_kwargs","ir","raman"]
#-------------------------------------------------
# Prompts
#-------------------------------------------------
PROMPT_RAMAN_ANALYSIS = "Below shows the result of the Raman spectrum calculation. It might contain multiple properties of the molecule, e.g., energy, frequencies, Raman intensities, etc. You should give explicitly the absolute path of the result file for further calculation of analysis. Please give a brief summary of the result in a few sentences. `Nan` in the result means the corresponding properties are not calculated. If there is any file path, please use <Path>file_path</Path> format to indicate the file path."

#-------------------------------------------------
# Schema
#-------------------------------------------------
# schema of raman should hold every properties needed
import copy
schema_replace = copy.deepcopy(replace)
schema_replace["arguments"].update({
    "molecule_file_name":{"type":str, "default":None, "description":"path to the molecule file of local minimum"},
    "method": {"type":str, "default":None, "description":"The method to get energy, energy derivatives, Hessian, and polarizability derivatives to calculate the Raman spectrum."},
    "program": {"type":Optional[str], "default":None, "description":"The program to be used for the method to get energy, energy derivatives, Hessian, and polarizability derivatives."},
    "raman_result_files": {"type":List[str],"default":None,"description":""},
})
RamanState = schema_from_mlatom(
    mod_path=mod_path, func_path=func_path, doc_func_path=doc_func_path,
    replace=schema_replace, delete=delete, schema_name="RamanState"
)
#-------------------------------------------------
# Tool functions
#-------------------------------------------------
tool_replace = copy.deepcopy(replace)

raman_tool = tool_from_mlatom(
    mod_path=mod_path, func_path=func_path, doc_func_path=doc_func_path, 
    replace=replace, delete=delete+["program", "program_kwargs", "working_directory"], tool_name="raman"
)
llm = create_agent()
#-------------------------------------------------
# scripts
#-------------------------------------------------
# render script with jinja2 template
run_raman_py = """#!/bin/env python

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
freq = ml.freq(
    molecule = mol,
    model = MODEL,
    program = {{raman_program | pyrepr}},
    program_kwargs = {{raman_program_kwargs}},
    normal_mode_normalization = {{normal_mode_normalization | pyrepr}},
    working_directory = working_directory, 
    anharmonic = {{anharmonic}},
    ir = False,
    raman = True,
)

scaling_factor = {{scaling_factor}}
if not scaling_factor is None:
    mol.frequencies = mol.frequencies * scaling_factor
spectrum = ml.spectra.raman.lorentzian(molecule=mol,fwhm=30)
spectrum.plot(os.path.join(working_directory,'raman.png'))
mol.dump(os.path.join(working_directory,'ramanmol.json'),format='json')

"""

run_raman_inp = """
raman
method={{method}}
{% if molecule_format == 'xyz' %}
xyzfile={{molecule_file_name}}
{% elif molecule_format == 'json' %}
jsonfile={{molecule_file_name}}
{% endif %} 
"""

#-------------------------------------------------
# Subgraphs
#-------------------------------------------------
# Read Raman settings
def raman_settings_node(state):
    logger.info("raman settings node")

    logger.debug("Input state:"); pretty_dict(state.model_dump(), logger)

    raman_settings_agent = create_llm(tools=[raman_tool],tool_kwargs={"tool_choice":"any"})
    raman_settings_tool = ToolNode([raman_tool])

    agent_response = raman_settings_agent.invoke(state.current_task_messages[-1])
    logger.debug("Response from raman settings agent:")
    logger.debug(agent_response)

    raman_kwargs = raman_settings_tool.invoke({"messages":[agent_response]})
    logger.debug("Response from raman settings tool:")
    logger.debug(raman_kwargs)

    raman_kwargs = json.loads(raman_kwargs["messages"][-1].content)
    raman_kwargs['working_direcotry'] = state.working_directory_stack[-1]

    if 'raman_program' in raman_kwargs:
        if raman_kwargs['raman_program'] is None:
            logger.debug("Unrecognized raman program, use Pyscf instread")
            raman_kwargs['raman_program'] = 'Pyscf'
        if not raman_kwargs['raman_program'].casefold() in ['pyscf','gaussian']:
            logger.debug("Unrecognized raman program, use Pyscf instread")
            raman_kwargs['raman_program'] = 'Pyscf'
    if 'raman_program_kwargs' in raman_kwargs:
        raman_kwargs['raman_program_kwargs'] = r"{}"

    logger.debug("Output state:"); pretty_dict(state.model_dump(), logger)
    return raman_kwargs

# Raman coder graph
def raman_coder_node(state):
    from jinja2 import Environment

    logger.info("raman python code node")
    message = "Start generating Raman calculation script"
    logger.debug("Input state:"); pretty_dict(state.model_dump(), logger)

    input_args = state.model_dump().copy()
    if state.molecule_file_name[-4:] == ".xyz": input_args['molecule_format'] = 'xyz'
    elif state.molecule_file_name[-5:] == ".json": input_args['molecule_format'] = 'json' 
    else: raise ValueError("Unhandled molecule format")

    env = Environment(autoescape=False)
    env.filters["pyrepr"] = repr
    raman_code = env.from_string(run_raman_py).render(**input_args)

    working_directory = state.working_directory_stack[-1]
    python_script = os.path.join(working_directory, "run_raman.py")

    with open(python_script, 'w') as f:
        f.write(raman_code)

    logger.debug("Output state:"); logger.debug({"scripts":python_script})

    return {"scripts":python_script,
            "messages_to_user": [AIMessage(content=message)],}
  

def raman_exec_user(state) -> Command[Literal["raman_exec"]]:

    execute = interrupt({
        "messages_to_user": f"The generated executable script is located at {state.scripts}. Proceed to execute the generated script for Raman calculation? [yes/no]",
        "options": ["yes", "no"]
    })

    logger.debug("Response from user:")
    logger.debug(execute)

    return Command(goto="raman_exec" if execute.lower() == "yes" else END) 

def raman_exec_node(state):
    logger.info("Start calculating Raman spectrum")
    message = "Start calculating Raman spectrum"

    try:
        python_script = state.scripts
        result_log = os.path.join(state.working_directory_stack[-1], "raman.log")
        result_err = os.path.join(state.working_directory_stack[-1], "raman.err")
        result_mol = os.path.join(state.working_directory_stack[-1], "ramanmol.json")
        result_spec = os.path.join(state.working_directory_stack[-1], "raman.png")

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
            
            error_message = f"Raman calculation script failed with exit code {exit_code}"
            if error_content:
                error_message += f"\nError output:\n{error_content}"
            
            logger.error(error_message)
            return {
                "error": error_message,
                "has_error": True,
                "messages": [AIMessage(content=error_message)],
                "messages_to_user": [AIMessage(content=error_message)],
            }

        return {"raman_result_files": [result_log, result_err,result_mol,result_spec],
                "messages_to_user": [AIMessage(content=message)]}
    
    except Exception as e:
        error_message = f"Error in raman_exec_node: {str(e)}"
        error_info = traceback.format_exc()
        logger.error(error_message + error_info)
        # writer = get_stream_writer()
        # writer(f"❌ {error_message}")
        response = Analysis.error_analysis(error_message)
        error_analysis = error_message + '\n' + '\n' + response
        return {
            "error": error_message,
            "has_error": True,
            "messages": [AIMessage(content=f"Raman calculation failed: {error_message}")],
            "messages_to_user": [AIMessage(content=f"Raman calculation failed: {error_analysis}")],
        }

def raman_analysis_node(state):
    logger.info("Start analyzing the result of Raman calculation")
    message = "Analyzing the result of Raman calculation"
    
    try:
        result_files = state.raman_result_files
        log_file = result_files[0]
        log_file = open(log_file, 'r').readlines()
        output_log = ""
        start_ii = 0
        for ill, ll in enumerate(log_file):
            if "Finish Raman calculation" in ll:
                start_ii = ill
                break 

        # Output log
        output_log = "".join(log_file[start_ii-1:])
        output_log += f"\nExecutable file: {state.scripts}"
        output_log += f"\nLog file: {result_files[0]}"
        output_log += f"\nError file: {result_files[1]}"
        output_log += f"\nMolecule with frequencies and Raman intensities: {result_files[2]}"
        output_log += f"\nRaman spectrum: {result_files[3]}"

        # Prepare formatted summary
        mol = ml.data.molecule.load(result_files[2],format='json')
        analysis_prompt = SystemMessage(content=PROMPT_RAMAN_ANALYSIS)
        result_message_str = "Raman spectrum calculation result: \n"
        result_message_str += f"    Energy: {mol.energy} Hartree\n"
        result_message_str += f"    Frequencies (cm^-1): {pretty_array(mol.frequencies)}\n"
        result_message_str += f"    Number of imaginary frequencies: {np.sum(mol.frequencies<0)}\n"
        result_message_str += f"    Raman intensities (km/mol): {pretty_array(mol.raman_intensities)}\n"
        result_message_str += f"    Molecule with frequencies and Raman intensities is saved in <Path>{result_files[2]}</Path>\n"
        result_message = AIMessage(content=result_message_str)

        # Get analysis of formatted summary from LLM
        response = llm.invoke(state.current_task_messages[-1]+[analysis_prompt,result_message])
        task_messages = state.current_task_messages
        task_messages[-1].append(AIMessage(content=result_message_str))

        # Write formatted summary into summary.out
        with open(os.path.join(state.working_directory,"summary.out"),'w') as f:
            f.write(result_message_str)
            Analysis.add_summary({'current_task_summary':result_message_str})
        result_message_str += f"Detailed summary can be found in {os.path.join(state.working_directory,'summary.out')}"

        # writer = get_stream_writer()
        # writer(output_log + result_message_str + '\n' + response.content + '\n' + "Calculation completed")        
        return {"messages": [AIMessage(content=output_log + result_message_str + '\n' + response.content)],
                "messages_to_user": [AIMessage(content=message + '\n' + output_log + result_message_str + '\n' + response.content)],
                "current_task_messages":task_messages,}
    
    except Exception as e:
        error_message = f"Error in raman_analysis_node: {str(e)}"
        error_info = traceback.format_exc()
        logger.error(error_message + error_info)
        # writer = get_stream_writer()
        # writer(f"❌ {error_message}")
        return {
            "error": error_message,
            "has_error": True,
            "messages": [AIMessage(content=f"Failed to analyze Raman results: {error_message}")],
            "messages_to_user": [AIMessage(content=f"Failed to analyze Raman results: {error_message}")]
        }

#-------------------------------------------------
# Graph
#-------------------------------------------------
# from .agent_template import schema_to_str
# logger.debug("Generated Raman schema:")
# logger.debug(schema_to_str(RamanState))

raman_static_builder = StateGraph(RamanState)
prepare_molecule_graph = prepare_molecule_builder.compile()
method_graph = method_builder.compile()

raman_static_builder.add_node("get_folder_name_node",get_folder_name_node)
raman_static_builder.add_node("get_geom",prepare_molecule_graph)
raman_static_builder.add_node("get_method",method_graph)
raman_static_builder.add_node("raman_settings",raman_settings_node)
raman_static_builder.add_node("raman_coder",raman_coder_node)
raman_static_builder.add_node("raman_exec",raman_exec_node)
raman_static_builder.add_node("raman_analysis",raman_analysis_node)
raman_static_builder.add_node("get_result_file_node",get_result_file_node)

raman_static_builder.add_edge(START,"get_folder_name_node")
raman_static_builder.add_edge("get_folder_name_node","get_geom")
raman_static_builder.add_edge("get_geom","get_method")
raman_static_builder.add_edge("get_method","raman_settings")
raman_static_builder.add_edge("raman_settings","raman_coder")
raman_static_builder.add_edge("raman_coder","raman_exec")
raman_static_builder.add_edge("raman_exec","raman_analysis")
raman_static_builder.add_edge("raman_analysis","get_result_file_node")
raman_static_builder.add_edge("get_result_file_node",END)

raman_static_graph = raman_static_builder.compile()

from .agent_template import BaseAgent
raman_static_agent = BaseAgent(
    name='raman_static_agent',
    description='Perform Raman intensity calculation of a molecular structure, usually an optimized molecule, to get the Raman spectrum.',
    graph=raman_static_builder,
)