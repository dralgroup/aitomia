"""
    IR agent
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

mod_path = "mlatom.simulations"
func_path = "freq.__init__"
doc_func_path = "freq"
replace = {
    "arguments":{
        "ir_program":{
            "type":Optional[str], "default":None, "description":"The program that provides algorithm to calculate frequencies and infrared (IR) intensities instead of the program that provides energy, energy gradients, and Hessian. Only pyscf, and gaussian are available. By default pyscf should be used."
        },
        "ir_program_kwargs":{
            "type":Optional[str], "default":None,
            "description":"Control the behavior of the algorithm used in frequencies and infrared (IR) intensities calculations. Do not provide keywords that are not related to the algorithm."
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
PROMPT_IR_ANALYSIS = "Below shows the result of the IR spectrum calculation. It might contain multiple properties of the molecule, e.g., energy, frequencies, IR intensities, etc. You should give explicitly the absolute path of the result file for further calculation of analysis. Please give a brief summary of the result in a few sentences. `Nan` in the result means the corresponding properties are not calculated. If there is any file path, please use <Path>file_path</Path> format to indicate the file path."


#-------------------------------------------------
# Schema
#-------------------------------------------------
# schema of ir should hold every properties needed
import copy
schema_replace = copy.deepcopy(replace)
schema_replace["arguments"].update({
    "molecule_file_name":{"type":str, "default":None, "description":"path to the molecule file of local minimum"},
    "method": {"type":str, "default":None, "description":"The method to get energy, energy derivatives, Hessian, and dipole derivatives to calculate the IR spectrum."},
    "program": {"type":Optional[str], "default":None, "description":"The program to be used for the method to get energy, energy derivatives, Hessian, and dipole derivatives."},
    "ir_result_files": {"type":List[str],"default":None,"description":""},
})
IRState = schema_from_mlatom(
    mod_path=mod_path, func_path=func_path, doc_func_path=doc_func_path,
    replace=schema_replace, delete=delete, schema_name="IRState"
)
#-------------------------------------------------
# Tool functions
#-------------------------------------------------
tool_replace = copy.deepcopy(replace)

ir_tool = tool_from_mlatom(
    mod_path=mod_path, func_path=func_path, doc_func_path=doc_func_path, 
    replace=replace, delete=delete+["program", "program_kwargs", "working_directory"], tool_name="ir"
)
llm = create_agent()
#-------------------------------------------------
# scripts
#-------------------------------------------------
# render script with jinja2 template
run_ir_py = """#!/bin/env python

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

# 3. fill in settings of ir
working_directory = {{working_directory | pyrepr}}
MODEL.working_directory = working_directory
freq = ml.freq(
    molecule = mol,
    model = MODEL,
    program = {{ir_program | pyrepr}},
    program_kwargs = {{ir_program_kwargs}},
    #normal_mode_normalization = {{normal_mode_normalization | pyrepr}},
    working_directory = working_directory, 
    anharmonic = {{anharmonic}},
    ir = True,
    raman = False,
)

scaling_factor = {{scaling_factor}}
if not scaling_factor is None:
    mol.frequencies = mol.frequencies * scaling_factor
spectrum = ml.spectra.ir.lorentzian(molecule=mol,fwhm=30)
spectrum.plot(os.path.join(working_directory,'ir.png'))
mol.dump(os.path.join(working_directory,'irmol.json'),format='json')

"""

run_ir_inp = """
ir
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
# Read IR settings
def ir_settings_node(irstate):
    logger.info("ir settings node")


    logger.debug("Input state:"); pretty_dict(irstate.model_dump(), logger)

    ir_settings_agent = create_llm(tools=[ir_tool],tool_kwargs={"tool_choice":"any"})
    ir_settings_tool = ToolNode([ir_tool])

    agent_response = ir_settings_agent.invoke(irstate.current_task_messages[-1])
    logger.debug("Response from ir settings agent:")
    logger.debug(agent_response)

    ir_kwargs = ir_settings_tool.invoke({"messages":[agent_response]})
    logger.debug("Response from ir settings tool:")
    logger.debug(ir_kwargs)

    ir_kwargs = json.loads(ir_kwargs["messages"][-1].content)
    ir_kwargs['working_direcotry'] = irstate.working_directory_stack[-1]

    if 'ir_program' in ir_kwargs:
        if ir_kwargs['ir_program'] is None:
            logger.debug("Unrecognized ir program, use Pyscf instread")
            ir_kwargs['ir_program'] = 'Pyscf'
        elif not ir_kwargs['ir_program'].casefold() in ['pyscf','gaussian']:
            logger.debug("Unrecognized ir program, use Pyscf instread")
            ir_kwargs['ir_program'] = 'Pyscf'
    if 'ir_program_kwargs' in ir_kwargs:
        ir_kwargs['ir_program_kwargs'] = r"{}"

    logger.debug("Output state:"); pretty_dict(irstate.model_dump(), logger)
    return ir_kwargs

# IR coder graph
def ir_coder_node(irstate):
    from jinja2 import Environment

    logger.info("ir python code node")
    message = "Start generating IR calculation script"
    logger.debug("Input state:"); pretty_dict(irstate.model_dump(), logger)

    input_args = irstate.model_dump().copy()
    if irstate.molecule_file_name[-4:] == ".xyz": input_args['molecule_format'] = 'xyz'
    elif irstate.molecule_file_name[-5:] == ".json": input_args['molecule_format'] = 'json' 
    else: raise ValueError("Unhandled molecule format")

    env = Environment(autoescape=False)
    env.filters["pyrepr"] = repr
    ir_code = env.from_string(run_ir_py).render(**input_args)

    working_directory = irstate.working_directory_stack[-1]
    python_script = os.path.join(working_directory, "run_ir.py")

    with open(python_script, 'w') as f:
        f.write(ir_code)

    logger.debug("Output state:"); logger.debug({"scripts":python_script})
    return {"scripts":python_script,
            "messages_to_user": [AIMessage(content=message)],} 

def ir_exec_user(irstate) -> Command[Literal["ir_exec"]]:

    execute = interrupt({
        "messages_to_user": f"The generated executable script is located at {irstate.scripts}. Proceed to execute the generated script for IR calculation? [yes/no]",
        "options": ["yes", "no"]
    })

    logger.debug("Response from user:")
    logger.debug(execute)

    return Command(goto="ir_exec" if execute.lower() == "yes" else END) 

def ir_exec_node(irstate):
    logger.info("Start calculating IR spectrum")
    message = "Start calculating IR spectrum"

    try:
        python_script = irstate.scripts
        result_log = os.path.join(irstate.working_directory_stack[-1], "ir.log")
        result_err = os.path.join(irstate.working_directory_stack[-1], "ir.err")
        result_mol = os.path.join(irstate.working_directory_stack[-1], "irmol.json")
        result_spec = os.path.join(irstate.working_directory_stack[-1], "ir.png")

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
            
            error_message = f"IR calculation script failed with exit code {exit_code}"
            if error_content:
                error_message += f"\nError output:\n{error_content}"
            
            logger.error(error_message)
            return {
                "error": error_message,
                "has_error": True,
                "messages": [AIMessage(content=error_message)],
                "messages_to_user": [AIMessage(content=error_message)],
            }

        return {"ir_result_files": [result_log, result_err,result_mol,result_spec],
                "messages_to_user": [AIMessage(content=message)],}
    
    except Exception as e:
        error_message = f"Error in ir_exec_node: {str(e)}"
        error_info = traceback.format_exc()
        logger.error(error_message + error_info)
        response = Analysis.error_analysis(error_message)
        error_analysis = error_message + '\n' + '\n' + response

        # from langgraph.config import get_stream_writer
        # writer = get_stream_writer()
        # writer(f"IR calculation failed: {error_message}")

        return {
            "error": error_message,
            "has_error": True,
            "messages": [AIMessage(content=f"IR calculation failed: {error_message}")],
            "messages_to_user": [AIMessage(content=f"IR calculation failed: {error_analysis}")]
        }

def ir_analysis_node(irstate):
    logger.info("Start analyzing the result of IR calculation")
    message = "Start analyzing the result of IR calculation"
    
    try:
        result_files = irstate.ir_result_files
        log_file = result_files[0]
        log_file = open(log_file, 'r').readlines()
        output_log = ""
        start_ii = 0
        for ill, ll in enumerate(log_file):
            if "Finish IR calculation" in ll:
                start_ii = ill
                break 

        # Output log
        output_log = "".join(log_file[start_ii-1:])
        output_log += f"\nExecutable file: {irstate.scripts}"
        output_log += f"\nLog file: {result_files[0]}"
        output_log += f"\nError file: {result_files[1]}"
        output_log += f"\nMolecule with frequencies and IR intensities: {result_files[2]}"
        output_log += f"\nIR spectrum: {result_files[3]}"

        # Prepare formatted summary
        mol = ml.data.molecule.load(result_files[2],format='json')
        analysis_prompt = SystemMessage(content=PROMPT_IR_ANALYSIS)
        result_message_str = "IR spectrum calculation result: \n"
        result_message_str += f"    Energy: {mol.energy} Hartree\n"
        result_message_str += f"    Frequencies (cm^-1): {pretty_array(mol.frequencies)}\n"
        result_message_str += f"    Number of imaginary frequencies: {np.sum(mol.frequencies<0)}\n"
        result_message_str += f"    IR intensities (km/mol): {pretty_array(mol.infrared_intensities)}\n"
        result_message_str += f"    Molecule with frequencies and IR intensities is <Path>saved in {result_files[2]}</Path>\n"
        result_message = AIMessage(content=result_message_str)

        # Get analysis of formatted summary from LLM
        response = llm.invoke(irstate.current_task_messages[-1]+[analysis_prompt,result_message])
        task_messages = irstate.current_task_messages
        task_messages[-1].append(AIMessage(content=result_message_str))

        # Write formatted summary into summary.out
        with open(os.path.join(irstate.working_directory,"summary.out"),'w') as f:
            f.write(result_message_str)
            Analysis.add_summary({'current_task_summary':result_message_str})
        result_message_str += f"Detailed summary can be found in {os.path.join(irstate.working_directory,'summary.out')}"

        # from langgraph.config import get_stream_writer
        # writer = get_stream_writer()
        # writer(output_log + result_message_str + '\n' + response.content + '\n' + "Calculation completed")
        return {"messages": [AIMessage(content=output_log + result_message_str + '\n' + response.content)],
                "messages_to_user": [AIMessage(content=message + '\n' + output_log + result_message_str + '\n' + response.content)],
                "current_task_messages":task_messages}
    
    except Exception as e:
        error_message = f"Error in ir_analysis_node: {str(e)}"
        error_info = traceback.format_exc()
        logger.error(error_message + error_info)
        
        # from langgraph.config import get_stream_writer
        # writer = get_stream_writer()
        # writer(f"❌ {error_message}")
        
        return {
            "error": error_message,
            "has_error": True,
            "messages": [AIMessage(content=f"Failed to analyze IR results: {error_message}")],
            "messages_to_user": [AIMessage(content=f"Failed to analyze IR results: {error_message}")]
        }

#-------------------------------------------------
# Graph
#-------------------------------------------------
# from .agent_template import schema_to_str
# logger.debug("Generated IR schema:")
# logger.debug(schema_to_str(IRState))

ir_static_builder = StateGraph(IRState)
prepare_molecule_graph = prepare_molecule_builder.compile()
method_graph = method_builder.compile()

ir_static_builder.add_node("get_folder_name_node",get_folder_name_node)
ir_static_builder.add_node("get_geom",prepare_molecule_graph)
ir_static_builder.add_node("get_method",method_graph)
ir_static_builder.add_node("ir_settings",ir_settings_node)
ir_static_builder.add_node("ir_coder",ir_coder_node)
ir_static_builder.add_node("ir_exec",ir_exec_node)
ir_static_builder.add_node("ir_analysis",ir_analysis_node)
ir_static_builder.add_node("get_result_file_node",get_result_file_node)

ir_static_builder.add_edge(START,"get_folder_name_node")
ir_static_builder.add_edge("get_folder_name_node","get_geom")
ir_static_builder.add_edge("get_geom","get_method")
ir_static_builder.add_edge("get_method","ir_settings")
ir_static_builder.add_edge("ir_settings","ir_coder")
ir_static_builder.add_edge("ir_coder","ir_exec")
ir_static_builder.add_edge("ir_exec","ir_analysis")
ir_static_builder.add_edge("ir_analysis","get_result_file_node")
ir_static_builder.add_edge("get_result_file_node",END)

ir_static_graph = ir_static_builder.compile()

from .agent_template import BaseAgent
ir_static_agent = BaseAgent(
    name='ir_static_agent',
    description='Perform static infrared (IR) intensity calculation of a molecular structure, usually an optimized molecule, to get the IR spectrum.',
    graph=ir_static_builder,
)