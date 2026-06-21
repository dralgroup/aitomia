"""
    Transition state agent
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
func_path = "optimize_geometry.__init__"
doc_func_path = "optimize_geometry"
replace = {
    "arguments": {
        "ts_program":{
            "type":Optional[str], "default": None, "description": "The program that provide algorithm to optimize the geometry of the molecule to the transition state, instead of the program that provides energy and energy gradients. Available options are Gaussian, ase, and geometric. By default, you should use Gaussian."
        },
        "ts_program_kwargs":{
            "type":Optional[str], 'default': None, "description": "Control the behavior of the program used in transition state optimization. It should be None by default. Do not provide keywords that are not related to the algorithm, like method, program, and other keywords of TS calculation."
        }
    }
}
delete = ["initial_molecule","molecule","model","model_predict_kwargs","ts","reactants","products","dump_trajectory_interval","filename","format","constaints","optimization_algorithm"]
#-------------------------------------------------
# Prompts
#-------------------------------------------------
PROMPT_TS_ANALYSIS = "Below shows the result of the transition state search. It might contain multiple properties of the molecule, e.g., energy, XYZ coordinates, etc. You should give explicitly the absolute path of the result file for further calculation of analysis. Please give a brief summary of the result in a few sentences. `Nan` in the result means the corresponding properties are not calculated. If there is any file path, please use <Path>file_path</Path> format to indicate the file path."
#-------------------------------------------------
# Schema
#-------------------------------------------------
# schema of transition state search
import copy
schema_replace = copy.deepcopy(replace)
schema_replace["arguments"].update({
    "molecule_file_name":{"type":str, "default":None, "description":"path to the molecule file of initial guess of transition state"},
    "method": {"type":str, "default":None, "description":"The method to get energy, energy derivatives, and Hessian to get the transition state."},
    "program": {"type":Optional[str], "default":None, "description":"The program to be used for the method to get energy, energy derivatives, and Hessian."},
    "ts_result_files": {"type":List[str],"default":None,"description":""},
})
TSState = schema_from_mlatom(
    mod_path=mod_path, func_path=func_path, doc_func_path=doc_func_path,
    replace=schema_replace, delete=delete, schema_name="TSState"
)
#-------------------------------------------------
# Tool functions
#-------------------------------------------------
tool_replace = copy.deepcopy(replace)

ts_tool = tool_from_mlatom(
    mod_path=mod_path, func_path=func_path, doc_func_path=doc_func_path, 
    replace=replace, delete=delete+["program", "program_kwargs", "working_directory"], tool_name="ts"
)
llm = create_agent()
#-------------------------------------------------
# scripts
#-------------------------------------------------
# render script with jinja2 template
run_ts_py = """#!/bin/env python

import mlatom as ml 
import os

# 1. load initial molecule
molecule_path = {{molecule_file_name | pyrepr}}
{% if molecule_format == 'json' %}
mol = ml.molecule.load(molecule_path, format='json')
{% elif molecule_format == 'xyz' %}
mol = ml.molecule.from_xyz_file(molecule_path)
{% endif %}

# 2. define method
MODEL = ml.methods(method={{method | pyrepr}}, program={{program | pyrepr}})

# 3. fill in settings of transition state search
working_directory = {{working_directory | pyrepr}}
MODEL.working_directory = working_directory
ts = ml.optimize_geometry(
    initial_molecule = mol,
    model = MODEL,
    program = {{ts_program | pyrepr}},
    program_kwargs = {{ts_program_kwargs}},
    maximum_number_of_steps = {{maximum_number_of_steps}},
    convergence_criterion_for_forces = {{convergence_criterion_for_forces}},
    working_directory = working_directory,
    dump_trajectory_interval = 1,
    filename = os.path.join(working_directory, "tstraj.json"),
    format = "json",
    ts = True,
)

tsmol = ts.optimized_molecule
tsmol.dump(os.path.join(working_directory,'tsmol.json'),format='json')
tstraj = ts.optimization_trajectory.to_database()
tstraj.write_file_with_xyz_coordinates(os.path.join(working_directory,'tstraj.xyz'))


"""

run_ts_inp = """
ts
method={{method}}
xyzfile={{molecule_file_name}}
{% elif molecule_format == 'json' %}
jsonfile={{molecule_file_name}}
{% endif %} 
"""

#-------------------------------------------------
# Subgraphs
#-------------------------------------------------
def ts_settings_node(state):
    logger.info("ts settings ndoe")
    
    logger.debug("Input state:"); pretty_dict(state.model_dump(),logger)
    ts_settings_agent = create_llm(tools=[ts_tool],tool_kwargs={"tool_choice":"any"})
    ts_settings_tool = ToolNode([ts_tool])

    agent_response = ts_settings_agent.invoke(state.current_task_messages[-1])
    logger.debug("Response from ts settings agent:")
    logger.debug(agent_response)

    ts_kwargs = ts_settings_tool.invoke({"messages":[agent_response]})
    logger.debug("Response from ts settings tool:")
    logger.debug(ts_kwargs)

    ts_kwargs = json.loads(ts_kwargs["messages"][-1].content)
    ts_kwargs["working_directory"] = state.working_directory_stack[-1]

    if 'ts_program' in ts_kwargs:
        if ts_kwargs['ts_program'] is None:
            logger.debug("Unrecognized ts program, use Gaussian instread")
            ts_kwargs['ts_program'] = 'Gaussian'
        elif not ts_kwargs['ts_program'].casefold() in ['ase','gaussian','geometric']:
            logger.debug("Unrecognized ts program, use Gaussian instread")
            ts_kwargs['ts_program'] = 'Gaussian'
    if 'ts_program_kwargs' in ts_kwargs:
        ts_kwargs['ts_program_kwargs'] = r"{}"

    logger.debug("Output state:"); pretty_dict(state.model_dump(), logger)

    return ts_kwargs 

# TS coder graph 
def ts_coder_node(state):
    from jinja2 import Environment

    logger.info("ts python code node")
    message = "Generating transition state calculation script"

    logger.debug("Input state:"); pretty_dict(state.model_dump(), logger)

    input_args = state.model_dump().copy()
    if state.molecule_file_name[-4:] == ".xyz": input_args['molecule_format'] = 'xyz'
    elif state.molecule_file_name[-5:] == ".json": input_args['molecule_format'] = 'json' 
    else: raise ValueError("Unhandled molecule format")

    env = Environment(autoescape=False)
    env.filters["pyrepr"] = repr
    ts_code = env.from_string(run_ts_py).render(**input_args)

    working_directory = state.working_directory_stack[-1]
    python_script = os.path.join(working_directory, "run_ts.py")

    with open(python_script, 'w') as f:
        f.write(ts_code)

    logger.debug("Output state:"); logger.debug({"scripts":python_script})
    return {"scripts":python_script,
            "messages_to_user": [AIMessage(content=message)]} 

def ts_exec_user(state) -> Command[Literal["ts_exec"]]:

    execute = interrupt({
        "messages_to_user": f"The generated executable script is located at {state.scripts}. Proceed to execute the generated script for TS calculation? [yes/no]",
        "options": ["yes", "no"]
    })

    logger.debug("Response from user:")
    logger.debug(execute)

    return Command(goto="ts_exec" if execute.lower() == "yes" else END) 

def ts_exec_node(state):
    logger.info("Start calculating transition state")
    message = "Start calculating transition state"

    try:

        python_script = state.scripts
        result_log = os.path.join(state.working_directory_stack[-1], "ts.log")
        result_err = os.path.join(state.working_directory_stack[-1], "ts.err")
        result_mol = os.path.join(state.working_directory_stack[-1], "tsmol.json")
        result_traj_json = os.path.join(state.working_directory_stack[-1], "tstraj.json")
        result_traj_xyz = os.path.join(state.working_directory_stack[-1], "tstraj.xyz")

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
            
            error_message = f"Transition state calculation script failed with exit code {exit_code}"
            if error_content:
                error_message += f"\nError output:\n{error_content}"
            
            logger.error(error_message)
            return {
                "error": error_message,
                "has_error": True,
                "messages": [AIMessage(content=error_message)],
                "messages_to_user": [AIMessage(content=error_message)],
            }

        return {"ts_result_files": [result_log, result_err,result_mol,result_traj_json,result_traj_xyz],
                "messages_to_user": [AIMessage(content=message)]}
    
    except Exception as e:
        error_message = f"Error in ts_exec_node: {str(e)}"
        error_info = traceback.format_exc()
        logger.error(error_message + error_info)
        response = Analysis.error_analysis(error_message)
        error_analysis = error_message + '\n' + '\n' + response
        # writer = get_stream_writer()
        # writer(f"❌ {error_message}")
        return {
            "error": error_message,
            "has_error": True,
            "messages": [AIMessage(content=f"Transition state calculation failed: {error_message}")],
            "messages_to_user": [AIMessage(content=f"Transition state calculation failed: {error_analysis}")],
        }

def ts_analysis_node(state):
    logger.info("Start analyzing the result of TS calculation")
    message = "Analyzing the result of TS calculation"
    
    try:
        result_files = state.ts_result_files
        log_file = result_files[0]
        log_file = open(log_file, 'r').readlines()
        output_log = ""
        start_ii = 0
        for ill, ll in enumerate(log_file):
            if "Finish TS calculation" in ll:
                start_ii = ill
                break 
        
        # Output log
        output_log = "".join(log_file[start_ii-1:])
        output_log += f"\nExecutable file: {state.scripts}"
        output_log += f"\nLog file: {result_files[0]}"
        output_log += f"\nError file: {result_files[1]}"
        output_log += f"\nTransition state: {result_files[2]}"
        output_log += f"\nTransition state search trajectory in json format: {result_files[3]}"
        output_log += f"\nTransition state search trajectory in xyz format: {result_files[4]}"

        # Prepare formatted summary
        mol = ml.data.molecule.load(result_files[2],format='json')
        analysis_prompt = SystemMessage(content=PROMPT_TS_ANALYSIS)
        result_message_str = "Transition state calculation result: \n"
        result_message_str += f"    Energy: {mol.energy} Hartree\n"
        result_message_str += f"    Optimized transition state geometry (Angstrom): {mol.get_xyz_string()}\n"
        result_message_str += f"    Transition state is saved in <Path>{result_files[2]}</Path>\n"
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

        # writer = get_stream_writer()
        # writer(result_message_str + '\n' + response.content + '\n' + "Calculation completed")

        return {"messages": [AIMessage(content=result_message_str + '\n' + response.content)],
                "messages_to_user": [AIMessage(content=message + '\n' + result_message_str + '\n' + response.content)],
                "current_task_messages":task_messages,}
    
    except Exception as e:
        error_message = f"Error in ts_analysis_node: {str(e)}"
        error_info = traceback.format_exc()
        logger.error(error_message + error_info)
        # writer = get_stream_writer()
        # writer(f"❌ {error_message}")
        return {
            "error": error_message,
            "has_error": True,
            "messages": [AIMessage(content=f"Failed to analyze transition state results: {error_message}")],
            "messages_to_user": [AIMessage(content=f"Failed to analyze transition state results: {error_message}")]
        }

#-------------------------------------------------
# Graph
#-------------------------------------------------
# from .agent_template import schema_to_str
# logger.debug("Generated TS schema:")
# logger.debug(schema_to_str(TSState))

ts_builder = StateGraph(TSState)
prepare_molecule_graph = prepare_molecule_builder.compile()
method_graph = method_builder.compile()

ts_builder.add_node("get_folder_name_node",get_folder_name_node)
ts_builder.add_node("get_geom",prepare_molecule_graph)
ts_builder.add_node("get_method",method_graph)
ts_builder.add_node("ts_settings",ts_settings_node)
ts_builder.add_node("ts_coder",ts_coder_node)
ts_builder.add_node("ts_exec",ts_exec_node)
ts_builder.add_node("ts_analysis",ts_analysis_node)
ts_builder.add_node("get_result_file_node",get_result_file_node)
# ts_builder.add_node()

ts_builder.add_edge(START,"get_folder_name_node")
ts_builder.add_edge("get_folder_name_node","get_geom")
ts_builder.add_edge("get_geom","get_method")
ts_builder.add_edge("get_method","ts_settings")
ts_builder.add_edge("ts_settings","ts_coder")
ts_builder.add_edge("ts_coder","ts_exec")
ts_builder.add_edge("ts_exec","ts_analysis")
ts_builder.add_edge("ts_analysis","get_result_file_node")
ts_builder.add_edge("get_result_file_node",END)

ts_graph = ts_builder.compile()

from .agent_template import BaseAgent
ts_agent = BaseAgent(
    name='ts_agent',
    description="Perform transition state search of a molecular structure",
    graph=ts_builder,
)