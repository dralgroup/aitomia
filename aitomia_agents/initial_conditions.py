"""
    Initial conditions agent
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
from .prepare_molecule import prepare_molecule_builder
from .file_manager import get_folder_name_node, get_result_file_node

from .logger import logger 
from .utils import create_agent, pretty_dict, pretty_array, Analysis

from typing import Optional, Literal, List 

mod_path = "mlatom.initial_conditions"
func_path = "generate_initial_conditions"
doc_func_path = "generate_initial_conditions"

replace = {
    "arguments":{
        "generation_method":{
            "type":Optional[str],"default":None,"description":"Initial condition generation method. Random, Maxwell-Boltzmann, Wigner, and harmonic-quantum-Boltzmann are available. By default, Maxwell-Boltzmann should be used."
        }
    }
}
delete = ["molecule", "file_with_initial_xyz_coordinates","file_with_initial_xyz_velocities","eliminate_angular_momentum","use_hessian","filter_by_energy_window","window_filter_kwargs","random_seed"]

#-------------------------------------------------
# Prompts
#-------------------------------------------------
PROMPT_INITIAL_CONDITIONS_ANALYSIS = "You should give explicitly the absolute path of the result file for further calculation of analysis. If there is any file path, please use <Path>file_path</Path> format to indicate the file path."

#-------------------------------------------------
# Schema
#-------------------------------------------------
# schema of initial conditions should hold every properties needed
import copy 
schema_replace = copy.deepcopy(replace)
schema_replace["arguments"].update({
    "molecule_file_name":{"type":str, "default":None, "description":"path to the molecule file of local minimum"},
    "initial_conditions_result_files": {"type":List[str],"default":None,"description":""},
})
InitialConditionState = schema_from_mlatom(
    mod_path=mod_path, func_path=func_path, doc_func_path=doc_func_path,
    replace=schema_replace, delete=delete, schema_name="InitialConditionState"
)

#-------------------------------------------------
# Tool functions
#-------------------------------------------------
tool_replace = copy.deepcopy(replace)

tool = tool_from_mlatom(
    mod_path=mod_path, func_path=func_path, doc_func_path=doc_func_path, 
    replace=replace, delete=delete+[], tool_name="initial_conditions"
)
llm = create_agent()

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

# 2. generate initial conditions
working_directory = {{working_directory | pyrepr}}
init_cond_db = ml.generate_initial_conditions(
    molecule = mol,
    generation_method = {{generation_method | pyrepr}},
    number_of_initial_conditions = {{number_of_initial_conditions}},
    degrees_of_freedom = {{degrees_of_freedom}},
    initial_temperature = {{initial_temperature}},
    initial_kinetic_energy = {{initial_kinetic_energy}},
    reaction_coordinate_momentum = {{reaction_coordinate_momentum}},
)

filename = os.path.join(working_directory,"init_cond_db.json")
init_cond_db.dump(filename,format='json')

"""
#-------------------------------------------------
# Subgraphs
#-------------------------------------------------
# Read initial conditions settings
def initial_conditions_settings_node(state):
    logger.info("initial conditions settings node")
    
    logger.debug("Input state:"); pretty_dict(state.model_dump(), logger)

    settings_agent = create_llm(tools=[tool],tool_kwargs={"tool_choice":"any"})
    settings_tool = ToolNode([tool])

    agent_response = settings_agent.invoke(state.current_task_messages[-1])
    logger.debug("Response from initial conditions settings agent:")
    logger.debug(agent_response)

    kwargs = settings_tool.invoke({"messages":[agent_response]})
    logger.debug("Response from initial conditions settings tool:")
    logger.debug(kwargs)

    kwargs = json.loads(kwargs["messages"][-1].content)
    kwargs['working_direcotry'] = state.working_directory_stack[-1]

    logger.debug("Output state:"); pretty_dict(state.model_dump(), logger)
    return kwargs

# Initial conditions coder graph
def initial_conditions_coder_node(state):
    from jinja2 import Environment 

    logger.info("initial conditions coder node")
    message = "Start generating initial conditions script"
    logger.debug("Input state:"); pretty_dict(state.model_dump(), logger)

    input_args = state.model_dump().copy()
    if state.molecule_file_name[-4:] == ".xyz": input_args['molecule_format'] = 'xyz'
    elif state.molecule_file_name[-5:] == ".json": input_args['molecule_format'] = 'json' 
    else: raise ValueError("Unhandled molecule format")

    env = Environment(autoescape=False)
    env.filters["pyrepr"] = repr
    code = env.from_string(run_py).render(**input_args)

    working_directory = state.working_directory_stack[-1]
    python_script = os.path.join(working_directory, "run_init_cond.py")

    with open(python_script, 'w') as f:
        f.write(code)

    logger.debug("Output state:"); logger.debug({"scripts":python_script})
    return {"scripts":python_script,
            "messages_to_user": [AIMessage(content=message)],} 
    
def initial_conditions_exec_node(state):
    logger.info("Start getting initial conditions")
    message = "Start getting initial conditions"

    try:
        python_script = state.scripts
        result_log = os.path.join(state.working_directory_stack[-1], "init_cond.log")
        result_err = os.path.join(state.working_directory_stack[-1], "init_cond.err")
        result_mol = os.path.join(state.working_directory_stack[-1], "init_cond_db.json")

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
            
            error_message = f"Initial conditions calculation script failed with exit code {exit_code}"
            if error_content:
                error_message += f"\nError output:\n{error_content}"
            
            logger.error(error_message)
            return {
                "error": error_message,
                "has_error": True,
                "messages": [AIMessage(content=error_message)],
                "messages_to_user": [AIMessage(content=error_message)],
            }

        return {"initial_conditions_result_files": [result_log, result_err,result_mol],
                "messages_to_user": [AIMessage(content=message)],}
    
    except Exception as e:
        error_message = f"Error in initial_conditions_exec_node: {str(e)}"
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
            "messages": [AIMessage(content=f"Initial conditions calculation failed: {error_message}")],
            "messages_to_user": [AIMessage(content=f"Initial conditions calculation failed: {error_analysis}")]
        }
    
def initial_conditions_analysis_node(state):
    logger.info("Start analyzing the result of initial conditions")
    message = "Start analyzing the result of initial conditions"

    try:
        result_files = state.initial_conditions_result_files
        log_file = result_files[0]
        log_file = open(log_file, 'r').readlines()
        output_log = ""
        start_ii = 0
        for ill, ll in enumerate(log_file):
            if "Finish initial conditions calculation" in ll:
                start_ii = ill
                break 

        # Output log
        output_log = "".join(log_file[start_ii-1:])
        output_log += f"\nExecutable file: {state.scripts}"
        output_log += f"\nLog file: {result_files[0]}"
        output_log += f"\nError file: {result_files[1]}"
        output_log += f"\nInitial conditions with geometries and velocities: {result_files[2]}"

        # Prepare formatted summary
        moldb = ml.data.molecular_database.load(result_files[2],format='json')
        analysis_prompt = SystemMessage(content=PROMPT_INITIAL_CONDITIONS_ANALYSIS)
        result_message_str = "Initial conditions result: \n"
        result_message_str += f"    Number of points: {len(moldb)}"
        result_message_str += f"    Initial conditions (molecular database) with geometries and velocities are saved in <Path>{result_files[2]}</Path>"
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

        return {"messages": [AIMessage(content=output_log + result_message_str + '\n' + response.content)],
                "messages_to_user": [AIMessage(content=message + '\n' + output_log + result_message_str + '\n' + response.content)],
                "current_task_messages":task_messages}
    except Exception as e:
        error_message = f"Error in ir_analysis_node: {str(e)}"
        error_info = traceback.format_exc()
        logger.error(error_message + error_info)
        
        return {
            "error": error_message,
            "has_error": True,
            "messages": [AIMessage(content=f"Failed to analyze IR results: {error_message}")],
            "messages_to_user": [AIMessage(content=f"Failed to analyze IR results: {error_message}")]
        }
    
init_cond_builder = StateGraph(InitialConditionState)
prepare_molecule_graph = prepare_molecule_builder.compile(checkpointer=True)

init_cond_builder.add_node("get_folder_name_node",get_folder_name_node)
init_cond_builder.add_node("get_mol",prepare_molecule_graph)
init_cond_builder.add_node("init_cond_settings",initial_conditions_settings_node)
init_cond_builder.add_node("init_cond_coder",initial_conditions_coder_node)
init_cond_builder.add_node("init_cond_exec",initial_conditions_exec_node)
init_cond_builder.add_node("init_cond_analysis",initial_conditions_analysis_node)
init_cond_builder.add_node("get_result_file_node",get_result_file_node)

init_cond_builder.add_edge(START,"get_folder_name_node")
init_cond_builder.add_edge("get_folder_name_node","get_mol")
init_cond_builder.add_edge("get_mol","init_cond_settings")
init_cond_builder.add_edge("init_cond_settings","init_cond_coder")
init_cond_builder.add_edge("init_cond_coder","init_cond_exec")
init_cond_builder.add_edge("init_cond_exec","init_cond_analysis")
init_cond_builder.add_edge("init_cond_analysis","get_result_file_node")
init_cond_builder.add_edge("get_result_file_node",END)

init_cond_graph = init_cond_builder.compile()

from .agent_template import BaseAgent
init_cond_agent = BaseAgent(
    name='init_cond_agent',
    description="Get initial conditions (molecular database) with XYZ coordinates and velocities.",
    graph=init_cond_builder,
)