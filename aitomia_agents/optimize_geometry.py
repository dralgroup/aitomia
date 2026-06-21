"""
    Geometry optimization agent
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

from .method_judge import method_builder
from .prepare_molecule import prepare_molecule_builder
from .file_manager import get_folder_name_node, get_result_file_node

from .states import AitomiaState
from .logger import logger 
from .utils import create_agent, pretty_dict, pretty_array, Analysis

from langgraph.config import get_stream_writer

#-------------------------------------------------
# Schema
#-------------------------------------------------
class GeomoptState(AitomiaState): 
    optprog: str = None
    molecule_file_name: str = None 
    method: str = None
    program: Union[str,None] = None
    geomopt_result: Union[str,None] = None

#-------------------------------------------------
# Prompts
#-------------------------------------------------
PROMPT_MLATOM_SUPPORTED_OPTPROG = "Available programs for geometry optimization in MLatom are geomtric, Gaussian, ase, and scipy. By default, choose geometric, unless the program is specified."
PROMPT_GEOMOPT_ANALYSIS = "Below shows the result of geometry optimization. It contains the optimized geometry of the molecule, energy, and other properties. You should give explicitly the absolute path of the result file for further calculation of analysis. Please give a brief summary of the result in a few sentences. If there is any file path, please use <Path>file_path</Path> format to indicate the file path. If there is any file path, please use <Path>file_path</Path> format to indicate the file path."

#-------------------------------------------------
# Tool functions
#-------------------------------------------------
def get_optprog(optprog:str=""):
    """
    This function is used to get program for geometry optimization, The program here means Calculation program. 

    Args:
        optprog: The program for geometry optimization.
    """
    if optprog == "":
        optprog = "geometric"

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

# 3. Geometry optimization
working_directory = {{working_directory | pyrepr}}
MODEL.working_directory = working_directory
geomopt = ml.optimize_geometry(
    model=MODEL,
    initial_molecule=mol,
    program={{optprog | pyrepr}},
    working_directory=working_directory
)

# 4. Save results
optmol = geomopt.optimized_molecule
optmol.dump({{geomopt_result | pyrepr}},format='json')
traj = geomopt.optimization_trajectory
traj.dump(os.path.join(working_directory,'opttraj.json'),format='json')
traj = traj.to_database()
traj.write_file_with_xyz_coordinates(os.path.join(working_directory,'opttraj.xyz'))
"""
#-------------------------------------------------
# Agent
#-------------------------------------------------
llm = create_agent()
tools = [get_optprog]
get_optprog_agent = create_agent(tools=tools)
get_optprog_tool = ToolNode(tools)

#-------------------------------------------------
# Graph
#-------------------------------------------------
def get_optprog_node(state:GeomoptState):
    logger.info("Starting getting geomopt program")
    
    try:
        program_avail_prompt = SystemMessage(content=PROMPT_MLATOM_SUPPORTED_OPTPROG)

        response = get_optprog_agent.invoke(state.current_task_messages[-1]+[program_avail_prompt])

        logger.debug("respose from the method agent")
        logger.debug("\t"+response.content)

        output = get_optprog_tool.invoke({"messages":[response]})["messages"][-1].content
        logger.debug("response from get_optprog tool:")
        logger.debug("\t"+output)
        output = json.loads(output)

        optprog = output["optprog"]
        if not optprog.casefold() in ['ase','geometric','gaussian','scipy']:
            optprog = 'geometric'

        output_state = {
            "optprog": optprog
        }
        return output_state
    
    except Exception as e:
        error_message = f"Error in get_optprog_node: {str(e)}"
        error_info = traceback.format_exc()
        logger.error(error_message + error_info)
        # writer = get_stream_writer()
        # writer(f"❌ {error_message}")
        return {
            "error": error_message,
            "optprog": "geometric",  # Use default
            "messages": [AIMessage(content=f"Failed to get optprog, using default 'geometric': {error_message}")],
            "messages_to_user": [AIMessage(content=f"Failed to get optprog, using default 'geometric': {error_message}")]
        }

def geomopt_node(state:GeomoptState):
    logger.info("Start geometry optimization")
    message = "Start geometry optimization"

    try:
        molecule_file_name = state.molecule_file_name
        if '.xyz' in molecule_file_name:
            mol = ml.data.molecule.from_xyz_file(molecule_file_name)
        elif '.json' in molecule_file_name:
            mol = ml.data.molecule.load(molecule_file_name,format='json')
        else:
            raise ValueError("Unknown molecule file format")
        
        method = ml.models.methods(method=state.method,program=state.program)
        geomopt = ml.optimize_geometry(model=method,initial_molecule=mol,program=state.optprog)
        optmol = geomopt.optimized_molecule
        # Result file name
        filename = "optmol.json"
        if os.path.exists(os.path.join(state.working_directory,filename)):
            ii = 1
            while True:
                if os.path.exists(os.path.join(state.working_directory,f"optmol_{ii}.json")):
                    ii += 1
                else:
                    break 
            filename = f"optmol_{ii}.json"
        result_file_name = os.path.join(state.working_directory,filename)

        optmol.dump(result_file_name,format='json')
        traj = geomopt.optimization_trajectory
        traj.dump('opttraj.json',format='json')
        traj = traj.to_database()
        traj.write_file_with_xyz_coordinates('opttraj.xyz')

        return {"geomopt_result":result_file_name,
                "messages_to_user": [AIMessage(content=message)],}
    
    except Exception as e:
        error_message = f"Error in geomopt_node: {str(e)}"
        error_info = traceback.format_exc()
        logger.error(error_message + error_info)
        # writer = get_stream_writer()
        # writer(f"❌ {error_message}")
        return {
            "error": error_message,
            "has_error": True,
            "messages": [AIMessage(content=f"Geometry optimization failed: {error_message}")],
            "messages_to_user": [AIMessage(content=f"Geometry optimization failed: {error_message}")],
        }

def geomopt_coder_node(state):
    try:
        from jinja2 import Environment 

        logger.info("Start geometry optimization coder")
        message = "Start geometry optimization coder"

        # Prepare result filename
        filename = "optmol.json"
        if os.path.exists(os.path.join(state.working_directory,filename)):
            ii = 1
            while True:
                if os.path.exists(os.path.join(state.working_directory,f"optmol_{ii}.json")):
                    ii += 1
                else:
                    break 
            filename = f"optmol_{ii}.json"
        logger.debug("Result file name:");
        logger.debug("\t" + filename)
        result_file_name = os.path.join(state.working_directory,filename)
        state.geomopt_result = result_file_name

        input_args = state.model_dump().copy()
        if state.molecule_file_name[-4:] == ".xyz": input_args['molecule_format'] = 'xyz'
        elif state.molecule_file_name[-5:] == ".json": input_args['molecule_format'] = 'json' 
        else: raise ValueError("Unhandled molecule format")

        env = Environment(autoescape=False)
        env.filters["pyrepr"] = repr
        code = env.from_string(run_py).render(**input_args)

        working_directory = state.working_directory_stack[-1]
        python_script = os.path.join(working_directory, "run_geomopt.py")

        with open(python_script, 'w') as f:
            f.write(code)

        logger.debug("Output state:"); logger.debug({"scripts":python_script})
        return {"scripts":python_script,
                "geomopt_result": result_file_name,
                "messages_to_user": [AIMessage(content=message)]}
    except Exception as e:
        error_message = f"Error in geomopt_coder_node: {str(e)}"
        error_info = traceback.format_exc()
        logger.error(error_message + error_info)
        state.has_error = True
        state.error = error_message
        response = Analysis.error_analysis(error_message)
        error_analysis = error_message + '\n' + '\n' + response
        # writer = get_stream_writer()
        # writer(f"❌ {error_message}")
        return {
            "error": error_message,
            "has_error": True,
            "messages": [AIMessage(content=f"Geometry optimization coding failed: {error_message}")],
            "messages_to_user": [AIMessage(content=f"Geometry optimization coding failed: {error_analysis}")],
        }

def geomopt_exec_node(state):
    logger.info("Start geometry optimization")
    message = "Start geometry optimization"
    # writer = get_stream_writer()
    # writer(message)
    try:
        python_script = state.scripts
        result_log = os.path.join(state.working_directory_stack[-1], "geomopt.log")
        result_err = os.path.join(state.working_directory_stack[-1], "geomopt.err")

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
            
            error_message = f"Geometry optimization script failed with exit code {exit_code}"
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
        error_message = f"Error in geomopt_exec_node: {str(e)}"
        error_info = traceback.format_exc()
        logger.error(error_message + error_info)
        response = Analysis.error_analysis(error_message)
        error_analysis = error_message + '\n' + '\n' + response
        # writer = get_stream_writer()
        # writer(f"❌ {error_message}")
        return {
            "error": error_message,
            "has_error": True,
            "messages": [AIMessage(content=f"Geometry optimization failed: {error_message}")],
            "messages_to_user": [AIMessage(content=f"Geometry optimization failed: {error_analysis}")],
        }


def geomopt_analysis_node(state:GeomoptState):
    logger.info("Start analyzing the result of geometry optimization")
    message = "Start analyzing the result of geometry optimization"
    
    try:
        mol = ml.data.molecule.load(state.geomopt_result,format='json')
        energy = mol.energy
        xyz = mol.get_xyz_string()
        analysis_prompt = SystemMessage(content=PROMPT_GEOMOPT_ANALYSIS)

        # Prepare formatted summary
        result_message_str = f"Geometry optimization result: \n"
        result_message_str += f"    Energy: {energy} Hartree\n"
        result_message_str += f"    Optimized Geometry (Angstrom): {xyz}\n"
        result_file_name = state.geomopt_result
        result_message_str += f"    Optimized geometry is saved in <Path>{result_file_name}</Path>"
        result_message = AIMessage(content=result_message_str)

        # Get analysis of formatted summary from LLM
        response = llm.invoke(state.current_task_messages[-1]+[analysis_prompt,result_message])
        task_messages = state.current_task_messages
        task_messages[-1].append(AIMessage(content=response.content))

        # Write formatted summary into summary.out
        with open(os.path.join(state.working_directory,"summary.out"),'w') as f:
            f.write(result_message_str)
            Analysis.add_summary({'current_task_summary':result_message_str})
        result_message_str += f"Detailed summary can be found in <Path>{os.path.join(state.working_directory,'summary.out')}</Path>"

        # writer = get_stream_writer()
        # writer(result_message_str + '\n' + response.content + '\n' + "Calculation completed")

        return {
            "messages": [AIMessage(content=result_message_str + '\n' + response.content)],
            "messages_to_user": [AIMessage(content=message + '\n' + result_message_str + '\n' + response.content)],
            "current_task_messages":task_messages,
        }
    
    except Exception as e:
        error_message = f"Error in geomopt_analysis_node: {str(e)}"
        error_info = traceback.format_exc()
        logger.error(error_message + error_info)
        # writer = get_stream_writer()
        # writer(f"❌ {error_message}")
        return {
            "error": error_message,
            "has_error": True,
            "messages": [AIMessage(content=f"Failed to analyze geometry optimization results: {error_message}")],
            "messages_to_user": [AIMessage(content=f"Failed to analyze geometry optimization results: {error_message}")]
        }

#-------------------------------------------------
# Error checking function
#-------------------------------------------------
def check_error(state:GeomoptState):
    """Check if an error has occurred and route accordingly"""
    if state.has_error:
        logger.warning("Error detected, skipping remaining steps")
        return END
    return "continue"

geomopt_builder = StateGraph(GeomoptState)

method_graph = method_builder.compile()
prepare_molecule_graph = prepare_molecule_builder.compile()

geomopt_builder.add_node("get_folder_name_node",get_folder_name_node)
geomopt_builder.add_node("method",method_graph)
geomopt_builder.add_node("prepare_molecule",prepare_molecule_graph)
geomopt_builder.add_node("get_optprog_node",get_optprog_node)
# geomopt_builder.add_node("geomopt_node",geomopt_node)
geomopt_builder.add_node("geomopt_coder_node",geomopt_coder_node)
geomopt_builder.add_node("geomopt_exec_node",geomopt_exec_node)
geomopt_builder.add_node("geomopt_analysis_node",geomopt_analysis_node)
geomopt_builder.add_node("get_result_file_node",get_result_file_node)

geomopt_builder.add_edge(START,"get_folder_name_node")
geomopt_builder.add_conditional_edges("get_folder_name_node", check_error, {"continue": "method", END: END})
geomopt_builder.add_conditional_edges("method", check_error, {"continue": "prepare_molecule", END: END})
geomopt_builder.add_conditional_edges("prepare_molecule", check_error, {"continue": "get_optprog_node", END: END})
# geomopt_builder.add_conditional_edges("get_optprog_node", check_error, {"continue": "geomopt_node", END: END})
# geomopt_builder.add_conditional_edges("geomopt_node", check_error, {"continue": "geomopt_analysis_node", END: END})
geomopt_builder.add_conditional_edges("get_optprog_node", check_error, {"continue": "geomopt_coder_node", END: END})
geomopt_builder.add_edge("geomopt_coder_node","geomopt_exec_node")
geomopt_builder.add_conditional_edges("geomopt_exec_node", check_error, {"continue": "geomopt_analysis_node", END: END})
geomopt_builder.add_conditional_edges("geomopt_analysis_node", check_error, {"continue": "get_result_file_node", END: END})
geomopt_builder.add_edge("get_result_file_node",END)

geomopt_graph = geomopt_builder.compile()

from .agent_template import BaseAgent
geomopt_agent = BaseAgent(
    name='geomopt_agent',
    description='Perform geometry optimization of a molecular structure and optimize it to the local minimum.',
    graph=geomopt_builder,
)