"""
    IRC agent

choose working directory ->
check molecule availability ->
check method and its programs -> 
get IRC settings ->
generate IRC script ->
execute IRC script (optional) ->
summary of IRC (optional)
"""

from .agent_template import BaseAgent, tool_from_mlatom, create_llm, schema_from_mlatom
import json, os
import traceback
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END 
from langgraph.types import interrupt, Command


from langchain_core.messages import AIMessage

# load other graphs
from .method_judge import method_builder
from .prepare_molecule import prepare_molecule_builder
from .file_manager import get_folder_name_node, get_result_file_node

from .logger import logger 
from .utils import pretty_dict, Analysis

from langgraph.config import get_stream_writer


from typing import Optional, Literal, List


mod_path = "mlatom.irc"
func_path = "irc.__init__"
doc_func_path = "irc"
replace = {
    "arguments": {
        "irc_program":{
            "type":Optional[str], "default":None, "description":"the program that provide algorithm to generate IRC instead of program that provides energy and its derivatives. Available options are geometric, gaussian, pysisyphus and None. By default None will be used which is the convenient and fast built-in implementation."},
        "irc_program_kwargs":{
            "type":Optional[dict], "default":None, "description":"Control the behavior of the algorithms used in IRC program. Do not provide keywords is they are not related to the behavior of algorithms"},
        },    
    }
delete = ["molecule", "model", "model_predict_kwargs"]

#-------------------------------------------------
# Schema
#-------------------------------------------------
# schema of irc should hold every properties needed
import copy
schema_replace = copy.deepcopy(replace)
schema_replace["arguments"].update({
    "molecule_file_name":{"type":str, "default":None, "description":"path to the molecule file of transition state"},
    "method": {"type":str, "default":None, "description":"The method to get energy and energy derivatives to propagate IRC trajectories."},
    "program": {"type":Optional[str], "default":None, "description":"The program to be used for the method to get energy and energy derivatives."},
    "irc_result_files": {"type":List[str],"default":None,"description":""},
})
IRCState = schema_from_mlatom(
    mod_path=mod_path, func_path=func_path, doc_func_path=doc_func_path,
    replace=schema_replace, delete=delete, schema_name="IRCState"
)

#-------------------------------------------------
# Tool functions
#-------------------------------------------------
irc_tool = tool_from_mlatom(
    mod_path=mod_path, func_path=func_path, doc_func_path=doc_func_path, 
    replace=replace, delete=delete+["program", "program_kwargs", "working_directory"], tool_name="irc")

#-------------------------------------------------
# scripts
#-------------------------------------------------
# render script with jinja2 template

run_irc_py = """#!/bin/env python

import mlatom as ml 

# 1. load ts molecule
molecule_path = {{molecule_file_name | pyrepr}}
{% if molecule_format == 'json' %}
ts = ml.molecule.load(molecule_path, format='json)
{% elif molecule_format == 'xyz' %}
ts = ml.molecule.from_xyz_file(molecule_path)
{% endif %}

# 2. define method
MODEL = ml.methods(method={{method | pyrepr}}, program={{program | pyrepr}})

# 3. fill in settings of irc
working_directory = {{working_directory | pyrepr}}
MODEL.working_directory = working_directory
irc_results = ml.irc(
    molecule = ts,
    model = MODEL,
    program = {{irc_program | pyrepr}},
    program_kwargs = {{irc_program_kwargs}},
    forward = {{forward}},
    backward = {{backward}},
    working_directory = {{working_directory | pyrepr}}, 
    overwrite = {{overwrite}},
    plot = {{plot}}, 
    plot_filename = {{plot_filename | pyrepr}},
    dump = {{dump}}, 
    dump_filename = {{dump_filename | pyrepr}},
    dump_format = {{dump_format | pyrepr}},
    verbose = {{verbose}}
)
"""

run_irc_inp = """
irc
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

def irc_settings_node(ircstate):
    logger.info("irc settings node")

    logger.debug("Input state:"); pretty_dict(ircstate.model_dump(), logger)

    irc_settings_agent = create_llm(tools=[irc_tool],tool_kwargs={"tool_choice":"any"})
    irc_settings_tool = ToolNode([irc_tool])

    agent_response = irc_settings_agent.invoke(ircstate.current_task_messages[-1])
    logger.debug("Response from irc settings agent:")
    logger.debug(agent_response)

    irc_kwargs = irc_settings_tool.invoke({"messages":[agent_response]})
    logger.debug("Response from irc settings tool:")
    logger.debug(irc_kwargs)

    irc_kwargs = json.loads(irc_kwargs["messages"][-1].content)
    irc_kwargs['working_direcotry'] = ircstate.working_directory_stack[-1]

    logger.debug("Output state:"); pretty_dict(ircstate.model_dump(), logger)

    # writer = get_stream_writer()
    # writer(f"Generating parameters for IRC calculation:\nThe generated parameters:\n{json.dumps(irc_kwargs, indent=4)}")

    return irc_kwargs

# IRC coder graph
# currently, it's not generated by LLM
def irc_coder_node(ircstate):

    from jinja2 import Environment
    
    logger.info("irc python code node")

    logger.debug("Input state:"); pretty_dict(ircstate.model_dump(), logger)

    input_args = ircstate.model_dump().copy()
    if ircstate.molecule_file_name[-4:] == ".xyz": input_args['molecule_format'] = 'xyz'
    elif ircstate.molecule_file_name[-4:] == ".json": input_args['molecule_format'] = 'json' 
    else: raise ValueError("Unhandled molecule format")

    env = Environment(autoescape=False)
    env.filters["pyrepr"] = repr
    irc_code = env.from_string(run_irc_py).render(**input_args)

    working_directory = ircstate.working_directory_stack[-2]
    python_script = os.path.join(working_directory, "run_irc.py")

    with open(python_script, 'w') as f:
        f.write(irc_code)

    logger.debug("Output state:"); logger.debug({"scripts":python_script})
    return {"scripts":python_script} 

def irc_exec_user(ircstate) -> Command[Literal["irc_exec"]]:

    resume = interrupt({
        "messages_to_user": f"The generated executable script is located at {ircstate.scripts}. Proceed to execute the generated script for IRC calculation? [yes/no]",
        "options": ["yes", "no"]
    })
    
    logger.debug("Response from user:")
    logger.debug(resume)

    return Command(goto="irc_exec" if resume["messages"].content.lower() == "yes" else "irc_exec_cancel") 

def irc_exec_node(ircstate):
    try:
        python_script = ircstate.scripts
        result_log = os.path.join(ircstate.working_directory_stack[-2], "irc.log")
        result_err = os.path.join(ircstate.working_directory_stack[-2], "irc.err")

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
            
            error_message = f"IRC calculation script failed with exit code {exit_code}"
            if error_content:
                error_message += f"\nError output:\n{error_content}"
            
            logger.error(error_message)
            return {
                "error": error_message,
                "has_error": True,
                "messages": [AIMessage(content=error_message)],
                "messages_to_user": [AIMessage(content=error_message)],
            }
        
        return {"irc_result_files": [result_log, result_err]}
    
    except Exception as e:
        error_message = f"Error in irc_exec_node: {str(e)}"
        error_info = traceback.format_exc()
        logger.error(error_message + error_info)
        response = Analysis.error_analysis(error_message)
        error_analysis = error_message + '\n' + '\n' + response
        # writer = get_stream_writer()
        # writer(f"IRC calculation failed: {error_message}")
        return {
            "error": error_message,
            "has_error": True,
            "messages": [AIMessage(content=f"IRC calculation failed: {error_message}")],
            "messages_to_user": [AIMessage(content=f"IRC calculation failed: {error_analysis}")]
        }

def irc_exec_cancel_node(ircstate):
    from langchain_core.messages import AIMessage
    return {"messages_to_user": AIMessage("Script execution cancelled.")} 

def irc_analysis_node(ircstate):
    try:
        result_files = ircstate.irc_result_files
        log_file = result_files[0]
        log_file = open(log_file, 'r').readlines()
        output_log = ""
        for ill, ll in enumerate(log_file):
            if "Finish IRC calculation" in ll:
                start_ii = ill
                break 
        output_log = "".join(log_file[start_ii-1:])
        output_log += f"\nLog file: {result_files[0]}"
        output_log += f"\nError file: {result_files[1]}"
        output_log += f"\nExecutable file: {ircstate.scripts}"
    
        # writer = get_stream_writer()
        # writer(output_log + '\n' + "Calculation completed")
        return {"messages": AIMessage(output_log),
                "messages_to_user": AIMessage(output_log)}
    
    except Exception as e:
        error_message = f"Error in irc_analysis_node: {str(e)}"
        error_info = traceback.format_exc()
        logger.error(error_message + error_info)
        # writer = get_stream_writer()
        # writer(f"❌ {error_message}")
        return {
            "error": error_message,
            "has_error": True,
            "messages": [AIMessage(content=f"Failed to analyze IRC results: {error_message}")],
            "messages_to_user": [AIMessage(content=f"Failed to analyze IRC results: {error_message}")]
        }


#-------------------------------------------------
# Graph
#-------------------------------------------------
# from .agent_template import schema_to_str
# logger.debug("Generated IRC schema:")
# logger.debug(schema_to_str(IRCState))

irc_builder = StateGraph(IRCState)
prepare_molecule_graph = prepare_molecule_builder.compile()
method_graph = method_builder.compile()

irc_builder.add_node("get_folder_name_node", get_folder_name_node)
irc_builder.add_node("get_geom", prepare_molecule_graph)
irc_builder.add_node("get_method", method_graph)
irc_builder.add_node("irc_settings", irc_settings_node)
irc_builder.add_node("irc_coder", irc_coder_node)
irc_builder.add_node("irc_exec", irc_exec_node)
irc_builder.add_node("irc_exec_user", irc_exec_user)
irc_builder.add_node("irc_exec_cancel", irc_exec_cancel_node)
irc_builder.add_node("irc_analysis", irc_analysis_node)

irc_builder.add_edge(START, "get_folder_name_node")
irc_builder.add_edge("get_folder_name_node","get_geom")
irc_builder.add_edge("get_geom", "get_method")
irc_builder.add_edge("get_method", "irc_settings")
irc_builder.add_edge("irc_settings", "irc_coder")
irc_builder.add_edge("irc_coder", "irc_exec_user")
irc_builder.add_edge("irc_exec", "irc_analysis")
irc_builder.add_edge("irc_analysis", END)
irc_builder.add_edge("irc_exec_cancel", END)

#-------------------------------------------------
# The Agent for Aitomia
#-------------------------------------------------

irc_agent = BaseAgent(
    name='IRC agent',
    description='perform IRC calculation',
    graph=irc_builder,
)




