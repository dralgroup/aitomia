"""
    file_manager is used to manage the working directory.
"""

import json 
import os 
from pathlib import Path
import shutil
from typing import Union, Optional 
import traceback
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END

from aitomia_agents import user_context 

from .states import AitomiaState 
from .logger import logger 
from .utils import create_agent, pretty_dict, Analysis
from aitomia_agents.user_context import user_context

#-------------------------------------------------
# Schema
#-------------------------------------------------
class FileManagerState(AitomiaState):
    pass 

#-------------------------------------------------
# Prompts
#-------------------------------------------------
def prompt_get_result_file():
    prompt = ""
    prompt += f"You should extract the result file from the following messages. There could be no result file, one result file or many result files. Below is the format of your output:\n"
    prompt += "\tEach file should be put in one line.\n"
    prompt += "\tYou must give the exactly the same absolute path of the file, as is provided in the messages.\n"
    prompt += "\tYou should also attach the computational details in brackets like method, molecule, task type, and other conditons in the same line as each file.\n"
    prompt += "\tThe path and the computational details should be separated using blank.\n"
    prompt += "**Examples:**\n"
    prompt += "/directory/opt/ethanol_optmol.json [AIQM2 method, ethanol, geometry optimization]"
    # prompt += "- The optimized ethanol molecule with AIQM2 method is in /directory/opt/ethanol_optmol.json\n"
    # prompt += "- The frequencies and thermodynamic properties of the methanol molecule calculated with AIQM2 method is in /directory/freq/methanol_freqmol.json\n"

    return prompt

PROMPT_GET_RESULT_FILE = prompt_get_result_file()


#-------------------------------------------------
# Tool functions
#-------------------------------------------------
def get_folder_name(folder_name:str=""):
    """
    This function is to generate a folder name according to the current task. You need to provide the folder name.

    Args:
        folder_name: The name of the folder.
    """

    if folder_name == "":
        folder_name = "task"

    return locals()

def get_working_directory(working_directory:str=""):
    """
    This function determines the working directory based on user input.

    The working directory must be an absolute path. The resolution follows this order:
    1. First, check the user input.  
    - If the input is a directory, use it directly.  
    - If the input is a file path, use the parent directory of that file.
    2. If the user does not provide a path, then check the chosen path.  
    - If it is a directory, use it directly.  
    - If it is a file, use its parent directory.
    3. If neither the user input nor the chosen path is provided, use the home directory.

    Args:
        working_directory: The absolute path of the working directory. Do not return a file path.

    """
    try:
        if working_directory == "":
            working_directory = user_context.home_dir or str(Path.home())
        
        # Validate that the directory is an absolute path
        if not os.path.isabs(working_directory):
            logger.warning(f"Working directory '{working_directory}' is not absolute, converting to absolute path")
            working_directory = os.path.abspath(working_directory)
        
        return locals()
    except Exception as e:
        error_info = traceback.format_exc()
        logger.error(f"Error in get_working_directory: {e}" + error_info)
        fallback_dir = str(user_context.home_dir or str(Path.home()))
        logger.warning(f"Using fallback directory: {fallback_dir}")
        return {"working_directory": fallback_dir}

#-------------------------------------------------
# Agent
#-------------------------------------------------
tools = [get_folder_name]
get_folder_name_agent = create_agent(tools=tools, tool_kwargs={"tool_choice":"any"})
get_folder_name_tool = ToolNode(tools)

tools = [get_working_directory]
get_working_directory_agent = create_agent(tools=tools, tool_kwargs={"tool_choice":"any"})
get_working_directory_tool = ToolNode(tools)

llm = create_agent()

#-------------------------------------------------
# Graph
#-------------------------------------------------
def get_folder_name_node(state:FileManagerState):
    try:
        logger.info("Start getting folder name, current working directory is "+ str(state.working_directory))
        message = "Start getting folder name"
        
        if state.current_task_messages is None:
            state.current_task_messages = [state.messages]
        
        if state.working_directory is None:
            try:
                current_workdir = user_context.home_dir or Path.home()
                prompt = SystemMessage(content=f"User's home directory is {user_context.home_dir or Path.home()}")
                response = get_working_directory_agent.invoke(state.current_task_messages[-1]+[prompt])
                output = get_working_directory_tool.invoke({"messages":[response]})["messages"][-1].content
                output = json.loads(output)
                current_workdir = output["working_directory"]
                
                if not os.path.exists(current_workdir):
                    try:
                        os.makedirs(current_workdir)
                        logger.info(f"Created working directory: {current_workdir}")
                    except OSError as e:
                        error_msg = f"Failed to create working directory {current_workdir}: {e}"
                        error_info = traceback.format_exc()
                        logger.error(error_msg+error_info)
                        raise Exception(error_msg)
            except json.JSONDecodeError as e:
                error_info = traceback.format_exc()
                error_msg = f"Failed to parse working directory from agent response: {e}"
                logger.error(error_msg+error_info)
                current_workdir = user_context.home_dir or str(Path.home())
                logger.warning(f"Falling back to default directory: {current_workdir}")
            except Exception as e:
                error_info = traceback.format_exc()
                error_msg = f"Error getting working directory: {e}"
                logger.error(error_msg+error_info)
                current_workdir = user_context.home_dir or str(Path.home())
                logger.warning(f"Falling back to default directory: {current_workdir}")
        else: 
            current_workdir = state.working_directory
        
        if state.working_directory_stack is None:
            stack = [state.working_directory]
        else: 
            stack = state.working_directory_stack
        
        # Here put all the histories in the agent. Probably it is a good idea to only input the last message
        try:
            response = get_folder_name_agent.invoke(state.current_task_messages[-1])
            logger.debug("response from get_folder_name_agent:")
            logger.debug("\t"+response.content)
        except Exception as e:
            error_info = traceback.format_exc()
            error_msg = f"Failed to invoke get_folder_name_agent: {e}"
            logger.error(error_msg+error_info)
            raise Exception(error_msg)

        try:
            output = get_folder_name_tool.invoke({"messages":[response]})["messages"][-1].content
            logger.debug("response from get_folder_name_tool:")
            logger.debug("\t"+output)
            folder_name = json.loads(output)["folder_name"]
        except (json.JSONDecodeError, KeyError) as e:
            error_info = traceback.format_exc()
            error_msg = f"Failed to parse folder name from tool response: {e}"
            logger.error(error_msg+error_info)
            folder_name = "task"
            logger.warning(f"Using default folder name: {folder_name}")
        except Exception as e:
            error_info = traceback.format_exc()
            error_msg = f"Error getting folder name from tool: {e}"
            logger.error(error_msg+error_info)
            raise Exception(error_msg)

        # Create the folder 
        try:
            if os.path.exists(os.path.join(current_workdir,folder_name)):
                ii = 1
                while True:
                    if os.path.exists(os.path.join(current_workdir,folder_name+"_"+str(ii))):
                        ii += 1 
                    else:
                        break 
                folder_name = folder_name +"_"+ str(ii)

            new_current_workdir = os.path.join(current_workdir,folder_name)
            os.mkdir(new_current_workdir)
            logger.info(f"Created new working directory: {new_current_workdir}")
        except OSError as e:
            error_info = traceback.format_exc()
            error_msg = f"Failed to create folder {new_current_workdir}: {e}"
            logger.error(error_msg+error_info)
            raise Exception(error_msg)
        except Exception as e:
            error_info = traceback.format_exc()
            error_msg = f"Unexpected error creating folder: {e}"
            logger.error(error_msg+error_info)
            raise Exception(error_msg)

        stack.append(new_current_workdir)

        return {
            "working_directory": new_current_workdir,
            "working_directory_stack": stack,
            "current_task_messages": state.current_task_messages,
        }
    
    except Exception as e:
        error_info = traceback.format_exc()
        error_msg = f"Critical error in get_folder_name_node: {e}"
        logger.error(error_msg+error_info)
        from langchain_core.messages import AIMessage
        error_message = AIMessage(content=f"Error creating folder: {error_msg}")
        
        response = Analysis.error_analysis(error_message)
        error_analysis = error_message + '\n' + '\n' + response
        # from langgraph.config import get_stream_writer
        # writer = get_stream_writer()
        # writer(f"❌ {error_msg}")
        
        # Return error state with original values preserved
        return {
            "working_directory": state.working_directory or (user_context.home_dir or str(Path.home())),
            "working_directory_stack": state.working_directory_stack or [],
            "current_task_messages": state.current_task_messages,
            "messages": [error_message],
            "messages_to_user": [error_analysis],
        }

def get_result_file_node(state:FileManagerState):
    try:
        logger.info("Start getting result file")
        # message = "Start getting result file"

        logger.debug("Input state:"); 
        try:
            pretty_dict(state.model_dump(), logger)
        except Exception as e:
            logger.warning(f"Failed to dump state for logging: {e}")

        if state.result_files is None: 
            result_files = []
        else: 
            result_files = state.result_files

        try:
            get_result_file_prompt = SystemMessage(content=PROMPT_GET_RESULT_FILE)
            
            if not state.current_task_messages or len(state.current_task_messages) == 0:
                logger.warning("No current_task_messages available, using empty list")
                response = llm.invoke([get_result_file_prompt])
            else:
                response = llm.invoke([get_result_file_prompt] + state.current_task_messages[-1][-1:])
            
            logger.debug("response from get_result_file:")
            logger.debug(response.content)
        except Exception as e:
            error_info = traceback.format_exc()
            error_msg = f"Failed to invoke LLM for result file extraction: {e}"
            logger.error(error_msg+error_info)
            raise Exception(error_msg)

        try:
            sequence = [each.strip() for each in response.content.split('\n') if each.strip()]
            logger.debug("Parsed result files:")
            logger.debug(sequence)
        except Exception as e:
            error_info = traceback.format_exc()
            error_msg = f"Failed to parse result file response: {e}"
            logger.error(error_msg+error_info)
            sequence = []
            logger.warning("Using empty sequence due to parsing error")

        # Deal with the working directory and the stack at the end of the task
        try:
            if state.working_directory_stack is None or len(state.working_directory_stack) == 0:
                stack = []
                current_working_directory = user_context.home_dir or Path.home()
                logger.warning(f"Empty working_directory_stack, using current directory: {current_working_directory}")
            else:
                stack = state.working_directory_stack.copy()
                stack.pop()
                if len(stack) == 0:
                    current_working_directory = user_context.home_dir or Path.home()
                else:
                    current_working_directory = stack[-1]
            
            logger.info(f"Updated working directory to: {current_working_directory}")
        except Exception as e:
            error_info = traceback.format_exc()
            error_msg = f"Error managing working directory stack: {e}"
            logger.error(error_msg+error_info)
            stack = state.working_directory_stack or []
            current_working_directory = state.working_directory or user_context.home_dir or Path.home()
            logger.warning(f"Using fallback working directory: {current_working_directory}")


        return {
            "result_files": result_files + sequence,
            "working_directory": current_working_directory,
            "working_directory_stack": stack,
        }
    
    except Exception as e:
        error_info = traceback.format_exc()
        error_msg = f"Critical error in get_result_file_node: {e}"
        logger.error(error_msg+error_info)
        from langchain_core.messages import AIMessage
        error_message = AIMessage(content=f"Error getting result files: {error_msg}")

        # from langgraph.config import get_stream_writer
        # writer = get_stream_writer()
        # writer(f"❌ {error_msg}")
        response = Analysis.error_analysis(error_message)
        error_analysis = error_message + '\n' + '\n' + response        
        # Return error state with original values preserved
        return {
            "result_files": state.result_files or [],
            "working_directory": state.working_directory or Path.home(),
            "working_directory_stack": state.working_directory_stack or [],
            "messages": [error_message],
            "messages_to_user": [error_analysis],
        }

def get_current_result_files_prompt(result_files):
    try:
        if result_files is None or len(result_files) == 0:
            prompt = "No result file is found.\n"
        else:
            prompt = "Current available result files from previous tasks:\n"
            for each in result_files:
                if each:  # Skip empty strings
                    prompt += f"\t{each}\n"
        
        return SystemMessage(content=prompt)
    except Exception as e:
        error_info = traceback.format_exc()
        logger.error(f"Error in get_current_result_files_prompt: {e}"+error_info)
        return SystemMessage(content="No result file is found.\n")