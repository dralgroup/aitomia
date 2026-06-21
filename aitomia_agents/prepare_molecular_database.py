"""
    Prepare database agent
"""
import json 
import os 
import shutil
import traceback 

from langchain_core.messages import SystemMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END 
from aitomia_agents.settings import settings
from aitomia_agents.user_context import user_context
from pathlib import Path
from langgraph.types import interrupt

from .states import AitomiaState
from .logger import logger 
from .utils import create_agent, pretty_dict, FileManager

from langgraph.config import get_stream_writer

#-------------------------------------------------
# Schema
#-------------------------------------------------
class PrepareMolecularDatabaseState(AitomiaState):
    molecular_database_file_name: str = None 

#-------------------------------------------------
# Prompts
#-------------------------------------------------
prepare_molecular_database_prompts = """
You need to retrieve all the molecular database information avaible from user and you have access to the files provided by users with tools. You should select the molecular database that is needed for the current task.
"""
prepare_molecular_database_prompts = SystemMessage(prepare_molecular_database_prompts)

#-------------------------------------------------
# Tool functions
#-------------------------------------------------
def get_molecular_database_from_file(filename:str=""):
    """
    This function is used to get the absolute path of the user provided molecular database file or molecular database in the result files.

    Args:
        filename: The absolute path of the molecular database file.
    """

    logger.debug("In get_molecular_database_from_file")
    return {"molecular_database_file_name":filename}

#-------------------------------------------------
# Agent
#-------------------------------------------------
tools = [get_molecular_database_from_file]
prepare_molecular_database_agent = create_agent(tools=tools,tool_kwargs={"tool_choice":"any"})
prepare_molecular_database_tool = ToolNode(tools)

#-------------------------------------------------
# Graph
#-------------------------------------------------
def prepare_molecular_database_node(state:PrepareMolecularDatabaseState):
    logger.info("Start preparing molecular database")
    message = "Start preparing molecular database"

    try:
        logger.debug("Input State:"); pretty_dict(state.model_dump(),logger)

        response = prepare_molecular_database_agent.invoke(state.current_task_messages[-1]+[prepare_molecular_database_prompts])
        function_name = response.tool_calls[0]['name']
        logger.debug("Response from the molecular database agent:")
        logger.debug("\t"+response.content)

        output = prepare_molecular_database_tool.invoke({"messages":[response]})["messages"][-1].content
        logger.debug("Response from the molecular database tool:")
        logger.debug("\t"+output)
        output = json.loads(output)

        source_path = output['molecular_database_file_name']
        
        # Validate that the path exists and is a file
        if not os.path.exists(source_path):
            error_message = f"Molecular database file path does not exist: {source_path}"
            logger.error(error_message)
            return {
                "error": error_message,
                "has_error": True,
                "messages": [AIMessage(content=error_message)],
                "messages_to_user": [AIMessage(content=error_message)],
            }
        
        if not os.path.isfile(source_path):
            error_message = f"Path is not a file: {source_path}. Please provide a direct file path."
            logger.error(error_message)
            return {
                "error": error_message,
                "has_error": True,
                "messages": [AIMessage(content=error_message)],
                "messages_to_user": [AIMessage(content=error_message)],
            }
        
        # Use shutil.copy2 instead of os.system for better error handling
        try:
            dest_path = os.path.join(state.working_directory, os.path.basename(source_path))
            shutil.copy2(source_path, dest_path)
            filename = dest_path
            logger.debug(f"Successfully copied molecular database file from {source_path} to {dest_path}")
        except Exception as e:
            error_message = f"Failed to copy molecular database file: {str(e)}"
            logger.error(error_message)
            return {
                "error": error_message,
                "has_error": True,
                "messages": [AIMessage(content=error_message)],
                "messages_to_user": [AIMessage(content=error_message)],
            }

        output_state = {
            "molecular_database_file_name": filename, 
            "messages_to_user":[AIMessage(content=message)],
            }
        
        logger.debug("Output state:"); pretty_dict(output_state, logger)
        return output_state
    except Exception as e:
        error_message = f"Error in prepare_molecular_database_node: {str(e)}"
        error_info = traceback.format_exc()
        logger.error(error_message + error_info)
        # writer = get_stream_writer()
        # writer(f"❌ {error_message}")
        return {
            "error": error_message,
            "molecular_database_file_name": "error",
            "messages": [AIMessage(content=f"Failed to prepare molecular database: {error_message}")],
            "messages_to_user": [AIMessage(content=f"Failed to prepare molecular database: {error_message}")],
        }


prepare_molecular_database_builder = StateGraph(PrepareMolecularDatabaseState)
prepare_molecular_database_builder.add_node("prepare_molecular_database_node",prepare_molecular_database_node)

prepare_molecular_database_builder.add_edge(START,"prepare_molecular_database_node")
prepare_molecular_database_builder.add_edge("prepare_molecular_database_node",END)

prepare_molecular_database_graph = prepare_molecular_database_builder.compile()