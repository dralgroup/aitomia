"""
    Label database agent
"""
import json 
import os 
import mlatom as ml 
from typing import Union
import traceback
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END 

from .method_judge import method_builder
from .prepare_molecular_database import prepare_molecular_database_builder
from .file_manager import get_folder_name_node, get_result_file_node

from .states import AitomiaState
from .logger import logger 
from .utils import create_agent, Analysis#, pretty_dict

from langgraph.config import get_stream_writer

#-------------------------------------------------
# Schema
#-------------------------------------------------
class LabelDatabaseState(AitomiaState):
    molecular_database_file_name: str = None
    method: str = None
    program: Union[str,None] = None 
    label_database_result: Union[str,None] = None 

#-------------------------------------------------
# Prompts
#-------------------------------------------------

#-------------------------------------------------
# Tool functions
#-------------------------------------------------

#-------------------------------------------------
# Agent
#-------------------------------------------------
llm = create_agent()

#-------------------------------------------------
# Graph
#-------------------------------------------------
def label_database_node(state:LabelDatabaseState):
    logger.info("Start labelling database")
    message = "Start labelling database"

    try:
        molecular_database_file_name = state.molecular_database_file_name 
        if '.xyz' in molecular_database_file_name:
            moldb = ml.data.molecular_database.from_xyz_file(molecular_database_file_name)
        elif '.json' in molecular_database_file_name:
            moldb = ml.data.molecular_database.load(molecular_database_file_name,format='json')
        else:
            raise ValueError("Unknown molecular database file format")
        
        method = ml.models.methods(method=state.method,program=state.program)
        method.predict(molecular_database=moldb,calculate_energy=True,calculate_energy_gradients=True)

        # Result file name 
        filename = "labeled_moldb.json"
        if os.path.exists(os.path.join(state.working_directory,filename)):
            ii = 1
            while True:
                if os.path.exists(os.path.join(state.working_directory,f"labeled_moldb_{ii}.json")):
                    ii += 1
                else:
                    break 
            filename = f"labeled_moldb_{ii}.json"
        result_file_name = os.path.join(state.working_directory,filename)
        moldb.dump(result_file_name,format='json')

        return {
            "label_database_result":result_file_name,
            "messages_to_user": [AIMessage(content=message)],
        }
    
    except Exception as e:
        error_message = f"Error in label_point_node"
        error_info = traceback.format_exc()
        logger.error(error_message + error_info)
        response = Analysis.error_analysis(error_message)
        error_analysis = error_message + '\n' + '\n' + response
        # writer = get_stream_writer()
        # writer(f"❌ {error_message}")
        return {
            "error": error_message,
            "has_error": True,
            "messages": [AIMessage(content=f"Labelling database failed: {error_message}")],
            "messages_to_user": [AIMessage(content=f"Labelling database failed: {error_analysis}")],
        }

def label_database_analysis_node(state:LabelDatabaseState):
    logger.info("Start analyzing the result of the database labelling")
    message = "Start analyzing the result of the database labelling"

    try:
        result_file_name = state.label_database_result
        result_message = f"Database labelling result: \n"
        result_message += f"    Labelled database: {result_file_name}"

        task_messages = state.current_task_messages
        task_messages[-1].append(AIMessage(content=result_message))

        return {
            "messages":[AIMessage(content=result_message)],
            "messages_to_user":[AIMessage(content=message + '\n' + result_message)],
            "current_task_messages":task_messages
        }
    except Exception as e:
        error_message = f"Error in label_database_analysis_node: {str(e)}"
        error_info = traceback.format_exc()
        logger.error(error_message + error_info)
        response = Analysis.error_analysis(error_message)
        error_analysis = error_message + '\n' + '\n' + response
        # writer = get_stream_writer()
        # writer(f"❌ {error_message}")
        return {
            "error": error_message,
            "has_error": True,
            "messages": [AIMessage(content=f"Failed to analyze database labelling results: {error_message}")],
            "messages_to_user": [AIMessage(content=f"Failed to analyze database labelling results: {error_analysis}")]
        }

#-------------------------------------------------
# Error checking function
#-------------------------------------------------
def check_error(state:LabelDatabaseState):
    """Check if an error has occurred and route accordingly"""
    if state.has_error:
        logger.warning("Error detected, skipping remaining steps")
        return END
    return "continue"


label_database_builder = StateGraph(LabelDatabaseState)

method_graph = method_builder.compile()
prepare_molecular_database_graph = prepare_molecular_database_builder.compile()

label_database_builder.add_node("get_folder_name_node",get_folder_name_node)
label_database_builder.add_node("method",method_graph)
label_database_builder.add_node("prepare_molecular_database",prepare_molecular_database_graph)
label_database_builder.add_node("label_database_node",label_database_node)
label_database_builder.add_node("label_database_analysis_node",label_database_analysis_node)

label_database_builder.add_edge(START,"get_folder_name_node")
label_database_builder.add_edge("get_folder_name_node","method")
label_database_builder.add_edge("method","prepare_molecular_database")
label_database_builder.add_edge("prepare_molecular_database","label_database_node")
label_database_builder.add_edge("label_database_node","label_database_analysis_node")
label_database_builder.add_edge("label_database_analysis_node",END)

label_database_graph = label_database_builder.compile()

from .agent_template import BaseAgent 
label_database_agent = BaseAgent(
    name='label_database_agent',
    description='Perform the single point calculations of molecular database.',
    graph=label_database_builder,
)