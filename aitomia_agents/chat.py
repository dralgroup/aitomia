"""
    Chat agent
"""

from langchain_core.messages import SystemMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END 
from langgraph.types import Command
import os
import traceback
import json
from .rag import rag_builder
from .states import AitomiaState 
from .logger import logger 
from .utils import create_agent, FileManager, load_knowledge_base, pretty_dict


llm = create_agent()
knowledge_base = load_knowledge_base()
#-------------------------------------------------
# Schema
#-------------------------------------------------
# Use AitomiaState

#-------------------------------------------------
# Prompts
#-------------------------------------------------
PROMPT_CHAT = """You are an expert of quantum chemistry with solid chemical knowledge.
"""
#-------------------------------------------------

class ChatState(AitomiaState):
    chat_next_node: str = None
#-------------------------------------------------
# Graph
#-------------------------------------------------
def judge_chat_context(
    add_file_content: bool=False,
    chat_directly: bool=False
):
    """
    Determine whether the user's question is related to a file.

    Args:
        add_file_content: Set to True if the user is asking a question about 
                          the currently opened file, or if the user provides 
                          a file path and asks a question about that file.
        chat_directly: Set to True if the user's question is unrelated to any file, 
                       i.e., a general chat question.
    """
    if add_file_content: return "add_file_content_node"
    if chat_directly: return "chat_node"
    return "chat_node"
#-------------------------------------------------
# Agent
#-------------------------------------------------
tools = [judge_chat_context]
judge_chat_context_agent = create_agent(tools=tools, tool_kwargs={'tool_choice':'judge_chat_context'})
judge_chat_tool = ToolNode(tools)
#-------------------------------------------------




def get_chat_file(chat_file:str=""):
    """
    This function determines the chat file based on user input.

    The chat file must be an absolute path. The resolution follows this order:
    1. First, check the user input.  
    - If the input contains a file path, use it directly.  
    2. If the user does not provide a path, then check the chosen path.
    - If the choose file is related to solve user's question, then use it.
    3. If neither the user input nor the chosen path is provided, use '' as chat_file.

    Args:
        chat_file: The absolute path of the working directory. Do not return a file path.

    """
    try:
        if chat_file != '':
            if not os.path.isabs(chat_file):
                logger.warning(f"chat_file '{chat_file}' is not absolute, converting to absolute path")
                chat_file = os.path.abspath(chat_file)
            return locals()
        else:
            return {'chat_file':''}
        
    except Exception as e:
        error_info = traceback.format_exc()
        logger.error(f"Error in get_working_directory: {e}" + error_info)


tools = [get_chat_file]
get_chat_file_agent = create_agent(tools=tools, tool_kwargs={"tool_choice":"any"})
get_chat_file_tool = ToolNode(tools)




#----------------------------
def judge_chat_context_node(state:ChatState): 
    logger.info('Start judging chat context')
    try:

        open_file_msg = AIMessage(content=f"The opened file is:{FileManager.read_path()}")

        response = judge_chat_context_agent.invoke(state.messages+[open_file_msg])
        output = judge_chat_tool.invoke({'messages':[response]})['messages'][-1].content
        if output not in {"chat_node", "add_file_content_node"}:
            logger.warning(f"Invalid route {output}, fallback to chat_node")
            output = "chat_node"
            
        logger.debug(output)
        return {
            "chat_next_node": output 
        }
    
    except Exception as e:
        import traceback
        error_message = f"Error in chat_node: {str(e)}"
        error_info = traceback.format_exc()
        logger.error(error_message + error_info)

        return {
            "messages": [AIMessage(content=f"Chat failed: {error_message}")],
            "messages_to_user": [AIMessage(content=f"Chat failed: {error_message}")],
            "error": error_message,
            "has_error": True
        }
    

def add_file_content_node(state:ChatState):
    logger.info('add file content to the context for answering user question')
    try:
        open_file_msg = AIMessage(content=f"The opened file is:{FileManager.read_path()}")
        response = get_chat_file_agent.invoke(state.messages+[open_file_msg])
        output = get_chat_file_tool.invoke({"messages":[response]})["messages"][-1].content
        output = json.loads(output)
        chat_file_content = output['chat_file']
        with open(chat_file_content, 'r') as f:
            file_content = f.read()
        return {
            "messages":[SystemMessage(content=f"The file content the user is asking about is: {file_content}")]
        }

    except Exception as e:
        import traceback
        error_message = f"Error in chat_node: {str(e)}"
        error_info = traceback.format_exc()
        logger.error(error_message + error_info)

        # from langgraph.config import get_stream_writer
        # writer = get_stream_writer()
        # writer(f"❌ {error_message}")

        return {
            "messages": [AIMessage(content=f"Chat failed: {error_message}")],
            "messages_to_user": [AIMessage(content=f"Chat failed: {error_message}")],
            "error": error_message,
            "has_error": True
        }
    



# def chat_node(state:ChatState):
#     logger.info("Start chatting")
#     message = "Start chatting"

#     try:
#         # chat_prompt = SystemMessage(content=PROMPT_CHAT)
        
#         docs = knowledge_base.similarity_search(state.messages[-1].content,k=10)
#         context = "\n\n".join([doc.page_content for doc in docs])
#         chat_prompt = PROMPT_CHAT + f"\nYou need to chat with the user given the above history, according to the context below: {context}\n"
        
#         # It should take in all the history
#         response = llm.invoke(state.messages+[chat_prompt])

#         logger.debug("response from the LLM")
#         logger.debug("\t"+response.content)
#         output_state = {
#             "messages": [AIMessage(content=response.content)],
#             "messages_to_user": [AIMessage(content=message + '\n' + response.content)],
#             "status":'completed'
#         }

#         # from langgraph.config import get_stream_writer
#         # writer = get_stream_writer()
#         # writer(response.content + '\n' + "Answer completed")        
#         return output_state
    
#     except Exception as e:
#         import traceback
#         error_message = f"Error in chat_node: {str(e)}"
#         error_info = traceback.format_exc()
#         logger.error(error_message + error_info)

#         # from langgraph.config import get_stream_writer
#         # writer = get_stream_writer()
#         # writer(f"❌ {error_message}")

#         return {
#             "messages": [AIMessage(content=f"Chat failed: {error_message}")],
#             "messages_to_user": [AIMessage(content=f"Chat failed: {error_message}")],
#             "error": error_message,
#             "has_error": True
#         }

rag_graph = rag_builder.compile()
chat_builder = StateGraph(AitomiaState)
chat_builder.add_node("judge_chat_context_node", judge_chat_context_node)
chat_builder.add_node("add_file_content_node", add_file_content_node)
chat_builder.add_node("chat_node",rag_graph)
chat_builder.add_node("judge_chat_context", judge_chat_context)

chat_builder.add_edge(START,"judge_chat_context_node")
chat_builder.add_conditional_edges(
    "judge_chat_context_node",
    lambda state: state.chat_next_node,
    {
        "add_file_content_node": "add_file_content_node",
        "chat_node": "chat_node",
    }, 
)
chat_builder.add_edge("add_file_content_node", "chat_node")
chat_builder.add_edge("chat_node",END)

chat_graph = chat_builder.compile()

from .agent_template import BaseAgent 
chat_agent = BaseAgent(
    name="chat_agent",
    description="Chat with the user.",
    graph=chat_builder,
)


