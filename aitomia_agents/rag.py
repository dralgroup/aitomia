"""
Adaptive RAG for chat 
"""

from langchain_core.messages import SystemMessage, AIMessage, AnyMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END 
from langgraph.types import Command
import os
import traceback
import json
from typing import Annotated, List
from .states import AitomiaState 
from .logger import logger 
from .utils import create_agent, FileManager, load_knowledge_base 


#-------------------------------------------------
# Schema
#-------------------------------------------------
class RAGState(AitomiaState):
    rag_messages: list[AnyMessage] = []
    judge_stop_message: AnyMessage | None = None
    context: str|None = None
    updated_query: str|None = None,
    initial_query: str|None = None,
    new_query: str|None = None,
    retrieval: int = 0,

#-------------------------------------------------
# Prompts
#-------------------------------------------------
def prompt_rewrite_query():
    prompt = "You are an intelligent query optimization assistant. Here are the instructions:\n"
    prompt += "\tPlease analyze the user's current question and the previous conversation history to determine whether the current question relies on context. If it does, please reformulate the current question into a complete, independent question that includes all necessary contextual information.\n"
    prompt += "\tPlease analyze the user's current question and the previous conversation history to determine whether the current question contains multiple objects that need to be compared. Do not do comparison unless you are explicitly asked to. If it does, rewrite the original question into a more explicity query that is more suitable for retrieval in the knowledge database.\n"
    prompt += """\tPlease analyze the user's current question and the previous conversation history to determine whether the current question contains any specific referents of ambiguous pronouns, such as "they", "it", "them", "this", "that", etc. If it does, replace these pronouns with clear object names to generate a new, unambiguous quesion.\n"""
    prompt += "\tYou should always and only return the modified query. If the query does not need any modification, just give the original query.\n"
    return prompt 
PROMPT_REWRITE_QUERY = prompt_rewrite_query()

def prompt_check_database_need():
    prompt = "There is a database of documentation of the MLatom program, so it does not know any foudamental knowledge of basic concepts. It contains the knowledge of (U)AIQM method series and KREG model. It also contains tutorial how to do simulations, like geometry optimization, frequency calculation, molecular dynamics, etc., in MLatom. You need to check if there is a need to check the database according to the query below. If so, you should answer yes, otherwise, you should answer no. You should also give the reason why you answer yes or no in one or two sentences.\n"
    return prompt
PROMPT_CHECK_DATABASE_NEED = prompt_check_database_need()

PROMPT_CHAT = """You are an expert of quantum chemistry with solid chemical knowledge. If you are not sure about the answer because of the lack of knowledge, put it clear in your answer.\n"""

PROMPT_CHECK_ANSWER = """You need to check if the answer fully answers the initial query. Especially, if the database is provided, whether the database is enough to answer the query. If so, you should answer yes, otherwise, you should answer no. If the answer says not sure because of the inadequate database knowledge, it is also considered fully answered. You should also give the reason why you answer yes or no in one or two sentences. If the database does not provide any information, you should mention that the database is not enough and you need to rely on the knowledge of LLM directly.\n"""

PROMPT_GENERATE_NEW_QUERY = """You need to generate a new query to the database according to the remaining problem. The ultimate task is to answer the initial query. You only need to output the new query."""
#-------------------------------------------------
# Tool functions
#-------------------------------------------------
def judge_database(
    yes:bool = False,
):
    """  
    Check whether to check the database.
    
    Args:
        yes: Whether to check the database.
    """
    
    if yes: return "retrieve_database_node"
    else: return "llm_node"

def judge_stop(
    yes:bool = False,
):
    """  
    Check if the answer fully answers the query.
    
    Args:
        yes: Whether the answer fully answers the query.
    """
    
    if yes: return "summarize_node"
    else: return "generate_new_query_node"

#-------------------------------------------------
# Agent
#-------------------------------------------------
llm = create_agent()
knowledge_base = load_knowledge_base() 
tools = [judge_database]
judge_database_agent = create_agent(tools=tools,tool_kwargs={"tool_choice":"any"})
judge_database_tool = ToolNode(tools)
tools = [judge_stop]
judge_stop_agent = create_agent(tools=tools,tool_kwargs={"tool_choice":"any"})
judge_stop_tool = ToolNode(tools)

#-------------------------------------------------
# Graph
#-------------------------------------------------
def initialize_node(state:RAGState):
    logger.info("Initializing RAG")
    return {
        "rag_messages":[],
        "judge_stop_message":None,
        "context":None,
        "updated_query":None,
        "initial_query":None,
        "new_query":None,
        "retrieval":0,
    }
    
    
def rewrite_query_node(state:RAGState):
    logger.info("Rewriting query")
    
    updated_query = state.updated_query
    
    rewrite_query_prompt = SystemMessage(content=PROMPT_REWRITE_QUERY)
    retrieval = state.retrieval + 1
    if retrieval == 1:
        response = llm.invoke(state.messages+[rewrite_query_prompt])
        
        logger.debug("response from the LLM")
        logger.debug("\t"+response.content)
        return {
            "rag_messages": [AIMessage(content="Initial query:\n"+response.content)],
            "initial_query": response.content,
            "updated_query": response.content,
            "retrieval": retrieval,
        }
    else:
        response = llm.invoke([AIMessage(content=state.new_query),rewrite_query_prompt])
    
        logger.debug("response from the LLM")
        logger.debug("\t"+response.content)
        return {
            "updated_query": response.content,
            "retrieval":retrieval
        }
    
def check_database_need_node(state:RAGState):
    logger.info("Checking the need to get the database")
    
    check_database_need_prompt = SystemMessage(content=PROMPT_CHECK_DATABASE_NEED)
    judge_database_message = llm.invoke([check_database_need_prompt,AIMessage(content=state.updated_query)])
    
    logger.debug("response from the LLM")
    logger.debug("\t"+judge_database_message.content)
    
    response = judge_database_agent.invoke([judge_database_message])
    output = judge_database_tool.invoke({"messages":[response]})["messages"][-1].content
    logger.debug(output) 
    
    logger.info(f"Next node: {output}")
    return Command(
        goto = output,
    )

def retrieve_database_node(state:RAGState):
    logger.info("Retrieving database")
    context = ""
    
    # First get pure LLM answer
    # updated_query = AIMessage(content=state.updated_query)
    # response = llm.invoke([updated_query]+state.messages)
    
    # Second get database knowledge
    docs = knowledge_base.similarity_search(state.updated_query,k=10)
    context = "\n\n".join(doc.page_content for doc in docs)
    
    # context = response.content + "\n\n" + context
    
    return {"context":context}

def llm_node(state:RAGState):
    logger.info("LLM") 
    if state.context is not None:
        chat_prompt = PROMPT_CHAT + f"Answer the query according to the given context: \n{state.context}\nHistory of the Computational Process:{state.messages}.\n Answer the Query: {state.updated_query}"
    else:
        chat_prompt = PROMPT_CHAT + f"History of the Computational Process:{state.messages}. Answer the query: {state.updated_query}. "
    chat_prompt = SystemMessage(content=chat_prompt)
    response = llm.invoke([chat_prompt])
    logger.debug("response from the LLM")
    logger.debug("\t"+response.content)
    
    rag_messages = state.rag_messages
    rag_messages.append(AIMessage(content="Updated query:\n"+state.updated_query))
    rag_messages.append(AIMessage(content="Answer:\n"+response.content))
    
    return {
        "rag_messages": rag_messages,
    }
    
# Check if the LLM answers the query
def check_answer_node(state:RAGState):
    logger.info("Checking the LLM's answer")
    
    check_answer_prompt = SystemMessage(content=PROMPT_CHECK_ANSWER)
    
    judge_stop_message = llm.invoke(state.rag_messages+[check_answer_prompt])
    
    logger.debug("response from the LLM")
    logger.debug("\t"+judge_stop_message.content)
    
    response = judge_stop_agent.invoke([judge_stop_message])
    output = judge_stop_tool.invoke({"messages":[response]})["messages"][-1].content 
    logger.debug(output)
    
    if state.retrieval >= 4:
        logger.info("Number of iterations exceed 4")
        output = "summarize_node"
    logger.info(f"Next node: {output}")
    return Command(
        goto = output,
        update= {
            "judge_stop_message":judge_stop_message
        }
    )
    
def generate_new_query_node(state:RAGState):
    logger.info("Generating new query")
    generate_new_query_prompt = SystemMessage(content=PROMPT_GENERATE_NEW_QUERY)
    
    response = llm.invoke(state.rag_messages+[state.judge_stop_message,generate_new_query_prompt]) 
    logger.debug("response from the LLM")
    logger.debug("\t"+response.content)
    
    new_query = response.content 
    
    return {"new_query":new_query}

def summarize_node(state:RAGState):
    logger.info("Summarizing the answers")
    # if len(state.rag_messages) == 1:
    #     return {
    #         "messages": state.rag_messages,
    #         "messages_to_user": state.rag_messages,
    #     }
    # else:
    if state.retrieval > 1:
        summarize_prompt = f"You need to answer the initial query. Some messages are provided, you need to answer the initial query according to these messages.\n"
        summarize_prompt = SystemMessage(content=summarize_prompt)
        response = llm.invoke([summarize_prompt]+state.rag_messages)
        logger.debug("response from the LLM")
        logger.debug("\t"+response.content)
        return {
            "messages": [AIMessage(content=response.content)],
            "messages_to_user": [AIMessage(content=response.content)],
            "status":"completed"
        }
    else:
        message = "\n".join(state.rag_messages[-1].content.split("\n")[1:])
        return {
            "messages": [AIMessage(content=message)],
            "messages_to_user": [AIMessage(content=message)],
            "status":"completed"
        }
    
    
rag_builder = StateGraph(RAGState)

rag_builder.add_node("initialize_node",initialize_node)
rag_builder.add_node("rewrite_query_node",rewrite_query_node)
rag_builder.add_node("check_database_need_node",check_database_need_node)
rag_builder.add_node("retrieve_database_node",retrieve_database_node)
rag_builder.add_node("llm_node",llm_node)
rag_builder.add_node("check_answer_node",check_answer_node)
rag_builder.add_node("generate_new_query_node",generate_new_query_node)
rag_builder.add_node("summarize_node",summarize_node)

# rag_builder.add_node()
# rag_builder.add_node()
# rag_builder.add_node()


rag_builder.add_edge(START,"initialize_node")
rag_builder.add_edge("initialize_node","rewrite_query_node")
rag_builder.add_edge("rewrite_query_node","check_database_need_node")
# rag_builder.add_edge("check_database_need_node","retrieve_database_node")
rag_builder.add_edge("retrieve_database_node","llm_node")
rag_builder.add_edge("llm_node","check_answer_node")
rag_builder.add_edge("generate_new_query_node","rewrite_query_node")
rag_builder.add_edge("summarize_node",END)
# rag_builder.add_edge()
# rag_builder.add_edge()
# rag_builder.add_edge()

rag_graph = rag_builder.compile() 
# from .agent_template import BaseAgent
# rag_agent = BaseAgent(
#     name='rag_agent'
# )
    
    
    
    
