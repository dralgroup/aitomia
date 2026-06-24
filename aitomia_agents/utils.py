'''
    utils.py
'''

from langchain_openai import ChatOpenAI 
from langchain_core.callbacks import BaseCallbackHandler 
from langchain_core.messages import SystemMessage, HumanMessage
import numpy as np
import os
from typing import Optional
from urllib.parse import urlparse
from langchain_community.embeddings import DashScopeEmbeddings 
from langchain_community.vectorstores import FAISS
from .agent_cards import agent_cards
from .settings import settings
from .logger import logger
from .user_context import user_context

class CustomCallbackHandler(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs):
        logger.info(f"LLM ended with response: {response}")

class TokenUsageCallback(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs):
        llm_output = getattr(response, "llm_output", None) or {}
        token_usage = llm_output.get("token_usage") or {}
        if not token_usage:
            return

        model_name = llm_output.get("model_name") or llm_output.get("model")
        model_suffix = f" model={model_name}" if model_name else ""
        prompt_tokens = token_usage.get("prompt_tokens")
        completion_tokens = token_usage.get("completion_tokens")
        total_tokens = token_usage.get("total_tokens")
        logger.info(
            "Token usage%s prompt=%s completion=%s total=%s user=%s",
            model_suffix,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            user_context.email
        )

        dsn = settings.statistic_database_dsn
        if dsn:
            _write_token_usage_to_db(
                dsn=dsn,
                user_email=user_context.email,
                input_token=prompt_tokens,
                output_token=completion_tokens,
                usage_type=model_name or "unknown",
            )


def _write_token_usage_to_db(
    dsn: str,
    user_email: Optional[str],
    input_token: Optional[int],
    output_token: Optional[int],
    usage_type: str,
) -> None:
    input_value = int(input_token) if input_token is not None else 0
    output_value = int(output_token) if output_token is not None else 0

    try:
        parsed = urlparse(dsn)
        scheme = parsed.scheme.lower()
        if scheme.startswith("mysql") or scheme.startswith("mariadb"):
            conn = _connect_mysql(dsn, parsed)
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO token_usage (user_email, input_token, output_token, type, created_at)
                        VALUES (%s, %s, %s, %s, NOW())
                        """,
                        (user_email, input_value, output_value, usage_type),
                    )
                conn.commit()
            finally:
                conn.close()
        else:
            conn = _connect_postgres(dsn)
            conn.autocommit = True
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO token_usage (user_email, input_token, output_token, type, created_at)
                        VALUES (%s, %s, %s, %s, NOW())
                        """,
                        (user_email, input_value, output_value, usage_type),
                    )
            finally:
                conn.close()
    except Exception as exc:
        logger.warning("Token usage DB logging failed: %s", exc)


def _connect_postgres(dsn: str):
    try:
        import psycopg2
    except Exception as exc:
        raise RuntimeError("psycopg2 unavailable") from exc

    return psycopg2.connect(dsn)


def _connect_mysql(dsn: str, parsed):
    user = parsed.username or ""
    password = parsed.password or ""
    host = parsed.hostname or "localhost"
    port = parsed.port or 3306
    db_name = (parsed.path or "").lstrip("/") or None

    try:
        import pymysql
    except Exception:
        pymysql = None

    if pymysql is not None:
        return pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=db_name,
            port=port,
            charset="utf8mb4",
        )

    try:
        import mysql.connector
    except Exception as exc:
        raise RuntimeError("MySQL driver unavailable") from exc

    return mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=db_name,
        port=port,
    )


# Create an agent with or without tools
def create_agent(model_kwargs={'temperature':0.0},tools:list=None,tool_kwargs:dict={}, callbacks=None):
    if callbacks is None:
        callbacks = [TokenUsageCallback()]

    agent =  ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base,
        callbacks=callbacks,
        **model_kwargs
    )
    if tools is not None:
        agent_with_tools = agent.bind_tools(tools, **tool_kwargs)
        return agent_with_tools
    else:
        return agent
    
def create_embeddings():
    embeddings = DashScopeEmbeddings(model="text-embedding-v3", dashscope_api_key=settings.openai_api_key)
    return embeddings

def load_knowledge_base():
    # The pre-built FAISS knowledge base is optional and is not shipped with the
    # open-source release. When it is absent, return None so that importing the
    # RAG/chat modules does not fail; the literature-RAG path is simply disabled
    # and the assistant answers from the LLM's own knowledge instead.
    file_directory = os.path.dirname(os.path.abspath(__file__))
    knowledge_dir = os.path.join(file_directory, "rag_database/knowledge")
    if not os.path.exists(os.path.join(knowledge_dir, "index.faiss")):
        logger.warning(
            "RAG knowledge base not found at %s; literature retrieval is disabled.",
            knowledge_dir,
        )
        return None
    embeddings = create_embeddings()
    knowledgeBase = FAISS.load_local(knowledge_dir, embeddings, allow_dangerous_deserialization=True)
    return knowledgeBase
    
def pretty_dict(d, logger):
    for kk, vv in d.items():
        logger.debug(f"\t{kk}: {vv}")

def pretty_array(array):
    array = np.array(array)
    if array.ndim == 0:
        string = "{:-11.5f}".format(array)
    elif array.ndim == 1:
        string = " ".join(["{:-11.5f}".format(each) for each in array])
    elif array.ndim == 2:
        string = ""
        for ii,each1 in enumerate(array):
            string += " ".join(["{:-11.5f}".format(each) for each in each1])
            if ii < len(array)-1: string += "\n"
    else:
        string = str(array)
    return string

        
def graph_vis(graph, tool:str="mermaid", tool_kwargs:dict={}):
    """Visialization of the langgraph
    
    Args:
        graph: the graph to be visualized
        tool: str
            'ascii': print the graph with ascii code
            'mermaid': return the url that can visualize the graph with mermaid
    """
    import base64, sys
    assert tool.lower() in ['mermaid', 'ascii'], 'tool must be either mermaid or ascii'

    if tool.lower() == 'mermaid':
        mermaid_syntax = graph.get_graph().draw_mermaid()
        mermaid_syntax_encoded = base64.b64encode(mermaid_syntax.encode("utf8")).decode("ascii")
        background_color=tool_kwargs.get('background_color', 'white')
        url = f"https://mermaid.ink/img/{mermaid_syntax_encoded}&bgColor=!{background_color}"
        print(f'Graph visualization url: \n{url}\n'); sys.stdout.flush()
        return url 
    
    if tool.lower() == 'ascii':
        graph_ascii = graph.get_graph().draw_ascii()
        print(f"Graph visualization with ascii\n{graph_ascii}")
        sys.stdout.flush()
        return graph_ascii
    

class FileManager:
    '''A simple file manager to read and write files'''
    _path = None
    @classmethod
    def set_path(cls, path):
        cls._path = path
    @classmethod
    def read_path(cls):
        return cls._path
    @classmethod
    def get_init_sturcture(cls):
        return []
        # import os
        # if cls._path != None:
        #     if os.path.isdir(cls._path):
        #         file_list = [
        #             os.path.join(cls._path, f) 
        #             for f in os.listdir(cls._path)
        #             if f.endswith(".json") or f.endswith(".xyz")
        #         ]
        #         return file_list
        # else:
        #     return []





    


class Analysis:
    '''
    Analysis for special need(error analysis, conbination analysis)
    '''
    _llm = None
    # def __init__(self):
    #     self.llm = create_agent()
    _global_summary = {'tasks':[]} #reset以及启动的时候清空global_analysis
    _user_query = []
    _time_consuming = {'start_time':0, 'end_time':''}
    
    @classmethod
    def initialize_analysis(cls):
        cls._llm = None
        cls._global_summary = {'tasks':[]} 
        cls._user_query = []
        cls._time_consuming = {'start_time':0, 'end_time':''}        

    @classmethod
    def set_start_time(cls, start_time):
        cls._time_consuming['start_time']=start_time
    @classmethod
    def get_start_time(cls):
        return cls._time_consuming['start_time']
    @classmethod
    def set_end_time(cls, end_time):
        cls._time_consuming['end_time']=end_time

    @classmethod
    def get_time_consuming(cls):
        start_time = cls._time_consuming['start_time']
        end_time = cls._time_consuming['end_time']
        time_consuming = end_time-start_time
        return time_consuming
    

    @classmethod
    def _get_llm(cls):
        if cls._llm is None:
            cls._llm = create_agent()
        return cls._llm
    @classmethod
    def error_analysis(cls, error_msg: str):
        PROMPT_ERROR = (
            """You are an expert in computational chemistry workflows and scientific
            computing environments.

            Given an error message or traceback from a computational chemistry workflow,
            provide a concise and high-level analysis of the most likely cause(s)
            of the problem.

            Your response should:
            - Identify whether the issue is more likely due to
            (a) the scientific calculation (e.g. convergence failure, invalid geometry,
                numerical instability, incompatible methods), or
            (b) the computational environment (e.g. missing files, permission issues,
                resource limits, software configuration).
            - Focus on likely physical, numerical, algorithmic, or system-level reasons,
            not line-by-line code debugging.
            - Avoid repeating the traceback.
            - Keep the explanation brief (1-3 sentences).
            - Use cautious language such as "likely", "may indicate", or "could be caused by".
            - Do not hallucinate specific results or claim certainty.

            The goal is to diagnose common failure modes in computational workflows,
            rather than to debug Python syntax or implementation details.
            """
        )

        llm = cls._get_llm()

        system_prompt = SystemMessage(content=PROMPT_ERROR)
        error_message = HumanMessage(
            content=f"The error message is: {error_msg}"
        )

        response = llm.invoke([system_prompt, error_message])
        return response.content


    #analysis
    @classmethod
    def add_summary(cls,msg):
        cls._global_summary['tasks'].append(msg)

    @classmethod
    def add_user_query(cls, user_inp):
        cls._user_query.append(user_inp)

    @classmethod
    def get_summary(cls):
        return cls._global_summary
    
    @classmethod
    def get_user_query(cls):
        return cls._user_query

    @classmethod
    def clear_summary_and_query(cls):
        cls._global_summary = {'tasks':[]}
        cls._user_query = []

    @classmethod
    def result_analysis(cls):
        
        def prompt_calc_judge():
            prompt = """
            You are an expert in computational chemistry.
            Note: This is a strict computational task. All calculations must be done with full numerical precision, and all reported results must exactly reflect the computed values. Do not perform any rounding or truncation of decimal points.

            You are given:

            1. The user's query as a list of messages (user intentions and instructions).
            2. The computational results from the workflow, including task details and task summaries.

            Your task is to provide an accurate, high-level, and chemically informed answer to the user query,
            based strictly on the computational results provided.

            Guidelines:

            - Identify and explain the key results from each task, such as:
                - Optimized geometry
                - Energies
                - Gradients
                - Dipole moments
            - Provide constructive insights based on chemical knowledge, for example:
                - Commenting on energy values and their reasonableness
                - Noting features of the optimized structure
                - Highlighting any notable trends or results
            - Always base your statements on the computation output; do not make assumptions or invent results.
            - Keep your explanation structured, readable, and informative.
            - If relevant, clearly reference which task corresponds to which result
            (e.g., "Geometry optimization result", "Single point calculation result").

            The goal is to give the user a **chemically meaningful interpretation** of their requested calculations,
            providing insight that a knowledgeable chemist would find useful, while remaining strictly faithful to the computed data.

            Time analysis (optional but allowed):
            - You may include a brief, high-level comment on the total time consumption.
            - Keep the time analysis qualitative and non-speculative.
            - Do not attribute performance to specific hardware, parallelization, or software internals unless explicitly supported by the input.
            - Use cautious language such as "may reflect", "is consistent with", or "is reasonable for".

            If necessary and the agents are available, further computations can be suggested based on the available agents and their capabilities. However, all additional tasks should remain within the current capabilities of the system, do not suggest solvent effects calculation. 
            Here is an overview of the available agents and their capabilities:
            """
            # Add agent descriptions
            for agent_name, agent_card in agent_cards.items():
                if agent_name in ["sp_agent","uvvis_agent","geomopt_agent","ts_agent","freq_agent","ir_agent","raman_agent","reaction_agent"]:
                    prompt += "\t" + agent_card.description_for_planner + "\n"
            prompt += '''
            If further computations are unrelated to the user's query or if the available agents are unable to handle them, no suggestions for additional computations will be made. In such cases, only an analysis of the calculation results is enough
            '''            
            return prompt

        PROMPT_FINAL_RESULT = prompt_calc_judge()

        llm = cls._get_llm()
        user_query = cls.get_user_query()
        time_consuming = cls.get_time_consuming()
        result_msg = cls.get_summary()


        system_prompt = SystemMessage(content=PROMPT_FINAL_RESULT)
        result_message = HumanMessage(
            content=f"The User query is: {user_query}, The total time consuming(agent thinking time + claculation time) is:{time_consuming}, The computational results is:{result_msg}"
        )

        response = llm.invoke([system_prompt, result_message])
        return response.content
    

class GlobalFlag:
    '''
    A simple global state manager
    '''
    _flag = {}

    @classmethod
    def set_flag(cls, key, value):
        cls._flag[key] = value
    @classmethod
    def get_flag(cls, key, default=None):
        return cls._flag.get(key, default)
    @classmethod
    def clear_flag(cls):
        cls._flag = {}