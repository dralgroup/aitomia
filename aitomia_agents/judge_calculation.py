
from .utils import create_agent
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END 
from .states import AitomiaState
from langgraph.types import Command
from .logger import logger 
from .prepare_molecule import prepare_molecule_builder
from .chat import chat_builder
from typing import List
from .agent_cards import agent_cards

llm = create_agent()

class JudgeCalculationState(AitomiaState):
    judge_calculation: str = None
    nocalc_task_type: str = None
    task_title: str = None
    prelim_result: str = None
    molecule_file_name: str = None
    # current_task_messages: List[str] = None


def generate_task_title_node(state: JudgeCalculationState):
    prompt = SystemMessage(content="Generate a concise and descriptive title in one sentence for the following task based on the user's messages:"+''.join([msg.content for msg in state.messages if isinstance(msg, HumanMessage)]) + "**ONLY RETURN THE TITLE WITHOUT ANY ADDITIONAL TEXT.**")
    title = llm.invoke(state.messages+[prompt]).content
    logger.info(f'Task title:{title}')
    return {'task_title':title}


def prompt_calc_judge():
    prompt = """
    You are a quantum chemistry planning agent. Your task is NOT to answer the user's question, but to decide whether the user intends to run a new quantum chemistry or ML-based molecular calculation.

    System goal:
    Prefer molecule-specific, quantitative calculations whenever the user mentions specific molecules, reactions, or properties that could be computed.

    Default behavior:
    Return "calculation" unless the user's request clearly matches one of the "no_calculation" conditions.

    Classification rules:

    Return "calculation" if:
    - The user expresses any intention to perform a computation.
    - Answering the request requires a new quantum chemistry or ML-based calculation (energy, geometry optimization, TS, frequency, PES, MD, benchmarking, etc.).
    - The user explicitly or implicitly wants physically meaningful molecular properties that require computation.
    - The user asks how to set up, run, or parameterize a calculation for a specific molecule, reaction, or target system.

    - A quantitative calculation would provide a more rigorous, predictive, or molecule-specific answer than qualitative reasoning.
    - The user asks about reaction difficulty, feasibility, rate, barrier, selectivity, or favorability for specific named reactants, even if a qualitative textbook answer exists.
    - The user compares specific molecules, reactions, pathways, or conformers in a way that could be resolved quantitatively.

    Return "no_calculation" only if:
    1. The user only wants to inspect, summarize, or process existing files, without running new calculations.
    2. The user is discussing or interpreting existing results without intending a new calculation.
    3. The user asks general conceptual, theoretical, or factual questions that do NOT reference any specific molecule, molecular system, reaction chemical formula, or target structure.
    4. A molecule-specific quantum chemistry or ML-based calculation would NOT improve the quality, rigor, or usefulness of the answer.
    5. The user only requests molecular structures without computational processing
    (e.g., "give me XYZ of methane"), unless explicitly asking for calculation.
    6. The user requests auxiliary or engineering tasks
    (plotting, visualization, data post-processing, script generation, file manipulation),
    where no new quantum chemistry calculation is needed.
    7. The user asks broad, high-level, or conceptual questions without specifying any concrete molecule, reaction, or target system. In this case, you should not assume or invent a molecular system, and should return "no_calculation".
    8. A new calculation would be required, but none of the available agents can support it
    (e.g., comparison to experimental data outside agent scope).

    Available agents and their capabilities:
    """
    # Add agent descriptions
    for agent_name, agent_card in agent_cards.items():
        if agent_name in ["sp_agent","uvvis_agent","geomopt_agent","ts_agent","freq_agent","ir_agent","raman_agent","reaction_agent"]:
            prompt += "\t" + agent_card.description_for_planner + "\n"

    prompt += """
    Output rules:
    - Only output two lines.
    - First line: "calculation" or "no_calculation"
    - Second line: A concise explanation.
    - Do NOT include any other text or extra reasoning.
    """
    return prompt

PROMPT_CALC_JUDGE = prompt_calc_judge()

def determin_prompt():
    prompt = """
    You are the final judge for whether to start a calculation.

    You are given:
    1. The model's preliminary judgment.
    2. The user's explicit confirmation or correction.
    3. The recent conversation context.

    Your task:
    Determine whether the user INTENDS to run an actual calculation, not merely agree with the assistant.

    Important:
    A simple "yes" or "ok" does NOT automatically mean the user wants to calculate.
    You must interpret the user's reply in context.

    Decision criteria:
    - Choose CALCULATE only if the user expresses intent to perform, run, start, or request a computation.
    - Choose NOT_CALCULATE if the user is only agreeing, acknowledging, chatting, or discussing concepts.

    Rules:
    - If the user's reply is a bare "yes"/"ok"/"sure", use the conversation context to infer intent.
    - If the user confirms a calculation-related understanding (e.g., "yes, go ahead and run it"), choose CALCULATE.
    - If the user confirms a non-calculation statement (e.g., "yes, that's right" about a conceptual summary), choose NOT_CALCULATE.

    First line give your judgement: CALCULATE or NOT_CALCULATE
    Seconde line explain the reason you make this judgement


    """
    return prompt

DETERMIN_PROMPT = determin_prompt()
    # Output STRICTLY one of the following tokens and nothing else:

    # CALCULATE
    # NOT_CALCULATE 


def natural_prompt():
    prompt = """
    You are a friendly assistant. 
    Rewrite the following AI judgment result into a conversational message that you would naturally say to a user. 
    The intention: CALCULATE means the user expresses intent to perform, run, start, or request a computation. NOT_CALCULATE means the user is only agreeing, acknowledging, chatting, or discussing concepts. 
    Keep the meaning exactly the same. 
    Ask the user to confirm or correct the interpretation.
    Important: when asking the user to confirm, focus on **whether the user wants to perform a calculation**, 
    not on confirming the steps, procedures, or workflow of the calculation. 
    Make it clear, polite, and easy to read. 
    Do not add or remove content, just rephrase naturally.
    """
    return prompt
NATURAL_PROMPT = natural_prompt()


def judge_calc(calculation: bool = False):
    """
    Determine whether the user intends to perform a calculation.

    Args:
        calculation: True if the user intends to perform or set up a calculation,
                            False otherwise.
    """
    if calculation:
        return "true"
    else:
        return "false"

tools = [judge_calc]
judge_calc_agent = create_agent(tools=tools, tool_kwargs={"tool_choice":"any"})
judge_calc_tool = ToolNode(tools)


def judge_calculation_node(state: JudgeCalculationState):
    prelim_prompt = SystemMessage(content="AI judge result:"+PROMPT_CALC_JUDGE)
    prelim_result = llm.invoke(state.messages + [prelim_prompt]).content
    logger.info(prelim_result)
    logger.info(f"prelim_result:{prelim_result}")
    return {'prelim_result':prelim_result}

def confirm_calculation_node(state: JudgeCalculationState):
    prelim_result = state.prelim_result
    natural_msg = SystemMessage(content=NATURAL_PROMPT)
    msg_need_natural = SystemMessage(
        content=f"""I think the user's intention is:
        {prelim_result}
        Need to ask user to confirm:
        - If this interpretation is correct, please confirm.
        - If it is incorrect, please correct me in one sentence.
        - Or clarify your intention in one sentence."""
        )


    msg_to_user_decision = llm.invoke([natural_msg]+[msg_need_natural]).content
    user_decision = interrupt({
        "messages_to_user": msg_to_user_decision
    })

    result1 = SystemMessage(
        content="Model preliminary judgment:\n" + prelim_result
    )
    result2 = SystemMessage(
        content="User confirmation:\n" + user_decision["messages"].content
    )
    final_prompt = SystemMessage(content=DETERMIN_PROMPT)

    final_judge_result = llm.invoke(
        state.messages + [result1, result2, final_prompt]
    ).content
    logger.info(final_judge_result)
    format_result = SystemMessage(content='The final decision by user is:'+final_judge_result)
    response = judge_calc_agent.invoke([format_result])
    output = judge_calc_tool.invoke({'messages':[response]})['messages'][-1].content
    if output=='true':
        return {'judge_calculation':"calculation"}
    else:
        return {'judge_calculation':"no_calculation"}
    

def decide_nocalc_task_type(
        chat: bool=False,
        prepare_molecule:bool=False,
):
    """
    Decide which task to perform. It is necessary that one and only one task is chosen to be True.

    Args:
        prepare_molecule: Whether to get the structure of the molecule only. it should be True ONLY if the user explicitly requests to generate, load, retrieve, or output a molecular structure or geometry(e.g., XYZ coordinates, SMILES, optimized structure).
        chat: Whether to chat with llm.
    """
    argvals = list(locals().values())
    if sum(argvals) != 1:
        logger.warning("More than one task or no task is choosen")

    if prepare_molecule: return "prepare_molecule_node"
    if chat: return "chat_node"


tools = [decide_nocalc_task_type]
nocalc_task_type_agent = create_agent(tools=tools,tool_kwargs={"tool_choice":"any"})
nocalc_task_type_tool = ToolNode(tools)
def nocalc_type_node(state: JudgeCalculationState):
    prompt = [SystemMessage("You must use the decide_task_type function to decide the task type.")]
    response = nocalc_task_type_agent.invoke(state.messages + prompt)  
    output = nocalc_task_type_tool.invoke({"messages":[response]})["messages"][-1].content 
    logger.debug(output)  
    task_type = output
    logger.info(f'Task type:{task_type}')
    return Command(
        goto = task_type,
        update = {
            "nocalc_task_type": task_type.replace("_node", ""),
            "current_task_messages":[state.messages] 
        }
    )

def wrap_status_for_single_prepare_molecule(state:JudgeCalculationState):
        return {
            "messages_to_user": [AIMessage(content=f'The molecule was successfully prepared in :<Path>{state.molecule_file_name}</Path>')],
            "status":"completed"
        }
prepare_molecule_graph = prepare_molecule_builder.compile()
chat_graph = chat_builder.compile()

judge_calculation_builder = StateGraph(JudgeCalculationState)
judge_calculation_builder.add_node('generate_task_title_node', generate_task_title_node)
judge_calculation_builder.add_node('judge_calculation_node', judge_calculation_node)
judge_calculation_builder.add_node('confirm_calculation_node', confirm_calculation_node)
judge_calculation_builder.add_node('nocalc_type_node', nocalc_type_node)
judge_calculation_builder.add_node("prepare_molecule_node",prepare_molecule_graph)
judge_calculation_builder.add_node('wrap_status_for_single_prepare_molecule',wrap_status_for_single_prepare_molecule )
judge_calculation_builder.add_node("chat_node",chat_graph)

judge_calculation_builder.add_edge(START,"generate_task_title_node")
judge_calculation_builder.add_edge("generate_task_title_node","judge_calculation_node")
judge_calculation_builder.add_edge("judge_calculation_node","confirm_calculation_node")
judge_calculation_builder.add_conditional_edges(
    "confirm_calculation_node",
    lambda state: state.judge_calculation,
    {
        "calculation":END,
        "no_calculation":"nocalc_type_node"
    }

)
judge_calculation_builder.add_edge("prepare_molecule_node",'wrap_status_for_single_prepare_molecule')
judge_calculation_builder.add_edge('wrap_status_for_single_prepare_molecule', END)
judge_calculation_builder.add_edge("chat_node",END)






