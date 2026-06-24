
from .utils import create_agent
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END 
from .states import AitomiaState
from .utils import FileManager
from .logger import logger
from pathlib import Path
import os

llm = create_agent()

class MethodConfirmState(AitomiaState):
    method_judge_result: str = None
    method_get_result: str = None
    method_confirm: str = None

def prompt_suggestion():
    prompt = """
    You are a professional computational chemistry assistant.

    Task:
    - Read the user's latest calculation request and the provided molecule list (if any).
    - If the molecule_list is empty, rely solely on the user_request.
    - Assess the system composition and size, then recommend a suitable computational method.
    - For the chosen method, separately recommend:
    1) a **method program** (the program used to implement the method, e.g., PySCF for many QM methods), and
    2) a **calculation program** (the program or tool used to perform the specific task such as geometry optimization or frequency calculation, e.g., Geometric, Gaussian, ASE, SciPy).
    - If the chosen method is ML-based (AIQM1, AIQM2, ANI-1ccx, UAIQM, etc.), state that no specific method program is required (or list typical ML toolkits if relevant), but still provide a recommended calculation program when applicable.

    Program selection notes:

    2. **Method program**:
    - If a program was specified for the method, include it.
    - If no program was specified, state: "Method program is not specified."
    - For ML-based methods (e.g., AIQM1, AIQM2, ANI-1ccx, UAIQM), indicate that no specific method program is required, but optionally list typical ML toolkits if relevant.
    - For quantum mechanical methods, if the program is not specified, use PySCF by default.

    3. **Calculation program**:
    - Only choose from: PySCF, Gaussian, Geometric, ASE.
    - If the user asks to do geometry optimization, do not plan for frequency calculations. If the user asks to do frequency calculations, do not plan to do geometry optimizations.
    - If the user requests both geometry optimization and UV-vis spectrum calculation, use default calculation program for optimization (unless otherwise specified) but DO NOT chooes calculation program for UV-vis calculation.
    - For each task (e.g., geometry optimization, frequency analysis), provide the recommended program.
        The types of tasks you define here will directly influence the downstream workflow design. Therefore, when determining which tasks require a recommended program, you must also follow the same task-design principles:
            "The task should be user-oriented. You may add additional tasks only when they genuinely help answer the user's underlying question or achieve the user's explicit goal, but never simply because they are commonly paired steps in computational chemistry.\nDo not follow typical workflows rigidly. Instead, infer whether an additional calculation is logically useful for the user's intent. For example, a frequency calculation may be added after a geometry optimization only if it is necessary to confirm stability or relevant to what the user is trying to obtain. IR, Raman, IRC, or any other extra tasks should only appear when they meaningfully contribute to the user's purpose.\nAlways prioritize the user's intent, and ensure that any additional tasks remain tightly aligned with the user's question rather than standard procedural habits.\n"

    - For single-point energy calculations, no calculation program is required, as the method program directly performs the task. Do not specify a calculation program in this case; just mark as 'Not applicable'.
    - Never recommend xtb as calculation program unless, you suggest GFN2-xTB as method. 
    


    Guidelines for output:
    1. Start with a concise **Conclusion** line that clearly contains three labeled items:
    - **Method**: the recommended computational method (e.g., AIQM2, B3LYP/6-31G).
    - **Method program**: the recommended program to implement the method (or “none”/“not required” if ML method).
    - **Calculation program (for the requested task)**: the program recommended for the specific calculation (e.g., Geometric for geometry optimization).
    Example (informative, not rigid):  
    `Conclusion — Method: AIQM2; Method program: none required; Calculation program (geometry optimization): Geometric.`

    2. On the next line, provide a short, natural **Reason** (1-3 sentences) explaining why these choices fit the system and the user's goals (mention system size, element types, accuracy vs. cost trade-off, or time constraints if provided).

    3. Optionally, include a one-line **Alternatives** field listing 1-2 viable alternatives and their trade-offs.

    Decision rules:
    - Ground-state calculations:
    - Molecules containing only C, H, O, N → favor ML/QM hybrid methods, specifically AIQM2.
    - Molecules containing other elements (heavy atoms, halogens beyond F, transition metals, I, Br, etc.) → favor UAIQM as a general-purpose method for ground states.
    - Excited-state calculations and Molecules containing only C, H, O, N → favor AIQM1.
    - High-accuracy emphasis:
    - If the user explicitly prioritizes accuracy over speed, recommend other high-level QM methods (e.g., DFT, CC, or other suitable QM approaches).
    - Alternatives:
    - Suggest other reasonable DFT or QM methods.
    - When selecting a DFT/QM method, apply chemical knowledge for heavy elements:
        - Use basis sets that properly support the element (e.g., def2-SVP, def2-TZVP, LANL2DZ/SDD with effective core potentials).
        - Avoid light-element-only basis sets (e.g., 6-31G(d)) for heavy atoms.
    - Time constraints:
    - If strict time limits are mentioned, bias toward faster methods (ML or lower-cost QM approaches).

    Style constraints:
    - Be concise and helpful. Do not rigidly follow a fixed sentence template — the Conclusion should be short and clear, the Reason should read like expert advice.
    - Always produce both **Method program** and **Calculation program** entries (use “none required” where applicable for the method program).
    - If some required information is missing (e.g., user did not provide molecule list or task type), ask a single clarifying question rather than guessing.

    If anything above is unclear, ask one short clarifying question before making a final recommendation.
    """
    return prompt
PROMPT_SUGGESTION = prompt_suggestion()



def uv_prompt_conclusion():
    prompt = """
    You are a professional quantum chemistry expert. Based on the previous discussion, provide a **final summary** of the computational choices. Your response should include:

    1. **Method**: 
    - Clearly state the selected computational method.
        ***Step-Method Binding Schema (STRICT)***

        All computational methods MUST be bound to specific workflow steps.
        Methods are NEVER global by default.

        Workflow steps include (but are not limited to):
        - geometry optimization (geomopt)
        - spectra calculation (e.g., IR, Raman, or UV-vis)
        - single-point energy
        - gradients / forces

        --------------------------------
        Step-method binding rules
        --------------------------------
        1. A method is considered used for a step ONLY if it is explicitly assigned to that step.
        2. If a method is assigned to one step, it MUST NOT be inferred to apply to any other step.
        3. The presence of a method name alone does NOT imply it is used for all steps.

        --------------------------------
        Explicit assignment patterns (authoritative)
        --------------------------------
        Treat the following as explicit step-method bindings:
        - `Use X for geomopt`
        - `X for geometry optimization`
        - `Use Y for spectra / UV-vis calculations`
        - `on X-optimized geometry`
        - `geometry optimized at X`

        These bindings override any default assumptions.

        --------------------------------
        Pre-optimized geometry references
        --------------------------------
        If the user specifies that a calculation is performed
        "on X-optimized geometry" or "using geometry optimized at X":

        - Treat this as an explicit binding of:
            geomopt → X
        - Do NOT interpret this as a request to perform geometry optimization again.
        - Do NOT consider any other method mentioned in the workflow as applying to geomopt.

        In this case:
        - Geometry optimization is considered already resolved.
        - OMNI-P2x geomopt warning rules MUST NOT be triggered
        unless OMNI-P2x is explicitly assigned to geomopt.

        --------------------------------
        Unassigned steps
        --------------------------------
        If a workflow step has no explicitly assigned method:
        - Apply auto-selection rules for that step only.
        - Do NOT consider methods assigned to other steps.

        --------------------------------
        Prohibited inference
        --------------------------------
        Never infer that a method is used for geometry optimization, spectra,
        single-point, or gradients unless explicitly bound to that step.

        ***Single-Method Fallback Rule for UV-vis Tasks***
            If a UV-vis spectrum calculation is requested and the user provides exactly one computational method (other than OMNI-P2x), and does not explicitly assign methods to individual steps, then treat this method as applying to BOTH:
                - geometry optimization
                - spectra calculation
            This fallback rule does NOT apply if OMNI-P2x is the only method mentioned.
            This fallback rule is overridden by any explicit step-method binding.

        ***Step-Scoped Warning Rules for OMNI-P2x***
            1. Geometry optimization (geomopt):
            - OMNI-P2x MUST NOT be used.
            - If OMNI-P2x is assigned to geomopt or geomopt has no method specified while OMNI-P2x appears in the workflow:
                → ***YOU MUST*** show the blocking warning and require an alternative method at the top head in bold and underlined: **<u>Warning: The OMNI-P2x method is not suitable for geometry optimization as it cannot predict accurate forces. Please choose a different method for optimization.</u>**
                → Recommendation for optimization method:
                    - If the molecule contains only C, H, O, N → suggest **AIQM2** (a QM/ML hybrid) for small-to-medium organic systems, unless the user explicitly requires high-level QM.
                    - If other elements are present → suggest a reliable DFT method (e.g., **B3LYP/6-31G**), unless time constraints or user preferences dictate otherwise.
                    - If the user mentions strict time limits → suggest faster ML methods or lower-cost DFT functionals/basis sets.
            - Important: If the user does not choose a method after your suggestion, you must use the method you suggested. Never provide an empty string or unreasonable method.
            - Format: 
                - First, display the warning in a new line (bold and underlined).
                - Then, in a new line, show the suggested method and the reason for the suggestion.

            2. Spectra calculation (e.g., UV-vis):
            - OMNI-P2x IS allowed.
            - If OMNI-P2x is explicitly assigned to spectra:
                → Show a non-blocking warning informing the user of method limitations and provide recommended validation or alternative methods.

        - Note: Be careful to distinguish between `OM2` (semi-empirical) and `OMNI-P2x` (machine learning potential). Do not confuse these two methods.

    2. **Method program**:
    - If a program was specified for the method, include it.
    - If no program was specified, state: "Method program is not specified."
    - For ML-based methods (e.g., AIQM1, AIQM2, ANI-1ccx, UAIQM), indicate that no specific method program is required, but optionally list typical ML toolkits if relevant.
    - For quantum mechanical methods, if the program is not specified, use PySCF by default.

    3. **Calculation program**:
    - Only choose from: PySCF, Gaussian, Geometric, ASE.
    - If the user asks to do geometry optimization, do not plan for frequency calculations. If the user asks to do frequency calculations, do not plan to do geometry optimizations.
    - If the user requests both geometry optimization and UV-vis spectrum calculation, use default calculation program for optimization (unless otherwise specified) but DO NOT chooes calculation program for UV-vis calculation.
    - For each task (e.g., geometry optimization, frequency analysis), provide the recommended program.
        The types of tasks you define here will directly influence the downstream workflow design. Therefore, when determining which tasks require a recommended program, you must also follow the same task-design principles:
            "The task should be user-oriented. You may add additional tasks only when they genuinely help answer the user's underlying question or achieve the user's explicit goal, but never simply because they are commonly paired steps in computational chemistry.\nDo not follow typical workflows rigidly. Instead, infer whether an additional calculation is logically useful for the user's intent. For example, a frequency calculation may be added after a geometry optimization only if it is necessary to confirm stability or relevant to what the user is trying to obtain. IR, Raman, IRC, or any other extra tasks should only appear when they meaningfully contribute to the user's purpose.\nAlways prioritize the user's intent, and ensure that any additional tasks remain tightly aligned with the user's question rather than standard procedural habits.\n"

    - For single-point energy calculations, no calculation program is required, as the method program directly performs the task. Do not specify a calculation program in this case.
    
    - If multiple programs are reasonable, only list the primary recommendation.
    
    - If the user asks to do IR or Raman spectrum calculation, you should give both geometry optimization and frequency calculation program.

    4.If the user explicitly specifies the calculation program for any task, you must strictly use the program specified by the user. Do NOT replace it with a recommended program, even if the recommendation matches common practice in computational chemistry. The user's explicit choice always has the highest priority. If the user has not specified a program, choose the most reasonable one from the allowed list, For general QM tasks (single-point energy, frequency), usually choose PySCF (most commonly used and robust), For geometry optimization tasks, the optimizer can be Geometric.

    4. **Structure**: Clearly separate Method, Method program, and Calculation program. If different tasks require different programs, categorize them accordingly.

    5. Only include information directly relevant to the user's input and previous discussion. Do not make assumptions beyond what was discussed.

    6. Return only the conclusion (no explanations).
    """
    return prompt
UV_PROMPT_CONCLUSION = uv_prompt_conclusion()


def prompt_conclusion():
    prompt = """
    You are a professional quantum chemistry expert. Based on the previous discussion, provide a **final summary** of the computational choices. Your response should include:

    1. **Method**: 
    - Clearly state the selected computational method.
        ***Below is the suggestion and auto-selection for optimization method if user requires OMNI-P2x method. Display the following warning and recommendations only when the user needs OMNI-P2x.***
        - If the user asks for a UV-vis spectrum calculation AND chooses the `OMNI-P2x` method without specifying an optimization method, you must:
            1. First, issue the following warning to the user in bold and underlined: 
            **<u>Warning: The OMNI-P2x method is not suitable for geometry optimization as it cannot predict accurate forces. Please choose a different method for optimization.</u>**
            2. Then, ask the user to provide a method and program for geometry optimization.
            3. Provide a recommended optimization method based on the system composition (see below).
            4. If the user provides a method, accept it. Otherwise, use the method you suggested. Do not proceed with an empty or unreasonable method.

        - Recommendation for optimization method:
            - If the molecule contains only C, H, O, N → suggest **AIQM2** (a QM/ML hybrid) for small-to-medium organic systems, unless the user explicitly requires high-level QM.
            - If other elements are present → suggest a reliable DFT method (e.g., **B3LYP/6-31G**), unless time constraints or user preferences dictate otherwise.
            - If the user mentions strict time limits → suggest faster ML methods or lower-cost DFT functionals/basis sets.

        - Important: If the user does not choose a method after your suggestion, you must use the method you suggested. Never provide an empty string or unreasonable method.

        - Format: 
            - First, display the warning in a new line (bold and underlined).
            - Then, in a new line, show the suggested method and the reason for the suggestion.

        - Note: Be careful to distinguish between `OM2` (semi-empirical) and `OMNI-P2x` (machine learning potential). Do not confuse these two methods.

    2. **Method program**:
    - If a program was specified for the method, include it.
    - If no program was specified, state: "Method program is not specified."
    - For ML-based methods (e.g., AIQM1, AIQM2, ANI-1ccx, UAIQM), indicate that no specific method program is required, but optionally list typical ML toolkits if relevant.
    - For quantum mechanical methods, if the program is not specified, use PySCF by default.

    3. **Calculation program**:
    - Only choose from: PySCF, Gaussian, Geometric, ASE.
    - If the user asks to do geometry optimization, do not plan for frequency calculations. If the user asks to do frequency calculations, do not plan to do geometry optimizations.
    - If the user requests both geometry optimization and UV-vis spectrum calculation, use default calculation program for optimization (unless otherwise specified) but DO NOT chooes calculation program for UV-vis calculation.
    - For each task (e.g., geometry optimization, frequency analysis), provide the recommended program.
        The types of tasks you define here will directly influence the downstream workflow design. Therefore, when determining which tasks require a recommended program, you must also follow the same task-design principles:
            "The task should be user-oriented. You may add additional tasks only when they genuinely help answer the user's underlying question or achieve the user's explicit goal, but never simply because they are commonly paired steps in computational chemistry.\nDo not follow typical workflows rigidly. Instead, infer whether an additional calculation is logically useful for the user's intent. For example, a frequency calculation may be added after a geometry optimization only if it is necessary to confirm stability or relevant to what the user is trying to obtain. IR, Raman, IRC, or any other extra tasks should only appear when they meaningfully contribute to the user's purpose.\nAlways prioritize the user's intent, and ensure that any additional tasks remain tightly aligned with the user's question rather than standard procedural habits.\n"

    - For single-point energy calculations, no calculation program is required, as the method program directly performs the task. Do not specify a calculation program in this case.
    
    - If multiple programs are reasonable, only list the primary recommendation.
    
    - If the user asks to do IR or Raman spectrum calculation, you should give both geometry optimization and frequency calculation program.

    4.If the user explicitly specifies the calculation program for any task, you must strictly use the program specified by the user. Do NOT replace it with a recommended program, even if the recommendation matches common practice in computational chemistry. The user's explicit choice always has the highest priority. If the user has not specified a program, choose the most reasonable one from the allowed list, For general QM tasks (single-point energy, frequency), usually choose PySCF (most commonly used and robust), For geometry optimization tasks, the optimizer can be Geometric.

    4. **Structure**: Clearly separate Method, Method program, and Calculation program. If different tasks require different programs, categorize them accordingly.

    5. Only include information directly relevant to the user's input and previous discussion. Do not make assumptions beyond what was discussed.

    6. Return only the conclusion (no explanations).
    """
    return prompt
PROMPT_CONCLUSION = prompt_conclusion()



def judge_method(has_method: bool=False):
    """
    This function is to judge whether the user has specified a computational method in the latest calculation request.

    Args: 
        has_method: Whether user has specified the calculation method.
    """
    if has_method:
        return "true"
    else:
        return "false"


tools = [judge_method]
judge_method_agent = create_agent(tools=tools, tool_kwargs={"tool_choice":"any"})
judge_method_tool = ToolNode(tools)


def judge_next_node(
    confirm_method_node:bool=False,
    offer_suggestion_node:bool=False
    ):
    """
    The function is to decide whether the user provided a method or asked for recommendations.

    Args:
        confirm_method_node: if user explicitly gave method/program. 
        offer_suggestion_node: if user asked for guidance or didn't give method
    """
    if confirm_method_node: return "confirm_method_node"
    if offer_suggestion_node: return "offer_suggestion_node"
    else: return "confirm_method_node"
judge_tools = [judge_next_node]
judge_next_node_agent = create_agent(tools=judge_tools, tool_kwargs={"tool_choice":"any"})
judge_next_node_tool = ToolNode(judge_tools)


def judge_uv(uv: bool=False):
    """
    This function is to judge whether the user is performing UV-Vis related calculations..

    Args: 
        uv: Whether user is performing uvvis.
    """
    if uv:
        return "true"
    else:
        return "false"
tools = [judge_uv]
judge_uv_agent = create_agent(tools=tools, tool_kwargs={"tool_choice":"any"})
judge_uv_tool = ToolNode(tools)

def user_confirm_node(
    reconfirm_node: bool=False,
    end_node: bool=False
    ):
    """
    This function if user has confirmed final calculation parameters.

    Args:
        reconfirm_node: user is still modifying calculation settings.
        end_node: user explicitly confirms final parameters.
    """
    if reconfirm_node: return "confirm_method_node"
    if end_node: return "end"
    else: return "confirm_method_node"
confirm_tools = [user_confirm_node]
user_confirm_agent = create_agent(tools=confirm_tools, tool_kwargs={"tool_choice":"any"})
user_confirm_tool = ToolNode(confirm_tools)

def judge_method_node(state: MethodConfirmState):
    response = judge_method_agent.invoke(state.messages)
    output = judge_method_tool.invoke({"messages":[response]})["messages"][-1].content
    if output=='true':
        return {"method_judge_result":"confirm_method_node"}
    else:
        return {"method_judge_result":"get_method_node"}

def get_method_node(state: MethodConfirmState):
    rup_msg = "Please specify the computational method and program you want to use for this calculation. If you are unsure which method to choose, feel free to ask me for recommendations."
    interrupt_msg = interrupt({
        "messages_to_user":rup_msg
    })
    system_msg = rup_msg + "I get the feedback from user:" + interrupt_msg['messages'].content
    logger.info("Get method node in method_confirm")
    logger.debug(interrupt_msg['messages'].content)
    system_prompt = [SystemMessage(content=f"{system_msg}")]
    
    response_node = judge_next_node_agent.invoke(state.messages+system_prompt)
    output = judge_next_node_tool.invoke({"messages":[response_node]})["messages"][-1].content #噢噢是这里返回了none， 然后invoke就会出现给llm空的情况，因此如果没有兜底不仅仅langgraph节点会崩， 这里再次invoke也会
    return {'method_get_result':output, 
            "messages":system_prompt,
            "messages_to_user":system_prompt}


def offer_suggestion_node(state: MethodConfirmState):
    inp_msg = state.messages
    path = FileManager.read_path()
    if path != Path.home():
        strcture_list = FileManager.get_init_sturcture()
        if strcture_list != []:
            user_provide_mols = []
            for structure_path in strcture_list:
                mol_name = os.path.splitext(os.path.basename(structure_path))[0]
                user_provide_mols.append(mol_name)
            inp_msg +=  [SystemMessage(content=f"The molecule list that user provide for calculation:{[user_provide_mols]}")]
    suggestion_prompt = SystemMessage(content=PROMPT_SUGGESTION)
    suggestion = llm.invoke(inp_msg+[suggestion_prompt])
    return {'messages': [AIMessage(content=suggestion.content)],
            'messages_to_user': [AIMessage(content=suggestion.content)]}

    
def confirm_method_node(state: MethodConfirmState):
    response_node = judge_uv_agent.invoke([state.messages[0]])
    output = judge_uv_tool.invoke({"messages":[response_node]})["messages"][-1].content 

    if output == 'true': 
        conclusion_propmpt = SystemMessage(content=PROMPT_CONCLUSION)
    else:
        conclusion_propmpt = SystemMessage(content=PROMPT_CONCLUSION)
    final_conclusion = llm.invoke(state.messages+[conclusion_propmpt])
    user_final_check = interrupt({
        "messages_to_user":final_conclusion.content+"\n\n"+"Do you confirm these as the final computational parameters?",
        "options": ["yes", "no"]
    })
    system_prompt = [HumanMessage(content=f"{user_final_check}")]
    final_prompt = [AIMessage(content=f"Final conclusion: {final_conclusion.content}")]
    response_node = user_confirm_agent.invoke(final_prompt+system_prompt)
    output = user_confirm_tool.invoke({"messages":[response_node]})["messages"][-1].content
    if output == "confirm_method_node":
        reconfirm_msg = [AIMessage(content=f"User changed the parameters: {user_final_check['messages'].content}")]
        return {'method_confirm':"confirm_method_node", 
                "messages":reconfirm_msg,
                "messages_to_user":reconfirm_msg}
    else:
        final_msg = [AIMessage(content=f"User confirmed this as final calculation parameters:\n{final_conclusion.content}, continue calculation steps")]
        return {'method_confirm':'end', 
                'messages':final_msg,
                'messages_to_user':final_msg}



method_confirm_builder = StateGraph(MethodConfirmState)
method_confirm_builder.add_node("judge_method_node", judge_method_node)
method_confirm_builder.add_node("get_method_node", get_method_node)
method_confirm_builder.add_node("offer_suggestion_node", offer_suggestion_node)
method_confirm_builder.add_node("confirm_method_node", confirm_method_node)

method_confirm_builder.add_edge(START, "judge_method_node")
method_confirm_builder.add_conditional_edges(
    "judge_method_node", 
    lambda state: state.method_judge_result,
    {
        "confirm_method_node":"confirm_method_node",
        "get_method_node":"get_method_node"
    })


method_confirm_builder.add_conditional_edges(
    "get_method_node",
    lambda state: state.method_get_result,
    {
        "confirm_method_node":"confirm_method_node",
        "offer_suggestion_node":"offer_suggestion_node"
    }
)
method_confirm_builder.add_edge("offer_suggestion_node","confirm_method_node")
method_confirm_builder.add_conditional_edges(
    "confirm_method_node",
    lambda state: state.method_confirm,
    {
        "confirm_method_node":"confirm_method_node",
        "end":END
    }
)






















