"""
    Excited-state single-point agent
"""
# import json
# import ast
import os
import mlatom as ml
from typing import Union, Optional
import numpy as np
import pandas as pd
import json
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END 
from langgraph.types import interrupt, Command
from copy import deepcopy

from ..method_judge import method_builder
from ..prepare_molecule import prepare_molecule_builder
from ..file_manager import get_folder_name_node, get_result_file_node

from ..states import AitomiaState
from ..logger import logger 
from ..utils import create_agent, pretty_dict, pretty_array, Analysis

from langgraph.config import get_stream_writer

#-----------------------TODO-----------------------
# 1. incorrect nstates;
# 2. gradients are not calculated when gradeints are required

#-----------------------Schema-----------------------
class UVvisState(AitomiaState):
    # General settings
    molecule_file_name:str = None
    method:str = None
    program:Union[str,None] = None
    tddft:bool=False
    tdadft:bool=False
    nstates:int = None
    uvvis_result:Union[str,None] = None
    band_width:Union[float,None]=None
    contribution_threshold:Union[float,None]=None

    # Generation method settings
    generation_method:str="spc"


#-----------------------Prompt-----------------------
PROMPT_UVVIS_SPC_STATES = """The `nstates` is the total electronic states to be calculated including ground state. If the user require calculate n electronic states, the nstates is n; but if the user ask to calculate n EXCITED STATES, the nstates is n + 1."""
# PROMPT_UVVIS_SPC_GETPROP = """The `calculate_energy_gradients` is a boolean variable to decide whether the agent performs gradients calculation for the `current_state`, and so does the boolean variable `calculate_hessian`, which is used for hessian calculation for `current_state`"""
PROMPT_UVVIS_SPC_METHOD = """If the user requests TD-DFT method or TDA-DFT method for excited-state calculation, you need to add `TD-` (for TD-DFT) or `TDA-` (for TDA-DFT) at the beginning of the method string."""
PROMPT_UVVIS_SPEC_SETTINGS = """Two params: `band_width` and `contribution_threshold`. The band_width is the parameter to control the width of spctrum, the default value is 0.3; `contribution_threshold`, as its name suggest, output the excitations whose contribution to the absorption peak is greater than it."""
PROMPT_UVVIS_SPC_ANALYSIS = """Below shows the result of the excited-state calculation and UV-vis spectrum calculation. It might contain multiple properties of the molecule, e.g., electronic states, excitation energies, oscillator strengths, energy gradients, hessian, etc. You should give explicitly the absolute path in html format like <Path>abs_path</Path> of the result files for further calculation of analysis, especially calculation results (.json) and spectrum file (.png). `Nan` in the result means the corresponding properties are not calculated. For the UV-vis spectrum, you need to show user the parameters of spectrum plotting like band width. The precautions of analysis are listed:
    - The first part should be a brief summary of the UV-vis task, including what task is done, what method is uesd, what kind of results do we get.
        - Methods and program need to correspond strictly. For Machine learning method, there will be no program (aka program=None); for QM method, tell the user corresponding program. **BE REALLY CAREFUL** with this part, as it relates to proper citation in academic writing.
    - Spectrum details are also important, as they provide the absorption peaks' position and contributions from individual excitation. Please show the spectrum details in neat markdown format like:
        |Absorption wavelength|Individual excitation contributions|
        |500 nm|S1: 0.5; S2: 0.4|
        |300 nm|S10: 0.3; S11: 0.25|
    - When writing an analysis, add vibrant emojis that are twice the text size before each section, like 📊 (choose different emojis according to the content), to visually categorize the content. The analysis need to be really detailed, easy-to-understand, and properly emphasize key points (spectrum parameters like band_width and contribution_threshold, and absorption wavelength, ect.), treat the user as beginner of spectroscopy (Provide detailed and professional analysis, but never tell the user they are beginner).
    - Assume the calculation finished, you must show the user result files as clickable html format like <Path>file_path</Path>, for the convenience of users to check the spectrum plot directly.
"""


#-----------------------ToolFunc--------------------
def get_states(nstates:int=None):
    """
    This function is used to get nstates for UV-vis spectrum (single-point convolution, spc) calculation. The `nstates` is total number of `electronic state`s going to be calculated, includes the ground state and excited states (for example: when user requies {N} `electronic states`, the `nstates` is {N}; but when user requires 'N' `excited states`, the `nstates` is {N+1}), you need to check the difference between `electronic states` and `excited states` carefully. The default value of `nstates` should always be 21 if the user requests spc UV-vis spectrum, unless other specifies from user.

    Args:
        nstates: Number of states to be calculated.
    """

    if nstates == None:
        nstates = 21  # default nstates

    return locals()


# def get_prop2calc(calculate_energy_gradients:bool=None,
#                   calculate_hessian:bool=None):
#     """
#     This function is used to determine which properties need to be calculated. The `calculate_energy_gradients` is `True` if the user requires gradients of the state of interest, and so does the `calculate_hessian`.

#     Args:
#         calculate_energy_gradients: To decide whether the gradients of current_state will be calculated,
#         calculate_hessian: To determine whether the hessian matrix of current_state will be calculated.
#     """
#     if calculate_energy_gradients == None:
#         calculate_energy_gradients = False
#     if calculate_hessian == None:
#         calculate_hessian = False
    
#     return locals()


def get_qm_method(method:str=None,
                  program:Optional[str]=None,
                  tddft:bool=False,
                  tdadft:bool=False):
    """
    This function is used to judge the method type requested by user. The exapmle of allowed method types and corresponding methods are listed below:
        |Type|Method|Program|
        |----|----|----|
        |semi-empirical|DFTB|DFTBPLUS|
        |semi-empirical|ODM2|MNDO|
        |ML|AIQM1|None|
        |TD-DFT|f"td-{functional}/{basis}"|QMPROG|
        |TDA-DFT|f"tda-{functional}/{basis}"|QMPROG|
    If the user requests TD-DFT or TDA-DFT method for the excited-state calculation, you need to rewrite the method name as TD-method or TDA-method for TD-DFT and TDA-DFT, respectively. The default choice of DFT method in excited-state calculations is TD-DFT, unless the user specifies TDA.
    Notice that `tddft` and `tdadft` will be `True` only of TD-DFT or TDA-DFT are required. If the user ask for machine learning (ML), semi-empirical or ML/QM hybrid method, `tddft` and `tdadft` both should be `False`.
    Args:
        method: The method for excited-state calculation;
        program: The program for excited-state calculation, if it's gaussian or orca, use TD-DFT or TDA-DFT;
        tddft: To judge if TD-DFT calculation is required;
        tdadft: To judge if TDA-DFT calculation is required.
    """
    if isinstance(program, str):
        if program.casefold() in ("gaussian", "orca"):
            if tdadft:
                method = f"tda-{method}"
            else:
                method = f"td-{method}"

    return locals()    


def get_spectrum_settings(band_width:float=None,
                          contribution_threshold:float=None):
    """
    This function is used to retrive the parameters for UV-vis spectrum plotting. The parameter `band_width` controls the width of spectrum. `contribution_threshold` is the parameter for the spectrum analysis, ensuring that only excitations with an absorption contribution greater than this value will be printed.

    Args:
        band_width: Parameter to control the width of peaks, default value is 0.3;
        contribution_threshold: Parameter that control the excitation contribution to print (only excitations that meet this criteria will be counted);
    """
    if band_width == None:
        band_width = 0.3
    
    if contribution_threshold == None:
        contribution_threshold = 0.1
    
    return locals()


def print_peak_contributions(uvvis_peaks: dict = None):
    formatted_str = ""
    for absorption, properties in uvvis_peaks.items():
        formatted_str += f"\ncontributions to {absorption} absorption: \n\n"
        prop = pd.DataFrame.from_dict(properties, orient='index', columns=['Contribution'])
        formatted_str += f"\t\t{prop.to_string()}\n"
    
    return formatted_str.lstrip("\n")

#-----------------------Agent-----------------------
llm = create_agent()
state_tools = [get_states]
get_state_agent = create_agent(tools=state_tools)
get_state_tool = ToolNode(state_tools)

# prop_tools = [get_prop2calc]
# get_prop_agent = create_agent(tools=prop_tools)
# get_prop_tool = ToolNode(prop_tools)

method_tools = [get_qm_method]
get_method_agent = create_agent(tools=method_tools)
get_method_tool = ToolNode(method_tools)

spec_settings_tool = [get_spectrum_settings]
get_spec_settings_agent = create_agent(tools=spec_settings_tool)
get_spec_settings_tool = ToolNode(spec_settings_tool)


#-----------------------Script-----------------------
run_es_py = """#!/bin/env python

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
if {{method | pyrepr}}.casefold() in ('omnip2x', 'omni-p2x'):
    MODEL = ml.models.omnip2x()
else:
    MODEL = ml.methods(method={{method | pyrepr}}, program={{program | pyrepr}})

# 3. fill in settings of raman
working_directory = {{working_directory | pyrepr}}
MODEL.working_directory = working_directory
MODEL.predict(
            molecule=mol,
            calculate_energy=True,
            calculate_energy_gradients=False,
            nstates={{nstates}}
            )

# 4. plot uvvis spectrum
ml.spectra.plot_uvvis(molecule=mol,
                      spc=True,
                      labels=[{{method | pyrepr}}],
                      filename=os.path.join(working_directory, "uvvis_spectra.png"),
                      band_width={{band_width | pyrepr}},
                      analyze_component=True,
                      contribution_threshold={{contribution_threshold | pyrepr}}
                      )
spec = ml.spectra.uvvis.spc(molecule=mol, band_width={{band_width | pyrepr}})

# 5. Save results
spec.dump(os.path.join(working_directory, 'uvvis_spectra_curve.txt'),format='txt')
mol.dump(filename={{uvvis_result | pyrepr}}, format='json')

# 6. Remove redundant files from AIQM1
if {{method | pyrepr}}.casefold() == 'aiqm1':
    os.system('rm predict*.xyz')
"""


#-----------------------Graph------------------------
def get_state_node(state:UVvisState):
    logger.info("Started getting states")

    try:
        state_prompt = SystemMessage(content=PROMPT_UVVIS_SPC_STATES)
        response = get_state_agent.invoke(state.current_task_messages[-1]+[state_prompt])

        logger.debug(f"Response from the state agent: \n\t{response.content}")

        output = get_state_tool.invoke({"messages": [response]})["messages"][-1].content
        logger.debug(f"Response from get_state_tool: \n\t{output}")
        output = json.loads(output)

        output_state = {
            "nstates": output["nstates"],
            # "current_state": output["current_state"]
        }
        return output_state

    except Exception as e:
        errmsg = f"Error in get_state_node: {str(e)}"
        logger.error(errmsg)
        return {
            "error": errmsg,
            "nstates": 21,
            # "current_state": 1,
            "messages": [AIMessage(content=f"Failed to get nstate, using default values 21 {errmsg}")],
            "messages_to_user": [AIMessage(content=f"Failed to get nstate, using default values 21: {errmsg}")]
        }


# def get_prop2calc_node(state:UVvisState):
#     logger.info("Started getting properties to be calculated")

#     try:
#         prop2calc_propmt = SystemMessage(content=PROMPT_UVVIS_SPC_GETPROP)
#         response = get_prop_agent.invoke(state.current_task_messages[-1]+[prop2calc_propmt])

#         logger.debug(f"Response from the get_prop_agent: \n\t{response.content}")

#         output = get_prop_tool.invoke({"messages": [response]})["messages"][-1].content
#         logger.debug(f"Response from the get_prop_tool: \n\t{output}")
#         output = json.loads(output)

#         output_state = {
#             "calculate_energy_gradients": output["calculate_energy_gradients"],
#             "calculate_hessian": output["calculate_hessian"]
#             }
#         return output
    
#     except Exception as e:
#         errmsg = f"Error in get_prop2calc_node: {e}"
#         logger.error(errmsg)
#         return {
#             "error": errmsg,
#             "calculate_energy_gradients": False,
#             "calculate_hessian": False,
#             "messages": [AIMessage(content=f"Failed to get properties to be calculated, using default values `False`: {errmsg}")],
#             "messages_to_user": [AIMessage(content=f"Failed to get properties to be calculated, using default values `False`: {errmsg}")]
#             }


def get_esmethod_node(state:UVvisState):
    logger.info("Started getting method for excited-state calculation")

    try:
        es_method_prompt = SystemMessage(content=PROMPT_UVVIS_SPC_METHOD)
        response = get_method_agent.invoke(state.current_task_messages[-1]+[es_method_prompt])

        logger.debug(f"Response from the excited-state method agent \n\t{response.content}")

        output = get_method_tool.invoke({"messages": [response]})["messages"][-1].content
        logger.debug(f"Response from excited-state method tool: \n\t{output}")
        output = json.loads(output)

        output_state = {
            "tddft": output["tddft"],
            "tdadft": output["tdadft"],
            # "program": output["program"],
            "method": output["method"]
            }
        return output_state
    
    except Exception as e:
        errmsg = f"Error in get_esmethod_node: {e}"
        logger.error(errmsg)
        return {
            "error": errmsg,
            "tddft": False,
            "tdadft": False, 
            "program": None,
            "method": "omnip2x"
        }


def get_spec_settings_node(state:UVvisState):
    logger.info("Start retrive the UV-vis spectrum plotting settings")

    try:
        settings_prompt = SystemMessage(content=PROMPT_UVVIS_SPEC_SETTINGS)
        response = get_spec_settings_agent.invoke(state.current_task_messages[-1]+[settings_prompt])

        logger.debug(f"Response from the UV-vis setting agent: \n\t{response.content}")

        output = get_spec_settings_tool.invoke({"messages": [response]})["messages"][-1].content
        logger.debug(f"Response from get_spec_settings_tool: \n\t{output}")
        output = json.loads(output)

        output_state = {
            "band_width": output["band_width"],
            "contribution_threshold": output["contribution_threshold"],
            }
        return output_state
    
    except Exception as e:
        errmsg = f"Error in get_spec_settings_node: {e}"
        logger.error(errmsg)
        return {
            "band_width": 0.3,
            "contribution_threshold": 0.1,
            }


def excited_state_coder_node(state:UVvisState):
    from jinja2 import Environment

    logger.info("excited-state code node")
    message = "Start calculating excited-state and plot UV-vis spectrum."
    logger.debug("Input state: "); pretty_dict(state.model_dump(), logger)

    filename = "es_single_point.json"
    if os.path.exists(os.path.join(state.working_directory,filename)):
        ii = 1
        while True:
            if os.path.exists(os.path.join(state.working_directory,f"ex_single_point_{ii}.json")):
                ii += 1
            else:
                break 
        filename = f"es_single_point_{ii}.json"
    result_file_name = os.path.join(state.working_directory,filename)
    state.uvvis_result = result_file_name

    input_args = state.model_dump().copy()

    if state.molecule_file_name[-4:] == ".xyz": input_args['molecule_format'] = 'xyz'
    elif state.molecule_file_name[-5:] == ".json": input_args['molecule_format'] = 'json' 
    else: raise ValueError("Unhandled molecule format")

    env = Environment(autoescape=False)
    env.filters["pyrepr"] = repr
    es_code = env.from_string(run_es_py).render(**input_args)

    working_directory = state.working_directory_stack[-1]
    python_script = os.path.join(working_directory, "run_sp_es.py")

    with open(python_script, "w") as f:
        f.write(es_code)
    logger.debug("Output state:"); logger.debug({"scripts":python_script})
    return {"scripts":python_script,
            "uvvis_result": result_file_name,
            "messages_to_user": [AIMessage(content=message)]} 


#-----------------------Graph-----------------------
def excited_state_uvvis_spc_exec_node(state:UVvisState):
    message = "Start calculating excited-state single-point energies"
    logger.info(message)

    try:
        python_script = state.scripts
        result_log = os.path.join(state.working_directory_stack[-1], "es_sp.log")
        result_err = os.path.join(state.working_directory_stack[-1], "es_sp.err")
        # result_mol = os.path.join(state.working_directory_stack[-1], "es_mol.json")
        # result_spec = os.path.join(state.working_directory_stack[-1], "es_sp.png")

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
            
            error_message = f"Excited-state calculation script failed with exit code {exit_code}"
            if error_content:
                error_message += f"\nError output:\n{error_content}"
            
            logger.error(error_message)
            return {
                "error": error_message,
                "has_error": True,
                "messages": [AIMessage(content=error_message)],
                "messages_to_user": [AIMessage(content=error_message)],
            }

        return {"excited_state_result_files": [result_log, result_err],
                "messages_to_user": [AIMessage(content=message)]}
    
    except Exception as e:
        import traceback
        error_message = f"Error in excited_state_uvvis_spc_exec_node: {str(e)}"
        error_info = traceback.format_exc()
        logger.error(error_message + error_info)
        return {
            "error": error_message,
            "has_error": True,
            "messages": [AIMessage(content=f"excited-state single-point calculation failed: {error_message}")],
            "messages_to_user": [AIMessage(content=f"excited-state single-point calculation failed: {error_message}")]
        }


def excited_state_analysis_node(state:UVvisState):
    message = "Start analyzing the result of excited-state calculation"
    logger.info(message)

    try:
        molecule = ml.molecule.load(filename=state.uvvis_result, format="json")
        result_file_name = state.uvvis_result
        excited_state_sp_prompt = SystemMessage(content=PROMPT_UVVIS_SPC_ANALYSIS)

        # Formatted summary
        excitation_data = pd.DataFrame({"Excitation  ": list(range(1, len(molecule.excitation_energies)+1)),
                                        "Excitation energies (eV)  ": [round(x, ndigits=2) for x in molecule.excitation_energies * ml.constants.Hartree2eV],
                                        "Oscillator strengths  ": molecule.oscillator_strengths})
        uvvis_components = deepcopy(molecule.uvvis_peaks)
        for wavelength in uvvis_components.keys():
            uvvis_components[wavelength] = str(uvvis_components[wavelength])
        uvvis_peaks_data = pd.DataFrame(uvvis_components, index=[range(len(uvvis_components.keys()))])
        
        result_message_str = f"Excited-state calculation: \n"
        result_message_str += f"{excitation_data.to_markdown(index=False)}\n\n"
        result_message_str += f"UV-vis spectrum analysis:\nThe UV-vis spectrum is plotted using the above excitation energies and oscillator strengths with a band width of {state.band_width} eV, and for each peak, excitations with contributions larger than {state.contribution_threshold} are printed.\n"
        result_message_str += f"""The absorption peaks' wavelength and the individual excitation contributions: \n{uvvis_peaks_data.to_markdown(index=False).replace("'", "")}\n\n"""

        result_message_str += f"\tResult file: {result_file_name}\n"
        result_message_str += f"\tSpectrum png file are stored in: {os.path.join(state.working_directory, 'uvvis_spectra.png')}\n"
        result_message = AIMessage(content=result_message_str)

        # Analysis from LLM
        response = llm.invoke(state.current_task_messages[-1] + [excited_state_sp_prompt] + [result_message])
        task_message = state.current_task_messages
        task_message[-1].append(AIMessage(content=response.content))

        with open(os.path.join(state.working_directory, "summary.out"), "w") as f:
            f.write(result_message_str)
        result_message_str += f"Detailed summary can be found in {os.path.join(state.working_directory, 'summary.out')}\n"


        return {
            "messages": [AIMessage(content=result_message_str + "\n" + response.content)],
            "messages_to_user": [AIMessage(content=message + "\n" + result_message_str + "\n" + response.content)],
            "current_task_messages": task_message
        }

    except Exception as e:
        error_message = f"Error in excited_state_analysis_node: {str(e)}"
        logger.error(error_message)
        writer = get_stream_writer()
        writer(f"❌ {error_message}")
        return {
            "error": error_message,
            "has_error": True,
            "messages": [AIMessage(content=f"Failed to analyze excited-state results: {error_message}")],
            "messages_to_user": [AIMessage(content=f"Failed to analyze excited-state results: {error_message}")]
        }

def check_error(state:UVvisState):
    """Check if an error has occurred and route accordingly"""
    if state.has_error:
        logger.warning("Error detected, skipping remaining steps")
        return END
    return "continue"
        

uvvis_spc_builder = StateGraph(UVvisState)

method_graph = method_builder.compile()
prepare_molecule_graph = prepare_molecule_builder.compile()

uvvis_spc_builder.add_node("get_folder_name_node",get_folder_name_node)
uvvis_spc_builder.add_node("method",method_graph)
uvvis_spc_builder.add_node("prepare_molecule",prepare_molecule_graph)
uvvis_spc_builder.add_node("get_state_node", get_state_node)
# uvvis_spc_builder.add_node("get_prop2calc_node", get_prop2calc_node)
uvvis_spc_builder.add_node("get_esmethod_node", get_esmethod_node)
uvvis_spc_builder.add_node("get_spec_settings_node", get_spec_settings_node)
uvvis_spc_builder.add_node("excited_state_coder_node",excited_state_coder_node)
uvvis_spc_builder.add_node("excited_state_uvvis_spc_exec_node",excited_state_uvvis_spc_exec_node)
uvvis_spc_builder.add_node("excited_state_analysis_node",excited_state_analysis_node)
uvvis_spc_builder.add_node("get_result_file_node",get_result_file_node)

uvvis_spc_builder.add_edge(START,"get_folder_name_node")
uvvis_spc_builder.add_conditional_edges("get_folder_name_node", check_error, {"continue": "method", END: END})
uvvis_spc_builder.add_conditional_edges("method", check_error, {"continue": "prepare_molecule", END: END})
uvvis_spc_builder.add_conditional_edges("prepare_molecule", check_error, {"continue": "get_state_node", END: END})
# uvvis_spc_builder.add_edge("get_state_node", "get_prop2calc_node")
# uvvis_spc_builder.add_edge("get_prop2calc_node", "get_esmethod_node")
uvvis_spc_builder.add_edge("get_state_node", "get_esmethod_node")
uvvis_spc_builder.add_edge("get_esmethod_node", "get_spec_settings_node")
uvvis_spc_builder.add_edge("get_spec_settings_node", "excited_state_coder_node")
uvvis_spc_builder.add_edge("excited_state_coder_node","excited_state_uvvis_spc_exec_node")
uvvis_spc_builder.add_conditional_edges("excited_state_uvvis_spc_exec_node", check_error, {"continue": "excited_state_analysis_node", END: END})
uvvis_spc_builder.add_conditional_edges("excited_state_analysis_node", check_error, {"continue": "get_result_file_node", END: END})
uvvis_spc_builder.add_edge("get_result_file_node",END)


uvvis_spc_graph = uvvis_spc_builder.compile()

from ..agent_template import BaseAgent
uvvis_spc_agent = BaseAgent(
    name='uvvis_spc_agent',
    description=f'Calculate UV-vis spectrum via single-point convolution for a given structure.',
    graph=uvvis_spc_builder,
)
