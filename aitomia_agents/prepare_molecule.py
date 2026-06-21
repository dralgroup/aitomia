"""
    Prepare molecule agent
"""
# import json
import ast
import os
import re
import glob
import time
import traceback
import json
import mlatom as ml
import numpy as np
# from typing import Union, Optional
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END 
from .settings import settings
from .user_context import user_context
from pathlib import Path
from langgraph.types import interrupt
from .states import AitomiaState
from .logger import logger 
from .utils import create_agent, pretty_dict, FileManager, GlobalFlag, create_agent
from . import aitomia_pcp as pcp

from langgraph.config import get_stream_writer

#-------------------------------------------------
# Schema
#-------------------------------------------------
class PrepareMoleculeState(AitomiaState):
    molecule_file_name: str = None
    nocalc_task_type: str = None
    

#-------------------------------------------------
# Prompts
#-------------------------------------------------
prepare_molecule_prompts = """
You need to retrieve all the molecule information avaible from user and you have access to the files provided by users with tools. You should select the molecule that is needed for the current task.
"""
prepare_molecule_prompts = SystemMessage(prepare_molecule_prompts)

#-------------------------------------------------
# Tool functions
#-------------------------------------------------

# Get the molecule from file
def get_molecule_from_file(filename:str=""):
    """
    This function is used to get the absolute path of the user provided molecule file or result file. 
    IMPORTANT: You must provide a FILE path, not a directory path. The file should have extensions like .xyz, .json, etc.
    This function MUST NOT generate, infer, or fabricate file paths.

    Args:
        filename: The absolute path of the molecule FILE (not directory) provided by the user. Must be an actual file with extension.
    """
    logger.debug("In get_molecule_from_file")

    # os.system(f"cp {filename}")
    return {"molecule_file_name":filename}

def get_molecule_from_database(molecule_name:str=""):
    """
    This function is to get the molecular structure from the online database.

    Args:
        molecule_name: Name of the molecule.
    """
    logger.debug("In get_molecule_from_database")

    return {"molecule_file_name":molecule_name}
    

def get_xyz_from_pubchem(name, generating_path=None):      
    import pubchempy as pcp
    try:
        compounds = pcp.get_compounds(name, 'name', record_type='3d')

        if compounds:
            compound = compounds[0]
            atoms = compound.atoms
            # Clean quotes from molecule name to avoid file names like 'ethanol.xyz'
            clean_name = name.strip("'\"").strip()
            filename = f"{clean_name}.xyz"
            generating_path = generating_path or user_context.home_dir or Path.home()
            filename = f"{generating_path}/{filename}"
            with open(filename, 'w') as f:
                f.write(f"{len(atoms)}\n\n")
                for atom in atoms:
                    f.write(f"{atom.element:<2} {atom.x:>16.8f} {atom.y:>16.8f} {atom.z:>16.8f}\n")

            logger.info(f"XYZ coordinates saved to {filename}") #filename是绝对路径
            writer = get_stream_writer()
            writer(f"XYZ coordinates saved to {filename}")
            return filename
        else:
            logger.warning(f"No 3D structure found for {name} in PubChem, trying RDKit...")
            return get_xyz_from_rdkit(name, generating_path)
    except Exception as e:
        logger.error(f"An error occurred in get_xyz_from_pubchem: {e}, trying RDKit...")
        return get_xyz_from_rdkit(name, generating_path)


def get_xyz_from_rdkit(name, generating_path=None):
    try:
        import requests
        from rdkit import Chem
        from rdkit.Chem import AllChem

        def get_smiles_from_name(name):
            """Retrieve the SMILES string for the molecule from its name."""
            service_url = settings.molecule_service_url
            url = f"https://cactus.nci.nih.gov/chemical/structure/{name}/smiles"
            if service_url:
                url = f"{service_url}/chemical/structure/{name}/smiles"
            response = requests.get(url)
            if response.status_code == 200:
                smiles = response.text.strip()
                return smiles
            else:
                logger.warning(f"Failed to retrieve SMILES for {name}. Status code: {response.status_code}")
                return None

        def generate_3d_coordinates(smiles, molecule_name, generating_path=None):
            """Generate 3D coordinates and save them in XYZ format."""
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol)
            conf = mol.GetConformer()
            atoms = mol.GetAtoms()

            # Begin the output with the number of atoms and an empty line for better formatting
            xyz_lines = [f"{len(atoms)}", ""]

            for atom in atoms:
                pos = conf.GetAtomPosition(atom.GetIdx())
                # Format the output for each atom to match the desired output format
                xyz_lines.append(f"{atom.GetSymbol():<2} {pos.x: 15.8f} {pos.y: 15.8f} {pos.z: 15.8f}")

            # Clean quotes from molecule name to avoid file names like 'ethanol.xyz'
            clean_name = molecule_name.strip("'\"").strip()
            filename = f"{clean_name}.xyz"
            generating_path = generating_path or user_context.home_dir or Path.home()
            filename = f"{generating_path}/{filename}"
            # Save the XYZ coordinates to a file with the molecule name as the filename
            with open(filename, 'w') as file:
                file.write("\n".join(xyz_lines))
            
            writer = get_stream_writer()
            writer(f"XYZ coordinates for {molecule_name} saved to {molecule_name}.xyz")

            return filename

        # Example usage
        smiles = get_smiles_from_name(name)
        if smiles:
            filename = generate_3d_coordinates(smiles, name, generating_path)
            # success_message = f"XYZ coordinates for {name} saved to {name}.xyz"
            return filename
        else:
            logger.error(f"Could not retrieve SMILES for {name}")
            return None

    except Exception as e:
        logger.error(f"An error occurred in get_xyz_from_rdkit: {e}")
        return None

def extract_charge_multiplicity(charge:int=None,multiplicity:int=None):
    """
    This function is to extract the charge and multiplicity of the molecule. If they are explicitly specified in the message, you should extract exactly the same values. If not, use the default values. Determine reasonable default values based on the molecular composition.
    - For multiplicity, consider the number of electrons and common chemical knowledge (e.g., O2 ground state is triplet).
    
    Args:
        charge: Charge for the molcule.
        multiplicity: Multiplicity of the molecule.
    """
    return locals()

def seperate_atom_and_molecule(name:str=""):
    """
    Determine whether the input refers to an atom or a molecule.
    
    Rules:
        - If the input is about calculating a property of a single atom,
          including single-atom radicals or ions, e.g. 'C', 'H', 'Cl·', 'O·', '·H', 'Na+'),
          return the element symbol (e.g., 'H', 'C', 'O', 'Cl·').
        - If the input is a molecule, return the molecule name as-is.

    Args:
        name: Name of the molecule or atom (case-sensitive).

    """
    atom_list = ['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','','Bi','','Po','','At','','Rn','','Fr','','Ra','','Ac','','Th','','Pa','','U','','Np','','Pu','','Am','','Cm','','Bk','','Cf','','Es','','Fm','','Md','','No','','Lr','','Rf','','Db','','Sg','','Bh','','Hs','','Mt','','Ds','','Rg','','Cn','','Nh','','Fl','','Mc','','Lv','','Ts','','Og']

    base = re.sub(r'^[·•*+\-]+|[·•*+\-]+$', '', name)
    if re.search(r'\d', base):
        return {"molecule": name}
    for atom in atom_list:
        if base.startswith(atom):
            suffix = base[len(atom):]
            if all(c in '·+-' for c in suffix):
                return {"atom": base}
            if suffix == '':
                return {"atom": base}
            break

    return {"molecule": name}

#-------------------------------------------------
# Agent
#-------------------------------------------------
tools = [get_molecule_from_file,get_molecule_from_database]
prepare_molecule_agent = create_agent(tools=tools,tool_kwargs={"tool_choice":"any"})
prepare_molecule_tool = ToolNode(tools)
tools = [extract_charge_multiplicity]
extract_charge_multiplicity_agent = create_agent(tools=tools,tool_kwargs={"tool_choice":"any"})
extract_charge_multiplicity_tool = ToolNode(tools)

tools = [seperate_atom_and_molecule]
seperate_atom_and_molecule_agent = create_agent(tools=tools, tool_kwargs={'tool_choice':'any'})
seperate_atom_and_molecule_tool = ToolNode(tools)

#-------------------------------------------------
# Graph
#-------------------------------------------------

def prompt_extract_target_and_formula(current_task, user_original_query):
    prompt = f"""You are a chemistry task interpretation agent.

    Your ONLY job is to identify the intended target molecule / atom / system for this task and provide its standard neutral molecular formula.

    STRICT RULES:
    1. If Current Task explicitly names a molecule, ion, atom, or system, that is the target. In this case, COMPLETELY IGNORE User Original Query.
    2. Only if Current Task is vague, use User Original Query to infer the target.
    3. Provide the most common standard neutral chemical formula.
    4. Do NOT include reaction partners, dimers, adducts, or salts unless explicitly named.
    5. Do NOT compute or report atom counts.
    6. Do NOT provide multiple possibilities.

    Return ONLY a JSON object in this exact format:

    {{
    "target_name": "<string>",
    "formula": "<chemical formula string, e.g. C5H6>"
    }}

    Current Task: {current_task}
    User Original Query: {user_original_query}
    """
    return prompt

def check_molecule_structure(file:str="", current_task:str="", user_original_query:str="", working_directory:str=""):
    import mlatom as ml
    if file.endswith(".json"):
        molecule_coordinates = ml.molecule.load(file, format='json').get_xyz_coordinates()
        molecule_elements = ml.molecule.load(file, format='json').get_element_symbols()
        n_atoms = len(molecule_elements)
        lines = [str(n_atoms), " "]
        for elem, (x, y, z) in zip(molecule_elements, molecule_coordinates):
            lines.append(f"{elem:2s} {x: .8f} {y: .8f} {z: .8f}")
        current_structure = lines
        convert_to_xyz_file = "\n".join(lines)
    else:
        with open(file, 'r') as f:
            current_structure = f.readlines()
            convert_to_xyz_file = "".join(current_structure)
    atom_number = str(current_structure[0].strip())
    PROMPT = prompt_extract_target_and_formula(
        current_task=current_task[-1].content,
        user_original_query=user_original_query.content,
    )
    rule_prompt = SystemMessage(content=PROMPT)
    llm = create_agent()
    response = llm.invoke([rule_prompt])
    def parse_formula(formula: str):
        tokens = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
        counts = {}
        for elem, num in tokens:
            n = int(num) if num else 1
            counts[elem] = counts.get(elem, 0) + n
        return counts
    def total_atoms_from_formula(formula: str):
        counts = parse_formula(formula)
        return sum(counts.values())
    try:
        formula = json.loads(response.content)["formula"]
        expected_atom_count = total_atoms_from_formula(formula)
        if expected_atom_count == int(atom_number):
            logger.info("Molecule structure is correct.")
            return {'correct': True}
        else:
            current_mol_name = json.loads(response.content)["target_name"]
            # Clean quotes from molecule name to avoid file names like 'ethanol.xyz'
            current_mol_name = current_mol_name.strip("'\"").strip()
            with open(f'{working_directory}/{current_mol_name}.xyz', 'w') as f:
                f.writelines(convert_to_xyz_file)
            filename = f'{working_directory}/{current_mol_name}.xyz'
            return {'correct': False, 'filename': filename}
    except Exception as e:
        logger.error("LLM can not prase JSON:", response.content, 'error msg', e)
        return {'correct': False}


def prepare_molecule_node(state:PrepareMoleculeState):
    logger.info("Start preparing molecule")
    message = "Start preparing molecule"
    molecule_filename = GlobalFlag.get_flag('molecule_filename')
    
    # try:
    if molecule_filename == None:
        logger.debug("Input state:"); pretty_dict(state.model_dump(), logger)
        user_init_structure = SystemMessage(content=f"The initial structure provided by user is: {FileManager.read_path()}")
        logger.debug("User initial structure message:")
        logger.debug(f"\t{user_init_structure}")
        invoke_params = [user_init_structure]+state.current_task_messages[-1]+[prepare_molecule_prompts]
        logger.debug(f"Invoke parameters: {invoke_params}")
        response = prepare_molecule_agent.invoke(invoke_params) 
        function_name = response.tool_calls[0]['name']
        logger.debug("Response from the molecule agent:")
        logger.debug("\t"+response.content)
        logger.debug(f"\tmessage to tool: {[response]}")
        output = prepare_molecule_tool.invoke({"messages":[response]})["messages"][-1].content
        logger.debug("Response from the molecule tool:")
        logger.debug("\t"+output)
        output = ast.literal_eval(output)

        if function_name == "get_molecule_from_file": 
            source_path = output['molecule_file_name']
            if not os.path.exists(source_path):
                error_message = f"Molecule file path does not exist: {source_path}"
                logger.error(error_message)
                return {
                    "error": error_message,
                    "has_error": True,
                    "molecule_file_name": 'no_type',
                    "messages": [AIMessage(content=error_message)],
                    "messages_to_user": [AIMessage(content=error_message)],
                }
            
            if os.path.isdir(source_path):
                logger.warning(f"Path is a directory, searching for molecule files: {source_path}")
                molecule_files = []
                for ext in ['.xyz', '.json', '.pdb', '.mol', '.sdf']:
                    molecule_files.extend(glob.glob(os.path.join(source_path, f"*{ext}")))
                
                if not molecule_files:
                    error_message = f"No molecule files found in directory: {source_path}. Please provide a direct file path."
                    logger.error(error_message)
                    return {
                        "error": error_message,
                        "has_error": True,
                        "molecule_file_name": 'no_type',
                        "messages": [AIMessage(content=error_message)],
                        "messages_to_user": [AIMessage(content=error_message)],
                    }
                
                # Use the first found molecule file
                source_path = molecule_files[0]
                logger.info(f"Found molecule file in directory: {source_path}")
            
            # Use shutil.copy2 instead of os.system for better error handling
            try:
                import shutil
                dest_path = os.path.join(state.working_directory, os.path.basename(source_path))
                shutil.copy2(source_path, dest_path)
                filename = dest_path
                logger.debug(f"Successfully copied molecule file from {source_path} to {dest_path}")
            except Exception as e:
                error_message = f"Failed to copy molecule file: {str(e)}"
                logger.error(error_message)
                return {
                    "error": error_message,
                    "has_error": True,
                    "molecule_file_name": 'no_type',
                    "messages": [AIMessage(content=error_message)],
                    "messages_to_user": [AIMessage(content=error_message)],
                }
        elif function_name == "get_molecule_from_database":
            response = seperate_atom_and_molecule_agent.invoke(state.current_task_messages[-1])
            output_judge_atom = seperate_atom_and_molecule_tool.invoke({"messages":[response]})["messages"][-1].content
            output_judge_atom = ast.literal_eval(output_judge_atom)
            logger.info(output_judge_atom)
            if 'atom' in output_judge_atom:
                message = f"The target is an atom: {output_judge_atom['atom']}. Successfully stored in ."
                logger.debug(message)
                atom_name = output_judge_atom['atom']
                # Clean quotes from atom name to avoid file names like 'H.xyz'
                atom_name = atom_name.strip("'\"").strip()
                with open(os.path.join(state.working_directory,f'{atom_name}.xyz'),'w') as f:
                    f.write(f"1\n\n{atom_name} 0.00000000 0.00000000 0.00000000\n")
                filename = os.path.join(state.working_directory,f'{atom_name}.xyz')
            
            else:     
                pubchem_msg = get_xyz_from_pubchem(output['molecule_file_name'],state.working_directory) 
                if pubchem_msg is None:
                    rdkit_msg = get_xyz_from_rdkit(output['molecule_file_name'],state.working_directory)
                    if rdkit_msg is None:
                        GlobalFlag.set_flag('molecule_filename', None)
                        interrupt_msg = interrupt({
                            "messages_to_user":  f"Fail to retrieve {output['molecule_file_name']} from database\nPlease prepare the molecule file by yourself and provide the file path.\nSupported formats: .xyz, .json.",
                            "options": {
                                "type": "file",
                                "must_exist": True,
                                "extensions": [".xyz", ".json"]
                            }
                        })
                        user_inp = interrupt_msg['messages'].content
                        restore_msg = SystemMessage(content=f"User has provided information containing the target molecular structure at: {user_inp}. Use this file for the calculation.")
                        response = prepare_molecule_agent.invoke([restore_msg]) 
                        output = prepare_molecule_tool.invoke({"messages":[response]})["messages"][-1].content
                        output = ast.literal_eval(output)
                        if os.path.exists(output['molecule_file_name']) and os.path.isfile(output['molecule_file_name']):
                            try:
                                import shutil
                                source_path = output['molecule_file_name']
                                dest_path = os.path.join(state.working_directory, os.path.basename(source_path))
                                if source_path != dest_path:
                                    shutil.copy2(source_path, dest_path)
                                filename = dest_path
                                logger.debug(f"Successfully copied molecule file from {source_path} to {dest_path}")
                            except Exception as e:
                                error_message = f"Failed to copy molecule file: {str(e)}"
                                logger.error(error_message)
                                return {
                                    "error": error_message,
                                    "has_error": True,
                                    "molecule_file_name": 'no_type',
                                    "messages": [AIMessage(content=error_message)],
                                    "messages_to_user": [AIMessage(content=error_message)],
                                }
                        else:
                            message = 'The file path you provided could not be processed. Please ensure:\n• The file exists at the specified location\n• The file format is supported (.xyz or .json)\n• You have provided a file path, not a directory path'
                            return  {"molecule_file_name": 'no_type', "messages":[AIMessage(content=message)],"messages_to_user":[AIMessage(content=message)]}
                    else:
                        message = 'Successfully retrive molecule in rdkit'
                        filename = rdkit_msg
                else:
                    message = 'Successfully retrived molecule in pubchem'
                    filename = pubchem_msg
        GlobalFlag.set_flag('molecule_filename', filename)
                
    filename = GlobalFlag.get_flag('molecule_filename')
    check_structure_result = check_molecule_structure(file = filename, current_task = state.current_task_messages[-1], user_original_query =state.messages[0], working_directory=state.working_directory)
    if check_structure_result['correct']:
        logger.info("The molecule structure has been confirmed correct.")
    else:

        filename = check_structure_result['filename']
        GlobalFlag.set_flag('molecule_filename', filename)
        _ = interrupt({
            "messages_to_user":  f"Based on my check, the molecular structure in this file may have potential issues. This is just a notification. Please directly inspect <Path>{filename}</Path>. If you find any problem, modify the structure directly in this file. I will keep monitoring it. If no modifications are made, I will assume the current structure is confirmed and continue the calculation. Once you finish editing, simply confirm that the modification is done.",
            "options": {
                "type": "file",
                "must_exist": True,
                "extensions": [".xyz", ".json"]
            }
        })


    # Check charge and multiplicity
    logger.info("Start getting charge and multiplicity of the molecule")
    response = extract_charge_multiplicity_agent.invoke(state.current_task_messages[-1])
    logger.debug("response from extract charge and multiplicity agent:")
    logger.debug("\t"+response.content)
    
    output = extract_charge_multiplicity_tool.invoke({"messages":[response]})["messages"][-1].content 
    
    logger.debug("response from extract charge and multiplicity tool:")
    logger.debug("\t"+output)
    try:
        output = json.loads(output)
    except Exception as e:
        logger.error("LLM can not prase JSON for charge and multiplicity, use default value:", output)
        output = {'charge':0, 'multiplicity':1}

    charge = output["charge"]
    multiplicity = output["multiplicity"]

    # For .xyz file
    if filename[-4:] == '.xyz':
        mol = ml.data.molecule.load(filename,format='xyz')
        if charge is not None:
            mol.charge = charge 
        else:
            mol.charge = 0 
        logger.debug(f"The charge of the molecule is set to {mol.charge}")
        if multiplicity is not None:
            mol.multiplicity = multiplicity 
        else:
            Nelectrons = np.sum(mol.atomic_numbers) - charge
            mol.multiplicity = (Nelectrons%2 + 1) % 2 
        logger.debug(f"The multiplicity of the molecule is set to {mol.multiplicity}")
        filename = filename[:-4]+'.json'
    # For .json file
    elif filename[-5:] == '.json':
        mol = ml.data.molecule.load(filename,format='json')
        if charge is not None:
            mol.charge = charge 
            logger.debug(f"The charge of the molecule is set to {charge}")
        if multiplicity is not None:
            mol.multiplicity = multiplicity 
            logger.debug(f"The multiplicity of the molecule is set to {multiplicity}")
            
    # Double check the multiplicity
    charge = mol.charge 
    multiplicity = mol.multiplicity
    Nelectrons = np.sum(mol.atomic_numbers) - charge
    SS = Nelectrons % 2 + 1
    if SS%2 != mol.multiplicity%2:
        if mol.multiplicity == 1:
            mol.multiplicity += 1
        else:
            mol.multiplicity -= 1
        warning_message = f"The number of electrons ({Nelectrons}) and multiplicity ({multiplicity}) do not match\n"
        warning_message += f"Change the multiplicity of the molecule to {mol.multiplicity}."
        logger.debug(warning_message)
        message += "\n"+warning_message
    
    mol.dump(filename,format='json')

    output_state = {
        "molecule_file_name": filename, "messages_to_user":[AIMessage(content="Start preparing molecule")], "messages":[AIMessage(content=message)]}   
    logger.debug("Output state:"); pretty_dict(output_state, logger)
    GlobalFlag.set_flag('molecule_filename', None)
    return output_state

prepare_molecule_builder = StateGraph(PrepareMoleculeState)
prepare_molecule_builder.add_node("prepare_molecule_node",prepare_molecule_node)
prepare_molecule_builder.add_edge(START,"prepare_molecule_node")
prepare_molecule_builder.add_conditional_edges(
    'prepare_molecule_node',
        lambda state: 'prepare_molecule_node' if state.molecule_file_name == "no_type" else END
)

prepare_molecule_graph = prepare_molecule_builder.compile()

from .agent_template import BaseAgent 
molecule_retrive_agent = BaseAgent(
    name="molecule_retrive_agent",
    description="prepare molecule for user based on their need",
    graph=prepare_molecule_builder,
)