"""
    Agent cards that save all the information of available agents

"""

class AgentCard:

    def __init__(
            self, 
            name=None, 
            description=None, 
            description_for_planner=None, 
            entry=None, 
            graph_name=None
        ):
        self.name = name 
        self.description = description
        self.description_for_planner = description_for_planner
        self.entry = entry 
        self.graph_name = graph_name
        
    def __str__(self):
        width = 60
        border = "=" * width
        lines = [self.name, self.description, self.entry]
        centered_lines = [line.center(width) for line in lines]
        return "\n".join([border]+centered_lines+[border])


agent_cards = {
    "planner_agent": AgentCard(
        name = "planner_agent",
        description="planner",
        entry="aitomia_agents.planner",
        graph_name="planner_agent"
    ),

    "task_agent": AgentCard(
        name="task_agent", 
        description="handle different tasks",
        entry="aitomia_agents.task",
        graph_name="task_agent"
    ),

    "sp_agent": AgentCard(
        name="single_point_agent", 
        description="perform single point calculation",
        description_for_planner="Perform the single point calculation of a molecular structure to get the electronic energy.",
        entry="aitomia_agents.single_point",
        graph_name="single_point_agent"
    ),

    "uvvis_spc_agent": AgentCard(
        name="uvvis_spc_agent",
        description="performs static UV-vis spectrum calculation",
        description_for_planner="Performs static EXCITED-STATE calculation and plots UV-vis spectrum for a molecule structure.",
        entry="aitomia_agents.excited_state_agents.uvvis_spc",
        graph_name="uvvis_spc_agent",
    ),

    "uvvis_agent": AgentCard(
        name="uvvis_planner_agent",
        description="Calculate the UV-vis spectrum for a molecule.",
        description_for_planner="Calculate the UV-vis spectrum for a molecule.",
        entry="aitomia_agents.excited_state_agents.uvvis_planner",
        graph_name="uvvis_planner_agent",
    ),

    "geomopt_agent": AgentCard(
        name="geomopt_agent",
        description="perform geometry optimization",
        description_for_planner="Perform geometry optimization of a molecular structure and optimize it to the local minimum.",
        entry="aitomia_agents.optimize_geometry",
        graph_name="geomopt_agent"
    ),

    "ts_agent": AgentCard(
        name="ts_agent",
        description="perform transition state search",
        description_for_planner="Perform transition state search of a molecular structure, i.e., optimize the molecule to the saddle point.",
        entry="aitomia_agents.transition_state",
        graph_name="ts_agent",
    ),

    "freq_agent": AgentCard(
        name = "freq_agent",
        description="perform frequency calculation",
        description_for_planner="Perform frequency calculation of a molecular structure, usually an optimized molecule, to get the frequencies and thermodynamic properties.",
        entry='aitomia_agents.frequency',
        graph_name="freq_agent",
    ),

    "ir_static_agent": AgentCard(
        name = "ir_static_agent",
        description="perform static infrared (IR) spectrum calculation",
        description_for_planner="Perform static infrared (IR) intensity calculation of a molecular structure, usually an optimized molecule, to get the IR spectrum.",
        entry='aitomia_agents.ir_static',
        graph_name="ir_static_agent",
    ),
    
    "ir_agent": AgentCard(
        name = "ir_static_agent",
        description="perform static infrared (IR) spectrum calculation",
        description_for_planner="Calculate the infrared (IR) spectrum of a molecule.",
        entry='aitomia_agents.ir',
        graph_name="ir_agent",
    ),

    "raman_static_agent": AgentCard(
        name = "raman_static_agent",
        description="perform raman spectrum calculation",
        description_for_planner="Perform Raman intensity calculation of a molecular structure, usually an optimized molecule, to get the Raman spectrum.",
        entry="aitomia_agents.raman_static",
        graph_name="raman_static_agent",
    ),
    
    "raman_agent": AgentCard(
        name = "raman_agent",
        description="perform raman spectrum calculation",
        description_for_planner="Calculate the Raman spectrum of a molecule.",
        entry="aitomia_agents.raman",
        graph_name="raman_agent",
    ),

    "reaction_agent": AgentCard(
        name = "reaction_agent",
        description="perform reaction calculation",
        description_for_planner="Perform the calculation of a chemical reaction. This task already contains the geometry optimization and frequency calculation of each reactant and product.",
        entry='aitomia_agents.reaction',
        graph_name='reaction_agent',
    ),

    "irc_agent": AgentCard(
        name="IRC_agent",
        description="perform IRC calculation",
        description_for_planner="Perform the intrinsic reaction coordinate (IRC) calculation, which needs to start from a transition state of a reaction.",
        entry="aitomia_agents.irc",
        graph_name="irc_agent",
    ),

    #no calculation agent
    "molecule_retrive_agent": AgentCard(
        name="molecule_retrive_agent", 
        description_for_planner="get molecule based on user's need.",
        entry="aitomia_agents.prepare_molecule",
        graph_name="molecule_retrive_agent"
    ),

    "chat_agent": AgentCard(
        name="chat_agent",
        description="chat with the user",
        description_for_planner="Chat with the user and answer the user's question.",
        entry="aitomia_agents.chat",
        graph_name="chat_agent"
    ),
}