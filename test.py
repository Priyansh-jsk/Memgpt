import memgpt
from memgpt.config import Config

from memgpt.agent import Agent 
from memgpt.main import run_agent_loop
import gradio as gr

import memgpt.presets.presets as presets
from memgpt.persistence_manager import InMemoryStateManager

from memgpt.interface import CLIInterface as interface
import memgpt.personas.personas as personas
import memgpt.humans.humans as humans

DEFAULT_MEMGPT_MODEL = "gpt-4"
DEFAULT_PRESET = "memgpt_chat" 

DEFAULT = "sam_pov"
memgpt_persona = (
    DEFAULT,
    None,
)

DEFAULT = "cs_phd"
cfg = Config.legacy_flags_init(
    DEFAULT_MEMGPT_MODEL,
    memgpt_persona,
    DEFAULT,
)

human_persona = """This is what I know so far about the user, I should expand this as I learn more about them.

First name: Chad
Last name: ?
Gender: Male
Age: ?
Nationality: ?
Occupation: Computer science PhD student at UC Berkeley
Interests: Formula 1, Sailing, Taste of the Himalayas Restaurant in Berkeley, CSGO"""

persistence_manager = InMemoryStateManager()
print(human_persona)
print(persistence_manager)

def initialize_memgpt_agent():
    # Here I use the memgpt_agent with required parameters
    memgpt_agent = presets.use_preset(
        presets.DEFAULT_PRESET,
        None,
        cfg.model,
        human_persona,
        interface,
        persistence_manager,
        human_persona,
    )

    print_messages = interface.print_messages
    print("Human persona:", human_persona)
    print_messages(memgpt_agent.messages)
    return memgpt_agent

def memgpt_interface(self, input_text):
    
    global memgpt_agent

    # run the agent loop with the user input
    run_agent_loop(memgpt_agent, first=True, no_verify=True)

    # set assistant's response
    assistant_response = memgpt_agent.messages[-1]['content']
    return assistant_response

# Initialize MemGPT agent
memgpt_agent = initialize_memgpt_agent()

# Gradio interface
iface = gr.Interface(
    fn=memgpt_interface,
    inputs=gr.Textbox(),
    outputs=gr.Textbox(),
    ive=True,
    capture_session=True
)

iface.launch()

