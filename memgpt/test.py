from memgpt.config import Config
import memgpt.presets.presets as presets
from memgpt.persistence_manager import (
    InMemoryStateManager,
)
from memgpt.interface import CLIInterface as interface
import memgpt.personas.personas as personas
import memgpt.humans.humans as humans
from memgpt.humans import get_human_text

import memgpt.utils as utils
import questionary

DEFAULT_MEMGPT_MODEL = "gpt-4"
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

chosen_human = cfg.human_persona
chosen_persona = cfg.memgpt_persona

persistence_manager = InMemoryStateManager()

memgpt_agent = presets.use_preset(
    presets.DEFAULT_PRESET,
    None,  # no agent config to provide
    cfg.model,
    personas.get_persona_text(*chosen_persona),
    humans.get_human_text(*chosen_human),
    interface,
    persistence_manager,
)
print_messages = interface.print_messages
print_messages(memgpt_agent.messages)

    
# data_list = utils.read_database_as_list(cfg.archival_storage_files)
# user_message = f"Your archival memory has been loaded with a SQL database called {data_list[0]}, which contains schema {data_list[1]}. Remember to refer to this first while answering any user questions!"
# for row in data_list:
#     memgpt_agent.persistence_manager.archival_memory.insert(row)
#     print(f"Database loaded into archival memory.")

#     if cfg.agent_save_file:
#         load_save_file = questionary.confirm(f"Load in saved agent '{cfg.agent_save_file}'?").ask()
#         if load_save_file:
#             load(memgpt_agent, cfg.agent_save_file)

memgpt_agent.model = "gpt-3.5-turbo-16k"


def get_memgpt_memory(userId, chatbotId):
    # Check if the memory exists in the database
    memory = get_memory_from_database(userId, chatbotId)
    if memory is None:
        # Create a new memory object
        # memory = MemGPTObject(userId, chatbotId)
        # Save the memory object to the database
        save_memory_to_database(memory)
    return memory

# Reference- https://github.com/PromptEngineer48/MemGPT-AutoGEN-LLM/blob/main/app.py
# https://github.com/PromptEngineer48?tab=repositories

# https://www.cs.utexas.edu/~gdurrett/courses/online-course/materials.html
# https://github.com/PromptEngineer48/MemGPT-AutoGEN-LLM/blob/main/app.py



