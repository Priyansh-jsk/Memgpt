from asyncio import constants
import json
from os import system

from pytest import console_main
from wandb import agent
import memgpt
from memgpt.config import Config

from memgpt.agent import Agent 
from memgpt.main import run_agent_loop
import gradio as gr

import memgpt.presets.presets as presets
from memgpt.persistence_manager import InMemoryStateManager

from memgpt.interface import STRIP_UI, CLIInterface as interface
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
print('PERSISTENCE_MANAGER: ')
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

def memgpt_interface(input_text):
    
    global memgpt_agent

    user_input = input_text

    user_input = user_input.rstrip()

    if user_input.startswith("!"):
        print(f"Commands for CLI begin with '/' not '!'")
        return

    if user_input == "":
        # no empty messages allowed
        print("Empty input received. Try again!")
        return

    # Handle CLI commands
    # Commands to not get passed as input to MemGPT
    if user_input.startswith("/"):
      
        # updated agent save functions
        if user_input.lower() == "/exit":
            memgpt_agent.save()
            return
        elif user_input.lower() == "/save" or user_input.lower() == "/savechat":
            memgpt_agent.save()
            return

        if user_input.lower() == "/dump" or user_input.lower().startswith("/dump "):
            # Check if there's an additional argument that's an integer
            command = user_input.strip().split()
            amount = int(command[1]) if len(command) > 1 and command[1].isdigit() else 0
            if amount == 0:
                interface.print_messages(memgpt_agent.messages, dump=True)
            else:
                interface.print_messages(memgpt_agent.messages[-min(amount, len(memgpt_agent.messages)) :], dump=True)
            return

        elif user_input.lower() == "/dumpraw":
            interface.print_messages_raw(memgpt_agent.messages)
            return

        elif user_input.lower() == "/memory":
            print(f"\nDumping memory contents:\n")
            print(f"{str(memgpt_agent.memory)}")
            print(f"{str(memgpt_agent.persistence_manager.archival_memory)}")
            print(f"{str(memgpt_agent.persistence_manager.recall_memory)}")
            return

        elif user_input.lower() == "/model":
            if memgpt_agent.model == "gpt-4":
                memgpt_agent.model = "gpt-3.5-turbo-16k"
            elif memgpt_agent.model == "gpt-3.5-turbo-16k":
                memgpt_agent.model = "gpt-4"
            print(f"Updated model to:\n{str(memgpt_agent.model)}")
            return

        elif user_input.lower() == "/pop" or user_input.lower().startswith("/pop "):
            # Check if there's an additional argument that's an integer
            command = user_input.strip().split()
            amount = int(command[1]) if len(command) > 1 and command[1].isdigit() else 3
            print(f"Popping last {amount} messages from stack")
            for _ in range(min(amount, len(memgpt_agent.messages))):
                memgpt_agent.messages.pop()
            return

        elif user_input.lower() == "/retry":
            # TODO this needs to also modify the persistence manager
            print(f"Retrying for another answer")
            while len(memgpt_agent.messages) > 0:
                if memgpt_agent.messages[-1].get("role") == "user":
                    # we want to pop up to the last user message and send it again
                    user_message = memgpt_agent.messages[-1].get("content")
                    memgpt_agent.messages.pop()
                    break
                memgpt_agent.messages.pop()

        elif user_input.lower() == "/rethink" or user_input.lower().startswith("/rethink "):
            # TODO this needs to also modify the persistence manager
            if len(user_input) < len("/rethink "):
                print("Missing text after the command")
                return
            for x in range(len(memgpt_agent.messages) - 1, 0, -1):
                if memgpt_agent.messages[x].get("role") == "assistant":
                    text = user_input[len("/rethink ") :].strip()
                    memgpt_agent.messages[x].update({"content": text})
                    break
            return

        elif user_input.lower() == "/rewrite" or user_input.lower().startswith("/rewrite "):
            # TODO this needs to also modify the persistence manager
            if len(user_input) < len("/rewrite "):
                print("Missing text after the command")
                return
            for x in range(len(memgpt_agent.messages) - 1, 0, -1):
                if memgpt_agent.messages[x].get("role") == "assistant":
                    text = user_input[len("/rewrite ") :].strip()
                    args = json.loads(memgpt_agent.messages[x].get("function_call").get("arguments"))
                    args["message"] = text
                    memgpt_agent.messages[x].get("function_call").update({"arguments": json.dumps(args)})
                    break
            return

        # No skip options
        elif user_input.lower() == "/wipe":
            memgpt_agent = agent.Agent(interface)
            user_message = None

        elif user_input.lower() == "/heartbeat":
            user_message = system.get_heartbeat()

        elif user_input.lower() == "/memorywarning":
            user_message = system.get_token_limit_warning()

        elif user_input.lower() == "//":
            multiline_input = not multiline_input
            return

        elif user_input.lower() == "/" or user_input.lower() == "/help":
            return

        else:
            print(f"Unrecognized command: {user_input}")
            return

    else:
        # If message did not begin with command prefix, pass inputs to MemGPT
        # Handle user message and append to messages
        user_message = user_input
    

        skip_next_user_input = False

        def process_agent_step(user_message, no_verify):
            new_messages, heartbeat_request, function_failed, token_warning = memgpt_agent.step(
                user_message, first_message=False, skip_verify=no_verify
            )

            skip_next_user_input = False
            if token_warning:
                user_message = system.get_token_limit_warning()
                skip_next_user_input = True
            elif function_failed:
                user_message = system.get_heartbeat(constants.FUNC_FAILED_HEARTBEAT_MESSAGE)
                skip_next_user_input = True
            elif heartbeat_request:
                user_message = system.get_heartbeat(constants.REQ_HEARTBEAT_MESSAGE)
                skip_next_user_input = True

            return new_messages, user_message, skip_next_user_input

        while True:
            try:
                if STRIP_UI:
                    new_messages, user_message, skip_next_user_input = process_agent_step(user_message, True)
                    break
                else:
                    with console_main.status("[bold cyan]Thinking...") as status:
                        new_messages, user_message, skip_next_user_input = process_agent_step(user_message, True)
                        break
            except Exception as e:
                break


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
    live=True
)

iface.launch()

