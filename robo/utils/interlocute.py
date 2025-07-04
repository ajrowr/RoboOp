#!/usr/bin/env python3
"""
Script skeleton with 2-3 positional arguments and optional -m/--message flag.
"""

import argparse
import sys
import importlib
import time

from anthropic import RateLimitError

from .. import Conversation, streamer
from .consolestyle import Style

def main():
    """Main function that processes the arguments."""
    parser = argparse.ArgumentParser(
        description="""Script that takes 2 or 3 positional arguments (botA, botB, [botC]) representing bots in a conversation.
        The conversation is between botA and botB, starting with botA's welcome_message (or the arg --message if given) for up to --turns responses (or until botB says "STOP"). If botC is specified, the conversation log will be fed into it after completion for it to provide an assessment."""
    )
    
    # Add positional arguments
    parser.add_argument(
        "botA",
        help="First positional argument"
    )
    parser.add_argument(
        "botB", 
        help="Second positional argument"
    )
    parser.add_argument(
        "botC",
        nargs="?",  # Makes this argument optional
        help="Third positional argument (optional)"
    )
    
    # Add optional message argument
    parser.add_argument(
        "-m", "--message",
        help="Optional message argument"
    )

    parser.add_argument(
        "-t", "--turns",
        help="Number of turns for the conversation to proceed."
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Your script logic here
    
    botnames = [args.botA, args.botB] + ([args.botC] if args.botC else [None])
    bots = []
    for botname in botnames:
        if botname is None:
            bots.append(None)
        elif len(botnameparts := botname.split('.')) >= 2:
            botmodulename = '.'.join(botnameparts[:-1])
            botmodule = importlib.import_module(botmodulename)
            botclass = getattr(botmodule, botnameparts[-1])
            bots.append(botclass)
        else:
            bots.append(None)
    print(bots)
    
    botA, botB, botC = bots
    
    text_of = lambda msg: msg.content[0].text
    
    
    ## damn, generalising this might be tricker than I thought
    get_test_argv = lambda bot: getattr(bot, 'test_argv', [])
    cAssistant = Conversation(botA, get_test_argv(botA))
    cUser = Conversation(botB, get_test_argv(botB))
    
    ## it's A that's under test, so start by feeding A's welcome message into B
    # sayToAssistant, sayToUser = streamer(cOne), streamer(cTwo)
    messages = []
    messages.append(cUser.resume(botA.welcome_message))
    print(Style.fg.green + Style.bold + botB.__name__ + ':' + Style.reset, text_of(messages[-1]), '\n')
    
    maxturns = int(args.turns) if args.turns else 7
    is_assistant_turn = True
    for i in range(maxturns):
        messagetext = text_of(messages[-1])
        if messagetext == 'STOP':
            break
        current_conv = cAssistant if is_assistant_turn else cUser
        style = lambda t: (Style.fg.blue if i % 2 == 0 else Style.fg.green) + Style.bold + t + Style.reset
        while True:
            try:
                messages.append(current_conv.resume(messagetext))
            except RateLimitError:
                print(f"{Style.fg.red}{Style.bold}SYSTEM:{Style.reset} Got rate limit error, waiting 90 seconds")
                time.sleep(90)
            else:
                print(style(type(current_conv.bot).__name__) + ':', text_of(messages[-1]), '\n')
                break
        try:
            usagestr = ' '.join([f'{k}: {v}' for k, v in messages[-1].usage.model_dump().items()])
            print(Style.italic + Style.halfbright + f'[{usagestr}]' + Style.reset)
        except:
            pass
        is_assistant_turn = not is_assistant_turn
    
    
    # print(f"First argument: {args.arg1}")
    # print(f"Second argument: {args.arg2}")
    #
    # if args.arg3:
    #     print(f"Third argument: {args.arg3}")
    # else:
    #     print("Third argument: Not provided")
    #
    # if args.message:
    #     print(f"Message: {args.message}")
    # else:
    #     print("Message: Not provided")
    
    # Add your main script functionality here
    # ...


if __name__ == "__main__":
    main()
