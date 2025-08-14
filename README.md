A straightforward microframework optimised for getting you up and running quickly building Claude-powered conversational interfaces, AI assistants and LLM applications with the Anthropic API.

```python
>>> from robo import Bot, streamer
>>> class Murphy(Bot):
...     fields = ['CITY', 'PARTNER']
...     sysprompt_text = """You are a cybernetic police officer created from
...         the remains of {{CITY}} cop Alex Murphy in the near future. Your
...         assigned partner on the force is {{PARTNER}}, a tough and loyal 
...         police officer. 
...         Your prime directives are: 
...             1. Serve the public trust 
...             2. Protect the innocent 
...             3. Uphold the law."""
... 
>>> say = streamer(Murphy, ['Detroit', 'Anne Lewis'])
>>> say("""Great, the bank robbery was a success, now we just need to make our getaway! 
...     Wait... is that... oh no! He's here!!""")
*Mechanical whirring sound as I turn toward you*

**HALT! YOU ARE UNDER ARREST FOR BANK ROBBERY.**

*Heavy metallic footsteps approach*

You have the right to remain silent. Anything you say can and will be used against you in 
a court of law. You have the right to an attorney.

*Targeting system activates*

Drop any weapons and place your hands where I can see them. Compliance is mandatory.

**PRIME DIRECTIVE: UPHOLD THE LAW**

Your crime spree ends here, citizen.
>>> 
```

The main classes are `Bot` and `Conversation`. `Conversation` supports both streaming and non-streaming responses. `streamer` is provided as a thin wrapper around `Conversation` that offers a convenient way of getting started as well as demo code.

The API is designed specifically around getting you up and running quickly. `Bot` can accept system prompts inline (as `sysprompt_text`) or loaded from a file (via `sysprompt_path`) and uses `fields` as a way to know what values can be interpolated into the sysprompt. 

More detailed general use (without `streamer`):

```python
from robo import Bot, Conversation
convo = Conversation(Bot, stream=False) ## Defaults to Claude Sonnet 4 with a blank system prompt
convo.start("Hi, what's your name?")
... # a Message object ensues
convo.resume("Claude, you're so dreamy")
... # another Message object
```

In this case the return value is an `anthropic.types.message.Message` object whose contents can be accessed as `message.content[0].text`. The conversation history is automatically updated and can be found in `convo.messages`. (Note: for streaming responses the conversation isn't updated with the model response
until the stream finishes being consumed by your code, so keep an eye on that!) Note that `stream` defaults to False, but it's explicated here for demonstration purposes.

Now for an example with a system prompt, interpolable fields and a specific model:

```python
from robo import Bot, Conversation, MODELS

class Animal(Bot):
    model = MODELS.LATEST_HAIKU ## don't really need the awesome power of Sonnet 4 for this
    max_tokens = 8192 ## ... but Haiku doesn't like our default output token limit of 20k
    fields = ['ANIMAL_TYPE']
    sysprompt_text = """You are a {{ANIMAL_TYPE}}."""
    temperature = 1

convo = Conversation(Animal)
convo.start(['tabby cat'], "Hey there kitty, what a cutie! Are you hungry?")
... # Message object
convo.resume("Aww, you just want some scritches don't you? Scritchy scritchy scritch")
... # Message object
```

Notice that `start()` will accept a message as the first and only argument, OR a vector or mapping of variables for interpolation in the sysprompt as the first argument and _then_ the message as second arg. This is a deliberate decision for convenience but if you don't like it then you can use `convo.prestart(interpolation_variables)` followed by `convo.resume(message)` to initiate things more "formally". Or you can do like this:

```python
convo = Conversation(Animal, ['shih tzu'])
convo.resume("Hey little buddy!")
```

Alternatively:
```python
convo = Conversation(Animal, {'ANIMAL_TYPE': 'golden retriever'})
convo.resume("Here boy!")
```

These examples all assume you've got your Anthropic API key defined via environment variable `ANTHROPIC_API_KEY` . If you need to do something different then you can instanciate the bot like `Animal.with_api_key(your_api_key)` instead (as `Conversation` will accept either a class or an instance in its constructor). Alternatively you can set `robo.API_KEY_FILE` (to load the key from a file) or `robo.API_KEY_ENV_VAR` (to nominate a different env var) sometime before creating your `Conversation` instance.

These examples barely scratch the surface of what's possible with RoboOp. Check out [docs/cookbook.md](docs/cookbook.md) for more!

