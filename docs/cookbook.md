
# Cookbook

Welcome to the Cookbook, where you can learn by doing with concrete examples of how to build with RoboOp.

## Basic concepts

RoboOp is designed to make building Claude-powered applications straightforward and intuitive. Whether you're creating simple chatbots or sophisticated AI agents, understanding the core concepts behind how LLMs work will help you make the most of the framework's capabilities and design better bots.

While a conversation with Claude or ChatGPT might seem to the user to be seamless, what's really happening is that every time the model is prompted to continue the conversation by a message from the user (a "user prompt"), in fact the entire conversation history is fed back in to the model and combined with the newest user message, to generate the next response. RoboOp handles all of this conversation management automatically, but understanding what's happening behind the scenes will help you make better decisions about bot design and optimize for performance and cost.

While many people tend to think of LLMs as just being chatbots, they actually can be used in almost unlimited ways. Many of these applications are achieved through providing custom "system prompts" to the LLM that can direct its behaviour in ways that the user prompt cannot - you can think of the system prompt as being the LLM's "programming" within the context of a request.

Some terminology:
- LLM: Large Language Model. Examples include ChatGPT, Gemini (by Google), LLaMa (by Facebook) and Anthropic Inc's Claude (which RoboOp is designed to interface with).
- User prompt: a message sent from the user to the model that prompts the generation of a response.
- System prompt: a description of how the model should behave when responding to a user prompt, provided by the developer and invisible to (and unchangeable by) the user. Can include directives on persona, tone, output format, special capabilities and much more, as well as specialised information that the model may want or need to use in composing a response.
- Tools: functions that an LLM can call to act on behalf of, or retrieve additional information for, the user.
- Token: where humans tend to think in terms of words, LLMs process text as "tokens" - smaller chunks that might be whole words, parts of words, or even punctuation marks. For example, "understanding" might be split into "under" and "standing" as two tokens. This affects costs (you're often charged per token) and model limits (there's usually a maximum number of tokens the model can process at once).
- Turn: a single exchange in a conversation with an LLM, consisting of one user message and the model's response to it. For example, if you ask "What's the weather like?" and the model responds, that's one turn. The conversation history is often measured in turns, and some LLMs have limits on how many turns they can remember or process at once.

## Setting up

You'll need a developer account with Anthropic to get an API key for use with RoboOp. You can get this process underway at [https://console.anthropic.com/](https://console.anthropic.com/), you'll probably need to buy some credits but you can get started with as little as $5USD.

Once you have an API key you can add it to your add it to your environment like so (macOS and Linux, not sure about Windows sorry):

```sh
export ANTHROPIC_API_KEY="<the API key>"
```

To install RoboOp, download the .zip file from [here](https://github.com/ajrowr/RoboOp/archive/refs/heads/master.zip) and unzip it somewhere on your system. Then add it to your Python path:

```sh
export PYTHONPATH=$PYTHONPATH:/path/to/RoboOp
```

Then launch Python and you should be ready to go! The following examples assume that you start with:

```python
from robo import *
```

## Recap

Let's briefly revist the examples given in README.md .

The fundamental objects of RoboOp are `Bot` and `Conversation`. Almost every meaningful use of RoboOp starts by subclassing `Bot`; on the other hand, you'd probably be trying to do something pretty specialised before you'd need to consider subclassing `Conversation`.

### streamer() and other convenience tools

For convenience, the `robo` package includes a function called `streamer` which makes it easy to set up a conversation and stream responses. 

```python
>>> say = streamer(Bot)
>>> say("Hello!")
<a response from Claude ensues>
```

This also allows passing in of values for fields, as shown below. You can also pass `streamer` an instance of Bot or Conversation (including subclasses). 

Streaming generally gives the best user experience, but the default is "flat" mode where the entire message is returned as a single object once it has been completely generated, as an `anthropic.types.message.Message` object which contains additional info that may be useful for debugging. You can print the text content from this easily with `printmsg`:

```python
>>> convo = Conversation(Bot, [])
>>> msg = convo.resume('Hello!')
>>> printmsg(msg)
<text of the response message>
```

### Fields

Fields allow for interpolation of conversation-specific parameters into the system prompt.

```python
class PoetryBot(Bot):
    fields = ['POEM_TYPE']
    sysprompt_text = """A user will ask you about something. You respond with a {{POEM_TYPE}} about the subject."""

>>> say = streamer(PoetryBot, ['haiku'])
>>> say('New York')
Towering steel and glass,
Yellow cabs weave through the rush—
Dreams rise with the dawn.
>>> say = streamer(PoetryBot, ['limerick'])
>>> say('San Francisco')
There once was a city by the bay,
Where fog rolls in nearly each day,
With hills steep and grand,
And tech close at hand,
San Francisco's a sight, so they say!
>>> 
```

### Selecting specific models and setting output token limits

```python
from robo import Bot, MODELS, streamer
class AnimalBot(Bot):
    model = MODELS.LATEST_HAIKU
    max_tokens = 8192
    fields = ['ANIMAL_TYPE']
    sysprompt_text = """Respond with a stereotypical sound made by a {{ANIMAL_TYPE}}."""

>>> say = streamer(AnimalBot, ['duck'])
>>> say('hi')
Quack!
>>> 
```

Check `robo/models.py` for the models list, or just `dir(MODELS)` will do the trick. Default is `LATEST_SONNET` which represents a good tradeoff between price and performance for typical applications.

### Different ways of setting up a Conversation

There are a few different ways of setting up a `Conversation`, so you can use whichever feels most natural to you. Method 1 shows how to initiate a conversation with a bot that doesn't have any fields, and the others are roughly equivalent to each other.

```python
# Method 1
>>> conv1 = Conversation(Bot)
>>> printmsg(conv1.start('hello')) # If the bot doesn't have any fields that need values
Hello! How are you doing today? Is there anything I can help you with?
>>> printmsg(conv1.resume('hello'))
Hello again! I'm here if you'd like to chat about something or if there's anything specific you'd like help with. What's on your mind?

# Method 2
>>> conv2 = Conversation(AnimalBot)
>>> printmsg(conv2.start(['dog'], 'hello'))
Woof!
>>> printmsg(conv2.resume('hello'))
Woof! Bark!

# Method 3
>>> conv3 = Conversation(AnimalBot)
>>> conv3.prestart(['goose']) # Get the conversation ready for use without sending a message
>>> printmsg(conv3.resume('hello'))
Honk!

# Method 4
>>> conv4 = Conversation(AnimalBot, ['mouse']) # This prestarts the conversation automatically
>>> printmsg(conv4.resume('hello'))
Squeak!
```

And now, on to the new stuff!

## One-shot

As mentioned earlier, a typical conversation with an LLM proceeds by feeding the pre-existing conversation back into the model with a new user message appended. But some bots don't actually need that context (eg. bots that do one thing and one thing only, and whose behaviour is specified by the system prompt). For such bots, feeding in the prior conversational context is not only a waste of tokens (and hence, money) but can also confuse the bot. That's where `oneshot` comes in - setting a bot as one-shot bypasses these concerns by preventing conversational context from being included with a request.

```python
from robo import *

class YesMan(Bot):
    sysprompt_text = """You respond with either 'Yes!' or 'No!', whichever is the opposite of what you said last turn. Always say 'Yes!' on the first turn."""
    oneshot = True

>>> say = streamer(YesMan)
>>> say('hello')
Yes!
>>> say('hello')
Yes!
>>> YesMan.oneshot = False
>>> say = streamer(YesMan)
>>> say('hello')
Yes!
>>> say('hello')
No!
>>> 
```
There are myriad applications that simply need the LLM to assess a piece of input without maintaining the structure of a conversation, which is where `oneshot` comes in really handy. For example, sentiment analysis:

```python
class ReviewAssessmentBot(Bot):
    sysprompt_text = """You are a Google Review assessment assistant. Your task is to consider social media reviews for a restaurant and provide an estimate of how many stars (out of five) the review would seem to indicate, in the format {"stars_count": <number of stars>}. Do not add any explanations, context, or additional text before or after this response."""
    oneshot = True

>>> convo = Conversation(ReviewAssessmentBot, [])
>>> printmsg(convo.resume("""Absolutely delicious, would definitely dine here again"""))
{"stars_count": 5}
>>> printmsg(convo.resume("""Food was okay but the service was kinda slow despite the place being half empty."""))
{"stars_count": 3}
>>> printmsg(convo.resume("""The staff were rude and I found a slug in my salad! I won't be back!"""))
{"stars_count": 1}
```

# More to come, watch this space! :)
