
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

There are two basic types of responses supplied by the Anthropic API - complete messages (which we call "flat" responses) - in which the entire response is composed before being made available as an object - and streaming responses, in which you get to see the response come in chunk-by-chunk, which is generally preferable for conversational scenarios.

### `streamer()` and other convenience tools

For convenience, the `robo` package includes a function called `streamer` which makes it easy to set up a conversation and stream responses. 

```python
>>> say = streamer(Bot)
>>> say("Hello!")
<a response from Claude ensues>
```

This also allows passing in of values for fields, as demonstrated in the "Fields" section below. Alternatively to passing a `Bot` class as first argument, you can also pass `streamer` an _instance_ of `Bot` or one of its subclasses; or an instance of a subclass of `Conversation` (in which case, make sure that it's been instanciated with `stream=True`).

While streaming generally gives the best user experience for conversational applications, it can (absent `streamer`) be a bit more technical to work with so the default is "flat" mode, in which the fully-generated message is returned as an `anthropic.types.message.Message` object passed directly through from the underlying Anthropic API. This object contains additional info that may be useful for debugging. You can print the text content from this object easily with `printmsg`:

```python
>>> convo = Conversation(Bot, [])
>>> msg = convo.resume("Who are you?")
>>> printmsg(msg)
I'm Claude, an AI assistant created by Anthropic. I'm here to help with a wide variety of tasks like
answering questions, helping with analysis and research, creative projects, math and coding, and 
having conversations. Is there something specific I can help you with today?
>>> print(msg.usage)
Usage(cache_creation_input_tokens=0, cache_read_input_tokens=0, input_tokens=11, 
      output_tokens=60, server_tool_use=None, service_tier='standard')
```

We also have `gettext(msg)` which returns a `str` and `getjson(msg)` which parses JSON from the message content.

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

One more thing, if you wish to use method 4 to begin a conversation with a bot that doesn't take parameters, you'll need to pass an empty list (as the presence of that argument is what triggers the automatic prestart):

```python
>>> conv = Conversation(Bot, [])
```

And now, on to the new stuff!

## One-shot

As mentioned earlier, a typical conversation with an LLM proceeds by feeding the pre-existing conversation back into the model with a new user message appended. But some bots don't actually need that context (eg. bots that do one thing and one thing only, and whose behaviour is specified by the system prompt). For such bots, feeding in the prior conversational context is not only a waste of tokens (and hence, money) but can also confuse the bot. That's where `oneshot` comes in - setting a bot as one-shot bypasses these concerns by preventing conversational context from being included with a request.

```python
from robo import *

class YesMan(Bot):
    sysprompt_text = """You respond with either 'Yes!' or 'No!', whichever is the opposite of what 
                        you said last turn. Always say 'Yes!' on the first turn."""
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
    sysprompt_text = """You are a Google Review assessment assistant. Your task is to consider 
                        social media reviews for a restaurant and provide an estimate of how many 
                        stars (out of five) the review would seem to indicate, in the format 
                        {"stars_count": <number of stars>}. Do not add any explanations, context, 
                        or additional text before or after this response."""
    oneshot = True

>>> convo = Conversation(ReviewAssessmentBot, [])
>>> getjson(convo.resume("""Absolutely delicious, would definitely dine here again"""))
{'stars_count': 5}
>>> getjson(convo.resume("""Food was okay but the service was kinda slow despite the place being half empty."""))
{'stars_count': 3}
>>> getjson(convo.resume("""The staff were rude and I found a slug in my salad! I won't be back!"""))
{'stars_count': 1}
```

## Dynamic system prompts

So far we have seen system prompts that consist of inline text, which is fine for simple bots, but for more sophisticated applications you may want more flexibility about how to specify your sysprompts.

A basic form of this is `sysprompt_path`, which lets you read the system prompt from a textfile:

```python
class SyspromptFromFileBot(Bot):
    sysprompt_path = '/path/to/sysprompt.txt'
```

System prompts can quickly get large and elaborate, so defining them as inline text in a class can become unwieldy, in which case using a textfile in this way is a helpful option. But Claude also supports system prompts as structured data, which can unlock powerful features such as prompt caching and knowledge base integration. That's where `sysprompt_generate()` comes in.

```python
from pathlib import Path

class LiteraryAssistant(Bot):
    def sysprompt_generate(self): # Overriding the abstract method in Bot
        sysprompt_segments = [
            self._make_sysprompt_segment("""You are a knowledgeable literary assistant. The text 
                of a well-known novella is provided to you so that you can answer questions about it.""")
        ]
        datadir = Path('/path/to/data/')
        noveltext = (datadir / 'franz-kafka-metamorphosis.txt').read_text()
        sysprompt_segments.append(self._make_sysprompt_segment(noveltext, set_cache_control=True))
        return sysprompt_segments

>>> say = streamer(LiteraryAssistant)
>>> say("Man, this guy just can't catch a break huh?")
You're absolutely right! Gregor Samsa's situation is relentlessly bleak from start to finish. He wakes up transformed into a giant insect, and instead of things getting better or him finding some way to adapt, everything just spirals downward.

What's particularly tragic is how his initial concerns are so mundane and human - he's worried about 
being late for work and losing his job, even while dealing with having multiple legs and an insect 
body. Then his family, who he's been supporting financially, gradually shifts from shock and concern 
to viewing him as a burden and eventually as something that needs to be eliminated.

The "breaks" he does get are pitiful - like when his sister brings him food scraps, or when he gets 
to listen to her violin playing. Even those small comforts are eventually taken away. By the end, 
he's literally pelted with apples by his own father and left to waste away, ignored and unloved.

Kafka really doesn't offer Gregor any redemption or escape - it's a masterclass in depicting someone 
trapped in an impossible situation that only gets worse. The story's power comes partly from how 
thoroughly hopeless Gregor's circumstances become, making it a perfect example of Kafka's absurdist 
vision of modern alienation.
```

There are a few things here worth noting. `Bot._make_sysprompt_segment(...)` has been provided as a convenient way of generating the correct structure for a multi-segment text-only system prompt. The `set_cache_control` argument, when set `True`, adds cache control clauses to the segment that enable system prompt caching. For large system prompts, prompt caching is highly recommended as it can significantly reduce costs. Further recommended reading is [here](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) but essentially, cache control clauses are best placed on the last segment of the system prompt that is unlikely to change frequently between requests.

# More to come, watch this space! :)
