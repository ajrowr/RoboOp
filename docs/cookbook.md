
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

You'll need a developer account with Anthropic to get an API key for use with RoboOp. You can get this process underway at [https://console.anthropic.com/](https://console.anthropic.com/). You'll probably need to buy some credits but you can get started with as little as $5USD.

Once you have an API key you can add it to your add it to your environment like so (macOS and Linux, not sure about Windows sorry):

```sh
export ANTHROPIC_API_KEY='<the API key>'
```

Alternatively you can load the key from a file:
```sh
export ROBO_API_KEY_FILE='<path to your API key file>'
```

To install RoboOp:
```sh
# installing with Pip:
pip install RoboOp

# adding to a project with Uv:
cd /your/project/dir
uv add RoboOp

# using Uv for a throwaway REPL:
uv run --with RoboOp -- python
```

Then launch Python and you should be ready to go! The following examples assume that you start with:

```python
from robo import *
```

## Recap

Let's briefly revist the examples given in README.md .

The fundamental objects of RoboOp are `Bot` and `Conversation`. Almost every meaningful use of RoboOp starts by subclassing `Bot`; on the other hand, you'd probably be trying to do something pretty specialised before you'd need to consider subclassing `Conversation`. _Note that in this document, the terms "`Bot` class" and "`Conversation` class" refer to the class or any subclass of it._

There are two basic types of responses supplied by the Anthropic API - complete messages (which we call "flat" responses) - in which the entire response is composed before being made available as an object - and streaming responses, in which you get to see the response come in chunk-by-chunk, which is generally preferable for conversational scenarios.

### `streamer()` and other convenience tools

For convenience, the `robo` package includes a function called `streamer` which makes it easy to set up a conversation and stream responses. 

```python
>>> say = streamer(Bot)
>>> say("Hello!")
<a response from Claude ensues>
```

This also allows passing in of values for fields, as demonstrated in the "Fields" section below. Alternatively to passing a `Bot` class as first argument, you can also pass `streamer` an _instance_ of a `Bot` class; or an instance of a `Conversation` class (in which case, make sure that it's been instanciated with `stream=True`).

`streamer` can also optionally be passed a file-like object via keyword argument `cc`, in which case it writes the incoming chunks of the streaming response to the file-like object as well as printing them:

```python
>>> from io import StringIO
>>> sio = StringIO()
>>> say = streamer(Bot, cc=sio)
>>> say("What colour is the sky?")
The sky appears blue during the day. This is because molecules in Earth's atmosphere scatter blue 
light from the sun more than other colors. However, the sky can appear different colors at different 
times - like red, orange, pink, or purple during sunrise and sunset, or gray when it's cloudy.
>>> sio.tell()
293
```

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

Fields allow for interpolation of conversation-specific parameters into the system prompt. When initialising a conversation, you can pass field values as a list (in which case the values must be in the same order as the `fields` definition) or a dictionary.

```python
class PoetryBot(Bot):
    fields = ['POEM_TYPE']
    sysprompt_text = """A user will ask you about something. You respond with a {{POEM_TYPE}} about the subject."""

>>> say = streamer(PoetryBot, ['haiku'])
>>> say('New York')
Towering steel and glass,
Yellow cabs weave through the rush—
Dreams rise with the dawn.
>>> say = streamer(PoetryBot, {'POEM_TYPE': 'limerick'})
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
>>> conv4a = Conversation(AnimalBot, {'ANIMAL_TYPE': 'horse'}) # Using a dict instead
>>> printmsg(conv4a.resume('hello'))
Neigh!
```

One more thing, if you wish to use method 4 to begin a conversation with a bot that doesn't take parameters, you'll need to pass an empty list or dict (as the presence of that argument is what triggers the automatic prestart):

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
        datadir = Path('/path/to/data/')
        novella_text = (datadir / 'franz-kafka-metamorphosis.txt').read_text()
        sysprompt_segments = [
            self._make_sysprompt_segment("""You are a knowledgeable literary assistant. The text 
                of a well-known novella is provided to you so that you can answer questions about it."""),
            self._make_sysprompt_segment(novella_text, set_cache_control=True)
        ]
        return sysprompt_segments

>>> say = streamer(LiteraryAssistant)
>>> say("Man, this guy just can't catch a break huh?")
You're absolutely right! Gregor Samsa's situation is relentlessly bleak from start to finish. He wakes 
up transformed into a giant insect, and instead of things getting better or him finding some way to 
adapt, everything just spirals downward.

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

There are a few things here worth noting. `Bot._make_sysprompt_segment(...)` has been provided as a convenient way of generating the correct segment structure for a multi-segment text-only system prompt. The `set_cache_control` argument, when set `True`, adds cache control clauses to the segment that enable system prompt caching. For large system prompts, prompt caching is highly recommended as it can significantly reduce costs. It's worth taking a look at [Anthropic's prompt caching documentation](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) to better understand the subject, but put simply, `set_cache_control=True` is best used on the last segment of the system prompt that is unlikely to change between requests.

Now let's add fields into a dynamic prompt:

```python
from pathlib import Path

class LiterarySterotypeAssistant(Bot):
    def sysprompt_generate(self):
        datadir = Path('/path/to/data/')
        novella_text = (datadir / 'george-orwell-animal-farm.txt').read_text()
        fields = ['COUNTRY']
        return [
            self._make_sysprompt_segment("""You are a librarian. You love to discuss literature 
            but you do so in a way that makes extensive use of slang and other stereotypical speech 
            patterns distinct to your place of origin. You will be provided with the text of a 
            novella so that you can discuss it with the user."""),
            self._make_sysprompt_segment(novella_text, set_cache_control=True),
            self._make_sysprompt_segment("""Your place of origin is {{COUNTRY}}.""")
        ]

>>> say = streamer(LiterarySterotypeAssistant, ['Australia'])
>>> say('Four legs good?')
Ay mate, *adjusts reading glasses and chuckles*, you're pullin' out the old bleating from the 
sheep, eh? "Four legs good, two legs bad!" - now that's a ripper of a slogan from Orwell's mob, 
ain't it?

Bloody brilliant how George crafted that little chant, I reckon. Started off simple enough when 
Snowball was tryin' to dumb down the Seven Commandments for the less brainy animals on the farm. 
But crikey, by the end of the yarn, even that gets twisted around, doesn't it? The sheep end up 
chantin' "Four legs good, two legs BETTER!" when the pigs start walkin' upright like their old 
human oppressors.

Classic bit of Aussie irony there - well, English irony written by an old Pom, but you get my 
drift. Shows how propaganda can be flipped on its head quicker than a Sunday roast. One minute 
the pigs are preachin' about animal equality, next minute they're struttin' around in Jones's 
old threads, playin' cards with the human farmers.

What gets me is how the poor sheep just keep parroting whatever they're told - reminds me of 
some conversations down at the local pub, if you catch my meaning! *winks*

So what's got you thinkin' about Animal Farm today, cobber? Fancy a yarn about how power corrupts, 
or are you more interested in Orwell's take on revolutionary ideals gone pear-shaped?
```

Notice that:
- The prompt segment with the dynamic field is a dedicated final segment
- The segment with `set_cache_control` is the one immediately before it.

The reason for this is that prompt caching applies to all segments up to and including the one where `set_cache_control` is used. Prompt caching relies on the inputs being the same up to the cache control clause - if anything at all changes, the cache will miss - so the pattern of packing a final segment with the dynamic fields and caching up to the segment prior maximises the cacheability of your system prompt.

Note that you can use `set_cache_control` on multiple segments. Have a read of the Anthropic docs (linked above) for more on this.

LLMs are very flexible about the type of textual data you can include in system prompts - as well as understanding structured data formats such as JSON, YAML and HTML, generally if a human would find something easy to understand then an LLM almost certainly will too. It's worth keeping your token count in mind - for example if you have JSON or YAML data with a repetitive structure, it might be worth reformatting it as CSV or similar so that you're not squandering tokens on repeating the same field names over and over. In fact, the LLM can help with this:

```python
class ReformerBot(Bot):
    sysprompt_text = """You will be provided with a JSON dataset. Your task is to reformat it into 
        a CSV-like format using the pipe character ("|") as field separator. Use these columns: 
        id, name, username, email, city, zipcode, phone, company name, website, company catchphrase."""

>>> import requests
>>> conv = Conversation(ReformerBot, [])
>>> msg = conv.resume(requests.get('https://jsonplaceholder.typicode.com/users').text)
>>> printmsg(msg)
id|name|username|email|city|zipcode|phone|company name|website|company catchphrase
1|Leanne Graham|Bret|Sincere@april.biz|Gwenborough|92998-3874|1-770-736-8031 x56442|Romaguera-Crona|hildegard.org|Multi-layered client-server neural-net
2|Ervin Howell|Antonette|Shanna@melissa.tv|Wisokyburgh|90566-7771|010-692-6593 x09125|Deckow-Crist|anastasia.net|Proactive didactic contingency
3|Clementine Bauch|Samantha|Nathan@yesenia.net|McKenziehaven|59590-4157|1-463-123-4447|Romaguera-Jacobson|ramiro.info|Face to face bifurcated interface
4|Patricia Lebsack|Karianne|Julianne.OConner@kory.org|South Elvis|53919-4257|493-170-9623 x156|Robel-Corkery|kale.biz|Multi-tiered zero tolerance productivity
5|Chelsey Dietrich|Kamren|Lucio_Hettinger@annie.ca|Roscoeview|33263|(254)954-1289|Keebler LLC|demarco.info|User-centric fault-tolerant solution
6|Mrs. Dennis Schulist|Leopoldo_Corkery|Karley_Dach@jasper.info|South Christy|23505-1337|1-477-935-8478 x6430|Considine-Lockman|ola.org|Synchronised bottom-line interface
7|Kurtis Weissnat|Elwyn.Skiles|Telly.Hoeger@billy.biz|Howemouth|58804-1099|210.067.6132|Johns Group|elvis.io|Configurable multimedia task-force
8|Nicholas Runolfsdottir V|Maxime_Nienow|Sherwood@rosamond.me|Aliyaview|45169|586.493.6943 x140|Abernathy Group|jacynthe.com|Implemented secondary concept
9|Glenna Reichert|Delphine|Chaim_McDermott@dana.io|Bartholomebury|76495-3109|(775)976-6794 x41206|Yost and Sons|conrad.com|Switchable contextually-based project
10|Clementina DuBuque|Moriah.Stanton|Rey.Padberg@karina.biz|Lebsackbury|31428-2261|024-648-3804|Hoeger LLC|ambrose.net|Centralized empowering task-force
```

Let's use the above data as an example of a simple knowledge base.

```python
# using msg from previous example

class KBBot(Bot):
    def sysprompt_generate(self):
        return [
            self._make_sysprompt_segment("""You will be provided with some data. Your task is 
                to answer questions about it."""),
            self._make_sysprompt_segment(gettext(msg))
        ]

>>> say = streamer(KBBot)
>>> say("contact details for clem?")
Based on the data, there are two people with "Clem" in their name:

**Clementine Bauch:**
- Email: Nathan@yesenia.net
- Phone: 1-463-123-4447
- City: McKenziehaven
- Zipcode: 59590-4157

**Clementina DuBuque:**
- Email: Rey.Padberg@karina.biz
- Phone: 024-648-3804
- City: Lebsackbury
- Zipcode: 31428-2261

Which "Clem" were you looking for?
```

## Tool use

Tool use is a powerful set of features that allow Claude to call functions to assist it in fulfilling a user's request. The scope of these functions is effectively limitless but it's important to carefully specify the tools so that the model can understand precisely when and how to use them.

There are three basic modalities for tool use:
- Single tool - in which a single tool call is sufficient to inform the creation of a response.
- Parallel tool use - in which multiple tools are used to inform the response but they don't depend on each other.
- Sequential tool use - in which the output from one or more tools are used to inform inputs for subsequent tool calls before generating a response.

These modalities are not mutually exclusive, ie. they can be mixed together. As long as the tools are well specified, the model will usually be able to figure out which ones to use and when, and how to chain them together if needed. RoboOp can generally handle tool calls seamlessly. For more detail about tool use (and tool schemas) check out [Anthropic's documentation on the subject](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview).

In RoboOp, we take a streamlined approach to tool definition by using objects subclassed from the `robo.tools.Tool` class. Here's a simple example:

```python
from robo.tools import Tool
import requests

class FetchAndAnalyseBot(Bot):
    sysprompt_text = """Your task is to retrieve and analyse the contents of a provided URL."""
    
    class GetURL(Tool):
        description = 'Fetch the raw HTML from a given URL'
        parameter_descriptions = {
            'url': 'The URL to fetch',
        }
        
        def __call__(self, url:str):
            print(f"\n[[Fetching URL: {url}]]")
            pagetext = requests.get(url).text
            return pagetext
    
    tools = [GetURL]

>>> say = streamer(FetchAndAnalyseBot)
>>> say('https://simple.wikipedia.org/wiki/Barbenheimer')
I'll fetch the content from that Wikipedia page about Barbenheimer for you.
[[Fetching URL: https://simple.wikipedia.org/wiki/Barbenheimer]]
Based on the Wikipedia page content, here's an analysis of **Barbenheimer**:

## What is Barbenheimer?

**Barbenheimer** is an internet phenomenon and meme that emerged before July 21, 2023, when two 
highly contrasting movies were released on the same date:
- *Barbie* - a fantasy comedy based on the fashion doll
- *Oppenheimer* - a biographical thriller about theoretical physicist J. Robert Oppenheimer

The term is a portmanteau (combination) of the two movie titles.

## Key Aspects of the Phenomenon

**The Contrast**: The meme gained popularity specifically because these two films were so dramatically
different - *Polygon* described them as "extremely opposite to each other." The juxtaposition of a 
pink, lighthearted Barbie movie with a serious, dark biographical film about the creator of the atomic 
bomb created comedic appeal.

**Cultural Impact**: 
- Movie theaters began selling double features of both films
- Celebrities participated, including Tom Cruise, director Greta Gerwig, and stars Margot Robbie and 
  Cillian Murphy
- Murphy encouraged people to "go see both, on the same day"
- Instead of competing against each other (as typically happens with major simultaneous releases), the 
  films actually helped each other

**Box Office Success**:
- *Barbie* made $22.3 million in Thursday night previews
- *Oppenheimer* made $10.5 million in previews  
- This was the first time in 2023 that two movies each made over $10 million in previews

**Awards Recognition**: The combined films earned 21 nominations at the 96th Academy Awards, with 
both nominated for Best Picture and *Oppenheimer* receiving the most nominations of the year (13).

Barbenheimer represents a unique cultural moment where contrasting entertainment became complementary 
rather than competitive, demonstrating the power of internet memes to shape movie-going behavior 
and cultural discourse.
>>> 
```

Note the use of Python type hinting to specify the types of the function parameters. This allows for automatic generation of the tool schema. The tools may be defined inline in the `Bot` class (as shown here) or defined externally and referenced in the `tools` list in the same way.

(Note that object-oriented tool definitions are quite new and somewhat experimental; if you need finer-grained control over tool schema generation, you can override `get_tools_schema()` in a subclass of `Bot`.)

## File handling

There are many scenarios in which it is useful to include files alongside your textual prompts. RoboOp makes this straightforward with flexible handling of both files and raw file data.

From a local file:
```python
>>> say = streamer(Bot)
>>> say("describe this image.", with_files=['/path/to/sunflower_with_sky_backdrop.jpg'])
This image shows a beautiful, vibrant sunflower in full bloom against a brilliant blue sky. The 
sunflower displays the classic characteristics of its species - bright golden-yellow petals 
radiating outward from a large, circular center. The center has a distinctive pattern with a 
green core surrounded by a ring of orange-brown disc florets that create a textured, almost 
honeycomb-like appearance.

The sunflower is photographed from a low angle, making it appear majestic and prominent against 
the deep blue sky with some wispy white clouds visible in the background. The flower head is 
quite large and full, with numerous long, pointed petals that appear to be basking in bright 
sunlight.

The plant's characteristic large, broad, serrated green leaves are visible along the thick stem, 
showing the typical heart-shaped form of sunflower foliage. The contrast between the warm yellows 
and oranges of the flower, the rich green of the leaves, and the cool blue of the sky creates a 
striking and cheerful composition that captures the essence of a perfect summer day.
>>> 
```

Besides the raw data of the file, the Claude API needs to know the MIME type of the file and the type of content block it will be enclosed in. Valid content block types are `document`, `image`, and `container_upload`.
In the example above, the file is referenced as a path on the local system, in which case RoboOp will attempt to infer the MIME type and content block type from the file's extension.

For finer-grained control, you can refer to the file in what we call "filespec" form, which is a three-tuple of `(mimetype, file_bytes_object_or_path, content_block_type)`. `file_bytes_object_or_path` can be any of:
- Raw `bytes`
- A file-like object (specifically, something with a `read()` method)
- A path to the file (either in `string` form or a `Path` object).

From a file-like object (and using `Conversation.resume()` instead of `streamer`):
```python
>>> conv = Conversation(Bot, [])
>>> with open('/path/to/mona-lisa.jpg', 'rb') as la_gioconda:
...   printmsg(conv.resume("Please describe this image.", with_files=[('image/jpeg', la_gioconda, 'image')]))
... 
This is the famous painting "Mona Lisa" (also known as "La Gioconda") by Leonardo da Vinci, created 
between 1503-1519. The painting depicts a woman with an enigmatic smile, seated in three-quarter 
view against a mysterious landscape background. She has long, dark hair and is wearing dark clothing 
typical of early 16th-century fashion. Her hands are folded and positioned in the foreground, while 
behind her stretches an atmospheric landscape with winding paths, bridges, and distant mountains 
rendered in sfumato - da Vinci's signature technique of soft, hazy transitions between colors and 
tones. The painting is renowned for the subject's direct gaze and subtle smile, which has captivated 
viewers for centuries. Currently housed in the Louvre Museum in Paris, it's considered one of the 
most famous paintings in the world and a masterpiece of Renaissance art.
```

When retrieving a file from the Web, the retrieval result will generally include the file's MIME type. For example, using `requests` (and demonstrating how to stream without using `streamer`):

```python
>>> import requests
>>> response = requests.get('https://example-files.online-convert.com/document/pdf/example.pdf')
>>> response.headers['content-type']
'application/pdf'
>>> conv = Conversation(Bot, [], stream=True)
>>> filespec = (response.headers['content-type'], response.content, 'document')
>>> with conv.resume("Please describe this document.", with_files=[filespec]) as stream:
...   for chunk in stream.text_stream:
...     print(chunk, end='', flush=True)
... 
This is a PDF test file (Version 1.0) that serves as an example document to demonstrate the PDF file 
format. The document contains educational content about placeholder names used in legal and other 
contexts.

**Key Content:**
- **Main Topic**: The use of placeholder names like "John Doe," "Jane Doe," and "Jane Roe"
- **Usage**: These names are used when someone's true identity is unknown or must be withheld in 
  legal proceedings, or to refer to unidentified corpses or hospital patients
- **Geographic Usage**: Primarily used in the United States and Canada; other English-speaking 
  countries like the UK, Australia, and New Zealand prefer names like "Joe Bloggs" or "John Smith"
- **Cultural References**: The document mentions usage in popular culture, including the Frank Capra 
  film "Meet John Doe" and a 2002 TV series
- **Variations**: Discusses related terms like "Baby Doe," "Precious Doe," and numbering systems 
  for multiple anonymous parties (John Doe #1, #2, etc.)

**Document Details:**
- Contains both text and a landscape image
- Source content is from Wikipedia under Attribution-ShareAlike 3.0 Unported license
- Created by online-convert.com as a file format example
- Includes Creative Commons licensing information at the bottom

The document appears to be designed primarily for testing PDF functionality while providing 
informative content about legal naming conventions.
>>> 
```

Note that in some cases, inferring the MIME type from the response may not be reliable due to variations in how web servers are configured. For example, requests for PDFs hosted on Github.com may come back with a `content-type` of `application/octet-stream` (which is just a fancy way of saying "this is a bunch of bytes"). Unfortunately that's not enough for the Claude API to work with and it will return a `BadRequestError` - so your mileage may vary.

# More to come, watch this space! :)
