import anthropic

from .models import MODELS
from .exceptions import *

from pathlib import Path
import os
import json
import datetime
import time


API_KEY_FILE = None ## If you want to load it from a file 
API_KEY_ENV_VAR = None ## If you want to use a different env var instead of ANTHROPIC_API_KEY


def _get_api_key():
    if API_KEY_FILE:
        return open(API_KEY_FILE).read()
    elif API_KEY_ENV_VAR:
        return os.environ[API_KEY_ENV_VAR]
    ## If neither, then returning None will let Anthropic check its default of ANTHROPIC_API_KEY

def _get_client_class(async_mode=False):
    if async_mode:
        return anthropic.AsyncAnthropic
    return anthropic.Anthropic

class Bot(object):
    __slots__ = ['fields', 'sysprompt_path', 'sysprompt_text', 'client', 'model', 
            'temperature', 'max_tokens', 'oneshot', 'welcome_message', 'soft_start']
    """soft_start will inject the welcome_message into the conversation context as though 
                the agent had said it, making it think that the conversation has already
                begun. Beware of causing confusion by soft-starting with something the model 
                wouldn't say."""
    
    def sysprompt_generate(self):
        """Allow for generation of sysprompt via a function as an alternative to text. Ideal
        if you want a structured sysprompt instead of plain text, which is especially useful
        when you want to use special features such as prompt caching."""
        raise NotImplementedError("This method is not implemented")
    
    @property
    def sysprompt_clean(self):
        try:
            return self.sysprompt_generate()
        except NotImplementedError:
            pass
        if hasattr(self, 'sysprompt_text'):
            return self.sysprompt_text
        elif hasattr(self, 'sysprompt_path'):
            return open(self.sysprompt_path).read()
        else:
            return ''
    
    def preprocess_response(self, message_text):
        """Hook to potentially send a canned response. 
        
        Returns:
            None: Forward message to model as normal
            str: Send this as canned response (include_in_context=True)
            tuple: (response_text, include_in_context) for more control
        """
        return None
    
    def sysprompt_vec(self, argv):
        sysp = self.sysprompt_clean
        if not argv:
            return sysp
        remap = False if type(sysp) is str else True
        if remap:
            sysp = json.dumps(sysp)
            
        for k, v in zip(self.fields, argv):
            sysp = sysp.replace(f'{{{{{k}}}}}', v)
        
        return json.loads(sysp) if remap else sysp
    
    def __init__(self, client=None, async_mode=False):
        for f, v in [('model', MODELS.CLAUDE_4.SONNET), ('temperature', 1), ('fields', []), 
                    ('max_tokens', 8192), ('oneshot', False), ('welcome_message', None),
                    ('soft_start', False)]:
            if not hasattr(self, f):
                setattr(self, f, v)
        if not client:
            client = _get_client_class(async_mode)(api_key=_get_api_key())
        self.client = client
    
    @classmethod
    def with_api_key(klass, api_key, async_mode=False):
        client = _get_client_class(async_mode)(api_key=api_key)
        return klass(client)


class StreamWrapper:
    def __init__(self, stream, conversation_obj):
        self.stream = stream
        self.conversation_obj = conversation_obj
        self.accumulated_text = ""
        self.chunks = []
    
    def __enter__(self):
        self.stream_context = self.stream.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        result = self.stream.__exit__(exc_type, exc_val, exc_tb)
        if exc_type is None and self.accumulated_text:
            asst_message = self.conversation_obj._make_text_message('assistant', self.accumulated_text)
            self.conversation_obj.messages.append(asst_message)
            self.conversation_obj._post_stream_hook()
        
        return result
    
    @property
    def text_stream(self):
        for text in self.stream_context.text_stream:
            self.chunks.append(text)
            self.accumulated_text += text
            yield text


class AsyncStreamWrapper:
    def __init__(self, stream, conversation_obj):
        self.stream = stream
        self.conversation_obj = conversation_obj
        self.accumulated_text = ""
        self.chunks = []
    
    async def __aenter__(self):
        self.stream_context = await self.stream.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        result = await self.stream.__aexit__(exc_type, exc_val, exc_tb)
        if exc_type is None and self.accumulated_text:
            asst_message = self.conversation_obj._make_text_message('assistant', self.accumulated_text)
            self.conversation_obj.messages.append(asst_message)
            await self.conversation_obj._post_stream_hook_async()
        
        return result
    
    @property
    async def text_stream(self):
        async for text in self.stream_context.text_stream:
            self.chunks.append(text)
            self.accumulated_text += text
            yield text


class CannedResponse:
    """Wrapper for canned responses to distinguish them from API responses"""
    def __init__(self, text, include_in_context=True):
        self.text = text
        self.include_in_context = include_in_context
        # Mock the structure of an API response
        self.content = [type('Content', (), {'text': text})()]


class Conversation(object):
    __slots__ = ['messages', 'bot', 'sysprompt', 'argv', 'max_tokens', 'message_objects', 
                'is_streaming', 'started', 'is_async', 'oneshot', 'cache_user_prompt']
    def __init__(self, bot, argv=None, stream=False, async_mode=False, soft_start=None, cache_user_prompt=False):
        self.is_async = async_mode
        if type(bot) is type:
            self.bot = bot(async_mode=async_mode)
        else:
            self.bot = bot
        self.max_tokens = self.bot.max_tokens
        self.oneshot = self.bot.oneshot
        self.messages = []
        self.message_objects = []
        self.cache_user_prompt = cache_user_prompt
        if soft_start or (self.bot.soft_start and not soft_start is False):
            self.messages.append(self._make_text_message('assistant', self.bot.welcome_message))
            self.message_objects.append(None)
        self.is_streaming = stream
        self.started = False
        if argv is not None:
            self.prestart(argv)
            self.argv = argv
        else:
            self.argv = []
    
    def _make_text_message(self, role, content):
        return {
            'role': role,
            'content': [{
                'type': 'text',
                'text': content
            }]
        }
    
    def _get_conversation_context(self):
        """Oneshot is for bots that don't need conversational context"""
        if self.oneshot:
            return [self.messages[-1]]
        elif self.cache_user_prompt:
            from copy import deepcopy
            mymessages = deepcopy(self.messages)
            mymessages[-1]['content'][-1]['cache_control'] = {'type': 'ephemeral'}
            return mymessages
        return self.messages
    
    def prestart(self, argv):
        self.argv = argv
        self.sysprompt = self.bot.sysprompt_vec(argv)
        self.started = True
    
    def start(self, *args):
        if type(args[0]) is list:
            argv, message = args
        else:
            argv, message = [], args[0]
        if self.started:
            raise Exception('Conversation has already started')
            
        self.prestart(argv)
        return self.resume(message)

    async def astart(self, *args):
        if type(args[0]) is list:
            argv, message = args
        else:
            argv, message = [], args[0]
        if self.started:
            raise Exception('Conversation has already started')
        
        self.prestart(argv)
        return await self.aresume(message)

    async def aresume(self, message):
        # Check for canned response first
        canned_response = self.bot.preprocess_response(message)
        if canned_response is not None:
            return self._handle_canned_response(message, canned_response)
        
        if self.is_streaming:
            return await self._aresume_stream(message)
        else:
            return await self._aresume_flat(message)

    def _handle_canned_response(self, original_message, canned_response):
        """Handle canned responses (works for both sync and async)"""
        self.messages.append(self._make_text_message('user', original_message))
        
        # The Bot.preprocess_response method should return a tuple (response, include_in_context)
        # or just a string (defaulting to include_in_context=True)
        if isinstance(canned_response, tuple):
            response_text, include_in_context = canned_response
        else:
            response_text, include_in_context = canned_response, True
        
        response_obj = CannedResponse(response_text, include_in_context)
        self.message_objects.append(response_obj)
        
        if include_in_context:
            self.messages.append(self._make_text_message('assistant', response_text))
        
        return response_obj

    async def _aresume_stream(self, message):
        self.messages.append(self._make_text_message('user', message))
        # Remove 'await' here - stream() returns AsyncMessageStreamManager directly
        stream = self.bot.client.messages.stream(
            model=self.bot.model, max_tokens=self.max_tokens,
            temperature=self.bot.temperature, system=self.sysprompt,
            messages=self._get_conversation_context()
        )
        return AsyncStreamWrapper(stream, self)

    async def _aresume_flat(self, message):
        self.messages.append(self._make_text_message('user', message))
        message_out = await self.bot.client.messages.create(
            model=self.bot.model, max_tokens=self.max_tokens,
            temperature=self.bot.temperature, system=self.sysprompt,
            messages=self._get_conversation_context()
        )
        self.message_objects.append(message_out)
        self.messages.append(self._make_text_message('assistant', message_out.content[0].text))
        return message_out

    def resume(self, message):
        # Check for canned response first
        canned_response = self.bot.preprocess_response(message)
        if canned_response is not None:
            return self._handle_canned_response(message, canned_response)
        
        if self.is_streaming:
            return self._resume_stream(message)
        else:
            return self._resume_flat(message)

    def _resume_stream(self, message):
        self.messages.append(self._make_text_message('user', message))
        stream = self.bot.client.messages.stream(
            model=self.bot.model, max_tokens=self.max_tokens,
            temperature=self.bot.temperature, system=self.sysprompt,
            messages=self._get_conversation_context()
        )
        return StreamWrapper(stream, self)
    
    def _resume_flat(self, message):
        self.messages.append(self._make_text_message('user', message))
        message_out = self.bot.client.messages.create(
            model=self.bot.model, max_tokens=self.max_tokens,
            temperature=self.bot.temperature, system=self.sysprompt,
            messages=self._get_conversation_context()
        )
        self.message_objects.append(message_out)
        self.messages.append(self._make_text_message('assistant', message_out.content[0].text))
        return message_out
    
    def _post_stream_hook(self):
        pass
    
    async def _post_stream_hook_async(self):
        pass


class LoggedConversation(Conversation):
    __slots__ = ['conversation_id', 'logs_dir', 'first_saved_at']
    def __init__(self, bot, **kwargs):
        if 'conversation_id' in kwargs:
            self.conversation_id = kwargs.pop('conversation_id')
        else:
            import uuid
            self.conversation_id = str(uuid.uuid4())
        
        if 'logs_dir' in kwargs:
            self.logs_dir = kwargs.pop('logs_dir')
        else:
            self.logs_dir = None
        self.first_saved_at = None
        
        super().__init__(bot, **kwargs)
    
    def __repr__(self):
        return f'<{type(self).__name__} with ID {self.conversation_id}>'
    
    def _logfolder_path(self):
        if self.first_saved_at is None:
            self.first_saved_at = int(time.time()/10)
        dirname = f"{self.first_saved_at:x}__{self.conversation_id}"
        return Path(self.logs_dir) / dirname
    
    def _write_log(self):
        if self.logs_dir:
            logdir = self._logfolder_path()
            logdir.mkdir(parents=True, exist_ok=True)
            with open(logdir / 'conversation.json', 'w') as logfile:
                json.dump({
                    'when': str(datetime.datetime.now()),
                    'with': type(self.bot).__name__,
                    'argv': self.argv,
                    'messages': self.messages
                }, logfile, indent=4)
    
    def resume(self, message):
        resp = super().resume(message)
        self._write_log()
        return resp
    
    async def aresume(self, message):
        resp = await super().aresume(message)
        self._write_log()
        return resp
    
    def _post_stream_hook(self):
        self._write_log()
    
    async def _post_stream_hook_async(self):
        self._write_log()
    
    @classmethod
    def revive(klass, bot, conversation_id, logs_dir, argv, **kwargs):
        revenant = klass(bot, conversation_id=conversation_id, logs_dir=logs_dir, **kwargs)
        ## Find the chatlog to continue the conversation
        try:
            logdir_candidate = list(filter(lambda f: f.endswith(conversation_id), os.listdir(logs_dir)))[0]
        except IndexError as exc:
            excmsg = f"Conversation with ID {conversation_id} could not be found"
            raise UnknownConversationException(excmsg) from exc
        revenant.first_saved_at = int(logdir_candidate.split('__')[0], 16)
        with (Path(logs_dir) / logdir_candidate / 'conversation.json').open('r') as reader:
            logdata = json.load(reader)
        revenant.messages = logdata['messages']
        revenant.prestart(argv)
        return revenant


class ConversationWithFiles(Conversation):
    """This is experimental and needs work. Currently no async and no streaming. 
    Right now to use it you need to bypass resume() like:
        c = robo.ConversationWithFiles(Classifier, [])
        withjpeg = lambda fpath: c._resume_flat(None, [('image/jpeg', fpath)])
        withpng = lambda fpath: c._resume_flat(None, [('image/png', fpath)])
    """
    
    def _make_message_with_images(self, role, message, filespecs=[]):
        import base64
        content = []
        for ctype, fpath in filespecs:
            with open(fpath, 'rb') as inputfile:
                content.append({
                    'type': 'image',
                    'source': {
                        'type': 'base64',
                        'media_type': ctype,
                        'data': base64.b64encode(inputfile.read()).decode('utf-8')
                    }
                })
        if message:
            content.append({'type': 'text', 'text': message})
        return {'role': role, 'content': content}
        
    def _resume_flat(self, message, filespecs=[]):
        """filespecs are (content_type, filepath)"""
        if filespecs:
            self.messages.append(self._make_message_with_images('user', message, filespecs))
        else:
            self.messages.append(self._make_text_message('user', message))
        ## rest is the same
        message_out = self.bot.client.messages.create(
            model=self.bot.model, max_tokens=self.max_tokens,
            temperature=self.bot.temperature, system=self.sysprompt,
            messages=self._get_conversation_context()
        )
        self.message_objects.append(message_out)
        self.messages.append(self._make_text_message('assistant', message_out.content[0].text))
        return message_out


def streamer(bot_or_conversation, args=[]):
    """If you're passing in a conversation, make sure it's got stream=True!"""
    if type(bot_or_conversation) is Conversation:
        convo = bot_or_conversation
    else: ## in which case it should be either a bot instance or Bot class
        convo = Conversation(bot_or_conversation, stream=True)
    def streamit(message):
        if not convo.started:
            convo.prestart(args)
        with convo.resume(message) as stream:
            for chunk in stream.text_stream:
                print(chunk, end="", flush=True)
    return streamit

def streamer_async(bot, args=[]):
    convo = Conversation(bot, stream=True, async_mode=True)
    async def streamit(message):
        if not convo.started:
            convo.prestart(args)
        async with await convo.aresume(message) as stream:
            async for chunk in stream.text_stream:
                print(chunk, end="", flush=True)
    return streamit

"""
To use streamer_async :
>>> import asyncio
>>> from robo import *
>>> say = streamer_async(Bot)
>>> coro = say('who goes there?')
>>> asyncio.run(coro)    
"""

__all__ = ['Bot', 'Conversation', 'streamer', 'streamer_async', 'MODELS']
