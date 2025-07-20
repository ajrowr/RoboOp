import anthropic

from .models import MODELS
from .exceptions import *
from .streamwrappers import *

from pathlib import Path
import os
import json
import datetime
import time
from types import SimpleNamespace


API_KEY_FILE = None ## If you want to load it from a file 
API_KEY_ENV_VAR = None ## If you want to use a different env var instead of ANTHROPIC_API_KEY

STREAM_WRAPPER_CLASS_SYNC = StreamWrapperWithToolUse
STREAM_WRAPPER_CLASS_ASYNC = AsyncStreamWrapperWithToolUse

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

def _get_client(async_mode=False):
    return _get_client_class()(api_key=_get_api_key())


class Bot(object):
    __slots__ = ['fields', 'sysprompt_path', 'sysprompt_text', 'client', 'model', 
            'temperature', 'max_tokens', 'oneshot', 'welcome_message', 'soft_start']
    """soft_start will inject the welcome_message into the conversation context as though 
            the agent had said it, making it think that the conversation has already
            begun. Beware of causing confusion by soft-starting with something the model 
            wouldn't say.
        oneshot is for bots that don't need to maintain conversation context to do their job.
            Is NOT compatible with tool use!"""
    
    def sysprompt_generate(self):
        """Allow for generation of sysprompt via a function as an alternative to text. Ideal
        if you want a structured sysprompt instead of plain text, which is especially useful
        when you want to use special features such as prompt caching."""
        raise NotImplementedError("This method is not implemented")
    
    def get_tools_schema(self):
        """Return a schema describing the tools available to this bot. See
        https://docs.anthropic.com/en/api/messages#body-tools for more info on the structure
        of the return value.
        The actual tool call functions should be implemented in a subclass as methods such as
             def tool_<toolname>(self, paramname1=None, paramname2=None, ...)
        """
        return None
    
    def handle_tool_call(self, tooluseblock):
        if type(tooluseblock) is dict:
            tooluseblock = SimpleNamespace(**tooluseblock)
        toolfnname = f'tools_{tooluseblock.name}'
        tool = getattr(self, toolfnname, None)
        if tool is None:
            raise Exception(f'Tool function not found: {toolfnname}')
        return tool(**tooluseblock.input)
    
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
    
    def preprocess_response(self, message_text, conversation):
        """Hook to potentially send a canned response or manipulate the message that will
        be sent to the model.
        
        Returns:
            -- these ones engage the model --
            None: Forward message to model as normal
            dict: Append the dict to the conversation messages and then invoke 
                  the model (ie. add a custom message to the stack rather than 
                  using the provided message text as basis for one - useful for 
                  some types of tool calls, particularly those where the model has to 
                  wait for the client to do something before proceeding)
            -- these ones bypass the model --
            str: Send this to the client as a CannedResponse (include_in_context=True)
            tuple: (response_text, include_in_context) as above but with more control
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




class CannedResponse:
    """Wrapper for canned responses to distinguish them from API responses"""
    def __init__(self, text, include_in_context=True):
        self.text = text
        self.include_in_context = include_in_context
        # Mock the structure of an API response
        self.content = [type('Content', (), {'text': text})()]
    
    def __repr__(self):
        return f'<{type(self).__name__}: "{self.text}">'
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False
    
    @property
    def text_stream(self):
        """Yield the entire text as a single chunk for streaming compatibility"""
        yield self.text


class CannedResponseAsync(CannedResponse):
    @property
    async def text_stream(self):
        """Yield the entire text as a single chunk for streaming compatibility"""
        """Claude tells me I'm violating the Liskov Substitution Principle or something but IDC TBH"""
        """Haven't tested yet..."""
        yield self.text


class Conversation(object):
    __slots__ = ['messages', 'bot', 'sysprompt', 'argv', 'max_tokens', 'message_objects', 
                'is_streaming', 'started', 'is_async', 'oneshot', 'cache_user_prompt', 
                'soft_started', 'tool_use_blocks']
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
        self.tool_use_blocks = SimpleNamespace(pending=[], resolved=[])
        if soft_start or (self.bot.soft_start and not soft_start is False):
            self.messages.append(self._make_text_message('assistant', self.bot.welcome_message))
            self.message_objects.append(None)
            self.soft_started = True
        self.is_streaming = stream
        self.started = False
        if argv is not None:
            self.prestart(argv)
            self.argv = argv
        else:
            self.argv = []
    
    @staticmethod
    def _make_text_message(role, content):
        return {
            'role': role,
            'content': [{
                'type': 'text',
                'text': content
            }]
        }
    
    @staticmethod
    def _make_tool_result_message(toolblock, toolresult):
        if type(toolblock) is dict:
            toolblock = SimpleNamespace(**toolblock)
        return {
            'role': 'user',
            'content': [{
                'type': 'tool_result',
                'tool_use_id': toolblock.id,
                'content': toolresult,
            }]
        }
    
    @staticmethod
    def _make_tool_request_message(toolblock):
        if type(toolblock) is dict:
            toolblock = SimpleNamespace(**toolblock)
        return {
            'role': 'assistant',
            'content': [{
                'type': 'tool_use',
                'id': toolblock.id,
                'name': toolblock.name,
                'input': toolblock.input,
            }]
        }
    
    def _get_last_tool_use_id(self):
        tu_id = None
        for m in reversed(self.messages):
            if m['role'] == 'assistant':
                for cblock in m['content']:
                    if cblock['type'] == 'tool_use':
                        tu_id = cblock['id']
                        break
        return tu_id
    
    def _add_tool_request(self, request):
        if type(request) is dict:
            request = SimpleNamespace(**request)
        self.tool_use_blocks.pending.append(
            SimpleNamespace(
                name = request.name,
                id = request.id,
                request = request,
                response = None,
                status = 'PENDING',
            )
        )
    
    def _handle_pending_tool_requests(self):
        for tub in self.tool_use_blocks.pending:
            if tub.status == 'PENDING':
                tub.response = self.bot.handle_tool_call(tub.request)
                tub.status = 'READY'
    
    def _compile_tool_responses(self, mark_resolved=True):
        """Compile the responses for tubs with status READY into a single block suitable for adding into the message history"""
        blocks_out = []
        for tub in self.tool_use_blocks.pending:
            if tub.status == 'READY':
                blocks_out.append({
                    'type': 'tool_result',
                    'tool_use_id': tub.id,
                    'content': str(tub.response['message']),
                })
                tub.status = 'RESOLVED' if mark_resolved else tub.status
        if mark_resolved:
            for tub in list(filter(lambda tub: tub.status == 'RESOLVED', self.tool_use_blocks.pending)):
                self.tool_use_blocks.resolved.append(tub)
                self.tool_use_blocks.pending.remove(tub)
                
        return {
            'role': 'user',
            'content': blocks_out
        }
    
    def _is_exhausted(self):
        """Return True if the last message is from the assistant and consists only of 
        text content blocks."""
        lastmsg = self.messages[-1]
        return lastmsg['role'] == 'assistant' and \
            all([block['type'] == 'text' for block in lastmsg['content']])
    
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
        canned_response = self.bot.preprocess_response(message, self)
        if canned_response is not None:
            return self._handle_canned_response(message, canned_response)
        
        if self.is_streaming:
            return await self._aresume_stream(message)
        else:
            return await self._aresume_flat(message)

    def _handle_canned_response(self, original_message, canned_response):
        """Handle canned responses (works for both sync and async). If original_message
        is None, it isn't added to the conversation history (which is useful for
        certain types of tool calls, eg. ones where you need to send a system message 
        to the client that might confuse the model if it became part of the chat log)."""
        if original_message is not None:
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

    async def _aresume_stream(self, message, is_tool_message=False):
        if is_tool_message:
            self.messages.append(message)
        else:
            self.messages.append(self._make_text_message('user', message))
        stream = self.bot.client.messages.stream(
            model=self.bot.model, max_tokens=self.max_tokens,
            temperature=self.bot.temperature, system=self.sysprompt,
            messages=self._get_conversation_context(),
            tools=self.bot.get_tools_schema(),
        )
        return STREAM_WRAPPER_CLASS_ASYNC(stream, self)

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
        canned_response = self.bot.preprocess_response(message, self)
        is_tool_message = False
        if type(canned_response) is dict:
            message = canned_response
            is_tool_message = True
        elif canned_response is not None:
            return self._handle_canned_response(message, canned_response)
        
        if self.is_streaming:
            return self._resume_stream(message, is_tool_message=is_tool_message)
        else:
            return self._resume_flat(message, is_tool_message=is_tool_message)

    def _resume_stream(self, message, is_tool_message=False):
        if is_tool_message:
            self.messages.append(message)
        else:
            self.messages.append(self._make_text_message('user', message))
        stream = self.bot.client.messages.stream(
            model=self.bot.model, max_tokens=self.max_tokens,
            temperature=self.bot.temperature, system=self.sysprompt,
            messages=self._get_conversation_context(),
            tools=self.bot.get_tools_schema(),
        )
        return STREAM_WRAPPER_CLASS_SYNC(stream, self)

    def _resume_flat(self, message, is_tool_message=False):
        if is_tool_message:
            self.messages.append(message)
        else:
            self.messages.append(self._make_text_message('user', message))
        message_out = self.bot.client.messages.create(
            model=self.bot.model, max_tokens=self.max_tokens,
            temperature=self.bot.temperature, system=self.sysprompt,
            messages=self._get_conversation_context(),
            tools=self.bot.get_tools_schema(),
        )
        self.message_objects.append(message_out)
        for contentblock in message_out.content:
            blocktype = type(contentblock).__name__
            if blocktype == 'ToolUseBlock':
                self.messages.append(self._make_tool_request_message(contentblock))
                tooldat = self.bot.handle_tool_call(contentblock)
                if tooldat['target'] == 'model':
                    return self._resume_flat(self._make_tool_result_message(contentblock, tooldat), is_tool_message=True)
                elif tooldat['target'] == 'client':
                    return self._handle_canned_response(None, (tooldat['message'], False))
            elif hasattr(contentblock, 'text'):
                self.messages.append(self._make_text_message('assistant', contentblock.text))
            else:
                raise Exception(f"Don't know what to do with blocktype: {blocktype}")
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
        # Check if the bot wants to preprocess the response
        canned_response = self.bot.preprocess_response(message, self) if message else None
        if canned_response is not None:
            return self._handle_canned_response(message, canned_response)
        
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
