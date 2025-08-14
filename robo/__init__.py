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

MEDIA_TYPE_MAP = {
    '.jpg': ('image/jpeg', 'image'),
    '.jpeg': ('image/jpeg', 'image'),
    '.png': ('image/png', 'image'),
    '.webp': ('image/webp', 'image'),
    '.txt': ('text/plain', 'document'),
    '.pdf': ('application/pdf', 'document'),
}

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
    return _get_client_class(async_mode=async_mode)(api_key=_get_api_key())


class Bot(object):
    """A bot that can engage in conversations via Claude models.
    
    Provides a foundation for creating AI assistants with customizable system prompts,
    tool capabilities, and conversation handling. Can be subclassed to create specialized
    bots with specific behaviors and tool integrations.
    """
    __slots__ = ['fields', 'sysprompt_path', 'sysprompt_text', 'client', 'model', 
            'temperature', 'max_tokens', 'oneshot', 'welcome_message', 'soft_start']
    """soft_start will inject the welcome_message into the conversation context as though 
            the agent had said it, making it think that the conversation has already
            begun. Beware of causing confusion by soft-starting with something the model 
            wouldn't say.
        oneshot is for bots that don't need to maintain conversation context to do their job.
            Is NOT compatible with tool use!"""
    
    @staticmethod
    def _make_sysprompt_segment(text, set_cache_control=False):
        return {
            **{'type': 'text', 'text': text}, 
            **({'cache_control': {'type': 'ephemeral'}} if set_cache_control else {})
        }
    
    def sysprompt_generate(self):
        """Generate a system prompt dynamically as an alternative to static text.
        
        Override this method to create structured system prompts or prompts that
        change based on runtime conditions. Particularly useful for prompt caching
        or complex prompt structures.
        
        Returns:
            str or dict: The system prompt content
            
        Raises:
            NotImplementedError: If not overridden in subclass
        """
        raise NotImplementedError("This method is not implemented")
    
    def get_tools_schema(self):
        """Return the schema describing tools available to this bot.
        
        Override this method to define tools that the bot can use during conversations.
        Tool implementations should be provided as methods named 'tools_<toolname>'.
        
        Returns:
            list: Tool schema following Anthropic's API format
        
        See https://docs.anthropic.com/en/api/messages#body-tools for more info on the structure
        of the tool schema.
        
        The actual tool call functions are implemented in a subclass as methods such as
             def tool_<toolname>(self, paramname1=None, paramname2=None, ...)
        """
        return []
    
    def handle_tool_call(self, tooluseblock):
        """Execute a tool call based on the provided tool use block. Looks for a function to
        handle the block, having the format:
            def tool_<tooluseblock.name>(self, paramname1=None, paramname2=None, ...)
        
        The function returns a dict in the format: 
            {
                "target": "<'client' or 'model'>,
                "message": <message for the client or the model, flexible format>
            }
        
        Args:
            tooluseblock: Tool use block containing name and input parameters
            
        Returns:
            The result of the tool execution
            
        Raises:
            Exception: If the requested tool function is not found
        """
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
        """Hook to intercept and potentially modify messages before sending to the model.
        
        Override this method to implement custom message handling, canned responses,
        or message preprocessing logic.
        
        Args:
            message_text (str): The incoming message text
            conversation (Conversation): The conversation context
            
        Returns:
            None: Forward message to model as normal
            dict: Custom message to append and send to model
            str: Send as canned response (using default include_in_context=True)
            tuple: (response_text, include_in_context) for more control
        """
        return None
    
    def sysprompt_vec(self, argv):
        """Generate a system prompt with template variable substitution.
        
        Replaces template variables in the format {{field}} with values from argv
        based on the bot's fields configuration.
        
        Args:
            argv (list): Values to substitute into template variables
            
        Returns:
            str or dict: The system prompt with substituted values
        """
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
    """Wrapper for pre-defined responses that bypass the AI model.
    
    Used to return static responses while maintaining compatibility with
    the conversation flow and streaming interfaces.
    """
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


class Conversation(object):
    """Manages a conversation session between a user and a bot.
    
    Handles message history, tool use coordination, streaming responses,
    and conversation state management. Supports both synchronous and
    asynchronous operation modes.
    """
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
        if (soft_start or (self.bot.soft_start and not soft_start is False)) and self.bot.welcome_message:
            self.messages.append(self._make_text_message('assistant', self.bot.welcome_message))
            self.message_objects.append(None)
            self.soft_started = True
        else:
            self.soft_started = False
        self.is_streaming = stream
        self.started = False
        if argv is not None:
            self.prestart(argv)
        else:
            self.argv = []
    
    def _convert_argv_if_needed(self, args, strict=True):
        if type(args) is dict:
            field_values = []
            missing_fields = []
            for k in self.bot.fields:
                try:
                    field_values.append(args[k])
                except KeyError as exc:
                    field_values.append('<VALUE MISSING>')
                    missing_fields.append(k)
            if missing_fields and strict:
                fieldsmissing = ', '.join(missing_fields)
                raise FieldValuesMissingException(f"Values missing for fields: [{fieldsmissing}]")
            return field_values
        else:
            return args
    
    @classmethod
    def _make_text_message(klass, role, content):
        return klass._make_generic_message(
            role, 
            [klass._make_message_text_segment(content)]
        )
    
    @staticmethod
    def _make_generic_message(role, content):
        return {
            'role': role,
            'content': content
        }
    
    @staticmethod
    def _make_message_text_segment(content):
        return {
            'type': 'text',
            'text': content
        }
    
    @staticmethod
    def _make_message_file_segment(filespec):
        """filespec is (mimetype, filething, blocktype)
        filething can be a filepath or a file-like object.
        """
        """blocktype (per Claude API) is one of image, document, container_upload"""
        import base64
        mimetype, filething, blocktype = filespec
        if hasattr(filething, 'read') and callable(filething.read):
            filedat = filething.read()
        else:
            with open(filething, 'rb') as inputfile:
                filedat = inputfile.read()
            
        return {
            'type': blocktype,
            'source': {
                'type': 'base64',
                'media_type': mimetype,
                'data': base64.b64encode(filedat).decode('utf-8')
            }
        }
    
    @staticmethod
    def _infer_filespec_from_filename(filepath):
        p = Path(filepath)
        suffix = p.suffix.lower()
        try:
            media_type, blocktype = MEDIA_TYPE_MAP[suffix]
        except KeyError as exc:
            raise Exception(f"Unrecognised media type suffix: {suffix}") from exc
        return (media_type, filepath, blocktype)
    
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
            if tu_id:
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
                if (target := tub.response['target']) == 'model':
                    tub.status = 'READY'
                elif target == 'client':
                    tub.status = 'WAITING'
    
    def _handle_waiting_tool_requests(self):
        """Handle requests that are in 'WAITING' state, ie. that have target "client" but haven't sent
        their message yet.
        WAITING requests are a bit tricky because they kind of work outside the usual flow; for now, 
        it's probably best to try to avoid mixing them with tool calls that target "model", and aim for
        only one to be sent at a time.
        This function concatenates the 'message' keys of anything found to be in WAITING state into a
        CannedMessage and returns it; if nothing is found, it returns None.
        NB: if model- and client-targeted tool calls are mixed in parallel, behaviour may be 
        unpredictable (including double-execution of model-targeted tools). It's recommended that you
        structure your system prompt and/or tool descriptions accordingly.
        """
        waiting_messages = [str(tub.response['message']) for tub in self.tool_use_blocks.pending if tub.status == 'WAITING']
        if waiting_messages:
            return '\n'.join(waiting_messages)
    
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
        """Initialize the conversation with template arguments.
        
        Sets up the system prompt by substituting template variables
        and marks the conversation as started.
        
        Args:
            argv (list): Arguments for system prompt template substitution
        """
        self.argv = self._convert_argv_if_needed(argv)
        self.sysprompt = self.bot.sysprompt_vec(self.argv)
        self.started = True
    
    def start(self, *args):
        """Start a new conversation with an initial message.
        
        Args:
            *args: Either (argv, message) or just (message)
                argv (list): Template arguments for system prompt
                message (str): The initial user message
                
        Returns:
            The bot's response to the initial message
            
        Raises:
            Exception: If conversation has already started
        """
        if type(args[0]) is list:
            argv, message = args
        else:
            argv, message = [], args[0]
        if self.started:
            raise Exception('Conversation has already started')
            
        self.prestart(argv)
        return self.resume(message)

    async def astart(self, *args):
        """Start a new conversation asynchronously with an initial message.
        
        Args:
            *args: Either (argv, message) or just (message)
                argv (list): Template arguments for system prompt  
                message (str): The initial user message
                
        Returns:
            The bot's response to the initial message
            
        Raises:
            Exception: If conversation has already started
        """
        if type(args[0]) is list:
            argv, message = args
        else:
            argv, message = [], args[0]
        if self.started:
            raise Exception('Conversation has already started')
        
        self.prestart(argv)
        return await self.aresume(message)

    async def aresume(self, message, with_files=[]):
        """Continue the conversation asynchronously with a new message.
        
        Args:
            message (str): The user's message
            
        Returns:
            The bot's response, which may be a model response or canned response
        """
        # Check for canned response first
        canned_response = self.bot.preprocess_response(message, self)
        is_tool_message = False
        if type(canned_response) is dict:
            message = canned_response
            is_tool_message = True
        elif canned_response is not None:
            return self._handle_canned_response(message, canned_response)
        
        if self.is_streaming:
            return await self._aresume_stream(message, is_tool_message=is_tool_message, with_files=with_files)
        else:
            return await self._aresume_flat(message, is_tool_message=is_tool_message, with_files=with_files)

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

    async def _aresume_stream(self, message, is_tool_message=False, with_files=[]):
        if is_tool_message:
            self.messages.append(message)
        else:
            self.messages.append(self._compile_user_message(message, with_files=with_files))
        stream = self.bot.client.messages.stream(
            model=self.bot.model, max_tokens=self.max_tokens,
            temperature=self.bot.temperature, system=self.sysprompt,
            messages=self._get_conversation_context(),
            tools=self.bot.get_tools_schema(),
        )
        return STREAM_WRAPPER_CLASS_ASYNC(stream, self)

    async def _aresume_flat(self, message, is_tool_message=False, with_files=[]):
        if is_tool_message:
            self.messages.append(message)
        else:
            self.messages.append(self._compile_user_message(message, with_files=with_files))
    
        message_out = await self.bot.client.messages.create(
            model=self.bot.model, max_tokens=self.max_tokens,
            temperature=self.bot.temperature, system=self.sysprompt,
            messages=self._get_conversation_context(),
            tools=self.bot.get_tools_schema(),
        )
        self.message_objects.append(message_out)
    
        # Process all content blocks in the response
        accumulated_context = []
        response_text = ""
    
        for contentblock in message_out.content:
            blocktype = type(contentblock).__name__
            if blocktype == 'ToolUseBlock':
                treq = {
                    'type': 'tool_use',
                    'id': contentblock.id,
                    'name': contentblock.name,
                    'input': contentblock.input,
                }
                self._add_tool_request(treq)
                accumulated_context.append(treq)
            elif hasattr(contentblock, 'text'):
                ttxt = {
                    'type': 'text',
                    'text': contentblock.text,
                }
                accumulated_context.append(ttxt)
                response_text += contentblock.text
            else:
                raise Exception(f"Don't know what to do with blocktype: {blocktype}")
    
        # Add the assistant's response to messages
        self.messages.append({'role': 'assistant', 'content': accumulated_context})
    
        # Handle tool calls if conversation is not exhausted (i.e., if there are pending tool calls)
        if not self._is_exhausted():
            self._handle_pending_tool_requests()
        
            # Check for client-targeted responses first
            msg_out = self._handle_waiting_tool_requests()
            if msg_out is not None:
                return self._handle_canned_response(None, (msg_out, False))
            else:
                # Handle model-targeted tool responses
                resps = self._compile_tool_responses()
                return await self._aresume_flat(resps, is_tool_message=True)
    
        return message_out
    
    def resume(self, message, with_files=[]):
        """Continue the conversation with a new message.
        
        Args:
            message (str): The user's message
            
        Returns:
            The bot's response, which may be a model response or canned response
        """
        # Check for canned response first
        canned_response = self.bot.preprocess_response(message, self)
        is_tool_message = False
        if type(canned_response) is dict:
            message = canned_response
            is_tool_message = True
        elif canned_response is not None:
            return self._handle_canned_response(message, canned_response)
        
        if self.is_streaming:
            return self._resume_stream(message, is_tool_message=is_tool_message, with_files=with_files)
        else:
            return self._resume_flat(message, is_tool_message=is_tool_message, with_files=with_files)

    @classmethod
    def _compile_user_message(klass, message, with_files=[]):
        if with_files:
            message_blocks = []
            for fspec in with_files:
                if type(fspec) is not tuple:
                    fspec = klass._infer_filespec_from_filename(fspec)
                message_blocks.append(
                    klass._make_message_file_segment(fspec)
                )
            message_blocks.append(klass._make_message_text_segment(message))
            message_out = klass._make_generic_message('user', message_blocks)
        else:
            message_out = klass._make_text_message('user', message)
        return message_out

    def _resume_stream(self, message, is_tool_message=False, with_files=[]):
        if is_tool_message:
            self.messages.append(message)
        else:
            self.messages.append(self._compile_user_message(message, with_files=with_files))
        
        stream = self.bot.client.messages.stream(
            model=self.bot.model, max_tokens=self.max_tokens,
            temperature=self.bot.temperature, system=self.sysprompt,
            messages=self._get_conversation_context(),
            tools=self.bot.get_tools_schema(),
        )
        return STREAM_WRAPPER_CLASS_SYNC(stream, self)

    def _resume_flat(self, message, is_tool_message=False, with_files=[]):
        if is_tool_message:
            self.messages.append(message)
        else:
            self.messages.append(self._compile_user_message(message, with_files=with_files))
    
        message_out = self.bot.client.messages.create(
            model=self.bot.model, max_tokens=self.max_tokens,
            temperature=self.bot.temperature, system=self.sysprompt,
            messages=self._get_conversation_context(),
            tools=self.bot.get_tools_schema(),
        )
        self.message_objects.append(message_out)
    
        # Process all content blocks in the response
        accumulated_context = []
        response_text = ""
    
        for contentblock in message_out.content:
            blocktype = type(contentblock).__name__
            if blocktype == 'ToolUseBlock':
                treq = {
                    'type': 'tool_use',
                    'id': contentblock.id,
                    'name': contentblock.name,
                    'input': contentblock.input,
                }
                self._add_tool_request(treq)
                accumulated_context.append(treq)
            elif hasattr(contentblock, 'text'):
                ttxt = {
                    'type': 'text',
                    'text': contentblock.text,
                }
                accumulated_context.append(ttxt)
                response_text += contentblock.text
            else:
                raise Exception(f"Don't know what to do with blocktype: {blocktype}")
    
        # Add the assistant's response to messages
        self.messages.append({'role': 'assistant', 'content': accumulated_context})
    
        # Handle tool calls if conversation is not exhausted (i.e., if there are pending tool calls)
        if not self._is_exhausted():
            self._handle_pending_tool_requests()
        
            # Check for client-targeted responses first
            msg_out = self._handle_waiting_tool_requests()
            if msg_out is not None:
                return self._handle_canned_response(None, (msg_out, False))
            else:
                # Handle model-targeted tool responses
                resps = self._compile_tool_responses()
                return self._resume_flat(resps, is_tool_message=True)
    
        return message_out
    
    def _post_stream_hook(self):
        pass
    
    async def _post_stream_hook_async(self):
        pass


class LoggedConversation(Conversation):
    """A conversation that automatically logs all interactions to disk.
    
    Extends Conversation to provide persistent storage of conversation history
    in JSON format, enabling conversation resumption and analysis.
    """
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
    def revive(klass, bot, conversation_id, logs_dir, argv=[], **kwargs):
        """Restore a previously logged conversation from disk.
        
        Args:
            bot: The bot instance or class to use for the conversation
            conversation_id (str): The unique identifier of the conversation to restore
            logs_dir (str): Directory containing the conversation logs
            argv (list): Template arguments for system prompt
            **kwargs: Additional arguments to be passed to the superclass constructor
            
        Returns:
            LoggedConversation: The restored conversation instance
            
        Raises:
            UnknownConversationException: If the conversation ID is not found
        """
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
    """Experimental conversation class that supports file attachments.
    
    Extends Conversation to handle image and file uploads alongside text messages.
    Currently supports image files and has limitations around tool use and async operation.
    """
    """This is experimental and needs work. Currently no async and no streaming. 
    Right now to use it you need to bypass resume() like:
        c = robo.ConversationWithFiles(Classifier, [])
        withjpeg = lambda fpath: c._resume_flat(None, [('image/jpeg', fpath)])
        withpng = lambda fpath: c._resume_flat(None, [('image/png', fpath)])
    DOES NOT CURRENTLY SUPPORT TOOL USE
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


def streamer(bot_or_conversation, args=[], cc=None):
    """Create a streaming conversation function for real-time output.
    
    Args:
        bot_or_conversation: Either a Bot instance/class or a Conversation with stream=True
        args (list): Template arguments for system prompt
        cc (file-like object): If provided, write the response text to the object (using obj.write())
            as well as printing it.
        
    Returns:
        function: A function that takes a message and streams the response to stdout
    """
    """If you're passing in a conversation, make sure it's got stream=True!"""
    if issubclass(type(bot_or_conversation), Conversation):
        convo = bot_or_conversation
    else: ## in which case it should be either a bot instance or Bot class
        convo = Conversation(bot_or_conversation, stream=True)
    def streamit(message, with_files=[]):
        if not convo.started:
            convo.prestart(args)
        with convo.resume(message, with_files=with_files) as stream:
            for chunk in stream.text_stream:
                print(chunk, end="", flush=True)
                if cc:
                    cc.write(chunk)
        print()
    return streamit

def streamer_async(bot_or_conversation, args=[]):
    """Create an async streaming conversation function for real-time output.
    
    Args:
        bot_or_conversation: Either a Bot instance/class or a Conversation with stream=True
        args (list): Template arguments for system prompt
        
    Returns:
        coroutine function: An async function that takes a message and streams the response to stdout
    """
    if issubclass(type(bot_or_conversation), Conversation):
        convo = bot_or_conversation
    else:
        convo = Conversation(bot_or_conversation, stream=True, async_mode=True)
    async def streamit(message, with_files=[]):
        if not convo.started:
            convo.prestart(args)
        async with await convo.aresume(message, with_files=with_files) as stream:
            async for chunk in stream.text_stream:
                print(chunk, end="", flush=True)
    return streamit

def gettext(message):
    text_out = ''
    for contentblock in message.content:
        if hasattr(contentblock, 'text'):
            text_out += contentblock.text
    return text_out

def getjson(message):
    return json.loads(gettext(message))

def printmsg(message):
    print(gettext(message))

"""
To use streamer_async :
>>> import asyncio
>>> from robo import *
>>> say = streamer_async(Bot)
>>> coro = say('who goes there?')
>>> asyncio.run(coro)
"""

__all__ = ['Bot', 'Conversation', 'streamer', 'streamer_async', 'gettext', 'getjson', 'printmsg', 'MODELS']
