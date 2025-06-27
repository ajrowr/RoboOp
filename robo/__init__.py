import anthropic

API_KEY_FILE = None ## If you want to load it from a file
API_KEY_ENV_VAR = None ## If you want to use a different env var instead of ANTHROPIC_API_KEY

from .models import MODELS

def _get_api_key():
    if API_KEY_FILE:
        return open(API_KEY_FILE).read()
    elif API_KEY_ENV_VAR:
        import os
        return os.environ[API_KEY_ENV_VAR]
    ## If neither, then returning None will let Anthropic check its default of ANTHROPIC_API_KEY


class Bot(object):
    __slots__ = ['fields', 'sysprompt_path', 'sysprompt_text', 'client', 'model', 
            'temperature', 'max_tokens']
    
    @property
    def sysprompt_clean(self):
        if hasattr(self, 'sysprompt_text'):
            return self.sysprompt_text
        elif hasattr(self, 'sysprompt_path'):
            return open(self.sysprompt_path).read()
        else:
            return ''
    
    def sysprompt_vec(self, argv):
        sysp = self.sysprompt_clean
        for k, v in zip(self.fields, argv):
            sysp = sysp.replace(f'{{{{{k}}}}}', v)
        return sysp
    
    def __init__(self, client=None):
        for f, v in [('model', MODELS.CLAUDE_4.SONNET), ('temperature', 1), ('fields', []), 
                    ('max_tokens', 20000)]:
            if not hasattr(self, f):
                setattr(self, f, v)
        if not client:
            client = anthropic.Anthropic(api_key=_get_api_key())
        self.client = client
    
    @classmethod
    def with_api_key(klass, api_key):
        client = anthropic.Anthropic(api_key=api_key)
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
        
        return result
    
    @property
    def text_stream(self):
        for text in self.stream_context.text_stream:
            self.chunks.append(text)
            self.accumulated_text += text
            yield text


class Conversation(object):
    __slots__ = ['messages', 'bot', 'sysprompt', 'argv', 'max_tokens', 'message_objects', 
                'is_streaming', 'started']
    def __init__(self, bot, argv=None, stream=False):
        if type(bot) is type:
            self.bot = bot()
        else:
            self.bot = bot
        self.max_tokens = self.bot.max_tokens
        self.messages = []
        self.message_objects = []
        self.is_streaming = stream
        self.started = False
        if argv is not None:
            self.prestart(argv)
        # self.is_async = is_async
    
    def _make_text_message(self, role, content):
        return {
            'role': role,
            'content': [{
                'type': 'text',
                'text': content
            }]
        }
    
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
    
    async def astart(self, message):
        raise NotImplementedError()
    
    async def aresume(self, message):
        raise NotImplementedError()
    
    def resume(self, message):
        if self.is_streaming:
            return self._resume_stream(message)
        else:
            return self._resume_flat(message)
    
    def _resume_stream(self, message):
        self.messages.append(self._make_text_message('user', message))
        stream = self.bot.client.messages.stream(
            model=self.bot.model, max_tokens=self.max_tokens,
            temperature=self.bot.temperature, system=self.sysprompt,
            messages=self.messages
        )
        return StreamWrapper(stream, self)
    
    def _resume_flat(self, message):
        self.messages.append(self._make_text_message('user', message))
        message_out = self.bot.client.messages.create(
            model=self.bot.model, max_tokens=self.max_tokens,
            temperature=self.bot.temperature, system=self.sysprompt,
            messages=self.messages
        )
        self.message_objects.append(message_out)
        self.messages.append(self._make_text_message('assistant', message_out.content[0].text))
        return message_out
        

def streamer(bot, args):
    convo = Conversation(bot, stream=True)
    def streamit(message):
        if not convo.started:
            convo.prestart(args)
        with convo.resume(message) as stream:
            for chunk in stream.text_stream:
                print(chunk, end="", flush=True)
    return streamit


__all__ = ['Bot', 'Conversation', 'streamer', 'MODELS']
