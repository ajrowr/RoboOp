import anthropic

API_KEY_FILE = '/Users/alandatech/Research/2025/06-Claude/api_key.txt'

MODEL_SONNET4 = 'claude-sonnet-4-20250514'


class Bot(object):
    __slots__ = ['fields', 'sysprompt_path', 'client', 'model', 'temperature']
    
    @property
    def sysprompt_clean(self):
        return open(self.sysprompt_path).read()
    
    def sysprompt_vec(self, argv):
        sysp = self.sysprompt_clean
        for k, v in zip(self.fields, argv):
            sysp = sysp.replace(f'{{{{{k}}}}}', v)
        return sysp
    
    def __init__(self, client=None):
        if not client:
            client = anthropic.Anthropic(api_key=open(API_KEY_FILE).read())
        self.client = client


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
    def __init__(self, bot, stream=False):
        self.max_tokens = 20000
        if type(bot) is type:
            self.bot = bot()
        else:
            self.bot = bot
        self.messages = []
        self.message_objects = []
        self.is_streaming = stream
        self.started = False
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
    
    def start(self, argv, message):
        if self.started:
            raise Exception('Conversation has already started')
            
        self.prestart(argv)
        return self.resume(message)
    
    async def astart(self, message):
        ...
    
    async def aresume(self, message):
        ...
    
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


__all__ = ['Bot', 'Conversation', 'streamer']
