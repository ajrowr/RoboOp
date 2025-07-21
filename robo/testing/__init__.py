
from robo import *
from . import testbots
import asyncio

class BaseTester(object):
    def __init__(self, botclass=Bot, convclass=Conversation, argv=[]):
        self.botclass = botclass
        self.convclass = convclass
        self.argv = argv
        self.bot = None
        self.convo = None
        self.setup()
    
    def say(self, message, printout=True, returnmessage=True):
        raise NotImplementedError()
    
    def setup(self):
        raise NotImplementedError()

class SyncFlatTester(BaseTester):
    def setup(self):
        self.bot = self.botclass()
        self.convo = self.convclass(self.bot, self.argv)

    def say(self, message, printout=True, returnmessage=True):
        message_out = self.convo.resume(message)
        if printout:
            print(message_out.content[0].text)
        if returnmessage:
            return (message_out, message_out.content[0].text)

class SyncStreamTester(BaseTester):
    def setup(self):
        self.bot = self.botclass()
        self.convo = self.convclass(self.bot, self.argv, stream=True)
    
    def say(self, message, printout=True, returnmessage=True):
        text_out = ''
        with self.convo.resume(message) as stream:
            for chunk in stream.text_stream:
                if printout:
                    print(chunk, end='', flush=True)
                text_out += chunk
        if returnmessage:
            return (None, text_out)

class AsyncFlatTester(BaseTester):
    def setup(self):
        self.bot = self.botclass(async_mode=True)
        self.convo = self.convclass(self.bot, self.argv, async_mode=True)
    
    def say(self, message, printout=True, returnmessage=True):
        coro = self.convo.aresume(message)
        message_out = asyncio.run(coro)
        if printout:
            print(message_out.content[0].text)
        if returnmessage:
            return (message_out, message_out.content[0].text)

class AsyncStreamTester(BaseTester):
    def setup(self):
        self.bot = self.botclass(async_mode=True)
        self.convo = self.convclass(self.bot, self.argv, async_mode=True, stream=True)
    
    def say(self, message, printout=True, returnmessage=True):
        text_out = ''
        async def streamit(message):
            nonlocal text_out
            async with await self.convo.aresume(message) as stream:
                async for chunk in stream.text_stream:
                    text_out += chunk
                    if printout:
                        print(chunk, end='', flush=True)
        asyncio.run(streamit(message))
        if returnmessage:
            return (None, text_out)


def tester_variants():
    for klass in [SyncFlatTester, SyncStreamTester, AsyncFlatTester, AsyncStreamTester]:
        yield klass


## Scenarios
def scenario1():
    """Tests client-targeted tool use"""
    for tklass in tester_variants():
        tester = tklass(testbots.GuidedNavigationTester)
        print(f'\n\n --- {tklass.__name__} ---')
        msgobj, msgtxt = tester.say("Please navigate me to /docs/django")
        if msgtxt.startswith('@@@@NAVIGATE'):
            tester.say('@@@@RECONNECT')
        else:
            raise Exception("The bot didn't behave as expected.")

def scenario2():
    """Tests sequential tool use"""
    for tklass in tester_variants():
        tester = tklass(testbots.ToolsTesterTravelPlanner)
        print(f'\n\n --- {tklass.__name__} ---')
        tester.say("I'm planning a trip from New York to Los Angeles. Can you help me compare the weather conditions in both cities and then calculate the best route between them?")
        
