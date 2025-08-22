import pytest
import json
import os
import asyncio
import anthropic
from unittest.mock import Mock, patch, AsyncMock
# from robo import Bot, Conversation, MODELS
from robo import *
from robo.exceptions import *
import robo
from io import StringIO



from robo import *
import asyncio
from robo.tools import *
from robo.testing.fakeanthropic import *

class ToolTesterBot(Bot):
    class GetWeather(Tool):
        description = "Get weather"
        parameter_descriptions = {
            'location': 'the location'
        }
        
        def __call__(self, location:str):
            print('GetWeather called with', location)
            return "Sunny, 23° celcius"
    
    class Calculate(Tool):
        description = "Calculate"
        parameter_descriptions = {
            'expression': 'the expression'
        }
        
        def __call__(self, expression:str):
            print('Calculate called with', expression)
            return '4'
    
    tools = [GetWeather, Calculate]



scenarios = {
    "test input": ["expected response"],
    "tool test": [{
        'type': 'tool_use',
        'id': 'toolu_123',
        'name': 'my_tool',
        'input': {'param': 'value'}
    }]
}

fake_client = lambda: FakeAnthropic(response_scenarios=scenarios)
fake_client_async = lambda: FakeAsyncAnthropic(response_scenarios=scenarios)


class TestToolUse:
    def test_tooluse_sync_flat(self):
        conv = Conversation(ToolTesterBot(client=fake_client()), [])
        # conv.register_callback(response_complete, )
        msg = conv.resume('calculate')
        assert gettext(msg) == 'Tool response was:4'

    def test_tooluse_sync_stream(self):
        conv = Conversation(ToolTesterBot(client=fake_client()), [], stream=True)
        sio = StringIO()
        say = streamer(conv, cc=sio)
        say('weather')
        sio.seek(0)
        assert sio.read() == 'Tool response was:Sunny, 23° celcius'

    def test_tooluse_async_flat(self):
        conv = Conversation(ToolTesterBot(client=fake_client_async()), [], async_mode=True)
        coro = conv.aresume('calculate')
        msg = asyncio.run(coro)
        assert gettext(msg) == 'Tool response was:4'

    def test_tooluse_async_stream(self):
        conv = Conversation(ToolTesterBot(client=fake_client_async()), [], stream=True, async_mode=True)
        sio = StringIO()
        say = streamer_async(conv, cc=sio)
        coro = say('weather')
        asyncio.run(coro)
        sio.seek(0)
        assert sio.read() == 'Tool response was:Sunny, 23° celcius'

def make_sio_callback():
    sio = StringIO()
    def sio_callback(message):
        sio.write(gettext(message))
    return (sio, sio_callback)

class TestCallbacks:
    def test_callbacks_sync_flat(self):
        conv = Conversation(ToolTesterBot(client=fake_client()), [])
        sio, cb = make_sio_callback()
        conv.register_callback('response_complete', cb)
        conv.resume('test input')
        sio.seek(0)
        assert sio.read() == 'expected response'

    def test_callbacks_sync_stream(self):
        conv = Conversation(ToolTesterBot(client=fake_client()), [], stream=True)
        sio, cb = make_sio_callback()
        conv.register_callback('response_complete', cb)
        with conv.resume('test input') as streamwrapper:
            for chunk in streamwrapper.text_stream:
                pass
        
        sio.seek(0)
        assert sio.read() == 'expected response'
    
    def test_callbacks_async_flat(self):
        conv = Conversation(ToolTesterBot(client=fake_client_async()), [], async_mode=True)
        sio, cb = make_sio_callback()
        conv.register_callback('response_complete', cb)
        coro = conv.aresume('test input')
        msg = asyncio.run(coro)
        sio.seek(0)
        assert sio.read() == 'expected response'
    
    def test_callbacks_async_stream(self):
        conv = Conversation(ToolTesterBot(client=fake_client_async()), [], stream=True, async_mode=True)
        sio, cb = make_sio_callback()
        conv.register_callback('response_complete', cb)
        say = streamer_async(conv)
        coro = say('test input')
        asyncio.run(coro)
        sio.seek(0)
        assert sio.read() == 'expected response'
        
