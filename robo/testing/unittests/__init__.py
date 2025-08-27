import pytest
import json
import os
import asyncio
import anthropic
from unittest.mock import Mock, patch, AsyncMock, mock_open
import contextlib
import tempfile
from datetime import datetime

import robo
from robo import *
from robo.exceptions import *
from robo.tools import *
from robo.testing.fakeanthropic import *
from robo.streamwrappers import StreamWrapper, AsyncStreamWrapper

from io import StringIO, BytesIO
from types import SimpleNamespace


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


class TimeWeatherLocationTestBot(Bot):
    class GetUserLocation(Tool):
        description = """Get the user's location."""
        
        def __call__(self):
            return "Auckland, New Zealand"
    
    class GetLocationWeather(Tool):
        description = """Get the weather for the named location."""
        
        def __call__(self, location:str):
            return "Sunny, 23°"
        
        parameter_descriptions = {
            'location': "The location to fetch weather data for"
        }
    
    class GetLocationTime(Tool):
        description = """Get the current time for the named location."""
        def __call__(self, location:str):
            from datetime import datetime
            return str(datetime.now())
        
        parameter_descriptions = {
            'location': "The location to fetch weather data for"
        }
    
    tools = [GetUserLocation, GetLocationWeather, GetLocationTime]
    
    test_scenario = {
        'tool test': [
            {
                'type': 'tool_use',
                'id': 'toolu_00001',
                'name': 'GetUserLocation',
                'input': {}
            },
            {
                'type': 'tool_use',
                'id': 'toolu_00002',
                'name': 'GetLocationWeather',
                'input': {'location': 'San Francisco'}
            },
        
        ]
    }
    


class TimerBot(Bot):
    sysprompt_text = "You can help users time activities. Use the timer tool when they ask."
    
    class StartTimer(Tool):
        description = 'Start a timer for a specified number of seconds'
        parameter_descriptions = {
            'seconds': 'Number of seconds to time',
        }
        
        def call_sync(self, seconds: float):
            import time
            time.sleep(seconds)
            return f"Synchronous timer finished! {seconds} seconds have elapsed."
        
        async def call_async(self, seconds: float):
            await asyncio.sleep(seconds)
            return f"Asynchronous timer finished! {seconds} seconds have elapsed."
    
    tools = [StartTimer]
    
    test_scenario = {
        'tool test': [{
            'type': 'tool_use',
            'id': 'toolu_98765',
            'name': 'StartTimer',
            'input': {'seconds': 0.2}
        }],
        'tool test 1s': [{
            'type': 'tool_use',
            'id': 'toolu_98765',
            'name': 'StartTimer',
            'input': {'seconds': 1.0}
        }]
    }


class TestAsyncToolCalls:
    def test_async_tool_defs(self):
        class TestTool(Tool):
            parameter_descriptions = {
                'numeric': 'An integer number',
                'characters': 'A string',
            }
            description = 'A tool'
        
        correct_tool_schema = {'name': 'MyTool', 'description': 'A tool', 'input_schema': {'type': 'object', 'properties': {'numeric': {'type': 'number', 'description': 'An integer number'}, 'characters': {'type': 'string', 'description': 'A string'}}, 'required': ['numeric', 'characters']}}
        
        class Bot1(Bot):
            class MyTool(TestTool):
                def __call__(self, numeric:int, characters:str):
                    return str(numeric) + characters
        
        assert Bot1.MyTool.get_call_schema() == correct_tool_schema

        class Bot2(Bot):
            class MyTool(TestTool):
                def call_sync(self, numeric:int, characters:str):
                    pass
        
        assert Bot2.MyTool.get_call_schema() == correct_tool_schema

        class Bot3(Bot):
            class MyTool(TestTool):
                async def call_async(self, numeric:int, characters:str):
                    pass
        
        assert Bot3.MyTool.get_call_schema() == correct_tool_schema
        
        assert asyncio.run(Bot1.MyTool().call_async(numeric=1, characters='a')) == '1a'
    
    def test_async_tool_calls(self):
        tub = {'id': 'tu_12345', 'name': 'StartTimer', 'input': {'seconds': 0.01}}
        assert TimerBot().handle_tool_call(tub)['message'].startswith('Synchronous timer finished!')
        coro = TimerBot().ahandle_tool_call(tub)
        assert asyncio.run(coro)['message'].startswith('Asynchronous timer finished!')
    
    def test_concurrent_tool_calls(self):
        async def testset(n):
            coroutines = []
            for i in range(n):
                myconv = Conversation(TimerBot(client=FakeAsyncAnthropic(response_scenarios=TimerBot.test_scenario)), [], async_mode=True)
                coroutines.append(myconv.aresume('tool test'))

            g = await asyncio.gather(*coroutines)
            return list(map(gettext, g))
        d0 = datetime.now()
        ts = asyncio.run(testset(100))
        d1 = datetime.now()
        assert len(ts) == 100
        assert (d1 - d0).total_seconds() < 1


class ToolTesterBotOldStyle(Bot):
    def get_tools_schema(self):
        return [
            {
                "name": "get_url",
                "description": "Fetch the raw HTML from a given URL.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to fetch"
                        }
                    },
                    "required": ["url"]
                }
            },
            {
                "name": "get_url2", ## this one deliberately doesn't exist
                "description": "Fetch the raw HTML from a given URL.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to fetch"
                        }
                    },
                    "required": ["url"]
                }
            }
        ]
    
    def tools_get_url(self, url=None):
        pagetext = """<html><head><title>Fake page</title></head></html>"""
        return {
            'message': pagetext,
            'target': 'model'
        }


class FieldsTesterBot(Bot):
    sysprompt_text = """Respond with a stereotypical sound made by a {{ANIMAL_TYPE}}."""
    fields = ['ANIMAL_TYPE']


class ClientToolTestBot(Bot):
    def get_tools_schema(self):
        return [
            {
                'name': 'guided_navigate',
                'description': "Navigates a user to a specific link on the site.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "destination": {
                            "type": "string",
                            "description": "The destination link on the site to navigate the user to."
                        }
                    },
                    "required": ["destination"]
                }
            },
            
        ]
    
    def tools_guided_navigate(self, destination=None):
        return {
            "message": f'\n@@@@NAVIGATE {destination}',
            "target": "client"
        }
    
    def preprocess_response(self, message, conversation):
        """If we got a RECONNECT then we need to match the tool use ID up"""
        if message.startswith('@@@@RECONNECT'):
            tu_id = conversation._get_last_tool_use_id()
            return conversation._make_tool_result_message({'id': tu_id}, "@@@@RECONNECT")


_IN1 = 'test input'
_OUT1 = 'expected response'
_IN2 = 'test input 2'
_OUT2 = 'expected response 2'

scenarios = {
    _IN1: [_OUT1],
    _IN2: [_OUT2],
    "tool test": [{
        'type': 'tool_use',
        'id': 'toolu_123',
        'name': 'my_tool',
        'input': {'param': 'value'}
    }],
    "json test": [
        """{"one": "two", "three": "four"}"""
    ],
}

fake_client = lambda: FakeAnthropic(response_scenarios=scenarios)
fake_client_async = lambda: FakeAsyncAnthropic(response_scenarios=scenarios)

def make_sio_callback():
    sio = StringIO()
    def sio_callback(conversation, message):
        assert issubclass(type(conversation), Conversation)
        sio.write(gettext(message))
    return (sio, sio_callback)


class TestMisc:
    def test_count_tokens(self):
        conv = Conversation(Bot, [])
        tcount = conv.count_tokens('hello everybody')
        assert tcount.input_tokens < 15 and tcount.input_tokens > 5
        assert len(conv.messages) == 0


class TestLoggedConversation:
    def test_logged_conversation_sync_flat(self):
        bot = Bot(client=fake_client())
        with tempfile.TemporaryDirectory() as tmpdir:
            loggedconv1 = LoggedConversation(bot, logs_dir=tmpdir).prestart([])
            loggedconv1.resume('one')
            loggedconv1.resume('two')
            loggedconv2 = LoggedConversation.revive(bot, conversation_id=loggedconv1.conversation_id, logs_dir=tmpdir)
            loggedconv2.resume('three')
        y = lambda m: 'one' in (s := m['content'][0]['text']) or 'two' in s or 'three' in s
        assert all([y(m) for m in loggedconv2.messages])
        assert len(loggedconv2.messages) == 6
        assert repr(loggedconv1) == repr(loggedconv2)
    
    def test_logged_conversation_sync_stream(self):
        bot = Bot(client=fake_client())
        with tempfile.TemporaryDirectory() as tmpdir:
            loggedconv1 = LoggedConversation(bot, logs_dir=tmpdir, stream=True).prestart([])
            with loggedconv1.resume('one') as stream:
                for chunk in stream.text_stream:
                    pass
            with loggedconv1.resume('two') as stream:
                for chunk in stream.text_stream:
                    pass
                
            loggedconv2 = LoggedConversation.revive(bot, conversation_id=loggedconv1.conversation_id, 
                        logs_dir=tmpdir, stream=True)
            with loggedconv2.resume('three') as stream:
                for chunk in stream.text_stream:
                    pass
                
        y = lambda m: 'one' in (s := m['content'][0]['text']) or 'two' in s or 'three' in s
        assert all([y(m) for m in loggedconv2.messages])
        assert len(loggedconv2.messages) == 6
        assert repr(loggedconv1) == repr(loggedconv2)
    
    def test_logged_conversation_async_flat(self):
        bot = Bot(client=fake_client_async())
        with tempfile.TemporaryDirectory() as tmpdir:
            loggedconv1 = LoggedConversation(bot, logs_dir=tmpdir, async_mode=True).prestart([])
            asyncio.run(loggedconv1.aresume('one'))
            asyncio.run(loggedconv1.aresume('two'))
            loggedconv2 = LoggedConversation.revive(bot, conversation_id=loggedconv1.conversation_id, 
                logs_dir=tmpdir, async_mode=True)
            asyncio.run(loggedconv2.aresume('three'))
        y = lambda m: 'one' in (s := m['content'][0]['text']) or 'two' in s or 'three' in s
        assert all([y(m) for m in loggedconv2.messages])
        assert len(loggedconv2.messages) == 6
        assert repr(loggedconv1) == repr(loggedconv2)
    
    def test_logged_conversation_async_stream(self):
        bot = Bot(client=fake_client_async())
        with tempfile.TemporaryDirectory() as tmpdir:
            loggedconv1 = LoggedConversation(bot, logs_dir=tmpdir, stream=True, async_mode=True).prestart([])
            async def part1():
                async with await loggedconv1.aresume('one') as stream:
                    async for chunk in stream.text_stream:
                        pass
                async with await loggedconv1.aresume('two') as stream:
                    async for chunk in stream.text_stream:
                        pass
            
            asyncio.run(part1())
            
            loggedconv2 = LoggedConversation.revive(bot, conversation_id=loggedconv1.conversation_id, 
                        logs_dir=tmpdir, stream=True, async_mode=True)
            async def part2():
                async with await loggedconv2.aresume('three') as stream:
                    async for chunk in stream.text_stream:
                        pass
            
            asyncio.run(part2())
        
        y = lambda m: 'one' in (s := m['content'][0]['text']) or 'two' in s or 'three' in s
        assert all([y(m) for m in loggedconv2.messages])
        assert len(loggedconv2.messages) == 6
        assert repr(loggedconv1) == repr(loggedconv2)
    
    def test_lc_other(self):
        with pytest.raises(Exception, match='logs_dir required'):
            loggedconv1 = LoggedConversation(Bot)
        
        with pytest.raises(UnknownConversationException):
            with tempfile.TemporaryDirectory() as tmpdir:
                loggedconv = LoggedConversation.revive(Bot, conversation_id='INVALID', 
                        logs_dir=tmpdir)


class TestUtils:
    def test_sync_streamer(self):
        with patch.object(robo, '_get_client_class') as mock_client_class:
            mock_client_class.return_value = FakeAnthropic
            sio = StringIO()
            say = streamer(Bot, cc=sio)
            say('test input')
            sio.seek(0)
            msgtext = sio.read()
            assert msgtext.startswith("I understand you said: 'test input'")
    
    def test_async_streamer(self):
        with patch.object(robo, '_get_client_class') as mock_client_class:
            mock_client_class.return_value = FakeAsyncAnthropic
            sio = StringIO()
            asay = streamer_async(Bot, cc=sio)
            asyncio.run(asay('test input'))
            sio.seek(0)
            msgtext = sio.read()
            assert msgtext.startswith("I understand you said: 'test input'")
    
    def test_getjson(self):
        bot = Bot(client=fake_client())
        conv = Conversation(bot, [])
        msg = conv.resume('json test')
        assert getjson(msg) == {'one': 'two', 'three': 'four'}
    
    def test_printmsg(self):
        bot = Bot(client=fake_client())
        conv = Conversation(bot, [])
        msg = conv.resume('test input')
        buffer = StringIO()
        with contextlib.redirect_stdout(buffer):
            printmsg(msg)
        buffer.seek(0)
        assert buffer.read() == _OUT1 + '\n'


class TestFileHandling:
    def test_file_segment_generate(self):
        seg_ideal = {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': 'MTIzNDU2Nzg5MA=='}}
        
        seg1 = Conversation._make_message_file_segment(('image/png', b'1234567890', 'image'))
        assert seg1 == seg_ideal
        
        bio = BytesIO(b'1234567890')
        seg3 = Conversation._make_message_file_segment(('image/png', bio, 'image'))
        assert seg3 == seg_ideal
        
        with patch('builtins.open', mock_open(read_data=b'1234567890')):
            seg4 = Conversation._make_message_file_segment(('image/png', '/tmp/xyz', 'image'))
            assert seg4 == seg_ideal
    
    def test_filename_inference(self):
        assert Conversation._infer_filespec_from_filename('xyz.jpg') == ('image/jpeg', 'xyz.jpg', 'image')
        assert Conversation._infer_filespec_from_filename('xyz.png') == ('image/png', 'xyz.png', 'image')
        assert Conversation._infer_filespec_from_filename('xyz.pdf') == ('application/pdf', 'xyz.pdf', 'document')
        
        with pytest.raises(Exception, match='Unrecognised media type suffix'):
            Conversation._infer_filespec_from_filename('xyz.zip')
    
    def test_compile_user_messages_file_handling(self):
        filespecs = [('image/png', b'1234567890', 'image')]
        assert Conversation._compile_user_message('test input', with_files=filespecs) == {
            'role': 'user', 'content': [{'type': 'image', 'source': {'type': 'base64', 
            'media_type': 'image/png', 'data': 'MTIzNDU2Nzg5MA=='}}, {'type': 'text', 
            'text': 'test input'}]}
        
        with patch('builtins.open', mock_open(read_data=b'1234567890')):
            seg = Conversation._compile_user_message('test_input', with_files=['/tmp/xyz.png'])
            assert seg == {'role': 'user', 'content': [{'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': 'MTIzNDU2Nzg5MA=='}}, {'type': 'text', 'text': 'test_input'}]}


class TestGeneratorFunctions:
    def test_sysprompt_segment_generate(self):
        assert Bot._make_sysprompt_segment('test input1') == {'type': 'text', 'text': 'test input1'}
        assert Bot._make_sysprompt_segment('test input2', set_cache_control=True) == {'type': 'text', 'text': 'test input2', 'cache_control': {'type': 'ephemeral'}}
    
    _example_sysprompt = """example sysprompt"""
    
    @patch('builtins.open', mock_open(read_data=_example_sysprompt))
    def test_sysprompt_from_path(self):
        class SyspromptFromFileTestBot(Bot):
            sysprompt_path = '/tmp/sysprompt.txt'
        
        b = SyspromptFromFileTestBot()
        assert b.sysprompt_clean == self._example_sysprompt
    
    def test_sysprompt_generate_and_remap(self):
        class SyspromptGenerateTestBot(Bot):
            fields = ['TESTFIELD']
            def sysprompt_generate(self):
                return [
                    self._make_sysprompt_segment("""test1"""),
                    self._make_sysprompt_segment("""test2""", set_cache_control=True),
                    self._make_sysprompt_segment("""{{TESTFIELD}}"""),
                ]
        
        bot = SyspromptGenerateTestBot()
        assert bot.sysprompt_clean == [{'type': 'text', 'text': 'test1'}, 
            {'type': 'text', 'text': 'test2', 'cache_control': {'type': 'ephemeral'}}, 
            {'type': 'text', 'text': '{{TESTFIELD}}'}]
        assert bot.sysprompt_vec(['TESTVALUE']) == [{'type': 'text', 'text': 'test1'}, 
            {'type': 'text', 'text': 'test2', 'cache_control': {'type': 'ephemeral'}}, 
            {'type': 'text', 'text': 'TESTVALUE'}]
    
    def test_make_tool_request_and_result_message(self):
        assert Conversation._make_tool_result_message({'id': 'tu_12345'}, 'XYZZY') == {'role': 'user', 
            'content': [{'type': 'tool_result', 'tool_use_id': 'tu_12345', 'content': 'XYZZY'}]}
        
        assert Conversation._make_tool_request_message({'id': 'tu_12345', 'name': 'test_tool', 
            'input': {'param1': 'arg1'}}) == {'role': 'assistant', 'content': [{'type': 'tool_use', 
            'id': 'tu_12345', 'name': 'test_tool', 'input': {'param1': 'arg1'}}]}


class TestCannedResponse:
    class CannedResponseTesterBot(Bot):
        def preprocess_response(self, usermessage, conversation):
            if usermessage == 'X':
                return 'CANNED RESPONSE'
    
    def test_canned_response_from_bot(self):
        bot = self.CannedResponseTesterBot(client=fake_client())
        conv = Conversation(bot, [])
        msg1 = conv.resume(_IN1)
        assert gettext(msg1) == _OUT1
        msg2 = conv.resume('X')
        assert gettext(msg2) == 'CANNED RESPONSE'
        
        bot = self.CannedResponseTesterBot(client=fake_client_async())
        conv = Conversation(bot, [], async_mode=True)
        msg1 = asyncio.run(conv.aresume(_IN1))
        assert gettext(msg1) == _OUT1
        msg2 = asyncio.run(conv.aresume('X'))
        assert gettext(msg2) == 'CANNED RESPONSE'
        
    
    def test_canned_response_direct(self):
        ctext = 'canned text'
        with robo.CannedResponse(ctext) as canned:
            assert repr(canned) == '<CannedResponse: "canned text">'
            accum = ''
            for chunk in canned.text_stream:
                accum += chunk
            assert accum == ctext
        
        async def test_async_with():
            async with robo.CannedResponse(ctext) as canned_async:
                accum = ''
                for chunk in canned.text_stream:
                    accum += chunk
                assert accum == ctext
        asyncio.run(test_async_with())


class TestFrontendFeatures:
    class SoftStartBot(Bot):
        welcome_message = "Hello! And welcome!"
        soft_start = True
    
    def test_soft_start(self):
        bot = self.SoftStartBot(client=fake_client())
        conv = Conversation(bot, [])
        conv.resume(_IN1)
        assert(conv.messages) == [{'role': 'assistant', 'content': [{'type': 'text', 'text': 'Hello! And welcome!'}]}, {'role': 'user', 'content': [{'type': 'text', 'text': 'test input'}]}, {'role': 'assistant', 'content': [{'type': 'text', 'text': 'expected response'}]}]


class TestConversationContext:
    
    def test_oneshot_bot(self):
        class OneshotBot(Bot):
            oneshot = True
        bot = OneshotBot(client=fake_client())
        conv = Conversation(bot, [])
        conv.resume('test input')
        conv.resume('test input')
        assert len(conv.messages) == 4
        assert len(conv._get_conversation_context()) == 1
    
    def test_cache_user_prompt(self):
        c = Conversation(Bot(client=fake_client()), [], cache_user_prompt=True)
        c.resume('test input')
        c.resume('test input')
        assert c._get_conversation_context()[-1]['content'][0]['cache_control']['type'] == 'ephemeral'
        with pytest.raises(KeyError, match='cache_control'):
            c._get_conversation_context()[-3]['content'][0]['cache_control']['type'] == 'ephemeral'


class TestClassBasicAttributes:
    def test_basic_attributes(self):
        class NameTesterBot(Bot):
            pass

        assert NameTesterBot().name == 'NameTesterBot'
        assert 'NameTesterBot object' in repr(NameTesterBot())
        NameTesterBot.bot_name = 'Bob'
        assert NameTesterBot().name == 'Bob'
        assert repr(NameTesterBot()).startswith('<"Bob"')


class TestAPIKeyAndClientGet:
    @patch.dict('os.environ', {'ROBO_API_KEY_FILE': '/path/to/key'})
    @patch('builtins.open', mock_open(read_data='file_api_key'))
    def test_api_key_get_from_file(self):
        robo._populate_apikey_vars()
        assert robo._get_api_key() == 'file_api_key'

    @patch.dict('os.environ', {}, clear=True)
    @patch.dict('os.environ', {'CUSTOM_API_KEY': 'env_api_key'})
    def test_get_api_key_from_env_var(self):
        robo._populate_apikey_vars()
        with patch('robo.API_KEY_ENV_VAR', 'CUSTOM_API_KEY'):
            assert robo._get_api_key() == 'env_api_key'

    @patch.dict('os.environ', {}, clear=True)
    def test_get_api_key_returns_None_if_not_found(self):
        robo._populate_apikey_vars()
        assert robo._get_api_key() is None
    
    def test_bot_with_api_key(self):
        with patch.object(robo, '_get_client_class') as mock_client_class:
            mock_client_class.return_value = FakeAnthropic
            bot = Bot.with_api_key('xyzzy')
            assert bot.client.api_key == 'xyzzy'


class TestOOTools:
    def test_tool_structure(self):
        class MyTool1(robo.tools.Tool):
            description = 'Test tool'
            parameter_descriptions = {
                'param1': 'Required string parameter',
                'param2': 'Optional integer parameter'
            }
            
            def __call__(self, param1:str, param2:int = None):
                return {'param1_reversed': ''.join(reversed(param1)), 'param2_squared': param2 ** 2}
            
        assert MyTool1.get_call_schema() == {'name': 'MyTool1', 'description': 'Test tool', 
            'input_schema': {'type': 'object', 'properties': {'param1': {'type': 'string', 
            'description': 'Required string parameter'}, 'param2': {'type': 'number', 
            'description': 'Optional integer parameter'}}, 'required': ['param1']}}
        
        assert MyTool1()(**{'param1': 'time loop', 'param2':20}) == {'param1_reversed': 'pool emit', 'param2_squared': 400}
        
    def test_raises_if_call_not_defined(self):
        class MyTool1(robo.tools.Tool):
            description = 'Test tool'
        
        with pytest.raises(NotImplementedError):
            MyTool1()()


class TestToolUse:
    def test_tooluse_sync_flat(self):
        conv = Conversation(ToolTesterBot(client=fake_client()), [])
        # conv.register_callback(response_complete, )
        msg = conv.resume('calculate')
        printmsg(msg)
        assert gettext(msg) == '''Tool response was:['4']'''

    def test_tooluse_sync_stream(self):
        bot = ToolTesterBot(client=fake_client())
        conv = Conversation(bot, [], stream=True)
        print(bot, conv)
        sio = StringIO()
        say = streamer(conv, cc=sio)
        say('weather')
        sio.seek(0)
        msgtext = sio.read()
        assert msgtext == '''Tool response was:['Sunny, 23° celcius']'''

    def test_tooluse_async_flat(self):
        conv = Conversation(ToolTesterBot(client=fake_client_async()), [], async_mode=True)
        coro = conv.aresume('calculate')
        msg = asyncio.run(coro)
        print(gettext(msg))
        assert gettext(msg) == '''Tool response was:['4']'''

    def test_tooluse_async_stream(self):
        conv = Conversation(ToolTesterBot(client=fake_client_async()), [], stream=True, async_mode=True)
        sio = StringIO()
        say = streamer_async(conv, cc=sio)
        coro = say('weather')
        asyncio.run(coro)
        sio.seek(0)
        assert sio.read() == """Tool response was:['Sunny, 23° celcius']"""
    
    def test_tooluse_old_style(self):
        bot = ToolTesterBotOldStyle()
        tooldata1 = {'type': 'tool use', 'id': 'tu_12345', 'name': 'get_url', 'input': {'url': 'https://nothing.net'}}
        tooldata_missing = tooldata1.copy()
        tooldata_missing['name'] = 'get_url2'
        toolresp1 = bot.handle_tool_call(tooldata1)
        assert 'Fake page' in toolresp1['message']
        assert toolresp1['target'] == 'model'
        
        with pytest.raises(Exception, match='Tool function not found: tools_get_url2'):
            bot.handle_tool_call(tooldata_missing)
    
    def test_tooluse_client_targeted(self):
        scenario = {'navigate me': [{'type': 'tool_use', 'id': 'toolu_98765', 'name': 'guided_navigate', 'input': {'destination': '/xyz/xyz/'}}]}
        # sync flat
        tb = ClientToolTestBot(client=FakeAnthropic(response_scenarios=scenario))
        conv = Conversation(tb, [])
        msg = conv.resume('navigate me')
        printmsg(msg)
        assert type(msg) is robo.CannedResponse
        assert gettext(msg) == '\n@@@@NAVIGATE /xyz/xyz/'
        msg2 = conv.resume('@@@@RECONNECT')
        printmsg(msg2)
        assert gettext(msg2) == '''Tool response was:['@@@@RECONNECT']'''
        
        # async flat
        tb = ClientToolTestBot(client=FakeAsyncAnthropic(response_scenarios=scenario))
        conv = Conversation(tb, [], async_mode=True)
        msg = asyncio.run(conv.aresume('navigate me'))
        assert type(msg) is robo.CannedResponse
        assert gettext(msg) == '\n@@@@NAVIGATE /xyz/xyz/'
        msg2 = asyncio.run(conv.aresume('@@@@RECONNECT'))
        assert gettext(msg2) == """Tool response was:['@@@@RECONNECT']"""
        
        # sync stream
        tb = ClientToolTestBot(client=FakeAnthropic(response_scenarios=scenario))
        conv = Conversation(tb, [], stream=True)
        accum = ''
        with conv.resume('navigate me') as stream:
            for chunk in stream.text_stream:
                print(chunk)
                accum += chunk
        assert accum == '\n@@@@NAVIGATE /xyz/xyz/'
        
        accum = ''
        with conv.resume('@@@@RECONNECT') as stream:
            for chunk in stream.text_stream:
                print(chunk)
                accum += chunk
        assert accum == """Tool response was:['@@@@RECONNECT']"""
        
        # async stream
        tb = ClientToolTestBot(client=FakeAsyncAnthropic(response_scenarios=scenario))
        conv = Conversation(tb, [], stream=True, async_mode=True)
        
        async def one():
            accum = ''
            async with await conv.aresume('navigate me') as stream:
                async for chunk in stream.text_stream:
                    print(chunk)
                    accum += chunk
            assert accum == '\n@@@@NAVIGATE /xyz/xyz/'
        
        asyncio.run(one())
        
        async def two():
            accum = ''
            async with await conv.aresume('@@@@RECONNECT') as stream:
                async for chunk in stream.text_stream:
                    print(chunk, end=' ')
                    accum += chunk
            assert accum == """Tool response was:['@@@@RECONNECT']"""
        
        asyncio.run(two())


class CallbackConversationVariantTester:
    
    @staticmethod
    def _runner_sync_flat(convo, msgs_in):
        for msg in msgs_in:
            convo.resume(msg)
    
    @staticmethod
    def _runner_async_flat(convo, msgs_in):
        for msg in msgs_in:
            async def resumeconvo(msg):
                await convo.aresume(msg)
            asyncio.run(resumeconvo(msg))
    
    @staticmethod
    def _runner_sync_stream(convo, msgs_in):
        for msg in msgs_in:
            with convo.resume(msg) as stream:
                for chunk in stream.text_stream:
                    pass
    
    @staticmethod
    def _runner_async_stream(convo, msgs_in):
        for msg in msgs_in:
            async def resumeconvo(msg):
                async with await convo.aresume(msg) as stream:
                    async for chunk in stream.text_stream:
                        pass
            asyncio.run(resumeconvo(msg))
    
    def run_variant_tests(self, test_wrapper, success_check, msgs_input):
        success_check(test_wrapper(self._runner_sync_flat, msgs_input))
        success_check(test_wrapper(self._runner_async_flat, msgs_input, async_mode=True))
        success_check(test_wrapper(self._runner_sync_stream, msgs_input, stream=True))
        success_check(test_wrapper(self._runner_async_stream, msgs_input, stream=True, async_mode=True))
    
    @staticmethod
    def get_client(kwargs):
        if 'async_mode' in kwargs:
            return fake_client_async()
        else:
            return fake_client()


class TestMoreToolUse(CallbackConversationVariantTester):
    @staticmethod
    def get_client(kwargs):
        if 'async_mode' in kwargs:
            return FakeAsyncAnthropic(response_scenarios=TimeWeatherLocationTestBot.test_scenario)
        else:
            return FakeAnthropic(response_scenarios=TimeWeatherLocationTestBot.test_scenario)
    
    def test_tooluse_multiple_sequential_and_parallel(self):
        def test_wrapper(runner, msgs_in, **conv_args):
            conv = Conversation(TimeWeatherLocationTestBot(client=self.get_client(conv_args)), [], **conv_args)
            runner(conv, msgs_in)
            return (conv,)
        
        def check_successful(returned):
            conv, = returned
            print(conv.messages)
            assert conv.messages[-1]['content'][0]['text'] == \
                "Tool response was:['Auckland, New Zealand', 'Sunny, 23°']"
            
        
        self.run_variant_tests(test_wrapper, check_successful, ['tool test'])


class TestSyncAsyncToolUse(CallbackConversationVariantTester):
    @staticmethod
    def get_client(kwargs):
        if 'async_mode' in kwargs:
            return FakeAsyncAnthropic(response_scenarios=TimerBot.test_scenario)
        else:
            return FakeAnthropic(response_scenarios=TimerBot.test_scenario)

    def test_sync_async_tool_use(self):
        def test_wrapper(runner, msgs_in, **conv_args):
            conv = Conversation(TimerBot(client=self.get_client(conv_args)), [], **conv_args)
            runner(conv, msgs_in)
            return (conv,)
        
        def check_successful(returned):
            conv, = returned
            messagetext = conv.messages[-1]['content'][0]['text']
            if conv.is_async:
                assert 'Asynchronous timer' in messagetext
            else:
                assert 'Synchronous timer' in messagetext
        
        self.run_variant_tests(test_wrapper, check_successful, ['tool test'])


class TestCallbacksStructure(CallbackConversationVariantTester):
    def test_callbacks_structure(self):
        def test_wrapper(runner, msgs_in, **conv_args):
            sio = StringIO()
            conv_retained = None
            def callback_function(conv, data_tuple):
                nonlocal conv_retained, sio
                assert type(data_tuple) is tuple
                msg, = data_tuple
                sio.write(gettext(msg))
                conv_retained = conv
            conv = Conversation(Bot(client=self.get_client(conv_args)), [], **conv_args)
            conv.register_callback('response_complete', callback_function)
            runner(conv, msgs_in)
            return sio, conv, conv_retained
    
        def check_successful(returned):
            sio, conv, conv_retained = returned
            sio.seek(0)
            assert sio.read() == _OUT1
            assert conv_retained == conv
            print('assertions passed')
    
        self.run_variant_tests(test_wrapper, check_successful, [_IN1])


class TestResponseCompleteCallbacks(CallbackConversationVariantTester):
    def test_response_complete_callback(self):
        def test_wrapper(runner, msgs_in, **conv_args):
            final_msg, conv_retained = None, None
            def callback_function(conv, data_tuple):
                nonlocal final_msg, conv_retained
                assert type(data_tuple) is tuple
                msg, = data_tuple
                conv_retained = conv
                final_msg = msg
            conv = Conversation(Bot(client=self.get_client(conv_args)), [], **conv_args)
            conv.register_callback('response_complete', callback_function)
            runner(conv, msgs_in)
            return final_msg, conv, conv_retained
        
        def check_successful(returned):
            final_msg, conv, conv_retained = returned
            assert gettext(final_msg) == _OUT1
            assert issubclass(type(conv_retained), Conversation)
            assert conv_retained == conv
            print('assertions passed')
        
        self.run_variant_tests(test_wrapper, check_successful, [_IN1])


class TestToolExecutedCallbacks(CallbackConversationVariantTester):
    def test_tool_executed_callback(self):
        def test_wrapper(runner, msgs_in, **conv_args):
            tool_req, tool_resp, conv_retained = None, None, None
            def callback_function(conv, tub):
                nonlocal tool_req, tool_resp, conv_retained
                conv_retained = conv
                tool_req, tool_resp = tub
            conv = Conversation(ToolTesterBot(client=self.get_client(conv_args)), [], **conv_args)
            conv.register_callback('tool_executed', callback_function)
            runner(conv, msgs_in)
            return tool_req, tool_resp, conv, conv_retained
        
        def check_successful(returned):
            tool_req, tool_resp, conv, conv_retained = returned
            print(tool_req)
            print(tool_resp)
            assert conv_retained == conv
            assert tool_req.name == 'Calculate'
            assert tool_resp['message'] == '4'
            print('assertions passed')
        
        self.run_variant_tests(test_wrapper, check_successful, ['calculate'])


class TestBaseStreamWrappers:
    """Because the ToolUse StreamWrapper variants are used by default, some lines of code get overlooked
    in the usual flows. Testing directly for code coverage purposes.
    """
    def test_streamwrapper_base(self):
        fsm = FakeStreamManager(['Hello'])
        conv = Conversation(Bot(client=create_fake_client()), [], stream=True)
        with StreamWrapper(fsm, conv) as sw:
            accumulated = ''
            for chunk in sw.text_stream:
                accumulated += chunk
            assert accumulated == 'Hello'

    def test_streamwrapper_async_base(self):
        fsm = FakeAsyncStreamManager(['Hello'])
        conv = Conversation(Bot(client=create_fake_async_client()), [], async_mode=True, stream=True)
        
        fsm = FakeAsyncStreamManager(['Hello2'])
        async def wrapper():
            accumulated = ''
            async with AsyncStreamWrapper(fsm, conv) as stream:
                async for chunk in stream.text_stream:
                    accumulated += chunk
            return accumulated
        
        result = asyncio.run(wrapper())
        assert result == 'Hello2'


class TestConversationStartMethods:
    """Tests the various conversation startup methods as described in the cookbook.
    Method 1: conv = Conversation() ; conv.start(usermessage) ; conv.resume(usermessage)
    Method 2: conv = Conversation(BotWithFields) ; conv.start(argv_or_dict, usermessage) ; conv.resume(usermessage)
    Method 3: conv = Conversation(BotWithFields) ; conv.prestart(argv_or_dict) ; conv.resume(usermessage)
    Method 4: conv = Conversation(BotWithFields, argv_or_dict) ; conv.resume(usermessage)
    """
    def test_method1_sync_flat(self):
        with patch.object(robo, '_get_client_class') as mock_client:
            mock_client.return_value.return_value = fake_client()
            
            conv = Conversation(Bot)
            assert gettext(conv.start(_IN1)) == _OUT1
            assert gettext(conv.resume(_IN2)) == _OUT2
    
    def test_method2_sync_flat_with_list(self):
        with patch.object(robo, '_get_client_class') as mock_client:
            mock_client.return_value.return_value = fake_client()
            conv = Conversation(FieldsTesterBot)
            msg1 = conv.start(['cat'], _IN1)
            assert conv.started
            assert 'sound made by a cat' in conv.sysprompt
            assert gettext(msg1) == _OUT1
            msg2 = conv.resume(_IN2)
            assert gettext(msg2) == _OUT2
    
    def test_method2_sync_flat_with_dict(self):
        with patch.object(robo, '_get_client_class') as mock_client:
            mock_client.return_value.return_value = fake_client()
            conv = Conversation(FieldsTesterBot)
            msg1 = conv.start({'ANIMAL_TYPE': 'wolf'}, _IN1)
            assert conv.started
            assert 'sound made by a wolf' in conv.sysprompt
            assert gettext(msg1) == _OUT1
            msg2 = conv.resume(_IN2)
            assert gettext(msg2) == _OUT2
    
    def test_method3_sync_flat_prestart_no_argv(self):
        with patch.object(robo, '_get_client_class') as mock_client:
            mock_client.return_value.return_value = fake_client()
            conv = Conversation(Bot)
            conv.prestart()
            assert conv.started
            msg = conv.resume(_IN1)
            assert gettext(msg) == _OUT1
    
    def test_method3_sync_flat_with_list(self):
        with patch.object(robo, '_get_client_class') as mock_client:
            mock_client.return_value.return_value = fake_client()
            conv = Conversation(FieldsTesterBot)
            conv.prestart(['goose'])
            assert conv.started
            assert 'sound made by a goose' in conv.sysprompt
            msg = conv.resume(_IN1)
            assert gettext(msg) == _OUT1
            
            conv2 = Conversation(FieldsTesterBot).prestart(['owl'])
            assert conv2.started
            assert 'sound made by a owl' in conv2.sysprompt
            msg = conv2.resume(_IN2)
            assert gettext(msg) == _OUT2

    def test_method3_sync_flat_with_dict(self):
        with patch.object(robo, '_get_client_class') as mock_client:
            mock_client.return_value.return_value = fake_client()
            conv = Conversation(FieldsTesterBot)
            conv.prestart({'ANIMAL_TYPE': 'goose'})
            assert conv.started
            assert 'sound made by a goose' in conv.sysprompt
            msg = conv.resume(_IN1)
            assert gettext(msg) == _OUT1
            
            conv2 = Conversation(FieldsTesterBot).prestart({'ANIMAL_TYPE': 'owl'})
            assert conv2.started
            assert 'sound made by a owl' in conv2.sysprompt
            msg = conv2.resume(_IN2)
            assert gettext(msg) == _OUT2
    
    def test_method4_sync_flat_with_list(self):
        with patch.object(robo, '_get_client_class') as mock_client:
            mock_client.return_value.return_value = fake_client()
            conv = Conversation(FieldsTesterBot, ['sheep'])
            assert conv.started
            assert 'sound made by a sheep' in conv.sysprompt
            msg = conv.resume(_IN1)
            assert gettext(msg) == _OUT1
            
    def test_method4_sync_flat_with_dict(self):
        with patch.object(robo, '_get_client_class') as mock_client:
            mock_client.return_value.return_value = fake_client()
            conv = Conversation(FieldsTesterBot, {'ANIMAL_TYPE': 'sheep'})
            assert conv.started
            assert 'sound made by a sheep' in conv.sysprompt
            msg = conv.resume(_IN1)
            assert gettext(msg) == _OUT1
            
    def test_method1_async_flat(self):
        with patch.object(robo, '_get_client_class') as mock_client:
            mock_client.return_value.return_value = fake_client_async()
            conv = Conversation(Bot, async_mode=True)
            coro1 = conv.astart(_IN1)
            msg1 = asyncio.run(coro1)
            coro2 = conv.aresume(_IN2)
            msg2 = asyncio.run(coro2)
            assert gettext(msg1) == _OUT1
            assert gettext(msg2) == _OUT2

    def test_method2_async_flat_with_list(self):
        with patch.object(robo, '_get_client_class') as mock_client:
            mock_client.return_value.return_value = fake_client_async()
            conv = Conversation(FieldsTesterBot, async_mode=True)
            coro1 = conv.astart(['cat'], _IN1)
            msg1 = asyncio.run(coro1)
            assert conv.started
            assert 'sound made by a cat' in conv.sysprompt
            assert gettext(msg1) == _OUT1
            coro2 = conv.aresume(_IN2)
            msg2 = asyncio.run(coro2)
            assert gettext(msg2) == _OUT2
    
    def test_method2_async_flat_with_dict(self):
        with patch.object(robo, '_get_client_class') as mock_client:
            mock_client.return_value.return_value = fake_client_async()
            conv = Conversation(FieldsTesterBot, async_mode=True)
            coro1 = conv.astart({'ANIMAL_TYPE': 'wolf'}, _IN1)
            msg1 = asyncio.run(coro1)
            assert conv.started
            assert 'sound made by a wolf' in conv.sysprompt
            assert gettext(msg1) == _OUT1
            coro2 = conv.aresume(_IN2)
            msg2 = asyncio.run(coro2)
            assert gettext(msg2) == _OUT2
    
    def test_method3_async_flat_with_list(self):
        with patch.object(robo, '_get_client_class') as mock_client:
            mock_client.return_value.return_value = fake_client_async()
            conv = Conversation(FieldsTesterBot, async_mode=True)
            conv.prestart(['goose'])
            assert conv.started
            assert 'sound made by a goose' in conv.sysprompt
            coro1 = conv.aresume(_IN1)
            msg1 = asyncio.run(coro1)
            assert gettext(msg1) == _OUT1
            
            conv2 = Conversation(FieldsTesterBot, async_mode=True).prestart(['owl'])
            assert conv2.started
            assert 'sound made by a owl' in conv2.sysprompt
            coro2 = conv2.aresume(_IN2)
            msg2 = asyncio.run(coro2)
            assert gettext(msg2) == _OUT2

    def test_method3_async_flat_with_dict(self):
        with patch.object(robo, '_get_client_class') as mock_client:
            mock_client.return_value.return_value = fake_client_async()
            conv = Conversation(FieldsTesterBot, async_mode=True)
            conv.prestart({'ANIMAL_TYPE': 'goose'})
            assert conv.started
            assert 'sound made by a goose' in conv.sysprompt
            coro1 = conv.aresume(_IN1)
            msg1 = asyncio.run(coro1)
            assert gettext(msg1) == _OUT1
            
            conv2 = Conversation(FieldsTesterBot, async_mode=True).prestart({'ANIMAL_TYPE': 'owl'})
            assert conv2.started
            assert 'sound made by a owl' in conv2.sysprompt
            coro2 = conv2.aresume(_IN2)
            msg2 = asyncio.run(coro2)
            assert gettext(msg2) == _OUT2
    
    def test_method4_async_flat_with_list(self):
        with patch.object(robo, '_get_client_class') as mock_client:
            mock_client.return_value.return_value = fake_client_async()
            conv = Conversation(FieldsTesterBot, ['sheep'], async_mode=True)
            assert conv.started
            assert 'sound made by a sheep' in conv.sysprompt
            coro = conv.aresume(_IN1)
            msg = asyncio.run(coro)
            assert gettext(msg) == _OUT1
            
    def test_method4_async_flat_with_dict(self):
        with patch.object(robo, '_get_client_class') as mock_client:
            mock_client.return_value.return_value = fake_client_async()
            conv = Conversation(FieldsTesterBot, {'ANIMAL_TYPE': 'sheep'}, async_mode=True)
            assert conv.started
            assert 'sound made by a sheep' in conv.sysprompt
            coro = conv.aresume(_IN1)
            msg = asyncio.run(coro)
            assert gettext(msg) == _OUT1
    
    def test_with_dict_missing_arg(self):
        with patch.object(robo, '_get_client_class') as mock_client:
            mock_client.return_value.return_value = fake_client()
            with pytest.raises(FieldValuesMissingException, match='ANIMAL_TYPE'):
                conv = Conversation(FieldsTesterBot, {})
    
    def test_wrong_context(self):
        with pytest.raises(SyncAsyncMismatchError):
            conv = Conversation(Bot, async_mode=True)
            conv.start()
        
        with pytest.raises(SyncAsyncMismatchError):
            conv = Conversation(Bot, async_mode=False)
            asyncio.run(conv.astart())
    
    def test_conversation_already_started(self):
        with pytest.raises(Exception, match='Conversation has already started'):
            conv = Conversation(Bot, [])
            conv.start([], 'test input')
        
        with pytest.raises(Exception, match='Conversation has already started'):
            conv = Conversation(Bot, [], async_mode=True)
            coro = conv.astart([], 'test input')
            asyncio.run(coro)

    def test_conversation_not_started(self):
        with pytest.raises(Exception, match='Attempting to resume a conversation that has not been started'):
            bot = Bot(client=fake_client())
            conv = Conversation(bot)
            conv.resume('test input')
        
        with pytest.raises(Exception, match='Attempting to resume a conversation that has not been started'):
            bot = Bot(client=fake_client_async())
            conv = Conversation(bot, async_mode=True)
            coro = conv.aresume('test input')
            asyncio.run(coro)
    
    def test_conversation_resume_context(self):
        with pytest.raises(SyncAsyncMismatchError):
            bot = Bot(client=fake_client())
            conv = Conversation(bot, [], async_mode=False)
            coro = conv.aresume('test input')
            asyncio.run(coro)
        
        with pytest.raises(SyncAsyncMismatchError):
            bot = Bot(client=fake_client_async())
            conv = Conversation(bot, [], async_mode=True)
            conv.resume('test input')
    
    def test_conversation_start_fail_if_no_api_key(self):
        with pytest.raises(Exception, match='Authentication method not valid'):
            bot = Bot.with_api_key(None)
            conv = Conversation(bot, [])
            conv.resume('test input')
        
        with pytest.raises(Exception, match='Authentication method not valid'):
            bot = Bot.with_api_key(None)
            conv = Conversation(bot, [], async_mode=True)
            asyncio.run(conv.aresume('test input'))
