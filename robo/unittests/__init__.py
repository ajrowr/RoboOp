import pytest
import json
import os
import asyncio
import anthropic
from unittest.mock import Mock, patch, AsyncMock
from robo import Bot, Conversation, MODELS
from robo.exceptions import *
import robo


class TestBot:
    """Test Bot class functionality"""
    
    def test_bot_initialization_defaults(self):
        """Test that Bot initializes with correct default values"""
        bot = Bot()
        assert bot.model == MODELS.LATEST_SONNET
        assert bot.temperature == 1
        assert bot.fields == []
        assert bot.max_tokens == 8192
        assert bot.oneshot == False
        assert bot.welcome_message is None
        assert bot.soft_start == False
    
    def test_bot_with_custom_attributes(self):
        """Test Bot with custom attributes set"""
        class CustomBot(Bot):
            model = MODELS.LATEST_HAIKU
            temperature = 0.5
            fields = ['name', 'role']
            max_tokens = 4096
            oneshot = True
            welcome_message = "Hello!"
            soft_start = True
        
        bot = CustomBot()
        assert bot.model == MODELS.LATEST_HAIKU
        assert bot.temperature == 0.5
        assert bot.fields == ['name', 'role']
        assert bot.max_tokens == 4096
        assert bot.oneshot == True
        assert bot.welcome_message == "Hello!"
        assert bot.soft_start == True


class TestBotSystemPrompt:
    """Test Bot system prompt functionality"""
    
    def test_sysprompt_text_attribute(self):
        """Test bot with sysprompt_text attribute"""
        class TextBot(Bot):
            sysprompt_text = "You are a helpful assistant."
        
        bot = TextBot()
        assert bot.sysprompt_clean == "You are a helpful assistant."
    
    def test_sysprompt_path_attribute(self, tmp_path):
        """Test bot with sysprompt_path attribute"""
        sysprompt_file = tmp_path / "sysprompt.txt"
        sysprompt_file.write_text("You are a file-based assistant.")
        
        class PathBot(Bot):
            sysprompt_path = str(sysprompt_file)
        
        bot = PathBot()
        assert bot.sysprompt_clean == "You are a file-based assistant."
    
    def test_sysprompt_generate_method(self):
        """Test bot with sysprompt_generate method"""
        class GeneratedBot(Bot):
            def sysprompt_generate(self):
                return "You are a dynamically generated assistant."
        
        bot = GeneratedBot()
        assert bot.sysprompt_clean == "You are a dynamically generated assistant."
    
    def test_sysprompt_generate_dict(self):
        """Test bot that generates dict system prompt"""
        class DictBot(Bot):
            def sysprompt_generate(self):
                return {
                    "type": "text",
                    "text": "You are a dict-based assistant."
                }
        
        bot = DictBot()
        expected = {
            "type": "text", 
            "text": "You are a dict-based assistant."
        }
        assert bot.sysprompt_clean == expected
    
    def test_empty_sysprompt(self):
        """Test bot with no system prompt"""
        bot = Bot()
        assert bot.sysprompt_clean == ""

class TestBotTemplateInterpolation:
    """Test Bot template variable interpolation"""
    
    def test_sysprompt_vec_with_string_template(self):
        """Test string template interpolation with list argv"""
        class TemplateBot(Bot):
            sysprompt_text = "You are {{role}} named {{name}}."
            fields = ['role', 'name']
        
        bot = TemplateBot()
        result = bot.sysprompt_vec(['assistant', 'Claude'])
        assert result == "You are assistant named Claude."
    
    def test_sysprompt_vec_with_dict_template(self):
        """Test dict template interpolation with list argv"""
        class DictTemplateBot(Bot):
            fields = ['role', 'name']
            def sysprompt_generate(self):
                return {
                    "type": "text",
                    "text": "You are {{role}} named {{name}}.",
                    "cache_control": {"type": "ephemeral"}
                }
        
        bot = DictTemplateBot()
        result = bot.sysprompt_vec(['assistant', 'Claude'])
        expected = {
            "type": "text",
            "text": "You are assistant named Claude.",
            "cache_control": {"type": "ephemeral"}
        }
        assert result == expected
    
    def test_sysprompt_vec_with_empty_argv(self):
        """Test template interpolation with empty argv"""
        class TemplateBot(Bot):
            sysprompt_text = "You are {{role}} named {{name}}."
            fields = ['role', 'name']
        
        bot = TemplateBot()
        result = bot.sysprompt_vec([])
        assert result == "You are {{role}} named {{name}}."
    
    def test_sysprompt_vec_with_none_argv(self):
        """Test template interpolation with None argv"""
        class TemplateBot(Bot):
            sysprompt_text = "You are {{role}} named {{name}}."
            fields = ['role', 'name']
        
        bot = TemplateBot()
        result = bot.sysprompt_vec(None)
        assert result == "You are {{role}} named {{name}}."
    
    def test_sysprompt_vec_partial_substitution(self):
        """Test template interpolation with fewer args than fields"""
        class TemplateBot(Bot):
            sysprompt_text = "You are {{role}} named {{name}}."
            fields = ['role', 'name']
        
        bot = TemplateBot()
        result = bot.sysprompt_vec(['assistant'])
        assert result == "You are assistant named {{name}}."


class TestConversation:
    """Test Conversation class functionality"""
    
    def test_conversation_initialization_with_bot_instance(self):
        """Test conversation initialization with bot instance"""
        bot = Bot()
        conv = Conversation(bot)
        assert conv.bot is bot
        assert conv.messages == []
        assert conv.message_objects == []
        assert conv.started == False
        assert conv.argv == []
    
    def test_conversation_initialization_with_bot_class(self):
        """Test conversation initialization with bot class"""
        conv = Conversation(Bot)
        assert isinstance(conv.bot, Bot)
        assert conv.messages == []
        assert conv.started == False
    
    def test_conversation_with_argv(self):
        """Test conversation initialization with argv"""
        class TemplateBot(Bot):
            sysprompt_text = "You are {{role}}."
            fields = ['role']
        
        conv = Conversation(TemplateBot, argv=['assistant'])
        assert conv.argv == ['assistant']
        assert conv.started == True
        assert conv.sysprompt == "You are assistant."
    
    def test_conversation_prestart(self):
        """Test conversation prestart method"""
        class TemplateBot(Bot):
            sysprompt_text = "You are {{role}}."
            fields = ['role']
        
        conv = Conversation(TemplateBot)
        assert conv.started == False
        
        conv.prestart(['helper'])
        assert conv.started == True
        assert conv.argv == ['helper']
        assert conv.sysprompt == "You are helper."
    
    def test_conversation_soft_start(self):
        """Test conversation with soft start enabled"""
        class SoftBot(Bot):
            welcome_message = "Hello! How can I help?"
            soft_start = True
        
        conv = Conversation(SoftBot)
        assert conv.soft_started == True
        assert len(conv.messages) == 1
        assert conv.messages[0]['role'] == 'assistant'
        assert conv.messages[0]['content'][0]['text'] == "Hello! How can I help?"
    
    def test_conversation_soft_start_override(self):
        """Test conversation with soft start override"""
        class SoftBot(Bot):
            welcome_message = "Hello! How can I help?"
            soft_start = True
        
        # Override soft_start to False
        conv = Conversation(SoftBot, soft_start=False)
        assert conv.soft_started == False
        assert len(conv.messages) == 0
        
        # Override soft_start to True for bot without it
        conv2 = Conversation(Bot, soft_start=True)
        assert conv2.soft_started == False  # No welcome_message, so no soft start
    
    def test_conversation_oneshot_mode(self):
        """Test conversation in oneshot mode"""
        class OneShotBot(Bot):
            oneshot = True
        
        conv = Conversation(OneShotBot)
        assert conv.oneshot == True


class TestConversationMessageHandling:
    """Test Conversation message handling"""
    
    def test_make_text_message(self):
        """Test _make_text_message static method"""
        message = Conversation._make_text_message('user', 'Hello')
        expected = {
            'role': 'user',
            'content': [{
                'type': 'text',
                'text': 'Hello'
            }]
        }
        assert message == expected
    
    def test_make_tool_result_message(self):
        """Test _make_tool_result_message static method"""
        from types import SimpleNamespace
        toolblock = SimpleNamespace(id='tool_123')
        message = Conversation._make_tool_result_message(toolblock, 'Tool result')
        expected = {
            'role': 'user',
            'content': [{
                'type': 'tool_result',
                'tool_use_id': 'tool_123',
                'content': 'Tool result'
            }]
        }
        assert message == expected
    
    def test_make_tool_request_message(self):
        """Test _make_tool_request_message static method"""
        from types import SimpleNamespace
        toolblock = SimpleNamespace(
            id='tool_123',
            name='test_tool', 
            input={'param': 'value'}
        )
        message = Conversation._make_tool_request_message(toolblock)
        expected = {
            'role': 'assistant',
            'content': [{
                'type': 'tool_use',
                'id': 'tool_123',
                'name': 'test_tool',
                'input': {'param': 'value'}
            }]
        }
        assert message == expected


class TestConversationContextHandling:
    """Test Conversation context handling"""
    
    def test_get_conversation_context_normal(self):
        """Test getting conversation context in normal mode"""
        conv = Conversation(Bot)
        conv.messages = [
            {'role': 'user', 'content': [{'type': 'text', 'text': 'Hello'}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Hi there'}]}
        ]
        
        context = conv._get_conversation_context()
        assert context == conv.messages
    
    def test_get_conversation_context_oneshot(self):
        """Test getting conversation context in oneshot mode"""
        class OneShotBot(Bot):
            oneshot = True
        
        conv = Conversation(OneShotBot)
        conv.messages = [
            {'role': 'user', 'content': [{'type': 'text', 'text': 'Hello'}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Hi there'}]},
            {'role': 'user', 'content': [{'type': 'text', 'text': 'How are you?'}]}
        ]
        
        context = conv._get_conversation_context()
        assert context == [conv.messages[-1]]  # Only last message
    
    def test_get_conversation_context_with_cache(self):
        """Test getting conversation context with user prompt caching"""
        conv = Conversation(Bot, cache_user_prompt=True)
        conv.messages = [
            {'role': 'user', 'content': [{'type': 'text', 'text': 'Hello'}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Hi there'}]},
            {'role': 'user', 'content': [{'type': 'text', 'text': 'How are you?'}]}
        ]
        
        context = conv._get_conversation_context()
        # Should add cache_control to the last message's last content block
        assert context[-1]['content'][-1]['cache_control'] == {'type': 'ephemeral'}
        # Original messages should be unchanged
        assert 'cache_control' not in conv.messages[-1]['content'][-1]


class TestToolHandling:
    """Test tool handling functionality"""
    
    def test_bot_get_tools_schema_default(self):
        """Test default tools schema is empty"""
        bot = Bot()
        assert bot.get_tools_schema() == []
    
    def test_bot_handle_tool_call_not_found(self):
        """Test handling tool call when function doesn't exist"""
        from types import SimpleNamespace
        bot = Bot()
        toolblock = SimpleNamespace(name='nonexistent', input={})
        
        with pytest.raises(Exception, match='Tool function not found: tools_nonexistent'):
            bot.handle_tool_call(toolblock)
    
    def test_bot_handle_tool_call_success(self):
        """Test successful tool call handling"""
        from types import SimpleNamespace
        
        class ToolBot(Bot):
            def tools_test_tool(self, param1=None):
                return {'target': 'model', 'message': f'Got {param1}'}
        
        bot = ToolBot()
        toolblock = SimpleNamespace(name='test_tool', input={'param1': 'value1'})
        
        result = bot.handle_tool_call(toolblock)
        assert result == {'target': 'model', 'message': 'Got value1'}

        bot = ToolBot()
        toolblock = dict(name='test_tool', input={'param1': 'value1'})
        
        result = bot.handle_tool_call(toolblock)
        assert result == {'target': 'model', 'message': 'Got value1'}


class TestBotPreprocessResponse:
    """Test Bot preprocess_response functionality"""
    
    def test_preprocess_response_default(self):
        """Test default preprocess_response returns None"""
        bot = Bot()
        conv = Conversation(bot)
        result = bot.preprocess_response("Hello", conv)
        assert result is None
    
    def test_preprocess_response_custom(self):
        """Test custom preprocess_response"""
        class CustomBot(Bot):
            def preprocess_response(self, message_text, conversation):
                if message_text == "ping":
                    return "pong"
                return None
        
        bot = CustomBot()
        conv = Conversation(bot)
        
        result = bot.preprocess_response("ping", conv)
        assert result == "pong"
        
        result = bot.preprocess_response("hello", conv)
        assert result is None


class TestConversationStartError:
    """Test Conversation start error handling"""
    
    def test_start_already_started_error(self):
        """Test that starting an already started conversation raises exception"""
        conv = Conversation(Bot, argv=[])
        assert conv.started == True
        
        with pytest.raises(Exception, match='Conversation has already started'):
            conv.start("Hello")
    
    def test_astart_already_started_error(self):
        """Test that async starting an already started conversation raises exception"""
        conv = Conversation(Bot, argv=[], async_mode=True)
        assert conv.started == True
        
        # We can't actually run async code in sync test, but we can test the sync part
        import asyncio
        async def test_async():
            with pytest.raises(Exception, match='Conversation has already started'):
                await conv.astart("Hello")
        
        asyncio.run(test_async())
        # Just ensure the method exists and would raise
        assert hasattr(conv, 'astart')


class TestConversationExhaustion:
    """Test conversation exhaustion detection"""
    
    def test_is_exhausted_true(self):
        """Test _is_exhausted returns True for text-only assistant message"""
        conv = Conversation(Bot)
        conv.messages = [{
            'role': 'assistant',
            'content': [{'type': 'text', 'text': 'Hello'}]
        }]
        assert conv._is_exhausted() == True
    
    def test_is_exhausted_false_user_message(self):
        """Test _is_exhausted returns False for user message"""
        conv = Conversation(Bot)
        conv.messages = [{
            'role': 'user', 
            'content': [{'type': 'text', 'text': 'Hello'}]
        }]
        assert conv._is_exhausted() == False
    
    def test_is_exhausted_false_tool_use(self):
        """Test _is_exhausted returns False for assistant message with tool use"""
        conv = Conversation(Bot)
        conv.messages = [{
            'role': 'assistant',
            'content': [
                {'type': 'text', 'text': 'I need to use a tool'},
                {'type': 'tool_use', 'id': 'tool_123', 'name': 'test', 'input': {}}
            ]
        }]
        assert conv._is_exhausted() == False

class TestConversationInitializationMethods:
    """Test all conversation initialization methods described in cookbook.md"""
    
    def test_method_1_bot_without_fields(self):
        """Test Method 1: Conversation with Bot that has no fields"""
        # This should work without needing to pass argv
        conv = Conversation(Bot)
        assert conv.bot.__class__ == Bot
        assert conv.started == False
        assert conv.argv == []
        assert not hasattr(conv, 'sysprompt') or conv.sysprompt is None
    
    def test_method_1_start_without_argv(self):
        """Test Method 1: start() call without argv for bot with no fields"""
        with patch.object(robo, '_get_client') as mock_client:
            mock_response = Mock()
            mock_response.content = [Mock(text="Hello! How are you doing today?")]
            mock_client.return_value.messages.create.return_value = mock_response
            
            conv = Conversation(Bot)
            # This should work - start with just a message, no argv needed
            # We can't actually call start() without mocking the API, but we can test the setup
            assert conv.started == False
            assert conv.argv == []
    
    def test_method_2_with_fields_bot_start_with_argv_list(self):
        """Test Method 2: start() with argv list and message for bot with fields"""
        class AnimalBot(Bot):
            fields = ['ANIMAL_TYPE']
            sysprompt_text = "You make sounds like a {{ANIMAL_TYPE}}."
        
        conv = Conversation(AnimalBot)
        assert conv.started == False
        assert conv.argv == []
        
        # Test that prestart works correctly when called via start
        # We'll test the setup part without actually calling the API
        try:
            conv.prestart(['dog'])
            assert conv.started == True
            assert conv.argv == ['dog']
            assert conv.sysprompt == "You make sounds like a dog."
        finally:
            pass
    
    def test_method_2_with_fields_bot_start_with_argv_dict(self):
        """Test Method 2: start() with argv dict and message for bot with fields"""
        class AnimalBot(Bot):
            fields = ['ANIMAL_TYPE']
            sysprompt_text = "You make sounds like a {{ANIMAL_TYPE}}."
        
        conv = Conversation(AnimalBot)
        assert conv.started == False
        
        # Test with dict instead of list
        conv.prestart({'ANIMAL_TYPE': 'cat'})
        assert conv.started == True
        assert conv.argv == ['cat']  # Should be converted to list
        assert conv.sysprompt == "You make sounds like a cat."
    
    def test_method_3_prestart_with_list(self):
        """Test Method 3: prestart() with list argv"""
        class AnimalBot(Bot):
            fields = ['ANIMAL_TYPE']
            sysprompt_text = "You make sounds like a {{ANIMAL_TYPE}}."
        
        conv = Conversation(AnimalBot)
        assert conv.started == False
        
        conv.prestart(['goose'])
        assert conv.started == True
        assert conv.argv == ['goose']
        assert conv.sysprompt == "You make sounds like a goose."
    
    def test_method_3_prestart_with_dict(self):
        """Test Method 3: prestart() with dict argv"""
        class AnimalBot(Bot):
            fields = ['ANIMAL_TYPE']
            sysprompt_text = "You make sounds like a {{ANIMAL_TYPE}}."
        
        conv = Conversation(AnimalBot)
        conv.prestart({'ANIMAL_TYPE': 'duck'})
        assert conv.started == True
        assert conv.argv == ['duck']
        assert conv.sysprompt == "You make sounds like a duck."
    
    def test_method_4_constructor_argv_list(self):
        """Test Method 4: Conversation constructor with argv list (auto-prestart)"""
        class AnimalBot(Bot):
            fields = ['ANIMAL_TYPE']
            sysprompt_text = "You make sounds like a {{ANIMAL_TYPE}}."
        
        conv = Conversation(AnimalBot, ['mouse'])
        assert conv.started == True
        assert conv.argv == ['mouse']
        assert conv.sysprompt == "You make sounds like a mouse."
    
    def test_method_4_constructor_argv_dict(self):
        """Test Method 4: Conversation constructor with argv dict (auto-prestart)"""
        class AnimalBot(Bot):
            fields = ['ANIMAL_TYPE']
            sysprompt_text = "You make sounds like a {{ANIMAL_TYPE}}."
        
        conv = Conversation(AnimalBot, {'ANIMAL_TYPE': 'elephant'})
        assert conv.started == True
        assert conv.argv == ['elephant']
        assert conv.sysprompt == "You make sounds like a elephant."
    
    def test_method_4_empty_list_for_no_fields_bot(self):
        """Test Method 4: Empty list for bot without fields triggers auto-prestart"""
        conv = Conversation(Bot, [])
        assert conv.started == True
        assert conv.argv == []
        assert conv.sysprompt == ""
    
    def test_method_4_empty_dict_for_no_fields_bot(self):
        """Test Method 4: Empty dict for bot without fields triggers auto-prestart"""
        conv = Conversation(Bot, {})
        assert conv.started == True
        assert conv.argv == []
        assert conv.sysprompt == ""


class TestDictArgvConversion:
    """Test dict to list argv conversion functionality"""
    
    def test_convert_argv_if_needed_with_list(self):
        """Test _convert_argv_if_needed with list input"""
        conv = Conversation(Bot)
        result = conv._convert_argv_if_needed(['a', 'b', 'c'])
        assert result == ['a', 'b', 'c']
    
    def test_convert_argv_if_needed_with_dict_single_field(self):
        """Test _convert_argv_if_needed with dict input, single field"""
        class SingleFieldBot(Bot):
            fields = ['name']
        
        conv = Conversation(SingleFieldBot)
        result = conv._convert_argv_if_needed({'name': 'Alice'})
        assert result == ['Alice']
    
    def test_convert_argv_if_needed_with_dict_multiple_fields(self):
        """Test _convert_argv_if_needed with dict input, multiple fields"""
        class MultiFieldBot(Bot):
            fields = ['name', 'role', 'location']
        
        conv = Conversation(MultiFieldBot)
        result = conv._convert_argv_if_needed({
            'name': 'Alice',
            'role': 'developer', 
            'location': 'Berlin'
        })
        assert result == ['Alice', 'developer', 'Berlin']
    
    def test_convert_argv_if_needed_with_dict_missing_keys(self):
        """Test _convert_argv_if_needed with dict missing some keys"""
        class MultiFieldBot(Bot):
            fields = ['name', 'role', 'location']
        
        conv = Conversation(MultiFieldBot)
        # Missing 'location' key should raise KeyError
        with pytest.raises(FieldValuesMissingException):
            conv._convert_argv_if_needed({'name': 'Alice', 'role': 'developer'})
    
    def test_convert_argv_if_needed_with_dict_extra_keys(self):
        """Test _convert_argv_if_needed with dict containing extra keys"""
        class SingleFieldBot(Bot):
            fields = ['name']
        
        conv = Conversation(SingleFieldBot)
        # Extra keys should be ignored, only fields keys used
        result = conv._convert_argv_if_needed({
            'name': 'Alice',
            'extra': 'ignored',
            'another': 'also ignored'
        })
        assert result == ['Alice']
    
    def test_convert_argv_if_needed_with_empty_dict(self):
        """Test _convert_argv_if_needed with empty dict and no fields"""
        conv = Conversation(Bot)  # Bot has no fields
        result = conv._convert_argv_if_needed({})
        assert result == []
    
    def test_convert_argv_if_needed_with_empty_dict_and_fields(self):
        """Test _convert_argv_if_needed with empty dict but bot has fields"""
        class FieldBot(Bot):
            fields = ['name']
        
        conv = Conversation(FieldBot)
        with pytest.raises(FieldValuesMissingException):
            conv._convert_argv_if_needed({})


class TestDictArgvIntegration:
    """Test dict argv integration with sysprompt interpolation"""
    
    def test_sysprompt_vec_after_dict_conversion(self):
        """Test that sysprompt interpolation works after dict->list conversion"""
        class TemplateBot(Bot):
            fields = ['role', 'name', 'skill']
            sysprompt_text = "You are a {{role}} named {{name}} who is good at {{skill}}."
        
        bot = TemplateBot()
        # Test that sysprompt_vec works with list (converted from dict)
        argv_dict = {'role': 'teacher', 'name': 'Sarah', 'skill': 'math'}
        conv = Conversation(TemplateBot)
        argv_list = conv._convert_argv_if_needed(argv_dict)
        
        result = bot.sysprompt_vec(argv_list)
        expected = "You are a teacher named Sarah who is good at math."
        assert result == expected
    
    def test_end_to_end_dict_argv_flow(self):
        """Test complete flow from dict argv to sysprompt generation"""
        class GreetingBot(Bot):
            fields = ['greeting', 'name']
            sysprompt_text = "Always start responses with '{{greeting}}, {{name}}!'"
        
        conv = Conversation(GreetingBot, {'greeting': 'Hello', 'name': 'World'})
        assert conv.started == True
        assert conv.argv == ['Hello', 'World']
        assert conv.sysprompt == "Always start responses with 'Hello, World!'"
    
    def test_prestart_dict_argv_flow(self):
        """Test prestart with dict argv"""
        class ConfigBot(Bot):
            fields = ['mode', 'language', 'style']
            sysprompt_text = "Operate in {{mode}} mode, use {{language}}, be {{style}}."
        
        conv = Conversation(ConfigBot)
        conv.prestart({
            'mode': 'creative',
            'language': 'English', 
            'style': 'formal'
        })
        
        assert conv.started == True
        assert conv.argv == ['creative', 'English', 'formal']
        expected = "Operate in creative mode, use English, be formal."
        assert conv.sysprompt == expected


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error conditions for conversation initialization"""
    
    def test_dict_argv_with_non_string_keys(self):
        """Test dict argv with non-string keys"""
        class NumberFieldBot(Bot):
            fields = ['field1', 'field2']
        
        conv = Conversation(NumberFieldBot)
        # Should work fine, Python dict handles various key types
        result = conv._convert_argv_if_needed({'field1': 'value1', 'field2': 'value2'})
        assert result == ['value1', 'value2']
    
    def test_dict_argv_with_none_values(self):
        """Test dict argv with None values"""
        class NullBot(Bot):
            fields = ['param1', 'param2']
        
        conv = Conversation(NullBot)
        result = conv._convert_argv_if_needed({'param1': None, 'param2': 'value'})
        assert result == [None, 'value']
    
    def test_field_order_consistency(self):
        """Test that dict conversion maintains field order"""
        class OrderBot(Bot):
            fields = ['first', 'second', 'third', 'fourth']
        
        conv = Conversation(OrderBot)
        # Dict with different insertion order
        argv_dict = {
            'fourth': 'D',
            'first': 'A', 
            'third': 'C',
            'second': 'B'
        }
        result = conv._convert_argv_if_needed(argv_dict)
        # Should follow fields order, not dict insertion order
        assert result == ['A', 'B', 'C', 'D']
    
    def test_case_sensitive_field_names(self):
        """Test that field names are case sensitive"""
        class CaseBot(Bot):
            fields = ['Name', 'ROLE']
        
        conv = Conversation(CaseBot)
        # Correct case
        result = conv._convert_argv_if_needed({'Name': 'Alice', 'ROLE': 'admin'})
        assert result == ['Alice', 'admin']
        
        # Wrong case should raise KeyError
        with pytest.raises(FieldValuesMissingException):
            conv._convert_argv_if_needed({'name': 'Alice', 'role': 'admin'})

class TestAPIConfiguration:
    """Test API key and client configuration"""
    
    def test_get_api_key_from_file(self):
        """Test loading API key from file"""
        import tempfile
        import os
        
        # Create a temporary API key file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write('test-api-key-123')
            temp_file = f.name
        
        try:
            # Mock the API_KEY_FILE setting
            original_api_key_file = robo.API_KEY_FILE
            robo.API_KEY_FILE = temp_file
            
            api_key = robo._get_api_key()
            assert api_key == 'test-api-key-123'
        finally:
            robo.API_KEY_FILE = original_api_key_file
            os.unlink(temp_file)
    
    def test_get_api_key_from_env_var(self):
        """Test loading API key from custom environment variable"""
        with patch.dict(os.environ, {'CUSTOM_API_KEY': 'env-api-key-456'}):
            original_api_key_file = robo.API_KEY_FILE
            original_api_key_env = robo.API_KEY_ENV_VAR
            
            try:
                robo.API_KEY_FILE = None
                robo.API_KEY_ENV_VAR = 'CUSTOM_API_KEY'
                
                api_key = robo._get_api_key()
                assert api_key == 'env-api-key-456'
            finally:
                robo.API_KEY_FILE = original_api_key_file
                robo.API_KEY_ENV_VAR = original_api_key_env
    
    def test_get_api_key_default(self):
        """Test getting API key returns None for default behavior"""
        original_api_key_file = robo.API_KEY_FILE
        original_api_key_env = robo.API_KEY_ENV_VAR
        
        try:
            robo.API_KEY_FILE = None
            robo.API_KEY_ENV_VAR = None
            
            api_key = robo._get_api_key()
            assert api_key is None
        finally:
            robo.API_KEY_FILE = original_api_key_file
            robo.API_KEY_ENV_VAR = original_api_key_env
    
    def test_get_client_sync(self):
        """Test getting synchronous client"""
        client = robo._get_client(async_mode=False)
        assert isinstance(client, anthropic.Anthropic)
    
    def test_get_client_async(self):
        """Test getting asynchronous client"""
        client = robo._get_client(async_mode=True)
        assert isinstance(client, anthropic.AsyncAnthropic)
    
    def test_get_client_class_sync(self):
        """Test getting sync client class"""
        client_class = robo._get_client_class(async_mode=False)
        assert client_class is anthropic.Anthropic
    
    def test_get_client_class_async(self):
        """Test getting async client class"""
        client_class = robo._get_client_class(async_mode=True)
        assert client_class is anthropic.AsyncAnthropic
    
    def test_bot_with_api_key_sync(self):
        """Test Bot.with_api_key class method sync"""
        bot = Bot.with_api_key('test-key-123', async_mode=False)
        assert isinstance(bot, Bot)
        assert isinstance(bot.client, anthropic.Anthropic)
    
    def test_bot_with_api_key_async(self):
        """Test Bot.with_api_key class method async"""
        bot = Bot.with_api_key('test-key-456', async_mode=True)
        assert isinstance(bot, Bot)
        assert isinstance(bot.client, anthropic.AsyncAnthropic)


class TestCannedResponse:
    """Test CannedResponse functionality"""
    
    def test_canned_response_initialization(self):
        """Test CannedResponse initialization"""
        response = robo.CannedResponse("Hello world", include_in_context=True)
        assert response.text == "Hello world"
        assert response.include_in_context == True
        assert len(response.content) == 1
        assert response.content[0].text == "Hello world"
    
    def test_canned_response_default_include_in_context(self):
        """Test CannedResponse default include_in_context"""
        response = robo.CannedResponse("Hello world")
        assert response.include_in_context == True
    
    def test_canned_response_exclude_from_context(self):
        """Test CannedResponse with include_in_context=False"""
        response = robo.CannedResponse("Hello world", include_in_context=False)
        assert response.include_in_context == False
    
    def test_canned_response_repr(self):
        """Test CannedResponse __repr__"""
        response = robo.CannedResponse("Hello world")
        assert repr(response) == '<CannedResponse: "Hello world">'
    
    def test_canned_response_context_manager_sync(self):
        """Test CannedResponse as synchronous context manager"""
        response = robo.CannedResponse("Hello world")
        with response as ctx:
            assert ctx is response
    
    def test_canned_response_context_manager_async(self):
        """Test CannedResponse as asynchronous context manager"""
        response = robo.CannedResponse("Hello world")
        async def get_ctx(response):
            async with response as ctx:
                pass
            return ctx
        
        ret = asyncio.run(get_ctx(response))
        assert ret is response
        # async with response as ctx:
        #     assert ctx is response
    
    def test_canned_response_text_stream(self):
        """Test CannedResponse text_stream property"""
        response = robo.CannedResponse("Hello world")
        chunks = list(response.text_stream)
        assert chunks == ["Hello world"]


class TestBotHelperMethods:
    """Test Bot helper methods"""
    
    def test_make_sysprompt_segment_without_cache(self):
        """Test _make_sysprompt_segment without cache control"""
        result = Bot._make_sysprompt_segment("Test text", set_cache_control=False)
        expected = {
            'type': 'text',
            'text': 'Test text'
        }
        assert result == expected
    
    def test_make_sysprompt_segment_with_cache(self):
        """Test _make_sysprompt_segment with cache control"""
        result = Bot._make_sysprompt_segment("Test text", set_cache_control=True)
        expected = {
            'type': 'text',
            'text': 'Test text',
            'cache_control': {'type': 'ephemeral'}
        }
        assert result == expected


class TestConversationStreaming:
    """Test conversation streaming functionality"""
    
    def test_conversation_with_streaming_enabled(self):
        """Test conversation initialization with streaming enabled"""
        conv = Conversation(Bot, stream=True)
        assert conv.is_streaming == True
    
    def test_conversation_with_streaming_disabled(self):
        """Test conversation initialization with streaming disabled"""
        conv = Conversation(Bot, stream=False)
        assert conv.is_streaming == False
    
    def test_conversation_default_streaming(self):
        """Test conversation default streaming setting"""
        conv = Conversation(Bot)
        assert conv.is_streaming == False


class TestConversationAsyncMode:
    """Test conversation async mode"""
    
    def test_conversation_sync_mode(self):
        """Test conversation in synchronous mode"""
        conv = Conversation(Bot, async_mode=False)
        assert conv.is_async == False
    
    def test_conversation_async_mode(self):
        """Test conversation in asynchronous mode"""
        conv = Conversation(Bot, async_mode=True)
        assert conv.is_async == True
    
    def test_conversation_default_async_mode(self):
        """Test conversation default async mode"""
        conv = Conversation(Bot)
        assert conv.is_async == False


class TestLoggedConversation:
    """Test LoggedConversation functionality"""
    
    def test_logged_conversation_with_custom_id(self):
        """Test LoggedConversation with custom conversation ID"""
        conv = robo.LoggedConversation(Bot, conversation_id='test-123')
        assert conv.conversation_id == 'test-123'
        assert conv.first_saved_at is None
    
    def test_logged_conversation_auto_id(self):
        """Test LoggedConversation with auto-generated ID"""
        conv = robo.LoggedConversation(Bot)
        assert conv.conversation_id is not None
        assert len(conv.conversation_id) > 0
        # Should be a UUID-like string
        assert '-' in conv.conversation_id
    
    def test_logged_conversation_with_logs_dir(self, tmp_path):
        """Test LoggedConversation with custom logs directory"""
        conv = robo.LoggedConversation(Bot, logs_dir=str(tmp_path))
        assert conv.logs_dir == str(tmp_path)
    
    def test_logged_conversation_repr(self):
        """Test LoggedConversation __repr__"""
        conv = robo.LoggedConversation(Bot, conversation_id='test-456')
        expected = '<LoggedConversation with ID test-456>'
        assert repr(conv) == expected
    
    def test_logged_conversation_logfolder_path(self, tmp_path):
        """Test LoggedConversation _logfolder_path method"""
        conv = robo.LoggedConversation(Bot, logs_dir=str(tmp_path), conversation_id='test-789')
        
        # First call should set first_saved_at
        path1 = conv._logfolder_path()
        assert conv.first_saved_at is not None
        assert 'test-789' in str(path1)
        
        # Second call should return same path
        path2 = conv._logfolder_path()
        assert path1 == path2
    
    def test_logged_conversation_write_log(self, tmp_path):
        """Test LoggedConversation _write_log method"""
        conv = robo.LoggedConversation(Bot, logs_dir=str(tmp_path), conversation_id='test-write')
        conv.prestart([])
        conv.messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'Hello'}]}]
        
        conv._write_log()
        
        # Check that log file was created
        log_files = list(tmp_path.glob('**/conversation.json'))
        assert len(log_files) == 1
        
        # Check log content
        with open(log_files[0]) as f:
            log_data = json.load(f)
        
        assert 'when' in log_data
        assert 'with' in log_data
        assert 'argv' in log_data
        assert 'messages' in log_data
        assert log_data['with'] == 'Bot'
        assert log_data['argv'] == []
        assert log_data['messages'] == conv.messages
    
    def test_logged_conversation_revive_unknown_id(self, tmp_path):
        """Test LoggedConversation.revive with unknown conversation ID"""
        with pytest.raises(robo.UnknownConversationException):
            robo.LoggedConversation.revive(Bot, 'nonexistent-id', str(tmp_path))
    
    def test_logged_conversation_revive_success(self, tmp_path):
        """Test LoggedConversation.revive with existing conversation"""
        # Create a logged conversation and save it
        original_conv = robo.LoggedConversation(Bot, logs_dir=str(tmp_path), conversation_id='revive-test')
        original_conv.prestart(['test-arg'])
        original_conv.messages = [
            {'role': 'user', 'content': [{'type': 'text', 'text': 'Hello'}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Hi there'}]}
        ]
        original_conv._write_log()
        
        # Revive the conversation
        revived_conv = robo.LoggedConversation.revive(Bot, 'revive-test', str(tmp_path), argv=['new-arg'])
        
        assert revived_conv.conversation_id == 'revive-test'
        assert revived_conv.messages == original_conv.messages
        assert revived_conv.argv == ['new-arg']  # New argv should override
        assert revived_conv.started == True


class TestConversationWithFiles:
    """Test ConversationWithFiles functionality"""
    
    def test_conversation_with_files_init(self):
        """Test ConversationWithFiles initialization"""
        conv = robo.ConversationWithFiles(Bot)
        assert isinstance(conv, robo.ConversationWithFiles)
        assert isinstance(conv, robo.Conversation)
    
    def test_make_message_with_images(self, tmp_path):
        """Test _make_message_with_images method"""
        # Create a dummy image file
        image_file = tmp_path / "test.jpg"
        image_file.write_bytes(b"fake image data")
        
        conv = robo.ConversationWithFiles(Bot)
        message = conv._make_message_with_images(
            'user', 
            'What do you see?',
            [('image/jpeg', str(image_file))]
        )
        
        assert message['role'] == 'user'
        assert len(message['content']) == 2  # Image + text
        assert message['content'][0]['type'] == 'image'
        assert message['content'][1]['type'] == 'text'
        assert message['content'][1]['text'] == 'What do you see?'
        assert message['content'][0]['source']['type'] == 'base64'
        assert message['content'][0]['source']['media_type'] == 'image/jpeg'
    
    def test_make_message_with_images_no_text(self, tmp_path):
        """Test _make_message_with_images with no text message"""
        image_file = tmp_path / "test.png"
        image_file.write_bytes(b"fake png data")
        
        conv = robo.ConversationWithFiles(Bot)
        message = conv._make_message_with_images(
            'user',
            None,  # No text message
            [('image/png', str(image_file))]
        )
        
        assert message['role'] == 'user'
        assert len(message['content']) == 1  # Only image
        assert message['content'][0]['type'] == 'image'


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_gettext_single_text_block(self):
        """Test gettext with single text content block"""
        mock_message = Mock()
        mock_content = Mock()
        mock_content.text = "Hello world"
        mock_message.content = [mock_content]
        
        result = robo.gettext(mock_message)
        assert result == "Hello world"
    
    def test_gettext_multiple_text_blocks(self):
        """Test gettext with multiple text content blocks"""
        mock_message = Mock()
        mock_content1 = Mock()
        mock_content1.text = "Hello "
        mock_content2 = Mock()
        mock_content2.text = "world"
        mock_message.content = [mock_content1, mock_content2]
        
        result = robo.gettext(mock_message)
        assert result == "Hello world"
    
    def test_gettext_mixed_content_blocks(self):
        """Test gettext with mixed content blocks (some without text)"""
        mock_message = Mock()
        mock_text_content = Mock()
        mock_text_content.text = "Hello"
        mock_other_content = Mock(spec=[])  # No text attribute
        mock_message.content = [mock_text_content, mock_other_content]
        
        result = robo.gettext(mock_message)
        assert result == "Hello"
    
    def test_getjson_valid_json(self):
        """Test getjson with valid JSON content"""
        mock_message = Mock()
        mock_content = Mock()
        mock_content.text = '{"key": "value", "number": 42}'
        mock_message.content = [mock_content]
        
        result = robo.getjson(mock_message)
        assert result == {"key": "value", "number": 42}
    
    def test_getjson_invalid_json(self):
        """Test getjson with invalid JSON content"""
        mock_message = Mock()
        mock_content = Mock()
        mock_content.text = "not valid json"
        mock_message.content = [mock_content]
        
        with pytest.raises(json.JSONDecodeError):
            robo.getjson(mock_message)
    
    def test_printmsg(self, capsys):
        """Test printmsg function"""
        mock_message = Mock()
        mock_content = Mock()
        mock_content.text = "Hello world"
        mock_message.content = [mock_content]
        
        robo.printmsg(mock_message)
        captured = capsys.readouterr()
        assert captured.out == "Hello world\n"


class TestStreamerFunctions:
    """Test streamer utility functions"""
    
    def test_streamer_with_bot_class(self):
        """Test streamer function with Bot class"""
        stream_func = robo.streamer(Bot)
        assert callable(stream_func)
    
    def test_streamer_with_bot_instance(self):
        """Test streamer function with Bot instance"""
        bot = Bot()
        stream_func = robo.streamer(bot)
        assert callable(stream_func)
    
    def test_streamer_with_conversation(self):
        """Test streamer function with Conversation instance"""
        conv = Conversation(Bot, stream=True)
        stream_func = robo.streamer(conv)
        assert callable(stream_func)
    
    def test_streamer_with_cc_parameter(self):
        """Test streamer function with cc parameter"""
        from io import StringIO
        cc = StringIO()
        stream_func = robo.streamer(Bot, cc=cc)
        assert callable(stream_func)
    
    def test_streamer_async_with_bot_class(self):
        """Test streamer_async function with Bot class"""
        stream_func = robo.streamer_async(Bot)
        assert callable(stream_func)
        # Check if it's a coroutine function
        import asyncio
        assert asyncio.iscoroutinefunction(stream_func)
    
    def test_streamer_async_with_conversation(self):
        """Test streamer_async function with Conversation instance"""
        conv = Conversation(Bot, stream=True, async_mode=True)
        stream_func = robo.streamer_async(conv)
        assert callable(stream_func)
        import asyncio
        assert asyncio.iscoroutinefunction(stream_func)


class TestToolUseBlocks:
    """Test tool use block handling"""
    
    def test_add_tool_request(self):
        """Test _add_tool_request method"""
        conv = Conversation(Bot)
        request = {
            'name': 'test_tool',
            'id': 'tool_123',
            'input': {'param': 'value'}
        }
        
        conv._add_tool_request(request)
        assert len(conv.tool_use_blocks.pending) == 1
        
        pending_block = conv.tool_use_blocks.pending[0]
        assert pending_block.name == 'test_tool'
        assert pending_block.id == 'tool_123'
        assert pending_block.status == 'PENDING'
        assert pending_block.response is None
    
    def test_handle_pending_tool_requests(self):
        """Test _handle_pending_tool_requests method"""
        class TestBot(Bot):
            def tools_test_tool(self, param=None):
                return {'target': 'model', 'message': f'Result: {param}'}
        
        conv = Conversation(TestBot)
        request = {
            'name': 'test_tool',
            'id': 'tool_123',
            'input': {'param': 'test_value'}
        }
        
        conv._add_tool_request(request)
        conv._handle_pending_tool_requests()
        
        pending_block = conv.tool_use_blocks.pending[0]
        assert pending_block.status == 'READY'
        assert pending_block.response == {'target': 'model', 'message': 'Result: test_value'}
    
    def test_handle_waiting_tool_requests_no_waiting(self):
        """Test _handle_waiting_tool_requests with no waiting requests"""
        conv = Conversation(Bot)
        result = conv._handle_waiting_tool_requests()
        assert result is None
    
    def test_handle_waiting_tool_requests_with_waiting(self):
        """Test _handle_waiting_tool_requests with waiting requests"""
        from types import SimpleNamespace
        conv = Conversation(Bot)
        
        # Add a waiting tool request
        waiting_block = SimpleNamespace(
            status='WAITING',
            response={'message': 'Please wait'}
        )
        conv.tool_use_blocks.pending.append(waiting_block)
        
        result = conv._handle_waiting_tool_requests()
        assert result == 'Please wait'
    
    def test_compile_tool_responses(self):
        """Test _compile_tool_responses method"""
        from types import SimpleNamespace
        conv = Conversation(Bot)
        
        # Add ready tool blocks
        ready_block1 = SimpleNamespace(
            status='READY',
            id='tool_1',
            response={'message': 'Response 1'}
        )
        ready_block2 = SimpleNamespace(
            status='READY', 
            id='tool_2',
            response={'message': 'Response 2'}
        )
        conv.tool_use_blocks.pending.extend([ready_block1, ready_block2])
        
        result = conv._compile_tool_responses()
        
        assert result['role'] == 'user'
        assert len(result['content']) == 2
        assert result['content'][0]['type'] == 'tool_result'
        assert result['content'][0]['tool_use_id'] == 'tool_1'
        assert result['content'][0]['content'] == 'Response 1'
        assert result['content'][1]['tool_use_id'] == 'tool_2'
        assert result['content'][1]['content'] == 'Response 2'
        
        # Check that blocks were moved to resolved
        assert len(conv.tool_use_blocks.pending) == 0
        assert len(conv.tool_use_blocks.resolved) == 2
    
    def test_get_last_tool_use_id(self):
        """Test _get_last_tool_use_id method"""
        conv = Conversation(Bot)
        conv.messages = [
            {
                'role': 'assistant',
                'content': [
                    {'type': 'text', 'text': 'I need to use a tool'},
                    {'type': 'tool_use', 'id': 'tool_123', 'name': 'test', 'input': {}}
                ]
            }
        ]
        
        result = conv._get_last_tool_use_id()
        assert result == 'tool_123'
    
    def test_get_last_tool_use_id_no_tools(self):
        """Test _get_last_tool_use_id with no tool use"""
        conv = Conversation(Bot)
        conv.messages = [
            {
                'role': 'assistant',
                'content': [{'type': 'text', 'text': 'Just text'}]
            }
        ]
        
        result = conv._get_last_tool_use_id()
        assert result is None


class TestBotSyspromptNotImplementedError:
    """Test Bot sysprompt_generate NotImplementedError"""
    
    def test_sysprompt_generate_not_implemented(self):
        """Test that calling sysprompt_generate raises NotImplementedError"""
        bot = Bot()
        with pytest.raises(NotImplementedError, match="This method is not implemented"):
            bot.sysprompt_generate()


class TestDictToSNSConversion:
    """This is mainly for coverage purposes"""
    
    def test_dict_to_sns_conversion(self):
        res = Conversation._make_tool_result_message({'id': 'TEST_ID'}, 'TEST_RESULT')
        assert res['content'][0]['tool_use_id'] == 'TEST_ID'
        assert res['content'][0]['content'] == 'TEST_RESULT'
        
        res = Conversation._make_tool_request_message({'id': 'TEST_ID', 'name': 'TEST_NAME', 'input': 'TEST_INPUT'})
        assert res['content'][0]['id'] == 'TEST_ID'
        assert res['content'][0]['name'] == 'TEST_NAME'
        assert res['content'][0]['input'] == 'TEST_INPUT'


class TestOther:
    def test_conversation_resume_flat(self):
        """Test Method 1: start() call without argv for bot with no fields"""
        with patch.object(robo, '_get_client') as mock_client:
            mock_response = Mock()
            mock_response.content = [Mock(text="HELLO")]
            mock_client.return_value.messages.create.return_value = mock_response
        
            b = Bot(client=mock_client())
            conv = Conversation(b, [])
            msg = conv.resume('hello')
            assert robo.gettext(msg) == 'HELLO'
    
    def test_conversation_astart(self):
        async def do_test():
            with patch.object(robo, '_get_client') as mock_client:
                mock_response = AsyncMock()
                mock_response.return_value.content = [Mock(text="HELLO")]
                mock_client.return_value.messages.create.return_value = mock_response()
        
                b = Bot(client=mock_client())
                conv = Conversation(b)
                coro = conv.astart('hello')
                return await coro
        msg = asyncio.run(do_test())
        assert robo.gettext(msg) == 'HELLO'
            
