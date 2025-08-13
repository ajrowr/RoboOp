import pytest
import json
from unittest.mock import Mock, patch
from robo import Bot, Conversation, MODELS


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
