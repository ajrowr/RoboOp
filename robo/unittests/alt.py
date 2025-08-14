import pytest
from unittest.mock import AsyncMock, Mock, patch
from types import SimpleNamespace
import asyncio

# Assuming your module is called 'robo'
from robo import Bot, Conversation


@pytest.fixture
def mock_bot():
    """Create a mock bot for testing"""
    bot = Mock(spec=Bot)
    bot.model = "claude-3-sonnet-20240229"
    bot.max_tokens = 1000
    bot.temperature = 1.0
    bot.oneshot = False
    bot.welcome_message = None
    bot.soft_start = False
    bot.fields = []
    
    # Mock the client
    bot.client = Mock()
    bot.client.messages = Mock()
    
    # Mock the methods that might be called
    bot.sysprompt_vec = Mock(return_value="Test system prompt")
    bot.get_tools_schema = Mock(return_value=[])
    bot.preprocess_response = Mock(return_value=None)
    
    return bot


@pytest.fixture
def mock_message_response():
    """Create a mock message response from the API"""
    # Create a mock text content block
    text_block = Mock()
    text_block.text = "Hello! This is a test response from Claude."
    
    # Create the mock message response
    message = Mock()
    message.content = [text_block]
    
    return message


@pytest.mark.asyncio
async def test_astart_with_message_only(mock_bot, mock_message_response):
    """Test astart() with just a message (no argv)"""
    # Setup the async mock for the API call
    mock_bot.client.messages.create = AsyncMock(return_value=mock_message_response)
    
    # Create conversation
    conv = Conversation(mock_bot, async_mode=True)
    
    # Test astart with just a message
    response = await conv.astart("Hello, how are you?")
    
    # Verify the conversation was properly initialized
    assert conv.started is True
    assert conv.argv == []
    assert len(conv.messages) == 2
    assert conv.messages[0]['role'] == 'user'
    assert conv.messages[0]['content'][0]['text'] == "Hello, how are you?"
    
    # Verify the API was called correctly
    mock_bot.client.messages.create.assert_called_once()
    call_kwargs = mock_bot.client.messages.create.call_args.kwargs
    assert call_kwargs['model'] == mock_bot.model
    assert call_kwargs['max_tokens'] == mock_bot.max_tokens
    assert call_kwargs['temperature'] == mock_bot.temperature
    assert len(call_kwargs['messages']) == 2
    assert call_kwargs['messages'][0]['role'] == 'user'
    
    # Verify the response
    assert response == mock_message_response


@pytest.mark.asyncio
async def test_astart_with_argv_and_message(mock_bot, mock_message_response):
    """Test astart() with both argv and message"""
    # Setup bot fields for template substitution
    mock_bot.fields = ['name', 'role']
    mock_bot.sysprompt_vec = Mock(return_value="Hello {{name}}, you are a {{role}}")
    
    # Setup the async mock
    mock_bot.client.messages.create = AsyncMock(return_value=mock_message_response)
    
    # Create conversation
    conv = Conversation(mock_bot, async_mode=True)
    
    # Test astart with argv and message
    response = await conv.astart(['Alice', 'assistant'], "What can you help me with?")
    
    # Verify initialization
    assert conv.started is True
    assert conv.argv == ['Alice', 'assistant']
    
    # Verify sysprompt_vec was called with the argv
    mock_bot.sysprompt_vec.assert_called_once_with(['Alice', 'assistant'])
    
    # Verify the message was added
    assert len(conv.messages) == 2
    assert conv.messages[0]['content'][0]['text'] == "What can you help me with?"
    
    assert response == mock_message_response


@pytest.mark.asyncio
async def test_astart_already_started_raises_exception(mock_bot):
    """Test that astart() raises exception if conversation already started"""
    conv = Conversation(mock_bot, async_mode=True)
    
    # Manually mark as started
    conv.started = True
    
    # Should raise exception
    with pytest.raises(Exception, match="Conversation has already started"):
        await conv.astart("This should fail")


@pytest.mark.asyncio
async def test_astart_with_tool_use_response(mock_bot):
    """Test astart() when the response contains tool use"""
    # Create a mock response with tool use
    class ToolUseBlock(Mock):
        pass
        
    tool_block = ToolUseBlock() ## to trick the block type detector
    tool_block.id = "tool_123"
    tool_block.name = "test_tool"
    tool_block.input = {"param": "value"}
    
    text_block = Mock()
    text_block.text = "I need to use a tool."
    
    message = Mock()
    message.content = [text_block, tool_block]
    
    # Mock the tool handling
    mock_bot.client.messages.create = AsyncMock(return_value=message)
    mock_bot.handle_tool_call = Mock(return_value={
        'target': 'model',
        'message': 'Tool result'
    })
    
    # Create second response for after tool execution
    final_response = Mock()
    final_text_block = Mock()
    final_text_block.text = "Based on the tool result, here's my answer."
    final_response.content = [final_text_block]
    
    # Set up multiple calls to create()
    mock_bot.client.messages.create = AsyncMock(side_effect=[message, final_response])
    
    conv = Conversation(mock_bot, async_mode=True)
    print(locals())
    
    # This should handle the tool use automatically
    response = await conv.astart("Please use a tool to help me")
    
    # Verify tool was called
    mock_bot.handle_tool_call.assert_called_once()
    
    # Verify multiple API calls were made
    assert mock_bot.client.messages.create.call_count == 2
    
    # Final response should be the second one
    assert response == final_response


@pytest.mark.asyncio
async def test_astart_with_canned_response(mock_bot):
    """Test astart() when preprocess_response returns a canned response"""
    from robo import CannedResponse
    
    # Mock preprocess_response to return a canned response
    mock_bot.preprocess_response = Mock(return_value="This is a canned response")
    
    conv = Conversation(mock_bot, async_mode=True)
    
    response = await conv.astart("Hello")
    
    # Should return a CannedResponse object
    assert isinstance(response, CannedResponse)
    assert response.text == "This is a canned response"
    
    # API should not have been called
    mock_bot.client.messages.create.assert_not_called()


# Helper test to verify the mock setup works
# @pytest.mark.asyncio
# async def test_mock_setup_verification(mock_bot, mock_message_response):
#     """Verify our mocks are set up correctly"""
#     # Test that we can await the mocked method
#     result = await mock_bot.client.messages.create()
#     assert result == mock_message_response
#
#     # Test that the mock tracks calls
#     mock_bot.client.messages.create.assert_called_once()


if __name__ == "__main__":
    # Run a simple test
    async def simple_test():
        from unittest.mock import Mock, AsyncMock
        
        # Create a simple mock bot
        bot = Mock()
        bot.model = "test-model"
        bot.max_tokens = 1000
        bot.temperature = 1.0
        bot.oneshot = False
        bot.welcome_message = None
        bot.soft_start = False
        bot.fields = []
        bot.client = Mock()
        bot.sysprompt_vec = Mock(return_value="System prompt")
        bot.get_tools_schema = Mock(return_value=[])
        bot.preprocess_response = Mock(return_value=None)
        
        # Mock response
        response = Mock()
        text_block = Mock()
        text_block.text = "Test response"
        response.content = [text_block]
        
        bot.client.messages.create = AsyncMock(return_value=response)
        
        # Test the conversation
        conv = Conversation(bot, async_mode=True)
        result = await conv.astart("Hello")
        
        print(f"Success! Got response: {result}")
        print(f"Response text: {result.content[0].text}")
    
    # Run the simple test
    asyncio.run(simple_test())
