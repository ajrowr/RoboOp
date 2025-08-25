"""
Fake Anthropic client classes for testing RoboOp framework.

These classes mimic the behavior of anthropic.Anthropic and anthropic.AsyncAnthropic
without making actual API calls, allowing for predictable testing.
"""

import json
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator, Generator
from dataclasses import dataclass
import uuid
import time


@dataclass
class Usage:
    """Mimics anthropic usage tracking"""
    input_tokens: int
    output_tokens: int
    
    def model_dump(self):
        return {
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens
        }


class TextBlock:
    """Mimics anthropic TextBlock"""
    def __init__(self, text: str):
        self.text = text
        self.type = "text"


class ToolUseBlock:
    """Mimics anthropic ToolUseBlock"""
    def __init__(self, id: str, name: str, input: dict):
        self.id = id
        self.name = name
        self.input = input
        self.type = "tool_use"


class StreamEvent:
    """Base class for stream events"""
    def __init__(self, event_type: str):
        self.type = event_type


class MessageStartEvent(StreamEvent):
    """Message start stream event"""
    def __init__(self):
        super().__init__("message_start")


class ContentBlockStartEvent(StreamEvent):
    """Content block start event"""
    def __init__(self, content_block):
        super().__init__("content_block_start")
        self.content_block = content_block


class TextEvent(StreamEvent):
    """Text delta event"""
    def __init__(self, text: str):
        super().__init__("text")
        self.text = text


class ContentBlockStopEvent(StreamEvent):
    """Content block stop event"""
    def __init__(self, content_block):
        super().__init__("content_block_stop")
        self.content_block = content_block


class MessageStopEvent(StreamEvent):
    """Message stop event"""
    def __init__(self):
        super().__init__("message_stop")


class FakeMessage:
    """Mimics anthropic Message object"""
    def __init__(self, content_blocks: List, usage: Usage = None):
        self.content = content_blocks
        self.usage = usage or Usage(input_tokens=100, output_tokens=50)
        self.id = f"msg_{uuid.uuid4().hex[:8]}"
        self.model = "claude-sonnet-4-20250514"
        self.role = "assistant"
        self.stop_reason = "end_turn"
    
    def __repr__(self):
        return f'<{self.__module__}.{self.__class__.__name__}: "{self.content}">'


class FakeStreamManager:
    """Mimics anthropic MessageStreamManager for sync streaming"""
    
    def __init__(self, response_generator):
        self.response_generator = response_generator
        self._final_message = None
        self._events = []
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
        
    def __iter__(self):
        content_blocks = []
        current_text = ""
        
        for response_item in self.response_generator:
            if isinstance(response_item, str):
                # Simple text response
                if not content_blocks:
                    text_block = TextBlock("")
                    content_blocks.append(text_block)
                    yield MessageStartEvent()
                    yield ContentBlockStartEvent(text_block)
                
                # Simulate character-by-character streaming
                for char in response_item:
                    current_text += char
                    text_block.text = current_text
                    event = TextEvent(char)
                    self._events.append(event)
                    yield event
                    
            elif isinstance(response_item, dict) and response_item.get('type') == 'tool_use':
                # Tool use block
                tool_block = ToolUseBlock(
                    response_item['id'],
                    response_item['name'], 
                    response_item['input']
                )
                content_blocks.append(tool_block)
                yield ContentBlockStartEvent(tool_block)
                yield ContentBlockStopEvent(tool_block)
        
        # Finalize any text blocks
        if content_blocks and isinstance(content_blocks[-1], TextBlock):
            yield ContentBlockStopEvent(content_blocks[-1])
            
        yield MessageStopEvent()
        self._final_message = FakeMessage(content_blocks)
        
    @property
    def text_stream(self) -> Generator[str, None, None]:
        """Stream just the text portions"""
        for event in self:
            if isinstance(event, TextEvent):
                yield event.text
    
    def get_final_message(self):
        """Get the final assembled message"""
        if self._final_message is None:
            # Force consumption of the stream
            list(self)
        return self._final_message


class FakeAsyncStreamManager:
    """Mimics anthropic AsyncMessageStreamManager for async streaming"""
    
    def __init__(self, response_generator):
        self.response_generator = response_generator
        self._final_message = None
        self._events = []
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False
        
    async def __aiter__(self):
        content_blocks = []
        current_text = ""
        
        for response_item in self.response_generator:
            if isinstance(response_item, str):
                # Simple text response
                if not content_blocks:
                    text_block = TextBlock("")
                    content_blocks.append(text_block)
                    yield MessageStartEvent()
                    yield ContentBlockStartEvent(text_block)
                
                # Simulate character-by-character streaming
                for char in response_item:
                    await asyncio.sleep(0.001)  # Small delay to simulate streaming
                    current_text += char
                    text_block.text = current_text
                    event = TextEvent(char)
                    self._events.append(event)
                    yield event
                    
            elif isinstance(response_item, dict) and response_item.get('type') == 'tool_use':
                # Tool use block
                tool_block = ToolUseBlock(
                    response_item['id'],
                    response_item['name'], 
                    response_item['input']
                )
                content_blocks.append(tool_block)
                yield ContentBlockStartEvent(tool_block)
                yield ContentBlockStopEvent(tool_block)
        
        # Finalize any text blocks
        if content_blocks and isinstance(content_blocks[-1], TextBlock):
            yield ContentBlockStopEvent(content_blocks[-1])
            
        yield MessageStopEvent()
        self._final_message = FakeMessage(content_blocks)
    
    @property
    async def text_stream(self) -> AsyncGenerator[str, None]:
        """Stream just the text portions"""
        async for event in self:
            if isinstance(event, TextEvent):
                yield event.text
    
    async def get_final_message(self):
        """Get the final assembled message"""
        if self._final_message is None:
            # Force consumption of the stream
            async for _ in self:
                pass
        return self._final_message


class FakeMessages:
    """Mimics the messages API interface"""
    
    def __init__(self, response_scenarios=None):
        self.response_scenarios = response_scenarios or {}
        self.call_count = 0
        
    def create(self, model: str, max_tokens: int, messages: List[Dict], 
               system: Optional[str] = None, temperature: float = 1.0, 
               tools: Optional[List] = None, **kwargs) -> FakeMessage:
        """Create a non-streaming response"""
        self.call_count += 1
        
        # Generate response based on the last user message
        user_message_parts = []
        is_tool_response = False
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                if isinstance(msg.get('content'), str):
                    user_message_parts.append(msg['content'])
                elif isinstance(msg.get('content'), list):
                    for block in msg['content']:
                        if block.get('type') == 'tool_result':
                            user_message_parts.append(block.get('content', ''))
                            is_tool_response = True
                        elif block.get('type') == 'text':
                            user_message_parts.append(block.get('text', ''))
                break
        
        if not is_tool_response:
            user_message = ' '.join(user_message_parts)
        else:
            user_message = user_message_parts
        response_content = self._generate_response(user_message, tools, is_tool_response)
        content_blocks = []
        
        for item in response_content:
            if isinstance(item, str):
                content_blocks.append(TextBlock(item))
            elif isinstance(item, dict) and item.get('type') == 'tool_use':
                content_blocks.append(ToolUseBlock(item['id'], item['name'], item['input']))
                
        return FakeMessage(content_blocks)
    
    def stream(self, model: str, max_tokens: int, messages: List[Dict],
               system: Optional[str] = None, temperature: float = 1.0,
               tools: Optional[List] = None, **kwargs) -> FakeStreamManager:
        """Create a streaming response"""
        self.call_count += 1
        
        # Generate response based on the last user message
        user_message_parts = []
        is_tool_response = False
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                if isinstance(msg.get('content'), str):
                    user_message_parts.append(msg['content'])
                elif isinstance(msg.get('content'), list):
                    for block in msg['content']:
                        if block.get('type') == 'tool_result':
                            user_message_parts.append(block.get('content', ''))
                            is_tool_response = True
                        elif block.get('type') == 'text':
                            user_message_parts.append(block.get('text', ''))
                break
        
        if not is_tool_response:
            user_message = ' '.join(user_message_parts)
        else:
            user_message = user_message_parts
        response_content = self._generate_response(user_message, tools, is_tool_response)
        return FakeStreamManager(response_content)
    
    def _generate_response(self, user_message: str, tools: Optional[List] = None, is_tool_response:bool = False) -> List:
        """Generate response content based on user message and available tools"""
        
        # Check for custom scenarios first
        if type(user_message) is str:
            if user_message in self.response_scenarios:
                return self.response_scenarios[user_message]
                
        if is_tool_response:
            return [f"Tool response was:{str(user_message)}"]
        
        # Simple keyword-based responses
        user_lower = user_message.lower()
        
        if 'hello' in user_lower or 'hi ' in user_lower:
            return ["Hello! How can I help you today?"]
        elif 'weather' in user_lower and tools:
            # Check if there's a weather tool
            for tool in tools:
                if 'weather' in tool.get('name', '').lower():
                    return [{
                        'type': 'tool_use',
                        'id': f'toolu_{uuid.uuid4().hex[:8]}',
                        'name': tool['name'],
                        'input': {'location': 'current location'}
                    }]
        elif 'calculate' in user_lower or 'math' in user_lower:
            if tools:
                for tool in tools:
                    if 'calc' in tool.get('name', '').lower() or 'math' in tool.get('name', '').lower():
                        return [{
                            'type': 'tool_use',
                            'id': f'toolu_{uuid.uuid4().hex[:8]}',
                            'name': tool['name'],
                            'input': {'expression': '2 + 2'}
                        }]
        
        # Default response
        return [f"I understand you said: '{user_message}'. How can I help you with that?"]


class FakeAsyncMessages:
    """Mimics the async messages API interface"""
    
    def __init__(self, response_scenarios=None):
        self.response_scenarios = response_scenarios or {}
        self.call_count = 0
        
    async def create(self, model: str, max_tokens: int, messages: List[Dict], 
                     system: Optional[str] = None, temperature: float = 1.0, 
                     tools: Optional[List] = None, **kwargs) -> FakeMessage:
        """Create a non-streaming response"""
        await asyncio.sleep(0.01)  # Simulate network delay
        self.call_count += 1
        
        # Generate response based on the last user message
        user_message_parts = []
        is_tool_response = False
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                if isinstance(msg.get('content'), str):
                    user_message_parts.append(msg['content'])
                elif isinstance(msg.get('content'), list):
                    for block in msg['content']:
                        if block.get('type') == 'tool_result':
                            user_message_parts.append(block.get('content', ''))
                            is_tool_response = True
                        elif block.get('type') == 'text':
                            user_message_parts.append(block.get('text', ''))
                break
        
        if not is_tool_response:
            user_message = ' '.join(user_message_parts)
        else:
            user_message = user_message_parts
        response_content = self._generate_response(user_message, tools, is_tool_response)
        content_blocks = []
        
        for item in response_content:
            if isinstance(item, str):
                content_blocks.append(TextBlock(item))
            elif isinstance(item, dict) and item.get('type') == 'tool_use':
                content_blocks.append(ToolUseBlock(item['id'], item['name'], item['input']))
                
        return FakeMessage(content_blocks)
    
    def stream(self, model: str, max_tokens: int, messages: List[Dict],
               system: Optional[str] = None, temperature: float = 1.0,
               tools: Optional[List] = None, **kwargs) -> FakeAsyncStreamManager:
        """Create a streaming response"""
        self.call_count += 1
        
        # Generate response based on the last user message
        user_message_parts = []
        is_tool_response = False
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                if isinstance(msg.get('content'), str):
                    user_message_parts.append(msg['content'])
                elif isinstance(msg.get('content'), list):
                    for block in msg['content']:
                        if block.get('type') == 'tool_result':
                            user_message_parts.append(block.get('content', ''))
                            is_tool_response = True
                        elif block.get('type') == 'text':
                            user_message_parts.append(block.get('text', ''))
                break
        
        if not is_tool_response:
            user_message = ' '.join(user_message_parts)
        else:
            user_message = user_message_parts
        response_content = self._generate_response(user_message, tools, is_tool_response)
        return FakeAsyncStreamManager(response_content)
    
    def _generate_response(self, user_message: str, tools: Optional[List] = None, is_tool_response:bool = False) -> List:
        """Generate response content based on user message and available tools"""
        
        # Check for custom scenarios first
        if type(user_message) is str:
            if user_message in self.response_scenarios:
                return self.response_scenarios[user_message]
                
        if is_tool_response:
            return [f"Tool response was:{str(user_message)}"]
        
        # Simple keyword-based responses
        user_lower = user_message.lower()
        
        if 'hello' in user_lower or 'hi ' in user_lower:
            return ["Hello! How can I help you today?"]
        elif 'weather' in user_lower and tools:
            # Check if there's a weather tool
            for tool in tools:
                if 'weather' in tool.get('name', '').lower():
                    return [{
                        'type': 'tool_use',
                        'id': f'toolu_{uuid.uuid4().hex[:8]}',
                        'name': tool['name'],
                        'input': {'location': 'current location'}
                    }]
        elif 'calculate' in user_lower or 'math' in user_lower:
            if tools:
                for tool in tools:
                    if 'calc' in tool.get('name', '').lower() or 'math' in tool.get('name', '').lower():
                        return [{
                            'type': 'tool_use',
                            'id': f'toolu_{uuid.uuid4().hex[:8]}',
                            'name': tool['name'],
                            'input': {'expression': '2 + 2'}
                        }]
        
        # Default response
        return [f"I understand you said: '{user_message}'. How can I help you with that?"]


class FakeAnthropic:
    """Fake Anthropic client for testing"""
    
    def __init__(self, api_key: str = "fake-key", response_scenarios: Optional[Dict] = None):
        self.api_key = api_key
        self.messages = FakeMessages(response_scenarios)
    
    def __repr__(self):
        return f"<FakeAnthropic(api_key='{self.api_key}')>"


class FakeAsyncAnthropic:
    """Fake AsyncAnthropic client for testing"""
    
    def __init__(self, api_key: str = "fake-key", response_scenarios: Optional[Dict] = None):
        self.api_key = api_key
        self.messages = FakeAsyncMessages(response_scenarios)
    
    def __repr__(self):
        return f"<FakeAsyncAnthropic(api_key='{self.api_key}')>"


# Convenience functions for creating clients with predefined scenarios
def create_fake_client(response_scenarios: Optional[Dict] = None) -> FakeAnthropic:
    """Create a fake sync Anthropic client with optional response scenarios"""
    return FakeAnthropic(response_scenarios=response_scenarios)


def create_fake_async_client(response_scenarios: Optional[Dict] = None) -> FakeAsyncAnthropic:
    """Create a fake async Anthropic client with optional response scenarios"""
    return FakeAsyncAnthropic(response_scenarios=response_scenarios)


# Example usage scenarios for common testing patterns
EXAMPLE_SCENARIOS = {
    "Hello": ["Hi there! How can I assist you?"],
    "What's 2+2?": [{
        'type': 'tool_use',
        'id': 'toolu_12345',
        'name': 'calculator',
        'input': {'expression': '2+2'}
    }],
    "Tell me a joke": ["Why don't scientists trust atoms? Because they make up everything!"],
    "Error test": []  # Empty response to test error handling
}
