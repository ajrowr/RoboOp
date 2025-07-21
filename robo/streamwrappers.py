
class StreamWrapper:
    def __init__(self, stream, conversation_obj):
        self.stream = stream
        self.conversation_obj = conversation_obj
        self.accumulated_text = ""
        self.chunks = []
        self.events = []
    
    def __enter__(self):
        self.stream_context = self.stream.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        result = self.stream.__exit__(exc_type, exc_val, exc_tb)
        if exc_type is None and self.accumulated_text:
            asst_message = self.conversation_obj._make_text_message('assistant', self.accumulated_text)
            self.conversation_obj.messages.append(asst_message)
            self.conversation_obj._post_stream_hook()
        
        return result
    
    @property
    def text_stream(self):
        for text in self.stream_context.text_stream:
            self.chunks.append(text)
            self.accumulated_text += text
            yield text
    
    @property
    def event_stream(self):
        for event in self.stream_context:
            self.events.append(event)
            yield event

class StreamWrapperWithToolUse(StreamWrapper):
    @property
    def text_stream(self):
        
        def exhaust_events(conv):
            current_block_type = None
            accumulated_text = ''
            accumulated_context = []
            ## Stream text responses, capture tool use requests
            for event in self.event_stream:
                if event.type == 'content_block_start':
                    current_block_type = type(event.content_block).__name__
                elif event.type == 'text':
                    yield event.text
                    accumulated_text += event.text
                elif event.type == 'content_block_stop' and current_block_type == 'ToolUseBlock':
                    treq = {
                        'type': 'tool_use',
                        'id': event.content_block.id,
                        'name': event.content_block.name,
                        'input': event.content_block.input,
                    }
                    conv._add_tool_request(treq)
                    accumulated_context.append(treq)
                elif event.type == 'content_block_stop' and current_block_type == 'TextBlock':
                    ttxt = {
                        'type': 'text',
                        'text': event.content_block.text,
                    }
                    accumulated_context.append(ttxt)
            conv.messages.append({'role': 'assistant', 'content': accumulated_context})
    
            if not conv._is_exhausted():
                conv._handle_pending_tool_requests()
                resps = conv._compile_tool_responses()
                with conv._resume_stream(resps, is_tool_message=True) as substream:
                    yield from substream.text_stream
                
        yield from exhaust_events(self.conversation_obj)

class AsyncStreamWrapper:
    def __init__(self, stream, conversation_obj):
        self.stream = stream
        self.conversation_obj = conversation_obj
        self.accumulated_text = ""
        self.accumulated_text_bypass = False
        self.chunks = []
        self.events = []
    
    async def __aenter__(self):
        self.stream_context = await self.stream.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        result = await self.stream.__aexit__(exc_type, exc_val, exc_tb)
        if exc_type is None and (self.accumulated_text or self.accumulated_text_bypass):
            if not self.accumulated_text_bypass:
                asst_message = self.conversation_obj._make_text_message('assistant', self.accumulated_text)
                self.conversation_obj.messages.append(asst_message)
            # finalmessage = await self.stream_context.get_final_message() ##
            # print(finalmessage.usage.model_dump()) ##
            await self.conversation_obj._post_stream_hook_async()
        
        return result
    
    @property
    async def text_stream(self):
        async for text in self.stream_context.text_stream:
            self.chunks.append(text)
            self.accumulated_text += text
            yield text
    
    @property
    async def event_stream(self):
        async for event in self.stream_context:
            self.events.append(event)
            yield event


class AsyncStreamWrapperWithToolUse(AsyncStreamWrapper):
    @property
    async def text_stream(self):
        
        async def exhaust_events(conv):
            current_block_type = None
            accumulated_text = ''
            accumulated_context = []
            ## Stream text responses, capture tool use requests
            async for event in self.event_stream:
                if event.type == 'content_block_start':
                    current_block_type = type(event.content_block).__name__
                elif event.type == 'text':
                    yield event.text
                    accumulated_text += event.text
                elif event.type == 'content_block_stop' and current_block_type == 'ToolUseBlock':
                    treq = {
                        'type': 'tool_use',
                        'id': event.content_block.id,
                        'name': event.content_block.name,
                        'input': event.content_block.input,
                    }
                    conv._add_tool_request(treq)
                    accumulated_context.append(treq)
                elif event.type == 'content_block_stop' and current_block_type == 'TextBlock':
                    ttxt = {
                        'type': 'text',
                        'text': event.content_block.text,
                    }
                    accumulated_context.append(ttxt)
            conv.messages.append({'role': 'assistant', 'content': accumulated_context})
            self.accumulated_text_bypass = True
    
            if not conv._is_exhausted():
                conv._handle_pending_tool_requests()
                msg_out = conv._handle_waiting_tool_requests()
                if msg_out is not None:
                    resp = conv._handle_canned_response(None, (msg_out, False))
                    for chunk in resp.text_stream:
                        yield chunk
                else:
                    resps = conv._compile_tool_responses()
                    async with await conv._aresume_stream(resps, is_tool_message=True) as substream:
                        async for chunk in substream.text_stream:
                            yield chunk
                
        async for chunk in exhaust_events(self.conversation_obj):
            yield chunk



__all__ = ['StreamWrapper', 'AsyncStreamWrapper', 'StreamWrapperWithToolUse', \
         'AsyncStreamWrapperWithToolUse']