
import inspect

_types_map = {
    str: 'string',
    int: 'number',
    float: 'number',
    bool: 'boolean',
    list: 'array',
    dict: 'object',
    type(None): 'null',
}

class Tool(object):
    __slots__ = ['name', 'description', 'parameter_descriptions', 'target']
    def __init__(self, *args, **kwargs):
        """If a tool_context exists, it will be passed in here as kwargs. If anything needs
        to be done with it (for example stashing object references on the Tool instance), 
        override __init__."""
        pass
    
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Please implement __call__ in a subclass")
    
    def call_sync(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)
    
    async def call_async(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)
    
    @classmethod
    def get_call_schema(klass):
        input_schema_properties = {}
        required = []
        for callname in ['__call__', 'call_sync', 'call_async']:
            sig = inspect.signature(getattr(klass, callname))
            for key, param in sig.parameters.items():
                if param.name in ['args', 'kwargs']:
                    break ## go to next signature since this appears to be a passthru
                attribs = {}
                if key == 'self':
                    continue
                attribs = {
                    'type': _types_map[param.annotation],
                    'description': klass.parameter_descriptions[key]
                }
                if param.default is inspect._empty:
                    required.append(key)
                input_schema_properties[key] = attribs
            if input_schema_properties:
                break
        
        return {
            'name': klass.name if hasattr(klass, 'name') and type(klass.name) is str else klass.__name__,
            'description': klass.description,
            'input_schema': {
                'type': 'object',
                'properties': input_schema_properties,
                'required': required
            }
        }


__all__ = ['Tool']