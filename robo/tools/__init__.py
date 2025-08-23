
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
    def __call__(self):
        raise NotImplementedError("Please implement __call__ in a subclass")
    
    @classmethod
    def get_call_schema(klass):
        input_schema_properties = {}
        required = []
        sig = inspect.signature(klass.__call__)
        for key, param in sig.parameters.items():
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