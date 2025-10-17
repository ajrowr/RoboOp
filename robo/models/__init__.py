
from .models_legacy import MODELS

from types import SimpleNamespace
import json
import re
from pathlib import Path

""" want to be able to do CLAUDE.SONNET['4.5'] but also CLAUDE.SONNET.LATEST"""

zzz = (Path(__file__).parent / '_models.json')
_get_models_json = lambda: json.loads((Path(__file__).parent / '_models.json').read_text())

class ModelFamily(dict):
    @staticmethod
    def modelcode(inp):
        return re.search(r'-(\d(-\d)?)-', inp).group(1).replace('-', '.')
        
    @classmethod
    def from_models_list(klass, filterphrase=''):
        a = {m['id']:m['id'] for m in _get_models_json()['data'] if filterphrase in m['id']}
        b = {klass.modelcode(m): m for m in a}
        obj = klass((a | b).items())
        obj._latest = list(obj.values())[0]
        return obj
    
    @property
    def LATEST(self):
        return self._latest
        

class CLAUDE:
    HAIKU = ModelFamily.from_models_list('haiku')
    SONNET = ModelFamily.from_models_list('sonnet')
    OPUS = ModelFamily.from_models_list('opus')


__all__ = ['CLAUDE', 'MODELS']