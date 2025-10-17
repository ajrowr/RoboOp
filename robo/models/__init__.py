import re

from . import cache
from .models_legacy import MODELS


def _modelcode(inp):
    return re.search(r'-(\d(-\d)?)-', inp).group(1).replace('-', '.')

def _make_dataset(filterphrase=''):
    a = {m['id']:m['id'] for m in cache.model_data()['data'] if filterphrase in m['id']}
    b = {_modelcode(m): m for m in a}
    c = {k+'.0':b[k] for k in b if '.' not in k}
    return dict(a | b | c)


class ModelFamily(dict):
    def __init__(self, *args, filterphrase='', **kwargs):
        self._dataset = None
        self._is_loaded = False
        self._filterphrase = filterphrase
        super().__init__(*args, **kwargs)
    
    def _ensure_loaded(self):
        if not self._is_loaded:
            self._load_dataset()
            self._is_loaded = True
    
    def _load_dataset(self):
        dataset = _make_dataset(self._filterphrase)
        self._dataset = dataset
        self._latest = list(dataset.values())[0]
        self.update(dataset)
        
    def __getattribute__(self, attrname):
        if attrname == '_latest':
            self._ensure_loaded()
        return super().__getattribute__(attrname)
    
    def __getitem__(self, key):
        self._ensure_loaded()
        return super().__getitem__(key)
    
    def __repr__(self):
        self._ensure_loaded()
        return super().__repr__()
        
    def __getitem__(self, key):
        self._ensure_loaded()
        return super().__getitem__(key)
    
    def __contains__(self, key):
        self._ensure_loaded()
        return super().__contains__(key)
    
    def __iter__(self):
        self._ensure_loaded()
        return super().__iter__()
    
    def __len__(self):
        self._ensure_loaded()
        return super().__len__()
    
    def get(self, key, default=None):
        self._ensure_loaded()
        return super().get(key, default)
    
    def keys(self):
        self._ensure_loaded()
        return super().keys()
    
    def values(self):
        self._ensure_loaded()
        return super().values()
    
    def items(self):
        self._ensure_loaded()
        return super().items()
    
    @property
    def LATEST(self):
        return self._latest


class CLAUDE:
    HAIKU = ModelFamily(filterphrase='haiku')
    SONNET = ModelFamily(filterphrase='sonnet')
    OPUS = ModelFamily(filterphrase='opus')


__all__ = ['CLAUDE', 'MODELS']