from platformdirs import user_cache_dir
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import json
import os
from pathlib import Path
from functools import lru_cache
from datetime import datetime

from robo.utils import _get_api_key

CACHE_MAX_AGE = int(os.environ.get('ROBO_MODELCACHE_MAX_AGE', 60*60*24*7)) ## Default 7 days

def get_latest():
    """Fetch latest models from Anthropic API."""
    url = "https://api.anthropic.com/v1/models"
    headers = {
        "x-api-key": _get_api_key(),
        "anthropic-version": "2023-06-01"
    }
    
    request = Request(url, headers=headers)
    
    try:
        with urlopen(request) as response:
            return json.loads(response.read().decode())
    except HTTPError as e:
        raise RuntimeError(f"API request failed: {e.code} {e.reason}")
    except URLError as e:
        raise RuntimeError(f"Network error: {e.reason}")

def get_cache_path():
    cache_dir = Path(user_cache_dir('RoboOp'))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / 'models-anthropic.json'

def get_cached_models():
    p = get_cache_path()
    should_refresh = False
    if p.exists():
        if (datetime.timestamp(datetime.now()) - p.stat().st_mtime) > CACHE_MAX_AGE:
            should_refresh = True
    else:
        should_refresh = True
    if should_refresh:
        with open(p, 'w') as fout:
            json.dump(get_latest(), fout)
    return json.loads(p.read_text())

@lru_cache(maxsize=1)
def model_data(refresh=False):
    if CACHE_MAX_AGE > 0:
        return get_cached_models()
    else:
        return get_latest()

