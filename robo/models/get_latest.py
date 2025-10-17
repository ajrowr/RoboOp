
import subprocess
import json
from pathlib import Path

from robo import _get_api_key

def get_latest():
    return json.loads(subprocess.check_output(f"""curl https://api.anthropic.com/v1/models -s --header "x-api-key: {_get_api_key()}" --header "anthropic-version: 2023-06-01" """, shell=True).decode())

def download_latest():
    foutpath = Path(__file__).parent / '_models.json'
    foutpath.write_text(json.dumps(get_latest(), indent=4))

if __name__ == '__main__':
    download_latest()