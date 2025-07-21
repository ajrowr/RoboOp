
from .. import Bot, Conversation
from anthropic import _exceptions as anthexc

class APIChecker(Bot):
    sysprompt_text = """You are a test assistant. Respond with "OK" and nothing more."""


def check_api():
    try:
        conv = Conversation(APIChecker, [])
        m = conv.resume('hello')
    except anthexc.OverloadedError as exc:
        print(f"529: API is overloaded")
    except:
        raise
    else:
        print(m.content[0].text)

if __name__ == '__main__':
    check_api()