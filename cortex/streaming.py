from langchain_core.callbacks import BaseCallbackHandler

class SreamHandler(BaseCallbackHandler):
    def __init__(self, on_token):
        self.on_token = on_token

    def on_llm_new_token(self, token, **kwargs):
        self.on_token(token)