import requests
from models.base import BaseChatbot, CompletionResponse

class Llama2(BaseChatbot):

    def __init__(
        self,
        prompt: str = "",
        system_prompt: str = None,
        system_prompt: str = None,
        api_key: str = "",
        ) -> None:
        self.system_prompt = system_prompt or "Answer politely"
        self.prompt = prompt
        self.system_prompt = system_prompt or "Answer politely"
        self.api_key = api_key
        API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
        headers = {"Authorization": f"Bearer {self.api_key}"}
    def get_completion(self) -> CompletionResponse:
        try:
            def query(payload):
                 response = requests.post(API_URL, headers=headers, json=payload)
                 return response.json()
            # Llama2 prompt template
            prompt_template = f"<s>[INST] <<SYS>>\n{{{{ {self.system_prompt} }}}}\n<</SYS>>\n{{{{ {self.prompt} }}}}\n[/INST]"

            payload = {
                "inputs": prompt_template
            }
            output = query(payload)
            
            # Llama2 prompt template
            prompt_template = f"<s>[INST] <<SYS>>\n{{{{ {self.system_prompt} }}}}\n<</SYS>>\n{{{{ {self.prompt} }}}}\n[/INST]"

            payload = {
                "inputs": prompt_template
            }
            output = query(payload)
            

        except Exception as e:
            raise Exception("Error Occur")
        
        return CompletionResponse(
            payload=output
        )
