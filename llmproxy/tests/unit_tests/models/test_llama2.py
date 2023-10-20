import os
from llmproxy.models.llama2 import Llama2, Llama2Model
from dotenv import load_dotenv

load_dotenv()
default_api_key = "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"


default_prompt ="What's 1+1?"
default_system_prompt = "Answer correctly"
default_model = Llama2Model.LLAMA_2_7B.value
def test_llama2_constructor() -> None:
    llama2_Con = Llama2().get_completion
    assert llama2_Con.message == "No prompt detected"

def test_llama2_prompt() -> None:
    llama2_emp_promt = ""
    output = Llama2(prompt=llama2_emp_promt,system_prompt=default_system_prompt,model=default_model)
    output.get_completion()
    assert output.message == "No prompt detected"

def test_llama2_api_key() -> None:
    api_key="LMAO-key"
    output = Llama2(prompt=default_prompt,system_prompt=default_system_prompt,api_key=api_key,model=default_model)
    output.get_completion()
    assert output.message == "Authorization header is correct, but the token seems invalid"
    
    

def test_llama2_model() -> None:
    test_model1 = Llama2Model.LLAMA_2_7B.value
    test_model2 = Llama2Model.LLAMA_2_13B.value
    test_model3 = Llama2Model.LLAMA_2_70B.value
    test_model4 = ""

    output1 = Llama2(prompt=default_prompt,system_prompt=default_system_prompt,api_key=default_api_key,model=test_model1)
    output2 = Llama2(prompt=default_prompt,system_prompt=default_system_prompt,api_key=default_api_key,model=test_model2)
    output3 = Llama2(prompt=default_prompt,system_prompt=default_system_prompt,api_key=default_api_key,model=test_model3)
    output4 = Llama2(prompt=default_prompt,system_prompt=default_system_prompt,api_key=default_api_key,model=test_model4)
    output1.get_completion()
    output2.get_completion()
    output3.get_completion()
    output4.get_completion()
    response = "Model requires a Pro subscription; check out hf.co/pricing to learn more. Make sure to include your HF token in your query."

    assert output1.message == response
    assert output2.message == response
    assert output3.message == response
    assert output4.message == "Invalide Model. Please use one of the following model: Llama-2-7b-chat-hf, Llama-2-13b-chat-hf, Llama-2-70b-chat-hf"
    