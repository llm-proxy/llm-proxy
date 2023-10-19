import os
from llmproxy.models.llama2 import Llama2, Llama2Model
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("LLAMA2_API_KEY")

def test_llama2_constructor() -> None:
    llama2_Con = Llama2()
    assert llama2_Con.api_key == ""

def test_llama2_prompt() -> None:
    llama2_emp_promt = ""
    output = Llama2(prompt=llama2_emp_promt)
    assert output.prompt == ""

def test_llama2_api_key() -> None:
    llama = Llama2(api_key="LMAO-key")
    

def test_llama2_model() -> None:
    test_model1 = Llama2Model.LLAMA_2_7B
    test_model2 = Llama2Model.LLAMA_2_13B
    test_model3 = Llama2Model.LLAMA_2_70B
    test_model4 = ""

    test_prompt = "What is 1+1?"
    output1 = Llama2(prompt=test_prompt,model=test_model1)
    output2 = Llama2(prompt=test_prompt,model=test_model2)
    output3 = Llama2(prompt=test_prompt,model=test_model3)
    output4 = Llama2(prompt=test_prompt,model=test_model4)
    
    