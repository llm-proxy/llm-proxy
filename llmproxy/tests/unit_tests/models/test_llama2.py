import os
from llmproxy.models.llama2 import Llama2, Llama2Model
from dotenv import load_dotenv

load_dotenv(".env.test")
default_api_key = os.getenv("LLAMA2_API_KEY")

default_prompt = "What's 1+1?"
default_system_prompt = "Answer correctly"
default_model = Llama2Model.LLAMA_2_7B.value


def test_llama2_constructor() -> None:
    llama2_Con = Llama2()
    output = llama2_Con.get_completion()

    assert output.message == "No prompt detected"


def test_llama2_emp_prompt() -> None:
    llama2_emp_promt = Llama2(
        prompt="", system_prompt=default_system_prompt, model=default_model
    )
    output = llama2_emp_promt.get_completion()
    assert output.message == "No prompt detected"


def test_llama2_api_key() -> None:
    test_api_key = Llama2(
        prompt=default_prompt,
        system_prompt=default_system_prompt,
        api_key="LMAO-key",
        model=default_model,
    )
    output = test_api_key.get_completion()
    assert (
        output.message == "Authorization header is correct, but the token seems invalid"
    )


def test_llama2_model() -> None:
    test_model_1 = Llama2(
        prompt=default_prompt,
        system_prompt=default_system_prompt,
        api_key=default_api_key,
        model=Llama2Model.LLAMA_2_7B.value,
    )
    test_model_2 = Llama2(
        prompt=default_prompt,
        system_prompt=default_system_prompt,
        api_key=default_api_key,
        model=Llama2Model.LLAMA_2_13B.value,
    )
    test_model_3 = Llama2(
        prompt=default_prompt,
        system_prompt=default_system_prompt,
        api_key=default_api_key,
        model=Llama2Model.LLAMA_2_70B.value,
    )

    output1 = test_model_1.get_completion()
    output2 = test_model_2.get_completion()
    output3 = test_model_3.get_completion()
    response = "Model requires a Pro subscription; check out hf.co/pricing to learn more. Make sure to include your HF token in your query."

    assert output1.message == response
    assert output2.message == response
    assert output3.message == response


def test_llama2_emp_model() -> None:
    test_emp_model = Llama2(
        prompt=default_prompt,
        system_prompt=default_system_prompt,
        api_key=default_api_key,
        model="",
    )
    output = test_emp_model.get_completion()

    assert (
        output.message
        == "Invalide Model. Please use one of the following model: Llama-2-7b-chat-hf, Llama-2-13b-chat-hf, Llama-2-70b-chat-hf"
    )
