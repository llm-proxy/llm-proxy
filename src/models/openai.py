import openai
from openai import error
import os
from dataclasses import dataclass

# Figure out a way to generalize this to all files
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class OpenAICompletionResponse:
    payload: str
    message: str
    err: str


# Might need to replace error handling with classes, since we are redundantly typing the openai errors
# We need to also decide on a standard return structure
def get_open_ai_completion(
    prompt: str, model: str = "gpt-3.5-turbo"
) -> OpenAICompletionResponse:
    try:
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
        )
    except error.AuthenticationError:
        return OpenAICompletionResponse(
            payload="",
            message="Sorry no API key was provided",
            err="Authentication Error",
        )
    except error.RateLimitError:
        return OpenAICompletionResponse(
            payload="",
            message="Sorry, you have exceeded your quota. Please check your billing details",
            err="RateLimitError",
        )
    except Exception:
        return OpenAICompletionResponse(
            payload="",
            message="Sorry, something went wrong. Please report this issue and try again later",
            err="Unknown Error",
        )
    return OpenAICompletionResponse(
        payload=response.choices[0].message["content"],
        message="OK",
        err="",
    )
