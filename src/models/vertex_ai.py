import os 

from google.cloud import aiplatform
import vertexai
from vertexai.language_models import TextGenerationModel

from dotenv import load_dotenv
load_dotenv()

project_id = os.getenv('GOOGLE_PROJECT_ID')

from google.cloud import aiplatform

aiplatform.init(
    project=project_id,
    location='us-central1'
)

def interview(
    temperature: float,
    project_id: str,
    location: str,
) -> str:
    """Ideation example with a Large Language Model"""

    vertexai.init(project=project_id, location=location)
    # TODO developer - override these parameters as needed:
    parameters = {
        "temperature": temperature,  # Temperature controls the degree of randomness in token selection.
        "max_output_tokens": 256,  # Token limit determines the maximum amount of text output.
        "top_p": 0.8,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
        "top_k": 40,  # A top_k of 1 means the selected token is the most probable among all tokens.
    }

    model = TextGenerationModel.from_pretrained("text-bison@001")
    response = model.predict(
        "define an llm"
    )
    print(f"Response from Model: {response.text}")

interview(0,project_id,'us-central1')