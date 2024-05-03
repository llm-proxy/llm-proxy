<a name="readme-top"></a>

<h3 align="center">LLM Proxy</h3>

  <p align="center">
    A low-code solution to efficiently manage multiple large language models
    <br />
<!--     <a href="https://github.com/github_username/repo_name"><strong>Explore the docs »</strong></a> -->
<!--     <br />
    <br /> -->
    <a href="https://youtube.com">View Demo</a>
    ·
    <a href="https://github.com/llm-proxy/llm-proxy/issues">Report Bug</a>
    ·
    <a href="https://github.com/llm-proxy/llm-proxy/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <!--  <li><a href="#contact">Contact</a></li>-->
   <!--  <li><a href="#contributors">Contributors</a></li>-->
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## What is LLM Proxy?

LLM Proxy is a tool that sits between your application and the different LLM providers. LLM Proxy's goal is to simplify the use of multiple LLMs through a TUI while providing cost and response optimization.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

There are 2 ways to get started. You can directly clone the repo into your project or install it as a library.

### Prerequisites

- Python 3.11+

<!-- ### Local Installation -->
<!---->
<!-- If you want to test the LLM Proxy you can use the following steps: -->
<!---->
<!-- 1. Clone the repository into the project of your choice -->
<!---->
<!-- ```shell -->
<!-- git clone https://github.com/llm-proxy/llm-proxy.git -->
<!-- ``` -->
<!---->
<!-- 2. Ensure that you have a `llmproxy.config.yaml` file set up in the root directory of your project -->
<!-- 3. Ensure that you have all of your API keys for each respective provider setup (You can utilize the .env.example for reference) -->
<!-- 4. Ensure that you only have providers and API keys for models you want active -->
<!---->
<!-- **Note:** For Google's models, you will need the path to application credentials, and the project ID inside of the .env -->
<!---->
<!-- <p align="right">(<a href="#readme-top">back to top</a>)</p> -->

### Installation

With pip:

```shell
pip install proxyllm
```

With poetry:

```shell
poetry add proxyllm
```

Run the install script for the default configuration file:

```shell
config --default-config
```

If you prefer poetry:

```shell
poetry run config --default-config
```

> If the installation scripts do not work, you can visit the [repo](https://github.com/llm-proxy/llm-proxy/blob/main/llmproxy.config.yml) to grab a copy manually.

**Note:**

- Ensure that you have all of your API keys for each respective provider in the .env file (You can utilize the .env.example for reference)
- For Google's models, you will need the path to application credentials, and the project ID inside of the .env

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage
### Basic Usage
Currently, the LLM Proxy provides 2 different route types: Cost and Category.

To get started import the LLMProxy client:

```python
from proxyllm import LLMProxy
```

After the setup is complete, you only need 1 line of code to get started:

```python
llmproxy_client = LLMProxy()
```

**Note**: You will need to specify your yaml configuration file if you did not use the default name:

```python
llmproxy_client = LLMProxy(path_to_user_configuration="llmproxy.config.yml")
```

To use the llmproxy, simply call the route function with your prompt:

```python
output = llmproxy_client.route(prompt=prompt)
```

The route function will return a CompletionResponse:

```python
print("RESPONSE MODEL: ", output.response_model)
print("RESPONSE: ", output.response)
print("ERRORS: ", output.errors)
```

- `response_model:` contains the model used for the request
- `response:` contains the string response from the model
- `errors:` contains an array of models that failed to make a request with their respective errors

**Important Note:** Although parameters changed programmatically, it is best to favor the YAML configuration file. Only use the constructor parameters when you must override the YAML configuration.

<!-- _For more examples, please refer to the [Documentation](https://example.com)_ -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Chat History
The user can pass in their own custom chat history to the proxy and it will route to the correct LLM with this chat history.
The data type of the chat history is:
`chat_history: List[Dict[str,str]]`

The chat history supports the following roles:
- `system`: Helps define the LLM's behavior. You can customize its personality or specify detailed instructions on how it should interact during conversations. It's important to note that while providing a system message is optional, if none is given, the assistant's behavior is likely to resemble that of a generic message, such as "You are a helpful assistant." If you plan on using the system role, place it in the first index of `chat_history` rather than later on in the conversation.
- `user`: Contain requests or comments for the LLM to address. 
- `assistant`: Are the response messages by the LLMs, and can also be authored by you to showcase desired behaviors.

An example of how chat history could be formatted is as such:

```python
chat_history = [
        {
            "role": "system",
            "content": "you are math bot, and you're responses must be short and sweet",
        },
        {
            "role": "user",
            "content": "what is 1 + 1",
        },
        {
            "role": "assistant",
            "content": "2",
        }
  ]
```

By default the chat history would be an empty array and will not need to be instantiated. But in order for you to retrieve the chat history after routing, simplyset your chat history variable to `output.chat_history` and then this chat history can be passed into the proxy's route() function.

The following example shows the usage of chat history with the proxy library:

```python
prompt = "what is 1 + 1"
proxy_client = LLMProxy(route_type="cost")
output = proxy_client.route(prompt=prompt)
chat_history = output.chat_history

prompt2 = "What was the first question that I asked you?"
output = proxy_client.route(prompt=prompt2, chat_history=chat_history)
chat_history = output.chat_history
```

Notes:
- When passing in a populated custom chat history, the first message in the chat history has to be either a `system` or `assistant` message. The last message has to be an `assistant` message.  The 'user' message cannot be appended to the chat history, it has to be passed in the form of the `prompt` to the proxy's `route` function.
- You must use only the following roles otherwise routing will not work: `user`, `assistant`, `system` 
- In order to keep track of your chat history, make sure to set it equal to the `output.chat_history` each time you make a call to the proxy's route() function

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [ ] Support for more providers
  - [ ] Replicate
  - [x] Claude
- [ ] Support for multimodal Models
- [ ] Custom, optimized model for general router
- [x] Elo Routing
- [ ] Context Injection
- [ ] Filter/Security Layer

See the [open issues](https://github.com/llm-proxy/llm-proxy/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->

## Contributing

LLM Proxy is open source, so we are open and grateful, for contributions. Open-source communities are what makes software great, so feel fork the repo and create a pull request with the feature tag. Thanks!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
<!-- ## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com

Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
- [Hadi Hemmati](https://www.linkedin.com/in/hhemmati/)
- [Adrien Laurent](https://www.linkedin.com/in/adrienlaurent/)
- [Alain Ballen](https://www.linkedin.com/in/alain-ballen/)
- [Mohamed Ahmed](https://www.linkedin.com/in/mohamed-ahmed-soft-eng/)
- [Ahmed Ali](https://www.linkedin.com/in/ahmed-ali00/)
- [Victor Chung](https://www.linkedin.com/in/victor-chung-ca/)
- [Ye-Ting Ke](https://www.linkedin.com/in/yi-ting-ke-8563a5204/)
- [Laxit Shahi](https://www.linkedin.com/in/laxitshahi/)
- [Dinuja Wattage](https://www.linkedin.com/in/dinuja-wattage/)

<!-- <p align="right">(<a href="#readme-top">back to top</a>)</p>-->
