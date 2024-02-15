<a name="readme-top"></a>

<h3 align="center">LLM Proxy</h3>

  <p align="center">
    A low code solution to managing multiple Large Language Models
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
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## What is LLM Proxy?
LLM Proxy is a tool that sits between your application and the different LLM providers. LLM Proxy's main goal is to simplify the use of multiple LLMs, while providing features such as cost and response optimization. 
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started
There are 2 ways to get started. You can directly clone the repo into your project, or you can install it as a library.

### Prerequisites
- Python 3.11+

### Local Installation
If you want to test the LLM Proxy locally, or set it up for contribution you can use the following steps:

1. Clone the repository into the project of your choice
```shell
git clone https://github.com/llm-proxy/llm-proxy.git
```
2. Ensure that you have a `llmproxy.config.yaml` file setup
3. Ensure that you have all of your API keys for each respective provider setup (You can utilize the .env.example for reference)

**Note:** For Google's models, you will need the path to application credentials, and the project ID inside of the .env
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### PyPi installation
-- IN PROGRESS
## Usage
Currently, the LLM Proxy provides 2 different route types: Cost and Category.
After the setup is complete, you only need 1 line of code to get started:
```python
proxy_client = LLMProxy()
```
**Note**: You will need to specific your yaml configuration file if you did not use the default name:
```python
client = LLMProxy(path_to_user_configuration="llmproxy.config.yml")
```

To route you can simply call the route function with your prompt:
```python
output = proxy_client.route(prompt=prompt)
```

The route function will return a CompletionResponse
``` python
    print("RESPONSE MODEL: ", output.response_model)
    print("RESPONSE: ", output.response)
    print("ERRORS: ", output.errors)
```
- `response_model:` contains the model that was used for the request
- `response:` contains the string response from the model
- `errors:` contains an array of models that failed to make a request with their respective errors 

**Important Note:** Although, certain parameters can be programmatically changed, it is best to favour the YAML configuration file, as it is prioritized in the configuration settings

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [ ] Support for more providers
  - [ ] Replicate
  - [ ] Claude
- [ ] Support for multimodal Models
- [ ] Custom, optimized model for category routing

See the [open issues](https://github.com/llm-proxy/llm-proxy/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

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
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com

Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p>