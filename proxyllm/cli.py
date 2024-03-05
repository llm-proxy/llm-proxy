import click
import requests

LINK_TO_CONFIG = (
    "https://raw.githubusercontent.com/llm-proxy/llm-proxy/main/llmproxy.config.yml"
)

FILE_NAME = "llmproxy.config.yml"


@click.command()
@click.option(
    "--default-config",
    is_flag=True,
    help="Install llmproxy.config.yml in the root directory",
)
def cli(default_config):
    if default_config:
        res = requests.get(LINK_TO_CONFIG, allow_redirects=True)

        if res.status_code == 200:
            with open(FILE_NAME, "wb") as file:
                file.write(res.content)
            click.echo(f"{FILE_NAME} has been installed successfully!")
        else:
            click.echo(
                f"{FILE_NAME} not installed! Please try copying the file manually."
            )


if __name__ == "__main__":
    cli()
