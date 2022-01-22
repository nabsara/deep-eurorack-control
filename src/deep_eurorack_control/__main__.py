import click
from deep_eurorack_control.cli import train_mnist_vae


@click.group()
def main():
    pass


main.command()(train_mnist_vae)

if __name__ == "__main__":
    main()
