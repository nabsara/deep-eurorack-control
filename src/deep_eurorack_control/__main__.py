import click
from deep_eurorack_control.cli import train_mnist_vae, train_rave, write_nsynth_json


@click.group()
def main():
    pass


main.command()(train_mnist_vae)
main.command()(train_rave)
main.command()(write_nsynth_json)

if __name__ == "__main__":
    main()
