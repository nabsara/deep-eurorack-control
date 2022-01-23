# -*- coding: utf-8 -*-

"""

"""

import click
from deep_eurorack_control.cli import train_ddsp


@click.group()
def main():
    pass

main.command()(train_ddsp)

if __name__ == "__main__":
    main()
    
    

