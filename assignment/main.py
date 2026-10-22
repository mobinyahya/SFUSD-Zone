import click

from student_assignment.market_generator.school_choice_market_generator import (
    MarketGenerator,
)


@click.command()
def generate():
    m = MarketGenerator()
    m.simulate()


if __name__ == "__main__":
    generate()
