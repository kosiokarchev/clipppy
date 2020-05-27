import click


@click.command()
@click.argument('config', type=click.File())
def cli(config, *args, **kwargs):
    print(config)
    print(args)
    print(kwargs)
