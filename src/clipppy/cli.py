from __future__ import annotations

from typing import get_args, Iterable, Literal

import click
import torch
from more_itertools import first

from . import autocli
from .autocli import AutoCLI
from .clipppy import Clipppy
from .yaml import ClipppyYAML


class LazyMultiCommand(click.MultiCommand):
    def __init__(self, *args, autocli=AutoCLI(), **kwargs):
        super().__init__(*args, **kwargs)
        self.autocli = autocli

    @staticmethod
    def ctx_to_obj(ctx: click.Context) -> Clipppy:
        ctx.ensure_object(Clipppy)
        return ctx.obj

    def list_commands(self, ctx: click.Context) -> Iterable[str]:
        return self.ctx_to_obj(ctx).commands.keys()

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command:
        obj = self.ctx_to_obj(ctx)

        try:
            cmd = getattr(obj, cmd_name)
        except AttributeError:
            raise click.UsageError(f'command {cmd_name} not available.', ctx)

        return click.Command(cmd_name, callback=cmd, help=getattr(cmd, '__doc__', None),
                             params=list(self.autocli.annotated_class(cmd).values()))


# TODO: migrate to 3.8
device_literal = Literal.__getitem__(
    (None, 'cpu') + (
        torch.cuda.is_available()
        and (('cuda',) + tuple(f'cuda:{i}' for i in range(torch.cuda.device_count())))
        or ()
    )
)


@click.version_option('0.1')  # TODO: version
@click.pass_context
def cli(ctx: click.Context, config: autocli._ExistingFile, device: device_literal = None):
    # print(locals())
    assert device in get_args(device_literal)
    if device in ('cpu', None):
        torch.set_default_tensor_type(torch.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        torch.cuda.set_device(torch.device(device))

    ctx.obj = ClipppyYAML().load(config)
    assert isinstance(ctx.obj, Clipppy)
    print(ctx.obj)


acli = AutoCLI(show_default=True)
cli = click.version_option('0.1')(  # TODO: version
    LazyMultiCommand(
        autocli=acli, name=cli.__name__, callback=cli,
        params=list(acli.function(cli).values()),
        invoke_without_command=True, no_args_is_help=False, chain=True,
    ))
setattr(first(p for p in cli.params if p.name == 'config'), 'metavar', 'config.yaml')

if __name__ == '__main__':
    # cli.main(['--version'], standalone_mode=False)
    cli.main(['/users/kosio/downloads/config.yaml', 'mock', '--savename', 'obs.pt'], standalone_mode=False)
    # cli.main(['examples/spectrum.yaml', 'ppd', '--help'], standalone_mode=False)
