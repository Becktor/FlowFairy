
import sys
import os
import pkgutil
from argparse import ArgumentParser
import importlib

def find_commands(management_dir):
    """
    Given a path to a management directory, return a list of all the command
    names that are available.
    """
    command_dir = os.path.join(management_dir, 'commands')
    return [name for _, name, is_pkg in pkgutil.iter_modules([command_dir])
            if not is_pkg and not name.startswith('_')]

def ensure_settings():
    if 'GLUE_SETTINGS_MODULE' in os.environ:
        return

    curdir = os.getcwd()
    stopat = [ os.path.expanduser('~') , '/' ]

    while 'settings.py' not in os.listdir(curdir) and curdir not in stopat:
        curdir = os.path.normpath(os.path.join(curdir, '..'))

    if curdir != stopat:
        sys.path.append(curdir)
        os.environ.setdefault('GLUE_SETTINGS_MODULE', 'settings')

    else:
        print('Warning, can not find any settings', file=sys.stderr)
        sys.exit()


def get_commands():
    command_names = find_commands(__path__[0])

def load_command(command):
    return importlib.import_module(f'flowfairy.core.management.commands.{command}').Command()

def execute():
    ensure_settings()

    parser = ArgumentParser()
    from flowfairy.conf import settings

    settings.add_arguments(parser)

    command = load_command(sys.argv[1])
    command.add_arguments(parser)

    args = parser.parse_args(sys.argv[2:])
    options = vars(args)

    settings.apply_arguments(**options)
    command.handle(**options)

