from flowfairy.core.management.base import BaseCommand
from flowfairy import app

"""
Run command
"""
class Command(BaseCommand):

    def __init__(self):
        pass

    def handle(self, **kwargs):
        print('Running network')
        app.run(**kwargs)

    def add_arguments(self, parser):
        pass
