




class BaseCommand:

    def add_arguments(self, parser):
        pass

    def handle(self, *args, **options):
        raise NotImplemented("Commands should define a handle(*args, **options) method")
