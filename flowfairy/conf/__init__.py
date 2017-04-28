
import importlib
import os
from . import global_settings as gl

class _Settings:

    def __init__(self):
        settings = os.environ.get("GLUE_SETTINGS_MODULE")
        self.settings = importlib.import_module(settings)
        self._setup()

    def _setup(self):

        for d in dir(self.settings):
            if d.isupper():
                attr = getattr(self.settings, d)
                self.__dict__[d] = attr

        for d in dir(gl):
            if not d in self.__dict__:
                attr = getattr(gl, d)
                self.__dict__[d] = attr


        if self.CUDA_VISIBLE_DEVICES != None:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(self.CUDA_VISIBLE_DEVICES))

    def add_arguments(self, parser):
        allowed_types = [str, bool, int, float]
        for arg in filter(lambda x: x.isupper(), dir(self)):

            attr = getattr(self, arg)
            arg_type = type(attr)

            if arg_type in allowed_types:
                arg = arg.lower()
                add_args = {'type': arg_type}
                if arg_type == bool:
                    add_args['action'] = 'store_true'

                parser.add_argument(f'--{arg}', **add_args)

    def apply_arguments(self, **kwargs):
        print(kwargs)
        for k, v in kwargs.items():
            if v == None:
                continue

            k = k.upper()

            if k in dir(self):
                setattr(self, k, v)


settings = _Settings()

