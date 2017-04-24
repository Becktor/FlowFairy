
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


settings = _Settings()

