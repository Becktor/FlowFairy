import itertools as it
import importlib


class Stage:
    def before(self, sess, net):
        pass

    def run(self, sess, i):
        raise NotImplemented("Stages need to implement run(session, step)")


class _StageManager:

    stages = dict()

    def before(self, sess, net):
        for stage in it.chain(*[stage for key, stage in self.stages.items()]):
            stage.before(sess, net)

    def run(self, sess, i):
        for stage in it.chain(*[self.stages[step] for step in self.stages if i % step == 0]):
            stage.run(sess, i)


    def register(self, stage, interval):
        if not interval in self.stages:
            self.stages[interval] = []

        self.stages[interval].append(stage)


stage = _StageManager()


def register(interval=1):
    def wrapped(cls):
        stage.register(cls(), interval)
    return wrapped


try:
    import stages
except ModuleNotFoundError:
    print(e)
    import sys
    print(sys.path)
    print("No stages")
