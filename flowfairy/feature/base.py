class MetaBase:
    ignored_fields = ()

class FeatureBase(type):
    def __new__(cls, clsname, bases, dct):

        metabase = MetaBase()
        meta = dct.get('Meta', None)
        if meta:
            metabase.__dict__.update(meta.__dict__)

        dct['_meta'] = metabase

        return type.__new__(cls, clsname, bases, dct)


class Feature(metaclass=FeatureBase):

    def feature(*args, **options):
        raise NotImplementedError

    def shapes(self):
        pass
