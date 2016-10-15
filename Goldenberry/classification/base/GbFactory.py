class GbFactory:
    def __init__(self, type, kargs = None ):
        self.type = type
        self.kargs = kargs

    def __call__(self):
        return self.type(**self.kargs)