class BaseGenarator(object):
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def generate(self):
        raise NotImplementedError
