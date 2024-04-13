from probes import Probe, LRProbe, MLP, MMProbe

class Monitor:
    def train(self, data):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError
    
    def get_loss(self, data):
        raise NotImplementedError

class TextMonitor(Monitor):
    def __init__(self):
        self.model = None

    def train(self, data):
        self.model = train_model(data)

    def predict(self, data):
        return self.model.predict(data)

    def get_loss(self, data):
        return self.model.get_loss(data)
    
class ActMonitor(Monitor):
    def __init__(self, probe, layer):
        self.probe = None
        self.layer = layer

    def train(self, acts, labels, lr=0.001, weight_decay=0.1, epochs=1000, device='cpu'):
        self.probe = train_probe(acts, labels, lr, weight_decay, epochs, device)

    def predict(self, acts):
        return self.probe.predict(acts)