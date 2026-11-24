"""Una sola CNN propia: no importa arquitecturas externas ni carga pesos."""
import torch
from torch import nn

ARCHITECTURE = "VisionCNN-v1"


class VisionCNN(nn.Module):
    def __init__(self, num_classes=6, dropout=0.3):
        super().__init__()
        if num_classes != 6:
            raise ValueError("Vision-IA utiliza seis categorias.")
        layers = []
        channels = (3, 16, 32, 64)
        for incoming, outgoing in zip(channels, channels[1:]):
            layers.extend((nn.Conv2d(incoming, outgoing, kernel_size=3, padding=1, bias=False),
                           nn.BatchNorm2d(outgoing), nn.ReLU(inplace=True), nn.MaxPool2d(2)))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64 * 4 * 4, 64), nn.ReLU(inplace=True),
                                        nn.Dropout(dropout), nn.Linear(64, num_classes))
        self.apply(self._initialize)

    @staticmethod
    def _initialize(layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 128, 128):
            raise ValueError("Se espera un tensor [batch, 3, 128, 128].")
        # Logits: CrossEntropyLoss aplica internamente log-softmax.
        return self.classifier(self.pool(self.features(images)))


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())
