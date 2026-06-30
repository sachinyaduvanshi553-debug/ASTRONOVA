
class ThresholdDetector:
    def __init__(self, threshold: float = 1e-5):
        self.threshold = threshold

    def detect(self, current_flux: float) -> bool:
        return current_flux >= self.threshold
