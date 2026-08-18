class UploadSessionStore:
    def __init__(self):
        self.predictor = None
        self.raw_data = []
        self.mapped_data = []

    def set_predictor(self, predictor):
        self.predictor = predictor

    def set_uploaded_voters(self, raw_data, mapped_data):
        self.raw_data = raw_data or []
        self.mapped_data = mapped_data or []


store = UploadSessionStore()
