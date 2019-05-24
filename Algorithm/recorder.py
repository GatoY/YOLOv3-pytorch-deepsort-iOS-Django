class Recorder():
    def __init__(self, id, label, first_frame):
        self.id = id
        self.label = label
        self.frames = [first_frame]

    def update(self, frame):
        self.frames.append(frame)

    def count(self):
        # TODO rule whether it should be counted
        pass
        if len(self.frames) < 5:
            return False

        return self.label
