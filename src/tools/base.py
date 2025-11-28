class Tool:
    """
    Base class for any tool the agent can call.
    """
    def __init__(self, name):
        self.name = name

    def run(self, input_text):
        raise NotImplementedError("Tool must implement a run() method")




