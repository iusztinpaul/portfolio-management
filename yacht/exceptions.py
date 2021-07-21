class EndOfDataStreamError(RuntimeError):
    def __init__(self):
        super().__init__('Your stream of data has reached its end')
