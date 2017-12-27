from timeit import default_timer as timer

class benchmark(object):
    """
    Benchmark some code.
    Usage:
    with benchmark("Measured time") as b:
        # do something

    >> 2.3 seconds
    """
    def __init__(self, msg, fmt="%0.3g"):
        self.msg = msg
        self.fmt = fmt

    def __enter__(self):
        self.start = timer()
        return self

    def __exit__(self, *args):
        t = timer() - self.start
        print(("%s : " + self.fmt + " seconds") % (self.msg, t))
        self.time = t