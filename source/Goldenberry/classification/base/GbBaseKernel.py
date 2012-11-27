import abc

class BaseKernel:
    """Base class for a kernel function."""
    __metaclass__ = abc.ABCMeta

    def __call__(self, x, y):
        return self.compute(x, y)

    @abc.abstractmethod
    def compute(self, x, y):
        """Computes the kernel function from the x and y vectors."""
        raise NotImplementedError()


