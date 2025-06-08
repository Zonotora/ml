class Tensor:
    def __init__(self, device: str = None, requires_grad: bool | None = None) -> None:
        self.shape = (1,)

        # Computed gradient if backward() is invoked
        self.grad: Tensor | None = None

        # Record all operations in order to compute gradients during backpropagation
        self.requires_grad: bool | None = requires_grad

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.shape})"

    def __add__(self, other) -> "Tensor":
        return self

    def realize(self) -> None:
        print(self)

    @staticmethod
    def empty(*shape) -> "Tensor":
        return Tensor()
