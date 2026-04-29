"""Sample Python file to test code chunking."""


def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide a by b.

    Raises:
        ValueError: If b is zero.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


class Calculator:
    """A simple calculator class."""

    def __init__(self) -> None:
        """Initialize the calculator with a history."""
        self.history: list[str] = []

    def compute(self, operation: str, a: float, b: float) -> float:
        """Perform a computation and store it in history.

        Args:
            operation: One of 'add', 'subtract', 'multiply', 'divide'.
            a: First operand.
            b: Second operand.

        Returns:
            The result of the operation.
        """
        if operation == "add":
            result = add(a, b)
        elif operation == "subtract":
            result = subtract(a, b)
        elif operation == "multiply":
            result = multiply(a, b)
        elif operation == "divide":
            result = divide(a, b)
        else:
            raise ValueError(f"Unknown operation: {operation}")

        self.history.append(f"{a} {operation} {b} = {result}")
        return result

    def get_history(self) -> list[str]:
        """Return the computation history."""
        return self.history

    def clear_history(self) -> None:
        """Clear the computation history."""
        self.history = []


class ScientificCalculator(Calculator):
    """A scientific calculator with extra operations."""

    def power(self, base: int, exp: int) -> float:
        """Raise base to the power of exp."""
        result = float(base ** exp)
        self.history.append(f"{base} ^ {exp} = {result}")
        return result

    def modulo(self, a: int, b: int) -> int:
        """Return the remainder of a divided by b."""
        if b == 0:
            raise ValueError("Cannot modulo by zero")
        return a % b
