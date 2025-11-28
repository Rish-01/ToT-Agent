from src.tools.base import Tool

from asteval import Interpreter
from math import pi, e
from sympy import Eq, solve, sympify

class EquationSolverTool(Tool):
    """Solve algebraic equations."""
    def __init__(self):
        super().__init__("equation_solver")

    def run(self, input_text):
        try:
            if "=" not in input_text:
                return "ERROR: Please provide an equation containing '='"
            
            input_text.replace("^", "**")
            lhs, rhs = input_text.split("=")

            lhs = sympify(lhs.strip())
            rhs = sympify(rhs.strip())

            # Create equation
            eq = Eq(lhs, rhs)

            # Detect all symbols in the equation
            symbols_in_eq = list(eq.free_symbols)

            if not symbols_in_eq:
                return "ERROR: No symbols found to solve for."

            sol = solve(eq, symbols_in_eq)

            return f"Symbols: {symbols_in_eq}, Solutions: {sol}"

        except Exception as e:
            return f"ERROR: An exception occured while using the equation solver tool, {e}"


class CalculatorTool(Tool):
    """Solve numerical questions."""
    def __init__(self):
        super().__init__("calculator")
        self.interpreter = Interpreter()
        self.interpreter.symtable['pi'] = pi
        self.interpreter.symtable['e'] = e


    def run(self, input_text):
        try:
            result = self.interpreter(input_text)
            if result is None:
                return "ERROR"
            return str(result)
        except Exception as e:
            return f"ERROR: An exception occured while using calculator, {e}"
        
if __name__ == "__main__":
    eq_tool = EquationSolverTool()
    calc_tool = CalculatorTool()

    # Test the equation solver
    eq_input = "x**2 - 4 = 0"
    print("EquationSolverTool test:")
    print(f"Input: {eq_input}")
    print(f"Output: {eq_tool.run(eq_input)}\n")

    # Test the calculator
    calc_input = "2 * pi + 3"
    print("CalculatorTool test:")
    print(f"Input: {calc_input}")
    print(f"Output: {calc_tool.run(calc_input)}")
