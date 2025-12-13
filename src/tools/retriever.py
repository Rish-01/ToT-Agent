import wikipedia
from src.tools.math_tools import Tool  

class WikipediaTool(Tool):
    """Retrieves summaries from Wikipedia."""
    def __init__(self):
        super().__init__("wiki")

    def run(self, input_text):
        try:
            return wikipedia.summary(input_text, sentences=5)
        except wikipedia.exceptions.DisambiguationError as e:
            return f"Wiki Error: Disambiguation. Try being more specific. Options: {e.options[:3]}"
        except wikipedia.exceptions.PageError:
            return "Wiki Error: Page not found."
        except Exception as e:
            return f"Wiki Error: {e}"