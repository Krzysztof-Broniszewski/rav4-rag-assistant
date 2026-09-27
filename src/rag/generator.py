import ollama

class Generator:
    def __init__(self):
        self.model_name = "SpeakLeash/bielik-11b-v2.3-instruct:Q8_0"
        self.system_prompt = """Jesteś asystentem technicznym.
        Odpowiadaj na pytanie wyłącznie na podstawie dostarczonego kontekstu.
        Jeżeli w kontekście nie ma odpowiedzi, powiedz, że nie znalazłeś
        tej informacji w dostarczonej dokumentacji."""

    def generate(self, question, results):
        context_parts = []