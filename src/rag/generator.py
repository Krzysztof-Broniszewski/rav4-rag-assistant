import ollama

class Generator:
    def __init__(self):
        self.model_name = "hf.co/SpeakLeash/Bielik-4.5B-v3.0-Instruct-GGUF:Q8_0"
        self.system_prompt = """Jesteś asystentem technicznym.

        Odpowiadaj bezpośrednio na zadane pytanie.

        Odpowiadaj na pytanie wyłącznie na podstawie dostarczonego kontekstu.

        Nie dodawaj informacji, które nie są potrzebne do udzielenia odpowiedzi,
        nawet jeżeli znajdują się w kontekście.

        Nie dodawaj kroków, nazw opcji, ustawień ani informacji,
        których nie ma w kontekście.

        Nie uzupełniaj brakujących informacji na podstawie własnej wiedzy
        ani przypuszczeń.

        Jeżeli kontekst zawiera tylko część procedury, podaj tylko tę część
        i zaznacz, że pozostałych kroków nie ma w dostarczonym kontekście.

        Jeżeli w kontekście nie ma odpowiedzi, powiedz, że nie znalazłeś
        tej informacji w dostarczonej dokumentacji.
        """

    def generate(self, question, results):
        context_parts = []
        for result in results:
            context_parts.append(result["text"])

        context = "\n\n".join(context_parts)

        user_prompt = f"""Kontekst:
        {context}

        Pytanie: 
        {question}"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = ollama.chat(
            model=self.model_name,
            messages=messages
        )
        return response["message"]["content"]