class PromptTemplates:
    PT = """Por favor, responda à pergunta baseando-se EXCLUSIVAMENTE no seguinte contexto:

{context}

---

Regras OBRIGATÓRIAS:
1. Use APENAS as informações fornecidas no contexto acima
2. Se a informação NÃO estiver EXPLICITAMENTE no contexto, responda "Não encontrei informações sobre isso no contexto fornecido"
3. NÃO FAÇA SUPOSIÇÕES ou adicione informações externas
4. Se o contexto não for relacionado à pergunta, responda "O contexto fornecido não contém informações relacionadas a esta pergunta"
5. Mantenha a resposta focada apenas no que está documentado
6. NÃO RESPONDA do que o contexto fornecido se trata, apenas que a pergunta não está nele

Pergunta: {question}"""

    ES = """Por favor, responde a la pregunta basándote EXCLUSIVAMENTE en el siguiente contexto:

{context}

---

Reglas OBLIGATORIAS:
1. Utiliza SOLO la información proporcionada en el contexto anterior
2. Si la información NO está EXPLÍCITAMENTE en el contexto, responde "No encontré información sobre esto en el contexto proporcionado"
3. NO HAGAS SUPOSICIONES ni agregues información externa
4. Si el contexto no está relacionado con la pregunta, responde "El contexto proporcionado no contiene información relacionada con esta pregunta"
5. Mantén la respuesta enfocada solo en lo que está documentado
6. NO RESPONDA sobre lo que trata el contexto proporcionado, solo que la pregunta no está en él.

Pregunta: {question}"""

    @classmethod
    def get_template(cls, language: str) -> str:
        """
        Retorna o template apropriado baseado no idioma.

        Args:
            language (str): Código do idioma ('pt' ou 'es')

        Returns:
            str: Template do prompt no idioma especificado

        Raises:
            ValueError: Se o idioma não for suportado
        """
        if language == 'pt':
            return cls.PT
        elif language == 'es':
            return cls.ES
        else:
            raise ValueError(f"Idioma não suportado: {language}")

    @classmethod
    def get_system_prompt(cls):
        """
        Retorna o system prompt.

        Returns:
            str: system prompt em português.
        """
        return """
            Você é um assistente técnico especializado em documentação de software.
            IMPORTANTE:
            1. Responda APENAS com informações presentes no contexto fornecido
            2. Se não houver informações suficientes, diga claramente
            3. NÃO USE conhecimento externo
            4. NÃO FAÇA suposições
            5. Seja direto e técnico em suas respostas
            6. Se a pergunta não estiver relacionada ao contexto, deixe isso claro
            7. NÃO RESPONDA do que o contexto fornecido se trata, apenas que a pergunta não está nele
            """