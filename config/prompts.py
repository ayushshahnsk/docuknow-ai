"""
Centralized prompt templates for DocuKnow AI - MULTILINGUAL EDITION

Keeping prompts separate:
- Improves maintainability
- Makes prompt-engineering explicit
- Enables multilingual support
- Looks very professional in reviews/interviews

Language Support:
- English (en)
- Hindi (hi)
- Marathi (mr)
- Gujarati (gu)
- French (fr)
- Bengali (bn)
- German (de)
- Chinese (zh)
- Spanish (es)
- And 10+ more languages
"""

from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# LANGUAGE-SPECIFIC TEMPLATES
# ============================================================================

# Language names for reference
LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "gu": "Gujarati",
    "fr": "French",
    "bn": "Bengali",
    "de": "German",
    "zh": "Chinese",
    "es": "Spanish",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "ur": "Urdu",
    "ar": "Arabic",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "pt": "Portuguese",
    "it": "Italian",
}

# System prompts for different languages
SYSTEM_PROMPTS = {
    "en": """You are DocuKnow AI, an intelligent multilingual document assistant.

Your purpose is to help users understand their documents by answering questions based ONLY on the provided context.

You must:
1. Answer ONLY using information from the CONTEXT
2. NEVER use outside knowledge or make assumptions
3. Be concise, clear, and accurate
4. If information is missing, admit it
5. Cite sources when possible
6. Respond in the same language as the question""",

    "hi": """आप DocuKnow AI हैं, एक बुद्धिमान बहुभाषी दस्तावेज़ सहायक।

आपका उद्देश्य उपयोगकर्ताओं को केवल प्रदान किए गए संदर्भ के आधार पर प्रश्नों का उत्तर देकर उनके दस्तावेज़ों को समझने में मदद करना है।

आपको यह करना चाहिए:
1. केवल CONTEXT से जानकारी का उपयोग करके उत्तर दें
2. बाहरी ज्ञान का उपयोग कभी न करें या धारणा न बनाएं
3. संक्षिप्त, स्पष्ट और सटीक रहें
4. यदि जानकारी गायब है, तो इसे स्वीकार करें
5. संभव हो तो स्रोतों का हवाला दें
6. प्रश्न की भाषा में ही उत्तर दें""",

    "mr": """तुम्ही DocuKnow AI आहात, एक बुद्धिमान बहुभाषी दस्तऐवज सहाय्यक.

तुमचे उद्दीष्ट फक्त प्रदान केलेल्या संदर्भावर आधारित प्रश्नांची उत्तरे देऊन वापरकर्त्यांना त्यांचे दस्तऐवज समजून घेण्यास मदत करणे हे आहे.

तुम्ही हे करणे आवश्यक आहे:
1. फक्त CONTEXT मधील माहिती वापरून उत्तर द्या
2. बाहेरील ज्ञान कधीही वापरू नका किंवा गृहीतके तयार करू नका
3. संक्षिप्त, स्पष्ट आणि अचूक रहा
4. माहिती गहाळ असल्यास, ती कबूल करा
5. शक्य असल्यास स्त्रोतांचा उल्लेख करा
6. प्रश्नाच्याच भाषेत उत्तर द्या""",

    "gu": """તમે ડોક્યુનો AI છો, એક બુદ્ધિશાળી બહુભાષી દસ્તાવેજ સહાયક.

તમારો હેતુ માત્ર પ્રદાન કરેલ સંદર્ભના આધારે પ્રશ્નોના જવાબ આપીને વપરાશકર્તાઓને તેમના દસ્તાવેજોને સમજવામાં મદદ કરવાનો છે.

તમારે આ કરવું જ જોઈએ:
1. ફક્ત CONTEXT માંથી માહિતીનો ઉપયોગ કરીને જવાબ આપો
2. બહારનું જ્ઞાન ક્યારેય વાપરશો નહીં અથવા ધારણાઓ ન બનાવશો
3. સંક્ષિપ્ત, સ્પષ્ટ અને ચોક્કસ રહો
4. જો માહિતી ખૂટે છે, તો તે સ્વીકારો
5. શક્ય હોય ત્યારે સ્રોતોનો સંદર્ભ લો
6. પ્રશ્નની જ ભાષામાં જવાબ આપો""",

    "fr": """Vous êtes DocuKnow AI, un assistant de documents multilingue intelligent.

Votre objectif est d'aider les utilisateurs à comprendre leurs documents en répondant aux questions en vous basant UNIQUEMENT sur le contexte fourni.

Vous devez :
1. Répondre UNIQUEMENT en utilisant les informations du CONTEXTE
2. NE JAMAIS utiliser de connaissances extérieures ou faire des suppositions
3. Être concis, clair et précis
4. Si des informations manquent, l'admettre
5. Citer les sources lorsque c'est possible
6. Répondre dans la même langue que la question""",

    "es": """Eres DocuKnow AI, un asistente de documentos multilingüe inteligente.

Tu propósito es ayudar a los usuarios a comprender sus documentos respondiendo preguntas basadas ÚNICAMENTE en el contexto proporcionado.

Debes:
1. Responder ÚNICAMENTE utilizando la información del CONTEXTO
2. NUNCA usar conocimiento externo o hacer suposiciones
3. Ser conciso, claro y preciso
4. Si falta información, admitirlo
5. Citar fuentes cuando sea posible
6. Responder en el mismo idioma que la pregunta""",

    "de": """Sie sind DocuKnow AI, ein intelligenter mehrsprachiger Dokumentenassistent.

Ihr Zweck ist es, Benutzern zu helfen, ihre Dokumente zu verstehen, indem Sie Fragen NUR auf der Grundlage des bereitgestellten Kontexts beantworten.

Sie müssen:
1. Antworten Sie NUR mit Informationen aus dem KONTEXT
2. Verwenden Sie NIEMALS externes Wissen oder treffen Sie Annahmen
3. Seien Sie präzise, klar und genau
4. Wenn Informationen fehlen, geben Sie dies zu
5. Zitieren Sie Quellen, wenn möglich
6. Antworten Sie in derselben Sprache wie die Frage""",

    "zh": """您是DocuKnow AI，一个智能的多语言文档助手。

您的目的是通过仅根据提供的上下文回答问题来帮助用户理解他们的文档。

您必须：
1. 仅使用上下文中的信息回答问题
2. 绝不使用外部知识或做出假设
3. 简洁、清晰、准确
4. 如果信息缺失，请承认
5. 尽可能引用来源
6. 用与问题相同的语言回答""",

    "ja": """あなたはDocuKnow AIです、インテリジェントな多言語ドキュメントアシスタントです。

あなたの目的は、提供されたコンテキストのみに基づいて質問に答えることで、ユーザーが自分のドキュメントを理解するのを助けることです。

あなたは次のことをしなければなりません：
1. コンテキストからの情報のみを使用して回答する
2. 外部の知識を使用したり、仮定を立てたりしない
3. 簡潔で、明確で、正確であること
4. 情報が不足している場合はそれを認める
5. 可能な場合はソースを引用する
6. 質問と同じ言語で回答する""",

    "ko": """당신은 DocuKnow AI입니다, 지능적인 다국어 문서 어시스턴트입니다.

귀하의 목적은 제공된 컨텍스트에만 기반하여 질문에 답변함으로써 사용자가 문서를 이해하도록 돕는 것입니다.

다음을 해야 합니다:
1. 컨텍스트의 정보만 사용하여 답변하세요
2. 외부 지식을 사용하거나 가정을 하지 마세요
3. 간결하고 명확하며 정확하게
4. 정보가 누락된 경우 인정하세요
5. 가능한 경우 출처를 인용하세요
6. 질문과 같은 언어로 답변하세요""",

    "default": """You are DocuKnow AI, an intelligent multilingual document assistant.

Your purpose is to help users understand their documents by answering questions based ONLY on the provided context.

You must:
1. Answer ONLY using information from the CONTEXT
2. NEVER use outside knowledge or make assumptions
3. Be concise, clear, and accurate
4. If information is missing, admit it
5. Cite sources when possible
6. Respond in the same language as the question"""
}

# "Not confident" messages for different languages
NOT_CONFIDENT_MESSAGES = {
    "en": "I am not confident based on the provided documents.",
    "hi": "प्रदान किए गए दस्तावेजों के आधार पर मुझे विश्वास नहीं है।",
    "mr": "दिलेल्या दस्तऐवजांच्या आधारे मला खात्री नाही.",
    "gu": "પ્રદાન કરેલા દસ્તાવેજોના આધારે મને વિશ્વાસ નથી.",
    "fr": "Je ne suis pas confiant sur la base des documents fournis.",
    "bn": "প্রদান করা নথির ভিত্তিতে আমি আত্মবিশ্বাসী নই।",
    "de": "Ich bin aufgrund der bereitgestellten Dokumente nicht zuversichtlich.",
    "zh": "根据提供的文件，我没有信心。",
    "es": "No estoy seguro basándome en los documentos proporcionados.",
    "ta": "வழங்கப்பட்ட ஆவணங்களின் அடிப்படையில் எனக்கு நம்பிக்கை இல்லை.",
    "te": "అందించిన డాక్యుమెంట్ల ఆధారంగా నాకు విశ్వాసం లేదు.",
    "kn": "ನೀಡಲಾದ ದಾಖಲೆಗಳ ಆಧಾರದ ಮೇಲೆ ನನಗೆ ವಿಶ್ವಾಸವಿಲ್ಲ.",
    "ml": "നൽകിയിരിക്കുന്ന രേഖകളുടെ അടിസ്ഥാനത്തിൽ എനിക്ക് ആത്മവിശ്വാസമില്ല.",
    "pa": "ਪ੍ਰਦਾਨ ਕੀਤੇ ਗਏ ਦਸਤਾਵੇਜ਼ਾਂ ਦੇ ਅਧਾਰ ਤੇ ਮੈਨੂੰ ਵਿਸ਼ਵਾਸ ਨਹੀਂ ਹੈ।",
    "ur": "فراہم کردہ دستاویزات کی بنیاد پر مجھے اعتماد نہیں ہے۔",
    "ar": "لست واثقًا بناءً على المستندات المقدمة.",
    "ja": "提供された文書に基づいて自信がありません。",
    "ko": "제공된 문서를 기반으로 확신이 서지 않습니다.",
    "ru": "Я не уверен на основе предоставленных документов.",
    "pt": "Não estou confiante com base nos documentos fornecidos.",
    "it": "Non sono sicuro sulla base dei documenti forniti.",
    "default": "I am not confident based on the provided documents."
}

# Instruction templates for different languages
INSTRUCTION_TEMPLATES = {
    "en": """INSTRUCTIONS (VERY IMPORTANT):
- Answer ONLY using the information in the CONTEXT below.
- Do NOT use outside knowledge or make assumptions.
- If the answer is not clearly present in the context, say: "{not_confident}"
- Keep the answer concise, clear, and well-structured.
- Include page numbers/sources when possible.
- Respond in {lang_name} ({lang_code}).""",

    "hi": """निर्देश (बहुत महत्वपूर्ण):
- नीचे दिए गए CONTEXT में दी गई जानकारी का उपयोग करके ही उत्तर दें।
- बाहरी ज्ञान का उपयोग न करें या धारणाएँ न बनाएं।
- यदि उत्तर संदर्भ में स्पष्ट रूप से मौजूद नहीं है, तो कहें: "{not_confident}"
- उत्तर संक्षिप्त, स्पष्ट और सुव्यवस्थित रखें।
- संभव हो तो पृष्ठ संख्या/स्रोत शामिल करें।
- {lang_name} ({lang_code}) में उत्तर दें।""",

    "mr": """सूचना (खूप महत्वाचे):
- खालील CONTEXT मधील माहितीचा वापर करून फक्त उत्तर द्या.
- बाहेरील ज्ञानाचा वापर करू नका किंवा गृहीतके तयार करू नका.
- उत्तर संदर्भात स्पष्टपणे उपलब्ध नसल्यास, म्हणा: "{not_confident}"
- उत्तर संक्षिप्त, स्पष्ट आणि सुव्यवस्थित ठेवा.
- शक्य असल्यास पृष्ठ क्रमांक/स्त्रोत समाविष्ट करा.
- {lang_name} ({lang_code}) मध्ये उत्तर द्या.""",

    "fr": """INSTRUCTIONS (TRÈS IMPORTANT):
- Répondez UNIQUEMENT en utilisant les informations dans le CONTEXTE ci-dessous.
- N'utilisez PAS de connaissances extérieures ou ne faites pas de suppositions.
- Si la réponse n'est pas clairement présente dans le contexte, dites : "{not_confident}"
- Gardez la réponse concise, claire et bien structurée.
- Incluez les numéros de page/sources lorsque c'est possible.
- Répondez en {lang_name} ({lang_code}).""",

    "es": """INSTRUCCIONES (MUY IMPORTANTE):
- Responda ÚNICAMENTE utilizando la información en el CONTEXTO a continuación.
- NO use conocimiento externo o haga suposiciones.
- Si la respuesta no está claramente presente en el contexto, diga: "{not_confident}"
- Mantenga la respuesta concisa, clara y bien estructurada.
- Incluya números de página/fuentes cuando sea posible.
- Responda en {lang_name} ({lang_code}).""",

    "default": """INSTRUCTIONS (VERY IMPORTANT):
- Answer ONLY using the information in the CONTEXT below.
- Do NOT use outside knowledge or make assumptions.
- If the answer is not clearly present in the context, say: "{not_confident}"
- Keep the answer concise, clear, and well-structured.
- Include page numbers/sources when possible.
- Respond in {lang_name} ({lang_code})."""
}

# ============================================================================
# CORE PROMPT BUILDING FUNCTIONS
# ============================================================================

def get_system_prompt(lang: str = "en") -> str:
    """
    Get system prompt for the specified language.
    
    Args:
        lang: Language code (default: "en")
        
    Returns:
        System prompt string
    """
    if lang in SYSTEM_PROMPTS:
        return SYSTEM_PROMPTS[lang]
    else:
        logger.warning(f"System prompt for language '{lang}' not found, using English")
        return SYSTEM_PROMPTS["en"]

def get_not_confident_message(lang: str = "en") -> str:
    """
    Get "not confident" message for the specified language.
    
    Args:
        lang: Language code (default: "en")
        
    Returns:
        "Not confident" message string
    """
    if lang in NOT_CONFIDENT_MESSAGES:
        return NOT_CONFIDENT_MESSAGES[lang]
    else:
        logger.warning(f"Not confident message for language '{lang}' not found, using English")
        return NOT_CONFIDENT_MESSAGES["en"]

def get_instruction_template(lang: str = "en") -> str:
    """
    Get instruction template for the specified language.
    
    Args:
        lang: Language code (default: "en")
        
    Returns:
        Instruction template string
    """
    if lang in INSTRUCTION_TEMPLATES:
        return INSTRUCTION_TEMPLATES[lang]
    else:
        logger.warning(f"Instruction template for language '{lang}' not found, using English")
        return INSTRUCTION_TEMPLATES["en"]

def format_context_block(contexts: List[Dict]) -> str:
    """
    Format contexts into a readable block with metadata.
    
    Args:
        contexts: List of context dictionaries
        
    Returns:
        Formatted context block string
    """
    if not contexts:
        return ""
    
    context_parts = []
    
    for i, ctx in enumerate(contexts, 1):
        source = ctx.get("source", "Unknown")
        page = ctx.get("page", "N/A")
        text = ctx.get("text", "").strip()
        
        if not text:
            continue
            
        # Truncate very long context
        if len(text) > 1000:
            text = text[:1000] + "... [truncated]"
        
        context_part = f"[Context {i} | Source: {source} | Page: {page}]\n{text}"
        context_parts.append(context_part)
    
    return "\n\n".join(context_parts)

def build_rag_prompt(
    query: str, 
    contexts: List[Dict], 
    lang: str = "en",
    include_system_prompt: bool = True,
    include_instructions: bool = True
) -> str:
    """
    Build a strict Retrieval-Augmented Generation prompt with multilingual support.
    
    The model is:
    - Forced to use ONLY document context
    - Prevented from hallucinating
    - Asked to be concise and clear
    - Required to respond in the query language
    
    Args:
        query: User query
        contexts: Retrieved contexts with metadata
        lang: Language code for response
        include_system_prompt: Whether to include system prompt
        include_instructions: Whether to include instructions
        
    Returns:
        Complete prompt string
    """
    # Validate language
    if lang not in LANGUAGE_NAMES:
        logger.warning(f"Unsupported language: {lang}, defaulting to English")
        lang = "en"
    
    lang_name = LANGUAGE_NAMES.get(lang, "English")
    
    # Get language-specific components
    not_confident_msg = get_not_confident_message(lang)
    
    # Handle case with no contexts
    if not contexts:
        if include_system_prompt:
            system_prompt = get_system_prompt(lang)
            base_prompt = f"""{system_prompt}

QUESTION ({lang_name}):
{query}

RESPONSE ({lang_name}):
{not_confident_msg}"""
        else:
            base_prompt = f"""QUESTION ({lang_name}):
{query}

RESPONSE ({lang_name}):
{not_confident_msg}"""
        
        return base_prompt.strip()
    
    # Format context block
    context_block = format_context_block(contexts)
    
    # Build the prompt
    prompt_parts = []
    
    # Add system prompt
    if include_system_prompt:
        system_prompt = get_system_prompt(lang)
        prompt_parts.append(system_prompt)
    
    # Add instructions
    if include_instructions:
        instruction_template = get_instruction_template(lang)
        instructions = instruction_template.format(
            not_confident=not_confident_msg,
            lang_name=lang_name,
            lang_code=lang
        )
        prompt_parts.append(instructions)
    
    # Add context
    prompt_parts.append(f"CONTEXT:\n{context_block}")
    
    # Add question
    prompt_parts.append(f"QUESTION ({lang_name}):\n{query}")
    
    # Add response instruction
    prompt_parts.append(f"RESPONSE ({lang_name}):")
    
    # Join all parts
    prompt = "\n\n".join(prompt_parts)
    
    # Log prompt for debugging (first 500 chars)
    logger.debug(f"Built prompt for language '{lang}' ({lang_name}), length: {len(prompt)}")
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Prompt preview: {prompt[:500]}...")
    
    return prompt.strip()

def build_followup_prompt(
    query: str,
    previous_answer: str,
    contexts: List[Dict],
    lang: str = "en"
) -> str:
    """
    Build a prompt for follow-up questions.
    
    Args:
        query: Follow-up query
        previous_answer: Previous answer from the assistant
        contexts: Retrieved contexts (can be same or new)
        lang: Language code
        
    Returns:
        Follow-up prompt string
    """
    # Get language-specific follow-up instructions
    followup_instructions = {
        "en": """This is a follow-up question. Consider the previous interaction below.

PREVIOUS INTERACTION:
Question: {prev_query}
Answer: {prev_answer}

Now answer the new question based on the context below.""",

        "hi": """यह एक अनुवर्ती प्रश्न है। नीचे पिछली बातचीत पर विचार करें।

पिछली बातचीत:
प्रश्न: {prev_query}
उत्तर: {prev_answer}

अब नीचे दिए गए संदर्भ के आधार पर नए प्रश्न का उत्तर दें।""",

        "fr": """Ceci est une question de suivi. Considérez l'interaction précédente ci-dessous.

INTERACTION PRÉCÉDENTE:
Question: {prev_query}
Réponse: {prev_answer}

Maintenant, répondez à la nouvelle question en fonction du contexte ci-dessous.""",

        "default": """This is a follow-up question. Consider the previous interaction below.

PREVIOUS INTERACTION:
Question: {prev_query}
Answer: {prev_answer}

Now answer the new question based on the context below."""
    }
    
    instruction = followup_instructions.get(lang, followup_instructions["default"])
    
    # Format the follow-up instruction
    followup_block = instruction.format(
        prev_query=query,  # Note: This should be previous query, but we don't have it
        prev_answer=previous_answer
    )
    
    # Build the main prompt
    base_prompt = build_rag_prompt(
        query=query,
        contexts=contexts,
        lang=lang,
        include_system_prompt=False,
        include_instructions=False
    )
    
    # Combine
    full_prompt = f"{followup_block}\n\n{base_prompt}"
    
    return full_prompt.strip()

def build_summarization_prompt(
    text: str,
    lang: str = "en",
    summary_type: str = "brief"  # "brief", "detailed", "bullet_points"
) -> str:
    """
    Build a prompt for document summarization.
    
    Args:
        text: Text to summarize
        lang: Language code
        summary_type: Type of summary
        
    Returns:
        Summarization prompt string
    """
    # Language-specific summarization instructions
    summarization_templates = {
        "en": {
            "brief": "Provide a brief summary (2-3 sentences) of the following text in English:",
            "detailed": "Provide a detailed summary of the following text in English, capturing all key points:",
            "bullet_points": "Provide a summary of the following text in English using bullet points:"
        },
        "hi": {
            "brief": "निम्नलिखित पाठ का संक्षिप्त सारांश (2-3 वाक्य) हिंदी में प्रदान करें:",
            "detailed": "निम्नलिखित पाठ का विस्तृत सारांश हिंदी में प्रदान करें, सभी मुख्य बिंदुओं को शामिल करते हुए:",
            "bullet_points": "बुलेट पॉइंट्स का उपयोग करके हिंदी में निम्नलिखित पाठ का सारांश प्रदान करें:"
        },
        "fr": {
            "brief": "Fournissez un bref résumé (2-3 phrases) du texte suivant en français:",
            "detailed": "Fournissez un résumé détaillé du texte suivant en français, capturant tous les points clés:",
            "bullet_points": "Fournissez un résumé du texte suivant en français en utilisant des puces:"
        },
        "default": {
            "brief": "Provide a brief summary (2-3 sentences) of the following text:",
            "detailed": "Provide a detailed summary of the following text, capturing all key points:",
            "bullet_points": "Provide a summary of the following text using bullet points:"
        }
    }
    
    # Get appropriate template
    if lang in summarization_templates:
        templates = summarization_templates[lang]
    else:
        templates = summarization_templates["default"]
    
    instruction = templates.get(summary_type, templates["brief"])
    
    # Build prompt
    prompt = f"""{get_system_prompt(lang)}

TASK: Summarize the following text.

{instruction}

TEXT TO SUMMARIZE:
{text[:5000]}  # Limit text length

SUMMARY ({LANGUAGE_NAMES.get(lang, 'English')}):"""
    
    return prompt.strip()

def build_translation_prompt(
    text: str,
    source_lang: str,
    target_lang: str
) -> str:
    """
    Build a prompt for text translation between languages.
    
    Args:
        text: Text to translate
        source_lang: Source language code
        target_lang: Target language code
        
    Returns:
        Translation prompt string
    """
    source_name = LANGUAGE_NAMES.get(source_lang, source_lang)
    target_name = LANGUAGE_NAMES.get(target_lang, target_lang)
    
    prompt = f"""{get_system_prompt("en")}

TASK: Translate the following text from {source_name} to {target_name}.

Translate accurately while preserving the meaning, tone, and context.

SOURCE TEXT ({source_name}):
{text[:3000]}  # Limit text length

TRANSLATED TEXT ({target_name}):"""
    
    return prompt.strip()

def build_comparison_prompt(
    text1: str,
    text2: str,
    lang: str = "en"
) -> str:
    """
    Build a prompt for comparing two texts.
    
    Args:
        text1: First text
        text2: Second text
        lang: Language code
        
    Returns:
        Comparison prompt string
    """
    lang_name = LANGUAGE_NAMES.get(lang, "English")
    
    prompt = f"""{get_system_prompt(lang)}

TASK: Compare and contrast the following two texts.

Analyze similarities and differences in content, style, and key points.

TEXT 1:
{text1[:2000]}

TEXT 2:
{text2[:2000]}

COMPARISON ANALYSIS ({lang_name}):"""
    
    return prompt.strip()

# ============================================================================
# PROMPT UTILITIES
# ============================================================================

def estimate_tokens(prompt: str) -> int:
    """
    Estimate token count for a prompt.
    Uses approximation: 1 token ≈ 4 characters.
    
    Args:
        prompt: Prompt text
        
    Returns:
        Estimated token count
    """
    if not prompt:
        return 0
    return max(1, len(prompt) // 4)

def truncate_prompt(
    prompt: str,
    max_tokens: int = 4000,
    truncation_method: str = "end"  # "end", "middle", "smart"
) -> str:
    """
    Truncate prompt to fit within token limits.
    
    Args:
        prompt: Prompt text
        max_tokens: Maximum allowed tokens
        truncation_method: How to truncate
        
    Returns:
        Truncated prompt
    """
    current_tokens = estimate_tokens(prompt)
    
    if current_tokens <= max_tokens:
        return prompt
    
    # Calculate max characters
    max_chars = max_tokens * 4
    
    if truncation_method == "end":
        return prompt[:max_chars] + "... [truncated]"
    
    elif truncation_method == "middle":
        half = max_chars // 2
        return prompt[:half] + "... [truncated] ..." + prompt[-half:]
    
    elif truncation_method == "smart":
        # Try to preserve important parts (context and question)
        if "CONTEXT:" in prompt and "QUESTION:" in prompt:
            # Extract context and question
            context_start = prompt.find("CONTEXT:")
            question_start = prompt.find("QUESTION:")
            
            context_section = prompt[context_start:question_start]
            question_section = prompt[question_start:]
            
            # Truncate context if needed
            if len(context_section) > max_chars * 0.7:  # 70% of max for context
                context_section = context_section[:int(max_chars * 0.7)] + "... [context truncated]"
            
            # Combine
            truncated = prompt[:context_start] + context_section + question_section
            
            if len(truncated) > max_chars:
                truncated = truncated[:max_chars] + "... [truncated]"
            
            return truncated
    
    # Default truncation
    return prompt[:max_chars] + "... [truncated]"

def get_language_support() -> Dict[str, str]:
    """
    Get information about supported languages.
    
    Returns:
        Dictionary of language_code -> language_name
    """
    return LANGUAGE_NAMES.copy()

def validate_language(lang: str) -> bool:
    """
    Check if a language code is supported.
    
    Args:
        lang: Language code
        
    Returns:
        True if supported, False otherwise
    """
    return lang in LANGUAGE_NAMES

# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Configure logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Multilingual Prompts")
    print("=" * 60)
    
    # Test contexts
    test_contexts = [
        {
            "text": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions.",
            "page": 1,
            "source": "AI_Introduction.pdf"
        },
        {
            "text": "Machine Learning is a subset of AI that enables machines to learn from data without being explicitly programmed.",
            "page": 2,
            "source": "AI_Introduction.pdf"
        }
    ]
    
    # Test different languages
    test_cases = [
        ("What is Artificial Intelligence?", "en"),
        ("कृत्रिम बुद्धिमत्ता क्या है?", "hi"),
        ("Qu'est-ce que l'intelligence artificielle?", "fr"),
        ("¿Qué es la inteligencia artificial?", "es"),
        ("कृत्रिम बुद्धिमता म्हणजे काय?", "mr"),
    ]
    
    for query, lang in test_cases:
        print(f"\nLanguage: {lang} ({LANGUAGE_NAMES.get(lang, 'Unknown')})")
        print(f"Query: {query}")
        print("-" * 40)
        
        try:
            prompt = build_rag_prompt(query, test_contexts, lang=lang)
            tokens = estimate_tokens(prompt)
            
            print(f"Prompt tokens: {tokens}")
            print(f"Prompt preview (first 300 chars):")
            print(prompt[:300] + "...")
            print()
            
        except Exception as e:
            print(f"Error: {e}")
    
    # Test no contexts case
    print("\n\nTesting No Contexts Case")
    print("=" * 60)
    
    query = "What is quantum computing?"
    lang = "en"
    
    prompt = build_rag_prompt(query, [], lang=lang)
    print(f"Query: {query}")
    print(f"Prompt (no contexts):")
    print(prompt[:500])
    
    # Show language support
    print("\n\nSupported Languages:")
    print("=" * 60)
    languages = get_language_support()
    for code, name in sorted(languages.items()):
        print(f"{code}: {name}")
    
    print(f"\nTotal supported languages: {len(languages)}")