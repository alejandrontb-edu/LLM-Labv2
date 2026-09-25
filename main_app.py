import hashlib
import html
import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from groq import Groq
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="NLP Lab + Groq",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .token-container {
        background: #111827;
        border-radius: 12px;
        padding: 18px;
        line-height: 2.4;
        margin-top: 10px;
        border: 1px solid #374151;
    }

    .token {
        display: inline-block;
        padding: 5px 9px;
        margin: 4px;
        border-radius: 7px;
        color: white;
        font-family: monospace;
        font-size: 14px;
        font-weight: 600;
    }

    .token-id {
        font-size: 10px;
        opacity: 0.75;
        margin-left: 5px;
    }

    .metric-card {
        background: #111827;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
    }

    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #60a5fa;
    }

    .metric-label {
        color: #9ca3af;
        font-size: 13px;
    }

    .info-box {
        background: #172033;
        border-left: 4px solid #60a5fa;
        padding: 12px 16px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UTILIDADES
# ============================================================

TOKEN_COLORS = [
    "#2563eb",
    "#7c3aed",
    "#db2777",
    "#dc2626",
    "#ea580c",
    "#ca8a04",
    "#16a34a",
    "#0891b2",
    "#4f46e5",
    "#9333ea",
]


def color_for_index(index: int) -> str:
    return TOKEN_COLORS[index % len(TOKEN_COLORS)]


def stable_id(text: str) -> int:
    """
    Genera un ID reproducible para esquemas de tokenización
    que no poseen IDs nativos.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def html_token(token: str, token_id: int, index: int) -> str:
    color = color_for_index(index)
    safe_token = html.escape(token)
    visible = safe_token.replace(" ", "␠")
    return (
        f'<span class="token" style="background:{color}">'
        f"{visible}"
        f'<span class="token-id">id={token_id}</span>'
        f"</span>"
    )


def render_tokens(tokens: List[Tuple[str, int]]) -> None:
    parts = []

    for i, (token, token_id) in enumerate(tokens):
        parts.append(html_token(token, token_id, i))

    st.markdown(
        '<div class="token-container">'
        + "".join(parts)
        + "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# TOKENIZADORES
# ============================================================

def tokenize_whitespace(text: str) -> List[Tuple[str, int]]:
    """
    Tokenización por espacios.
    Los IDs se generan de forma determinista para fines educativos.
    """
    tokens = text.split()

    return [
        (token, stable_id(token))
        for token in tokens
    ]


def tokenize_wordpunct(text: str) -> List[Tuple[str, int]]:
    """
    Separa palabras y puntuación.
    """
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

    return [
        (token, stable_id(token))
        for token in tokens
    ]


def tokenize_characters(text: str) -> List[Tuple[str, int]]:
    """
    Tokenización carácter por carácter.
    """
    tokens = list(text)

    return [
        (token, ord(token))
        for token in tokens
    ]


def tokenize_tiktoken(text: str) -> List[Tuple[str, int]]:
    """
    Tokenización BPE mediante tiktoken.

    cl100k_base se utiliza como tokenizer educativo local.
    No significa que sea necesariamente el tokenizer interno
    del modelo seleccionado en Groq.
    """
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        ids = encoding.encode(text)

        result = []

        for token_id in ids:
            token_bytes = encoding.decode_single_token_bytes(token_id)

            try:
                token = token_bytes.decode("utf-8")
            except UnicodeDecodeError:
                token = repr(token_bytes)

            result.append((token, token_id))

        return result

    except Exception as exc:
        raise RuntimeError(
            f"No se pudo cargar tiktoken: {exc}"
        )


TOKENIZERS = {
    "Whitespace": tokenize_whitespace,
    "Word + punctuation": tokenize_wordpunct,
    "Characters": tokenize_characters,
    "BPE - tiktoken cl100k_base": tokenize_tiktoken,
}


# ============================================================
# BAG OF WORDS
# ============================================================

def bag_of_words(texts: List[str]):
    vectorizer = CountVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
    )

    matrix = vectorizer.fit_transform(texts)

    vocabulary = vectorizer.get_feature_names_out()

    dataframe = pd.DataFrame(
        matrix.toarray(),
        columns=vocabulary,
    )

    return vectorizer, dataframe


# ============================================================
# COSENO
# ============================================================

def cosine_for_sentences(sentence_a: str, sentence_b: str):
    vectorizer = CountVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
    )

    matrix = vectorizer.fit_transform(
        [sentence_a, sentence_b]
    )

    similarity = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]

    return similarity, matrix.toarray(), vectorizer.get_feature_names_out()


# ============================================================
# GROQ
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_models(api_key: str) -> List[Dict]:
    """
    Consulta el catálogo dinámico de modelos de Groq.

    Se cachea unos minutos para no realizar una petición
    cada vez que Streamlit actualiza la interfaz.
    """
    client = Groq(api_key=api_key)

    response = client.models.list()

    models = []

    for model in response.data:
        model_id = getattr(model, "id", "")

        # Solo modelos relacionados con OpenAI/GPT.
        # Esto excluye Llama, Qwen, etc.
        if (
            "openai/" in model_id.lower()
            or "gpt" in model_id.lower()
        ):
            models.append(
                {
                    "id": model_id,
                    "owned_by": getattr(model, "owned_by", ""),
                    "context_window": getattr(
                        model,
                        "context_window",
                        None,
                    ),
                    "max_completion_tokens": getattr(
                        model,
                        "max_completion_tokens",
                        None,
                    ),
                }
            )

    return sorted(
        models,
        key=lambda x: x["id"],
    )


def generate_response(
    api_key: str,
    model: str,
    prompt: str,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_completion_tokens: int,
    seed: int | None,
    include_reasoning: bool,
):
    client = Groq(api_key=api_key)

    messages = []

    if system_prompt.strip():
        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )

    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_completion_tokens": max_completion_tokens,
    }

    if seed is not None:
        params["seed"] = seed

    # GPT-OSS permite controlar si se incluye reasoning.
    if model.startswith("openai/gpt-oss"):
        params["include_reasoning"] = include_reasoning

    response = client.chat.completions.create(**params)

    message = response.choices[0].message

    content = message.content or ""

    reasoning = getattr(
        message,
        "reasoning",
        None,
    )

    usage = getattr(
        response,
        "usage",
        None,
    )

    return {
        "content": content,
        "reasoning": reasoning,
        "usage": usage,
        "response": response,
    }


# ============================================================
# SIDEBAR - API KEY
# ============================================================

st.sidebar.title("🔐 Conexión con Groq")

api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    placeholder="gsk_...",
    help="La API key se utiliza únicamente durante esta sesión.",
)

if not api_key:
    st.title("🧠 NLP Lab + Groq")

    st.info(
        "Introduce tu Groq API Key en la barra lateral para comenzar."
    )

    st.markdown(
        """
        ### Funcionalidades

        - 🔤 Diferentes esquemas de tokenización.
        - 🆔 IDs de tokens.
        - 🎨 Visualización coloreada de tokens.
        - 📦 Bag of Words.
        - 📐 Similitud mediante distancia coseno.
        - 🤖 Generación de texto con modelos OpenAI GPT-OSS.
        - 🌡️ Temperature.
        - 🎯 Top-p.
        - 🔢 Máximo de tokens.
        - 🎲 Seed.
        - 🧠 Configuración de reasoning para GPT-OSS.
        - 📚 Catálogo dinámico de modelos disponibles en Groq.
        """
    )

    st.stop()


# ============================================================
# CARGAR MODELOS
# ============================================================

try:
    models = get_models(api_key)

except Exception as exc:
    st.error(
        "No se pudo consultar el catálogo de Groq."
    )
    st.exception(exc)
    st.stop()


if not models:
    st.warning(
        "No se encontraron modelos OpenAI/GPT disponibles "
        "para esta API key."
    )
    st.stop()


model_ids = [model["id"] for model in models]


# ============================================================
# HEADER
# ============================================================

st.title("🧠 NLP Lab + Groq")
st.caption(
    "Laboratorio educativo de tokenización, representación "
    "vectorial, similitud y generación de lenguaje."
)


# ============================================================
# INFORMACIÓN DE MODELOS
# ============================================================

with st.expander("📚 Catálogo de modelos OpenAI / GPT", expanded=False):

    catalog = pd.DataFrame(models)

    st.dataframe(
        catalog,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "La aplicación filtra el catálogo para priorizar modelos "
        "OpenAI/GPT-OSS y excluir Llama."
    )


# ============================================================
# TABS PRINCIPALES
# ============================================================

tab_token, tab_bow, tab_cosine, tab_generation = st.tabs(
    [
        "🔤 Tokenización",
        "📦 Bag of Words",
        "📐 Similitud Coseno",
        "🤖 Generación con Groq",
    ]
)


# ============================================================
# TAB 1 - TOKENIZACIÓN
# ============================================================

with tab_token:

    st.header("🔤 Tokenización")

    text = st.text_area(
        "Texto a tokenizar",
        value=(
            "Los modelos de lenguaje procesan texto "
            "mediante tokens."
        ),
        height=140,
    )

    tokenizer_name = st.selectbox(
        "Esquema de tokenización",
        list(TOKENIZERS.keys()),
    )

    if st.button(
        "Tokenizar",
        type="primary",
        key="tokenize_button",
    ):

        try:
            tokenizer = TOKENIZERS[tokenizer_name]

            tokens = tokenizer(text)

            st.subheader("Tokens")

            render_tokens(tokens)

            st.markdown("### Tabla de tokens")

            rows = []

            for position, (token, token_id) in enumerate(tokens):
                rows.append(
                    {
                        "posición": position,
                        "token": repr(token),
                        "token_id": token_id,
                    }
                )

            dataframe = pd.DataFrame(rows)

            st.dataframe(
                dataframe,
                use_container_width=True,
                hide_index=True,
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Cantidad de tokens",
                    len(tokens),
                )

            with col2:
                st.metric(
                    "Caracteres",
                    len(text),
                )

            with col3:
                if text:
                    ratio = len(tokens) / len(text)
                else:
                    ratio = 0

                st.metric(
                    "Tokens / carácter",
                    f"{ratio:.3f}",
                )

            if tokenizer_name != "BPE - tiktoken cl100k_base":
                st.caption(
                    "Los IDs de este esquema son IDs educativos "
                    "generados por la aplicación; no representan "
                    "IDs internos de un modelo de Groq."
                )

            if tokenizer_name == "BPE - tiktoken cl100k_base":
                st.caption(
                    "cl100k_base es un tokenizer local de referencia. "
                    "No debe interpretarse como una garantía de que "
                    "sea exactamente el tokenizer del modelo elegido "
                    "en Groq."
                )

        except Exception as exc:
            st.error("Error durante la tokenización.")
            st.exception(exc)


# ============================================================
# TAB 2 - BAG OF WORDS
# ============================================================

with tab_bow:

    st.header("📦 Bag of Words")

    st.write(
        "Introduce varios documentos/frases para convertirlos "
        "en vectores de frecuencia de palabras."
    )

    default_documents = (
        "Los modelos de lenguaje procesan texto.",
        "Los modelos de lenguaje utilizan tokens.",
        "Los tokens representan fragmentos de texto.",
    )

    documents_text = st.text_area(
        "Un documento por línea",
        value="\n".join(default_documents),
        height=180,
    )

    if st.button(
        "Calcular Bag of Words",
        type="primary",
        key="bow_button",
    ):

        documents = [
            line.strip()
            for line in documents_text.splitlines()
            if line.strip()
        ]

        if len(documents) < 1:
            st.warning(
                "Introduce al menos un documento."
            )
        else:

            try:
                vectorizer, dataframe = bag_of_words(
                    documents
                )

                st.subheader("Matriz Bag of Words")

                dataframe.index = [
                    f"Documento {i + 1}"
                    for i in range(len(documents))
                ]

                st.dataframe(
                    dataframe,
                    use_container_width=True,
                )

                st.subheader("Vocabulario")

                vocabulary = vectorizer.get_feature_names_out()

                vocabulary_df = pd.DataFrame(
                    {
                        "ID": range(len(vocabulary)),
                        "palabra": vocabulary,
                    }
                )

                st.dataframe(
                    vocabulary_df,
                    use_container_width=True,
                    hide_index=True,
                )

                st.subheader("Interpretación")

                st.write(
                    """
                    Cada fila representa un documento y cada columna
                    una palabra del vocabulario. El valor de cada celda
                    indica cuántas veces aparece esa palabra en el
                    documento.
                    """
                )

            except Exception as exc:
                st.error(
                    "No se pudo calcular Bag of Words."
                )
                st.exception(exc)


# ============================================================
# TAB 3 - COSINE SIMILARITY
# ============================================================

with tab_cosine:

    st.header("📐 Similitud mediante distancia coseno")

    sentence_a = st.text_area(
        "Frase A",
        value="Los modelos de lenguaje procesan texto.",
        height=100,
    )

    sentence_b = st.text_area(
        "Frase B",
        value="Los modelos de lenguaje trabajan con texto.",
        height=100,
    )

    if st.button(
        "Calcular similitud",
        type="primary",
        key="cosine_button",
    ):

        if not sentence_a.strip() or not sentence_b.strip():
            st.warning(
                "Debes introducir ambas frases."
            )
        else:

            try:
                similarity, matrix, vocabulary = (
                    cosine_for_sentences(
                        sentence_a,
                        sentence_b,
                    )
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Similitud coseno",
                        f"{similarity:.4f}",
                    )

                with col2:
                    st.metric(
                        "Porcentaje",
                        f"{similarity * 100:.2f}%",
                    )

                with col3:
                    st.metric(
                        "Palabras del vocabulario",
                        len(vocabulary),
                    )

                st.subheader(
                    "Representación vectorial"
                )

                vector_df = pd.DataFrame(
                    matrix,
                    columns=vocabulary,
                    index=["Frase A", "Frase B"],
                )

                st.dataframe(
                    vector_df,
                    use_container_width=True,
                )

                st.subheader(
                    "Interpretación geométrica"
                )

                st.write(
                    """
                    La similitud coseno compara el ángulo entre los
                    vectores de las dos frases. Un valor cercano a 1
                    indica que los vectores tienen una orientación
                    similar, mientras que un valor cercano a 0 indica
                    poca coincidencia bajo esta representación.
                    """
                )

                st.progress(
                    float(np.clip(similarity, 0, 1))
                )

            except Exception as exc:
                st.error(
                    "No se pudo calcular la similitud."
                )
                st.exception(exc)


# ============================================================
# TAB 4 - GENERACIÓN
# ============================================================

with tab_generation:

    st.header("🤖 Generación de respuestas con Groq")

    st.markdown(
        """
        <div class="info-box">
        Esta sección utiliza la API compatible con OpenAI de Groq.
        Los parámetros disponibles dependen del modelo seleccionado.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])

    with col1:

        selected_model = st.selectbox(
            "Modelo",
            model_ids,
            index=0,
        )

        model_info = next(
            (
                item
                for item in models
                if item["id"] == selected_model
            ),
            {},
        )

        st.caption(
            f"Modelo: `{selected_model}`"
        )

        if model_info.get("context_window"):
            st.caption(
                f"Context window: "
                f"{model_info['context_window']:,} tokens"
            )

    with col2:

        st.markdown("### Parámetros")

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.05,
            help=(
                "Valores bajos tienden a producir respuestas "
                "más deterministas; valores altos aumentan "
                "la variabilidad."
            ),
        )

        top_p = st.slider(
            "Top-p",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05,
            help=(
                "Controla la masa de probabilidad considerada "
                "durante el muestreo."
            ),
        )

        max_completion_tokens = st.number_input(
            "Max completion tokens",
            min_value=1,
            max_value=65536,
            value=1024,
            step=128,
        )

        use_seed = st.checkbox(
            "Usar seed",
            value=False,
        )

        seed = None

        if use_seed:
            seed = st.number_input(
                "Seed",
                min_value=0,
                max_value=2_147_483_647,
                value=42,
                step=1,
            )

    st.divider()

    system_prompt = st.text_area(
        "System prompt",
        value=(
            "Eres un asistente útil. Responde de forma clara "
            "y concisa en español."
        ),
        height=120,
    )

    prompt = st.text_area(
        "Prompt",
        value=(
            "Explica qué es la tokenización en los modelos "
            "de lenguaje con un ejemplo sencillo."
        ),
        height=180,
    )

    include_reasoning = False

    if selected_model.startswith("openai/gpt-oss"):
        include_reasoning = st.checkbox(
            "Incluir reasoning del modelo",
            value=False,
            help=(
                "GPT-OSS permite controlar si el campo reasoning "
                "se incluye en la respuesta."
            ),
        )

    st.info(
        "Learning rate no es un parámetro de inferencia de "
        "Groq Chat Completions. Es un hiperparámetro utilizado "
        "durante entrenamiento/ajuste de modelos."
    )

    generate = st.button(
        "🚀 Generar respuesta",
        type="primary",
        key="generate_button",
    )

    if generate:

        if not prompt.strip():
            st.warning(
                "Introduce un prompt."
            )
        else:

            with st.spinner(
                "Generando respuesta con Groq..."
            ):

                try:

                    result = generate_response(
                        api_key=api_key,
                        model=selected_model,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        top_p=top_p,
                        max_completion_tokens=(
                            max_completion_tokens
                        ),
                        seed=seed,
                        include_reasoning=include_reasoning,
                    )

                    st.subheader("Respuesta")

                    st.markdown(
                        result["content"]
                    )

                    if (
                        include_reasoning
                        and result["reasoning"]
                    ):
                        with st.expander(
                            "🧠 Reasoning devuelto por el modelo"
                        ):
                            st.markdown(
                                result["reasoning"]
                            )

                    st.divider()

                    st.subheader(
                        "📊 Información de la ejecución"
                    )

                    usage = result["usage"]

                    if usage:

                        prompt_tokens = getattr(
                            usage,
                            "prompt_tokens",
                            0,
                        )

                        completion_tokens = getattr(
                            usage,
                            "completion_tokens",
                            0,
                        )

                        total_tokens = getattr(
                            usage,
                            "total_tokens",
                            0,
                        )

                        c1, c2, c3 = st.columns(3)

                        with c1:
                            st.metric(
                                "Prompt tokens",
                                prompt_tokens,
                            )

                        with c2:
                            st.metric(
                                "Completion tokens",
                                completion_tokens,
                            )

                        with c3:
                            st.metric(
                                "Total tokens",
                                total_tokens,
                            )

                    with st.expander(
                        "⚙️ Parámetros utilizados"
                    ):

                        st.json(
                            {
                                "model": selected_model,
                                "temperature": temperature,
                                "top_p": top_p,
                                "max_completion_tokens": (
                                    max_completion_tokens
                                ),
                                "seed": seed,
                                "include_reasoning": (
                                    include_reasoning
                                ),
                            }
                        )

                except Exception as exc:

                    st.error(
                        "Error al generar la respuesta."
                    )

                    st.exception(exc)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "NLP Lab + Groq · Tokenización · Bag of Words · "
    "Cosine Similarity · Generación LLM"
)
