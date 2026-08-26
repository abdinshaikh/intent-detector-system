import streamlit as st
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Amazon MASSIVE Intent Detector",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "abdinshaikh/intenr-bert"

MAX_LENGTH = 128


# ============================================================
# INTENT LABELS
# ============================================================

intent_names = [
    "datetime_query",
    "iot_hue_lightchange",
    "transport_ticket",
    "takeaway_query",
    "qa_stock",
    "general_greet",
    "recommendation_events",
    "music_dislikeness",
    "iot_wemo_off",
    "cooking_recipe",
    "qa_currency",
    "transport_traffic",
    "general_quirky",
    "weather_query",
    "audio_volume_up",
    "email_addcontact",
    "takeaway_order",
    "email_querycontact",
    "iot_hue_lightup",
    "recommendation_locations",
    "play_audiobook",
    "lists_createoradd",
    "news_query",
    "alarm_query",
    "iot_wemo_on",
    "general_joke",
    "qa_definition",
    "social_query",
    "music_settings",
    "audio_volume_other",
    "calendar_remove",
    "iot_hue_lightdim",
    "calendar_query",
    "email_sendemail",
    "iot_cleaning",
    "audio_volume_down",
    "play_radio",
    "cooking_query",
    "datetime_convert",
    "qa_maths",
    "iot_hue_lightoff",
    "iot_hue_lighton",
    "transport_query",
    "music_likeness",
    "email_query",
    "play_music",
    "audio_volume_mute",
    "social_post",
    "alarm_set",
    "qa_factoid",
    "calendar_set",
    "play_game",
    "alarm_remove",
    "lists_remove",
    "transport_taxi",
    "recommendation_movies",
    "iot_coffee",
    "music_query",
    "play_podcasts",
    "lists_query"
]


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    # Streamlit Community Cloud will generally use CPU
    device = torch.device("cpu")

    model.to(device)
    model.eval()

    return tokenizer, model, device


# ============================================================
# LOAD MODEL
# ============================================================

with st.spinner("Loading BERT model..."):

    tokenizer, model, device = load_model()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .prediction-box {
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        background-color: #fafafa;
        text-align: center;
    }

    .prediction-label {
        color: #666;
        font-size: 0.9rem;
    }

    .prediction-value {
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: 0.3rem;
    }

    .footer {
        text-align: center;
        color: #777;
        font-size: 0.8rem;
        margin-top: 2rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🧠 Amazon MASSIVE Intent Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    BERT-based Natural Language Intent Classification<br>
    Fine-tuned on the Amazon MASSIVE English dataset with
    <b>60 intent classes</b>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INPUT AREA
# ============================================================

col_input, col_result = st.columns(
    [2, 1],
    gap="large"
)


with col_input:

    st.subheader("Enter your message")

    text = st.text_area(
        "Message",
        placeholder="Example: wake me up at 7 tomorrow",
        height=130,
        label_visibility="collapsed"
    )

    predict_button = st.button(
        "🔍 Predict Intent",
        type="primary",
        use_container_width=True
    )


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_intent(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0]

    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=3
    )

    predicted_id = top_indices[0].item()

    predicted_intent = intent_names[
        predicted_id
    ]

    predicted_probability = (
        top_probabilities[0].item() * 100
    )

    top3_data = []

    for probability, index in zip(
        top_probabilities,
        top_indices
    ):

        index = index.item()

        top3_data.append({
            "Intent": intent_names[index],
            "Probability": probability.item() * 100
        })

    top3_df = pd.DataFrame(top3_data)

    return (
        predicted_intent,
        predicted_probability,
        top3_df
    )


# ============================================================
# RESULT AREA
# ============================================================

with col_result:

    st.subheader("Prediction")

    if predict_button:

        if not text.strip():

            st.warning(
                "Please enter a message first."
            )

        else:

            (
                predicted_intent,
                predicted_probability,
                top3_df
            ) = predict_intent(text)

            st.markdown(
                f"""
                <div class="prediction-box">

                <div class="prediction-label">
                Predicted Intent
                </div>

                <div class="prediction-value">
                {predicted_intent}
                </div>

                <br>

                <div class="prediction-label">
                Prediction Probability
                </div>

                <div class="prediction-value">
                {predicted_probability:.2f}%
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# TOP 3 PREDICTIONS
# ============================================================

if predict_button and text.strip():

    st.markdown("---")

    st.subheader("Top 3 Predictions")

    # Reverse so highest probability appears at top
    chart_df = (
        top3_df
        .iloc[::-1]
        .reset_index(drop=True)
    )

    st.bar_chart(
        chart_df.set_index("Intent")["Probability"],
        horizontal=True,
        x_label="Probability (%)",
        y_label="Intent"
    )

    # Show exact values
    display_df = top3_df.copy()

    display_df["Probability"] = (
        display_df["Probability"]
        .map(lambda x: f"{x:.2f}%")
    )

    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "Prediction probability represents the softmax "
        "probability assigned by the model to each class. "
        "It is not a guarantee of correctness."
    )


# ============================================================
# EXAMPLE QUERIES
# ============================================================

st.markdown("---")

st.subheader("💡 Try an Example")

examples = [
    "wake me up at 7 tomorrow",
    "what time is my alarm",
    "cancel my alarm",
    "turn the lights off",
    "make the lights brighter",
    "play some music",
    "turn the volume down",
    "what is the weather today",
    "find me a taxi",
    "send an email to John",
    "what is the capital of France",
    "tell me a joke",
    "what's happening in LA?",
    "good night"
]

example_cols = st.columns(4)

for i, example in enumerate(examples):

    with example_cols[i % 4]:
        st.caption(example)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown("---")

with st.expander("📊 Model Information"):

    info_col1, info_col2 = st.columns(2)

    with info_col1:

        st.markdown(
            """
            **Model:** `abdinshaikh/intenr-bert`

            **Architecture:** BERT for Sequence Classification

            **Task:** Single-label intent classification

            **Number of intents:** 60

            **Maximum sequence length:** 128 tokens

            **Training epochs:** 3
            """
        )

    with info_col2:

        st.markdown(
            """
            **Test Accuracy:** 88.90%

            **Test Macro F1:** 86.33%

            **Test Weighted F1:** 88.86%

            **Baseline Accuracy:** 80.42%

            **Baseline Macro F1:** 75.45%
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
    Amazon MASSIVE Intent Detection • Fine-tuned BERT
    </div>
    """,
    unsafe_allow_html=True
)
