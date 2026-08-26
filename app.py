import streamlit as st
import pandas as pd
import torch
import plotly.express as px

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Intent Detector System",
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
# EXAMPLE UTTERANCES
# ============================================================

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


# ============================================================
# SESSION STATE
# ============================================================

if "message_input" not in st.session_state:
    st.session_state.message_input = ""

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


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

    # Streamlit Community Cloud uses CPU
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
        font-size: 2.3rem;
        font-weight: 700;
        margin-top: 0.5rem;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .prediction-card {
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #ddd;
        background-color: #fafafa;
        text-align: center;
        min-height: 190px;
    }

    .prediction-label {
        color: #777;
        font-size: 0.85rem;
        margin-bottom: 0.3rem;
    }

    .prediction-value {
        font-size: 1.45rem;
        font-weight: 700;
        margin-bottom: 1.2rem;
    }

    .footer {
        text-align: center;
        color: #888;
        font-size: 0.8rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Intent Detector System</div>',
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
# INPUT + PREDICTION
# ============================================================

col_input, col_prediction = st.columns(
    [2, 1],
    gap="large"
)


# ============================================================
# INPUT AREA
# ============================================================

with col_input:

    st.subheader("Enter your message")

    st.text_area(
        "Message",
        placeholder="Example: wake me up at 7 tomorrow",
        height=130,
        label_visibility="collapsed",
        key="message_input"
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
# RUN PREDICTION
# ============================================================

if predict_button:

    text = st.session_state.message_input

    if not text.strip():

        st.warning(
            "Please enter a message first."
        )

        st.session_state.prediction_result = None

    else:

        st.session_state.prediction_result = (
            predict_intent(text)
        )


# ============================================================
# DISPLAY PREDICTION
# ============================================================

with col_prediction:

    st.subheader("Prediction")

    if st.session_state.prediction_result is not None:

        (
            predicted_intent,
            predicted_probability,
            top3_df
        ) = st.session_state.prediction_result

        st.markdown(
            f"""
            <div class="prediction-card">

                <div class="prediction-label">
                    Predicted Intent
                </div>

                <div class="prediction-value">
                    {predicted_intent}
                </div>

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

    else:

        st.info(
            "Enter a message and click "
            "**Predict Intent**."
        )


# ============================================================
# TOP 3 PREDICTIONS
# ============================================================

if st.session_state.prediction_result is not None:

    (
        predicted_intent,
        predicted_probability,
        top3_df
    ) = st.session_state.prediction_result

    st.markdown("---")

    st.subheader("Top 3 Predictions")

    # --------------------------------------------------------
    # PREPARE CHART DATA
    # --------------------------------------------------------

    chart_df = top3_df.copy()

    chart_df["Probability"] = (
        chart_df["Probability"].round(2)
    )

    chart_df = chart_df.sort_values(
        "Probability",
        ascending=True
    )

    # --------------------------------------------------------
    # PLOTLY CHART
    # --------------------------------------------------------

    fig = px.bar(
        chart_df,
        x="Probability",
        y="Intent",
        orientation="h",
        text="Probability",
        labels={
            "Probability": "Probability (%)",
            "Intent": "Intent"
        },
        title="Top 3 Intent Predictions"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig.update_layout(
        height=300,
        margin=dict(
            l=20,
            r=50,
            t=55,
            b=20
        ),
        xaxis=dict(
            range=[
                0,
                max(
                    100,
                    chart_df["Probability"].max() * 1.15
                )
            ]
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # --------------------------------------------------------
    # TOP 3 TABLE
    # --------------------------------------------------------

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
        "It is not a guarantee that the prediction is correct."
    )


# ============================================================
# TRY AN EXAMPLE
# ============================================================

st.markdown("---")

st.subheader("Try an Example")

st.caption(
    "Click an example to place it in the message box."
)


example_cols = st.columns(4)

for i, example in enumerate(examples):

    with example_cols[i % 4]:

        if st.button(
            example,
            key=f"example_{i}",
            use_container_width=True
        ):

            # Put selected example into input box
            st.session_state.message_input = example

            # Clear previous prediction
            st.session_state.prediction_result = None

            # Rerun so the text area displays the example
            st.rerun()


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
