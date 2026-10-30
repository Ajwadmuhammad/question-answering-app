
import streamlit as st
from transformers import pipeline

# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Question Answering Application",
    page_icon="❓",
    layout="centered"
)

# --------------------------------------------------
# 2. Student Information
# --------------------------------------------------

st.title("Ajwad Muhammad Waseem | Student ID: b04-0923-000004")

st.subheader("Hugging Face Question Answering Application")

st.write(
    "Enter a paragraph (context) and ask a question "
    "related to the information in that paragraph. "
    "The AI model will extract the answer for you."
)

st.divider()

# --------------------------------------------------
# 3. Pretrained Model Configuration
# --------------------------------------------------

MODEL_NAME = "distilbert/distilbert-base-cased-distilled-squad"


# --------------------------------------------------
# 4. Load the Hugging Face Model
# --------------------------------------------------

@st.cache_resource
def load_qa_model():
    """
    Load the pretrained Question Answering model.
    Streamlit caches the model to avoid loading it
    again every time the user interacts with the app.
    """

    return pipeline(
        "question-answering",
        model=MODEL_NAME
    )


# --------------------------------------------------
# 5. User Input Section
# --------------------------------------------------

st.header("Enter Your Context")

context = st.text_area(
    "Context Paragraph",
    placeholder="Example: France is a country in Western Europe. Its capital city is Paris.",
    height=180
)

st.header("Ask Your Question")

question = st.text_input(
    "Question",
    placeholder="Example: What is the capital of France?"
)

st.divider()

# --------------------------------------------------
# 6. Generate Answer
# --------------------------------------------------

if st.button("Get Answer", type="primary", use_container_width=True):

    # Validate context
    if not context.strip():
        st.warning("Please enter a context paragraph.")

    # Validate question
    elif not question.strip():
        st.warning("Please enter a question.")

    else:
        try:
            # Load model
            with st.spinner("Loading Question Answering model..."):

                qa_model = load_qa_model()

            st.success("Question Answering model loaded successfully!")

            # Generate answer
            with st.spinner("Finding the answer..."):

                result = qa_model(
                    question=question.strip(),
                    context=context.strip()
                )

            # Extract answer
            answer = result["answer"]

            # Get confidence score from model
            confidence = result["score"] * 100

            # Display answer
            st.subheader("Question Answering Results")

            st.success("Answer generated successfully!")

            st.markdown("### Extracted Answer")

            st.info(answer if answer.strip() else "No clear answer found in the context.")

            # Display confidence score
            st.markdown("### Confidence Score")

            st.metric(
                label="Model Confidence",
                value=f"{confidence:.2f}%"
            )

            st.progress(float(result["score"]))

        except Exception as e:
            st.error("An error occurred while processing your question.")

            st.write("Error details:", str(e))

            st.info(
                "Please check your internet connection, "
                "model installation, and input text, then try again."
            )

# --------------------------------------------------
# 7. Additional Information
# --------------------------------------------------

st.divider()

st.caption(
    "Model: distilbert/distilbert-base-cased-distilled-squad"
)

st.caption(
    "Powered by Hugging Face Transformers and PyTorch."
)
