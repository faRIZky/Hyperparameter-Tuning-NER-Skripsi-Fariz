import streamlit as st
from NER_Modeling import NER_Modeling

class View:
    def __init__(self):
        # Initialize NER_Modeling
        CACHE_DIR = "C:/Users/lenovo/PycharmProjects/skripsi_draft"
        MODEL_NAME = "farizkuy/skripsi-bert-ner-Final_Configuration_XIII-model"
        try:
            self.NER_Modeling = NER_Modeling(model_name=MODEL_NAME, cache_dir=CACHE_DIR)
        except RuntimeError as e:
            st.error(f"Error loading model or tokenizer: {e}")
            st.stop()

    def run(self):
        # UI Input
        text_input = self.form()

        # Process text and display results
        if st.button('Run NER'):
            if text_input.strip():
                try:
                    ner_results = self.NER_Modeling.run_ner(text_input)
                    self.display_results(ner_results)
                except Exception:
                    st.warning("An error occurred during NER processing.")
            else:
                st.warning("Please enter some text for NER.")

    def form(self):
        # Create input form
        st.title("Named Entity Recognition (NER)")
        text_input = st.text_area("Enter text for NER", "Andrew Malik started working at Google in Southern Canada this morning.")
        return text_input

    def display_results(self, ner_results):
        # Define label-to-color mapping for Dark Mode
        LABEL_COLORS = {
            "B-PER": "background-color:#1f77b4; color:#ffffff;",
            "I-PER": "background-color:#1f77b4; color:#ffffff;",
            "B-LOC": "background-color:#ff7f0e; color:#000000;",
            "I-LOC": "background-color:#ff7f0e; color:#000000;",
            "B-GEO": "background-color:#2ca02c; color:#ffffff;",
            "I-GEO": "background-color:#2ca02c; color:#ffffff;",
            "B-ORG": "background-color:#d62728; color:#ffffff;",
            "I-ORG": "background-color:#d62728; color:#ffffff;",
            "B-GPE": "background-color:#9467bd; color:#ffffff;",
            "I-GPE": "background-color:#9467bd; color:#ffffff;",
            "B-TIM": "background-color:#8c564b; color:#ffffff;",
            "I-TIM": "background-color:#8c564b; color:#ffffff;",
            "B-ART": "background-color:#e377c2; color:#000000;",
            "I-ART": "background-color:#e377c2; color:#000000;",
            "B-EVE": "background-color:#7f7f7f; color:#ffffff;",
            "I-EVE": "background-color:#7f7f7f; color:#ffffff;",
            "B-NAT": "background-color:#bcbd22; color:#000000;",
            "I-NAT": "background-color:#bcbd22; color:#000000;"
        }

        # Extract results and description
        token_label_scores = ner_results["results"]
        description = ner_results["description"].replace("\n", "<br>")  # Fix di sini

        # Format and display results
        formatted_text = ""
        for token, label, score in token_label_scores:
            if label != 'O':  # Only style tokens with a valid label
                style = LABEL_COLORS.get(label, "")
                formatted_text += f'<span style="{style}">{token} [{label}] - {score}</span> '
            else:
                formatted_text += f'{token} '

        # Display formatted NER output
        st.markdown(f"<p style='line-height:1.8;'>{formatted_text.strip()}</p>", unsafe_allow_html=True)

        # Display description
        st.markdown("## Description :", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:14px;'>{description}</p>", unsafe_allow_html=True)


# Run the Streamlit app
app = View()
app.run()
