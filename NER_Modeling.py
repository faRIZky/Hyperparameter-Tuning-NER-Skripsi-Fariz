from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch


class NER_Modeling:
    def __init__(self, model_name: str, cache_dir: str):
        """Initialize the tokenizer and model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name, cache_dir=cache_dir)
        except Exception as e:
            raise RuntimeError(f"Error loading model or tokenizer: {e}")

    def run_ner(self, text: str):
        """
        Perform Named Entity Recognition (NER) on the input text.

        Args:
            text (str): Input text to process.

        Returns:
            dict: Dictionary containing token-label-score results and description.
        """
        # Tokenize input text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get predictions and confidence scores
        predictions = torch.argmax(outputs.logits, dim=-1)
        token_scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        labels = predictions[0].tolist()

        # Map label IDs to labels and confidence scores
        token_label_scores = []
        for token, label_id, scores in zip(tokens, labels, token_scores[0]):
            if token not in ["[CLS]", "[SEP]"]:  # Ignore special tokens
                label = self.model.config.id2label[label_id]
                score_percent = f"{scores[label_id] * 100:.1f}%"
                token_label_scores.append((token, label, score_percent))
        # Keterangan tambahan yang ditampilkan di hasil output
        description = """
                BIO Format:
                - B- (Beginning): Start of an entity
                - I- (Inside): Continuation of the same entity
                - O (Outside): Not an entity

                Entity Labels:
                - B-PER, I-PER: Person
                - B-LOC, I-LOC: Location
                - B-GEO, I-GEO: Geographical Entity
                - B-ORG, I-ORG: Organization
                - B-GPE, I-GPE: Geopolitical Entity
                - B-TIM, I-TIM: Time Expression
                - B-ART, I-ART: Artifact
                - B-EVE, I-EVE: Event
                - B-NAT, I-NAT: Natural Phenomenon
        """

        return {
            "results": token_label_scores,
            "description": description
        }

