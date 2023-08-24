"""
This script downloads the model resources necessary to execute de-identification while in standalone/offline mode
(i.e., not connected to the internet)
"""
from transformers import AutoTokenizer, AutoModelForTokenClassification

if __name__ == '__main__':
    # Huggingface
    tokenizer = AutoTokenizer.from_pretrained("obi/deid_roberta_i2b2")
    model = AutoModelForTokenClassification.from_pretrained("obi/deid_roberta_i2b2")
    tokenizer.save_pretrained("obi_deid_roberta_i2b2")
    model.save_pretrained("obi_deid_roberta_i2b2")

