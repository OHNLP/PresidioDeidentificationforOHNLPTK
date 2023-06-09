import json
from typing import Union, List

from ohnlp.toolkit.backbone.api import BackboneComponentDefinition, BackboneComponent, BackboneComponentOneToOneDoFn, \
    Row, Schema, SchemaField, FieldType, TypeName
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

# Huggingface/Transformers and Spacy Model Support
from transformers import AutoTokenizer, AutoModelForTokenClassification
from spacy.cli import download


class PresidioDeIDComponent(BackboneComponent):
    def __init__(self):
        super().__init__()
        self.input_col = None
        self.output_col = None

    def init(self, configstr: str) -> None:
        config = json.loads(configstr)

        pass

    def to_do_fn_config(self) -> str:
        pass

    def get_input_tag(self) -> str:
        return "Documents with PII"

    def get_output_tags(self) -> List[str]:
        return ["De-Identified Text"]

    def calculate_output_schema(self, input_schema: Schema) -> dict[str, Schema]:
        return {
            "De-Identified Text": input_schema
        }


class PresidioDeIDDoFn(BackboneComponentOneToOneDoFn):
    def __init__(self):
        super().__init__()
        self.note_text_col_name = None
        self.output_col = None
        self.analyzer: Union[None, AnalyzerEngine] = None
        self.anonymizer: Union[None, AnonymizerEngine] = None

    def init_from_driver(self, config_json_str: str) -> None:
        pass

    def on_bundle_start(self) -> None:
        # TODO make this configurable
        transformers_model = "obi/deid_roberta_i2b2"
        # Download models -- TODO offline mode
        AutoTokenizer.from_pretrained(transformers_model)
        AutoModelForTokenClassification.from_pretrained(transformers_model)
        download("en_core_web_sm")
        presidio_analyzer_configuration = {
            "nlp_engine_name": "transformers",
            "models": [
                {
                    "lang_code": "en",
                    "model_name": {
                        "spacy": "en_core_web_sm",
                        "transformers": transformers_model
                    }
                }
            ]
        }
        provider: NlpEngineProvider = NlpEngineProvider(nlp_configuration=presidio_analyzer_configuration)
        self.analyzer = AnalyzerEngine(nlp_engine=provider.create_engine())
        self.anonymizer = AnonymizerEngine()

    def on_bundle_end(self) -> None:
        pass

    def apply(self, input_row: Row) -> List[Row]:
        fields: List[SchemaField] = input_row.schema.fields
        fields.append(SchemaField(self.output_col, FieldType(TypeName.STRING)))
        input_text: str = str(input_row.get_value(self.note_text_col_name))
        identified_pii_entities = self.analyzer.analyze(
            text=input_text,
            language='en'
        )
        anonymizer_output = self.anonymizer.anonymize(
            text=input_text,
            analyzer_results=identified_pii_entities
        )
        anonymized_text = anonymizer_output.text
        input_values = input_row.values
        input_values.append(anonymized_text)
        return [Row(Schema(fields), input_values)]


class PresidioDeIDComponentDefinition(BackboneComponentDefinition):
    def get_component_def(self) -> BackboneComponent:
        return PresidioDeIDComponent()

    def get_do_fn(self) -> BackboneComponentOneToOneDoFn:
        return PresidioDeIDDoFn()
