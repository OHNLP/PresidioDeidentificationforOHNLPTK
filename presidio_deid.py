import json
from typing import Union, List

from ohnlp.toolkit.backbone.api import BackboneComponentDefinition, BackboneComponent, BackboneComponentOneToOneDoFn, \
    Row, Schema, SchemaField, FieldType, TypeName, BackboneComponentOneToManyDoFn, TaggedRow
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import RecognizerResult

# Huggingface/Transformers and Spacy Model Support
from transformers import AutoTokenizer, AutoModelForTokenClassification
from spacy.cli import download


class IdentifyPIIComponent(BackboneComponent):
    def __init__(self):
        super().__init__()
        self.config = None
        self.note_id_column_name = None
        self.input_col = None

    def init(self, configstr: str) -> None:
        self.config = configstr
        configobj = json.loads(self.config)
        self.note_id_column_name = configobj["id_col"]
        self.input_col = configobj["text_col"]

    def to_do_fn_config(self) -> str:
        return self.config

    def get_input_tag(self) -> str:
        return "Documents with PII"

    def get_output_tags(self) -> List[str]:
        return ["PII Positions", "Text w/ PII Redacted"]

    def calculate_output_schema(self, input_schema: Schema) -> dict[str, Schema]:
        pii_entity_schema: Schema = Schema([
            SchemaField(self.note_id_column_name, input_schema.get_field(self.note_id_column_name).field_type),
            SchemaField("entity_type", FieldType(TypeName.STRING)),
            SchemaField("start_offset", FieldType(TypeName.INT32)),
            SchemaField("end_offset", FieldType(TypeName.INT32)),
            SchemaField("confidence_score", FieldType(TypeName.DOUBLE))
        ])
        return {
            "PII Positions": pii_entity_schema,
            "De-Identified Text": input_schema
        }


class IdentifyPIIDoFn(BackboneComponentOneToManyDoFn):
    def __init__(self):
        super().__init__()
        self.acceptance_threshold = None
        self.note_id_col_name = None
        self.note_text_col_name = None
        self.analyzer: Union[None, AnalyzerEngine] = None
        self.anonymizer: Union[None, AnonymizerEngine] = None
        self.pii_entity_schema = None
        self.transformers_model = None

    def init_from_driver(self, config_json_str: str) -> None:
        config = json.loads(config_json_str)
        self.note_id_col_name = config['id_col']
        self.note_text_col_name = config['text_col']
        self.transformers_model = config['ner_model_path']
        self.acceptance_threshold = config['acceptance_threshold']

    def on_bundle_start(self) -> None:
        # Download models
        AutoTokenizer.from_pretrained(self.transformers_model)
        AutoModelForTokenClassification.from_pretrained(self.transformers_model)
        download("en_core_web_lg")  # TODO offline mode
        presidio_analyzer_configuration = {
            "nlp_engine_name": "transformers",
            "models": [
                {
                    "lang_code": "en",
                    "model_name": {
                        "spacy": "en_core_web_lg",
                        "transformers": self.transformers_model
                    }
                }
            ]
        }
        provider: NlpEngineProvider = NlpEngineProvider(nlp_configuration=presidio_analyzer_configuration)
        self.analyzer = AnalyzerEngine(
            nlp_engine=provider.create_engine(),
            default_score_threshold=self.acceptance_threshold)
        self.anonymizer = AnonymizerEngine()

    def on_bundle_end(self) -> None:
        pass

    def apply(self, row: Row) -> List[TaggedRow]:
        if self.pii_entity_schema is None:
            self.pii_entity_schema: Schema = Schema([
                SchemaField(self.note_id_col_name, row.get_schema().get_field(self.note_id_col_name).field_type),
                SchemaField("entity_type", FieldType(TypeName.STRING)),
                SchemaField("start_offset", FieldType(TypeName.INT32)),
                SchemaField("end_offset", FieldType(TypeName.INT32)),
                SchemaField("confidence_score", FieldType(TypeName.DOUBLE))
            ])

        ret: List[TaggedRow] = []
        input_text: str = str(row.get_value(self.note_text_col_name))
        identified_pii_entities: List[RecognizerResult] = self.analyzer.analyze(
            text=input_text,
            language='en'
        )
        # Output Identified Entities
        for pii_entity in identified_pii_entities:
            ret.append(TaggedRow(
                'PII Positions',
                Row(self.pii_entity_schema,
                    [
                        row.get_value(self.note_id_col_name),
                        pii_entity.entity_type,
                        pii_entity.start,
                        pii_entity.end,
                        pii_entity.score
                    ])
            ))

        anonymizer_output = self.anonymizer.anonymize(
            text=input_text,
            analyzer_results=identified_pii_entities
        )
        anonymized_text = anonymizer_output.text
        row.set_value(self.note_text_col_name, anonymized_text)
        ret.append(TaggedRow("De-Identified Text", row))
        return ret


class IdentifyPIIComponentDefinition(BackboneComponentDefinition):
    def get_component_def(self) -> BackboneComponent:
        return IdentifyPIIComponent()

    def get_do_fn(self) -> BackboneComponentOneToManyDoFn:
        return IdentifyPIIDoFn()
