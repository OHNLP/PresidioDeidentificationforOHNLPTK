import json
from typing import Union, List

from ohnlp.toolkit.backbone.api import BackboneComponentDefinition, BackboneComponent, Row, Schema, SchemaField, \
    FieldType, BackboneComponentOneToManyDoFn, TaggedRow, BackboneComponentOneToOneDoFn
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import RecognizerResult
# Huggingface/Transformers and Spacy Model Support
from transformers import AutoTokenizer, AutoModelForTokenClassification


def resolve_from_json_config(config, field: str):
    if field in config:
        return config[field]
    else:
        return None


class IdentifyPIIComponent(BackboneComponent):
    def __init__(self):
        super().__init__()
        self.config = None
        self.note_id_column_name = None
        self.input_col = None

    def init(self, configstr: Union[str, None]) -> None:
        if configstr is not None:
            self.config = configstr
            configobj = json.loads(self.config)
            self.note_id_column_name = resolve_from_json_config(configobj, 'id_col')
            if self.note_id_column_name is not None:
                self.note_id_column_name = self.note_id_column_name["sourceColumnName"]
            self.input_col = resolve_from_json_config(configobj, 'text_col')

    def to_do_fn_config(self) -> str:
        return self.config

    def get_input_tag(self) -> str:
        return "Documents with PII"

    def get_output_tags(self) -> List[str]:
        return ["PII Positions", "Text w/ PII Redacted"]

    def calculate_output_schema(self, input_schema: dict[str, Schema]) -> dict[str, Schema]:
        pii_entity_schema: Union[Schema, None] = None
        deid_text_schema: Union[Schema, None] = None
        for key in input_schema:
            schema = input_schema[key]
            pii_entity_schema = Schema([
                SchemaField(
                    self.note_id_column_name,
                    schema.get_field(self.note_id_column_name).get_field_type()
                ),
                SchemaField("entity_type", FieldType("STRING")),
                SchemaField("start_offset", FieldType("INT32")),
                SchemaField("end_offset", FieldType("INT32")),
                SchemaField("confidence_score", FieldType("DOUBLE"))
            ], self.gateway)
            deid_text_schema = schema
        return {
            "PII Positions": pii_entity_schema,
            "Text w/ PII Redacted": deid_text_schema
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

    def init_from_driver(self, config_json_str: Union[str, None]) -> None:
        if config_json_str is not None:
            config = json.loads(config_json_str)
            self.note_id_col_name = resolve_from_json_config(config, 'id_col')
            self.note_text_col_name = resolve_from_json_config(config, 'text_col')
            self.transformers_model = resolve_from_json_config(config, 'ner_model_path')
            self.acceptance_threshold = resolve_from_json_config(config, 'acceptance_threshold')

    def on_bundle_start(self) -> None:
        # Download models
        AutoTokenizer.from_pretrained(self.transformers_model)
        AutoModelForTokenClassification.from_pretrained(self.transformers_model)
        presidio_analyzer_configuration = {
            "nlp_engine_name": "transformers",
            "models": [
                {
                    "lang_code": "en",
                    "model_name": {
                        "spacy": "en_core_web_lg",  # Installed as part of
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
                SchemaField(self.note_id_col_name, row.get_schema().get_field(self.note_id_col_name).get_field_type()),
                SchemaField("entity_type", FieldType("STRING")),
                SchemaField("start_offset", FieldType("INT32")),
                SchemaField("end_offset", FieldType("INT32")),
                SchemaField("confidence_score", FieldType("DOUBLE"))
            ], self.gateway)

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


class ReplacePIIComponentDefinition(BackboneComponentDefinition):
    def get_component_def(self) -> BackboneComponent:
        pass

    def get_do_fn(self) -> Union[BackboneComponentOneToOneDoFn, BackboneComponentOneToManyDoFn]:
        pass
