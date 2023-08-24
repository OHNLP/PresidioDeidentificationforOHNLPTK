import abc
import json
import random
import re
import time
from datetime import datetime
from typing import Union, List, Callable

from ohnlp.toolkit.backbone.api import BackboneComponentDefinition, BackboneComponent, Row, Schema, SchemaField, \
    FieldType, BackboneComponentOneToManyDoFn, TaggedRow, BackboneComponentOneToOneDoFn
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import RecognizerResult
# Huggingface/Transformers and Spacy Model Support
from transformers import AutoTokenizer, AutoModelForTokenClassification

from transformers_recognizer import TransformersRecognizer, BERT_DEID_CONFIGURATION


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
            if self.note_id_col_name is not None:
                self.note_id_col_name = self.note_id_col_name["sourceColumnName"]
            self.note_text_col_name = resolve_from_json_config(config, 'text_col')
            if self.note_text_col_name is not None:
                self.note_text_col_name = self.note_text_col_name["sourceColumnName"]
            self.transformers_model = resolve_from_json_config(config, 'ner_model_path')
            self.acceptance_threshold = resolve_from_json_config(config, 'acceptance_threshold')

    def on_bundle_start(self) -> None:
        # Download models if necessary
        AutoTokenizer.from_pretrained(self.transformers_model)
        AutoModelForTokenClassification.from_pretrained(self.transformers_model)
        # Setup NLP Engine
        presidio_analyzer_configuration = {
            "nlp_engine_name": "spacy",
            "models": [
                {
                    "lang_code": "en",
                    "model_name": "en_core_web_sm",  # Installed as part of pip dependency direct from github
                }
            ]
        }
        # Inject Transformer Recognizer
        # See: https://huggingface.co/spaces/presidio/presidio_demo/blob/main/presidio_nlp_engine_config.py
        registry: RecognizerRegistry = RecognizerRegistry()
        registry.load_predefined_recognizers()
        transformers_recognizer: TransformersRecognizer = TransformersRecognizer(model_path=self.transformers_model)
        transformers_recognizer.load_transformer(**BERT_DEID_CONFIGURATION)
        provider: NlpEngineProvider = NlpEngineProvider(nlp_configuration=presidio_analyzer_configuration)
        registry.add_recognizer(transformers_recognizer)
        registry.remove_recognizer("SpacyRecognizer")

        self.analyzer = AnalyzerEngine(
            nlp_engine=provider.create_engine(),
            registry=registry,
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
        ret.append(TaggedRow("Text w/ PII Redacted", row))
        return ret


class IdentifyPIIComponentDefinition(BackboneComponentDefinition):
    def get_component_def(self) -> BackboneComponent:
        return IdentifyPIIComponent()

    def get_do_fn(self) -> BackboneComponentOneToManyDoFn:
        return IdentifyPIIDoFn()


recognized_presidio_types: List[str] = ["LOCATION", "PERSON", "ORGANIZATION", "AGE", "PHONE_NUMBER", "EMAIL",
                                        "DATE_TIME", "ZIP", "PROFESSION", "USERNAME", "ID"]


class Synthesizer(abc.ABC):
    @abc.abstractmethod
    def generate(self) -> str:
        pass


class RandomSelectionSynthesizer(Synthesizer):
    def __init__(self, file: str):
        self.selections: list[str] = []
        with open(file, 'r') as f:
            for line in f:
                self.selections.append(line.replace('\n', ''))

    def generate(self) -> str:
        if len(self.selections) > 0:
            return self.selections[random.randrange(0, len(self.selections))]
        else:
            return ""


class NumberFormatSynthesizer(Synthesizer):
    def __init__(self, num_format: str):
        self.base_format = ''
        self.format_decimal_lengths = []
        self.format_include_zero = []

        # Scan for number format/counts
        digit_count_scan = re.compile("\\{(\\d+)}")
        curr_start_pos = 0
        for m in digit_count_scan.finditer(num_format):
            count = m.group(1)
            self.format_include_zero.append(count.startswith('0'))
            self.format_decimal_lengths.append(int(count))
            end_pos, new_start_pos = m.span()
            self.base_format += num_format[curr_start_pos:end_pos]
            self.base_format += '{}'
            curr_start_pos = new_start_pos
        self.base_format = num_format[curr_start_pos:]

    def generate(self) -> str:
        # Generate integer list corresponding to the number of {} args in base_format
        num_format_args: list[str] = []
        for i in range(0, len(self.format_decimal_lengths)):
            num_digits = self.format_decimal_lengths[i]
            digits = []
            for j in range(0, num_digits):
                digits.append(random.randrange(0 if j != 0 or self.format_include_zero[j] else 1, 10))
            num_format_args.append(''.join([str(x) for x in digits]))
        return self.base_format.format(*num_format_args)


class DateFormatSynthesizer(Synthesizer):
    def __init__(self, date_format: str):
        self.date_format = date_format
        self.min_date = time.time() - 315_600_000  # 10 Years

    def generate(self) -> str:
        date_seconds = random.randrange(round(self.min_date), round(time.time()))
        return datetime.fromtimestamp(date_seconds).strftime(self.date_format)


class SynthesizePIIReplacementComponent(BackboneComponent):

    def __init__(self):
        super().__init__()
        self.configstr = None

    def init(self, configstr: Union[str, None]) -> None:
        if configstr is not None:
            self.configstr = configstr

    def to_do_fn_config(self) -> str:
        return self.configstr

    def get_input_tag(self) -> str:
        return "Tagged De-Identified Text"

    def get_output_tags(self) -> List[str]:
        return ["Synthesized Text"]

    def calculate_output_schema(self, input_schema: dict[str, Schema]) -> dict[str, Schema]:
        for key in input_schema:
            schema = input_schema[key]
        return {
            'Synthesized Text': schema
        }


class SynthesizePIIReplacementDoFn(BackboneComponentOneToOneDoFn):

    def __init__(self):
        super().__init__()
        self.synthesizer_config_file = None
        self.synthesizers: dict[str, Synthesizer] = {}
        self.note_text_col_name = None
        self.regex: Union[None, re.Pattern] = None

    def init_from_driver(self, config_json_str: Union[str, None]) -> None:
        config = json.loads(config_json_str)
        self.note_text_col_name = resolve_from_json_config(config, 'text_col')
        if self.note_text_col_name is not None:
            self.note_text_col_name = self.note_text_col_name["sourceColumnName"]
        self.synthesizer_config_file = resolve_from_json_config(config, 'synthesizer_config_file')

    def on_bundle_start(self) -> None:
        with open(self.synthesizer_config_file, 'r') as f:
            synthesizer_config_map = json.load(f)
        # Initialize synthesizers
        for presidio_type in recognized_presidio_types:
            if presidio_type in synthesizer_config_map:
                synthesizer_type = synthesizer_config_map[presidio_type]['type']
                synthesizer_config = synthesizer_config_map[presidio_type]['config']
                synthesizer: Union[Synthesizer, None] = None
                if synthesizer_type == 'TEXT':
                    synthesizer = RandomSelectionSynthesizer(synthesizer_config)
                elif synthesizer_type == 'NUMBER':
                    synthesizer = NumberFormatSynthesizer(synthesizer_config)
                elif synthesizer_type == 'DATETIME':
                    synthesizer = DateFormatSynthesizer(synthesizer_config)
                if synthesizer is not None:
                    self.synthesizers[presidio_type] = synthesizer
        self.regex = re.compile('<(' + '|'.join(self.synthesizers.keys()).upper() + ')>')

    def on_bundle_end(self) -> None:
        self.synthesizers.clear()
        return

    def apply(self, input_row: Row) -> List[Row]:
        text: str = str(input_row.get_value(self.note_text_col_name))
        synthetic_text = ''
        curr_start_pos = 0
        for m in self.regex.finditer(text):
            end_pos, new_start_pos = m.span()
            synthetic_text += text[curr_start_pos:end_pos]
            tag = m.group(1).upper()
            synthetic_value = '<' + tag + '>'  # Default operation is to not replace anything and leave it tagged
            if tag in self.synthesizers:
                synthetic_value = self.synthesizers[tag].generate()
            synthetic_text += synthetic_value
            curr_start_pos = new_start_pos
        synthetic_text += text[curr_start_pos:]
        input_row.set_value(self.note_text_col_name, synthetic_text)
        return [input_row]


class SynthesizePIIReplacementComponentDefinition(BackboneComponentDefinition):
    def get_component_def(self) -> BackboneComponent:
        return SynthesizePIIReplacementComponent()

    def get_do_fn(self) -> Union[BackboneComponentOneToOneDoFn, BackboneComponentOneToManyDoFn]:
        return SynthesizePIIReplacementDoFn()
