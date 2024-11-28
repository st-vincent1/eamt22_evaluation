from fire import Fire
import json
import logging
import os
import re
from tqdm import tqdm

import numpy as np
import spacy

logging.basicConfig(level=logging.INFO)


class Attributes:
    def __init__(self):
        self.types: dict[str, list[str]] = {
            "SpGender": ["<sp:feminine>", "<sp:masculine>"],
            "IlGender": ["<il:feminine>", "<il:masculine>", "<il:mixed>"],
            "IlNumber": ["<singular>", "<plural>"],
            "Formality": ["<formal>", "<informal>"]
        }
        self.attribute_list: list[str] = list(self.types.keys())
        self.type_list: list[str] = [
            a for b in self.types.values() for a in b
        ]
        self.reverse_map: dict[str, str] = {
            "<sp:feminine>": "<sp:masculine>",
            "<sp:masculine>": "<sp:feminine>",
            "<il:feminine>": "<il:masculine>",
            "<il:masculine>": "<il:feminine>",
            "<il:mixed>": "<il:feminine>",
            "<singular>": "<plural>",
            "<plural>": "<singular>",
            "<formal>": "<informal>",
            "<informal>": "<formal>",
            "": ""
        }

    def identify_from_type(self, attribute_type: str) -> str:
        for attribute, types in self.types.items():
            if attribute_type in types:
                return attribute
        logging.error(f"Tried to identify a type which does not exist: {attribute_type}")
        raise ValueError(f"Attribute type does not exist: {attribute_type}")

    def sort_group(self, group: list[str]) -> str:
        return " ".join([type_ for attrib in self.types.keys() for type_ in self.types[attrib] if type_ in group])

    @staticmethod
    def types_to_str(types: dict[str, str | None]) -> str:
        return ",".join(types.get(key, "") for key in ["SpGender", "IlGender", "IlNumber", "Formality"])


class Detector:
    def __init__(self):
        try:
            nlp = spacy.load("pl_spacy_model_morfeusz_big")
            self.nlp = nlp
        except ValueError:
            assert hasattr(self, "nlp")
        with open(os.path.join(os.getcwd(), "data/stopwords"), "r") as f:
            stopwords = f.read().splitlines()
        self.stopwords: list[str] = stopwords
        self.attributes = Attributes()

    def parse_sentence(self, sentence):
        try:
            parsed = self.nlp(sentence)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logging.info(f"{e}; skipping sentence: {sentence}")
            return ""
        return parsed

    def initialise_types(self) -> dict[str, str| None]:
        return {k: None for k in self.attributes.attribute_list}

    def calculate_type_agreement(self, sentences: list[str], en_sentences: list[str], attribute_type: list[str]):
        # Used for evaluation.
        reverse_type = [self.attributes.reverse_map[x] for x in attribute_type]
        # A list of bools depending on whether ith sentence agreed to the ith type
        correct = [
            self.verify_context(sentences[i], en_sentences[i], attribute_type[i]) for i in tqdm(range(len(sentences)))
        ]
        incorrect = [
            self.verify_context(sentences[i], en_sentences[i], reverse_type[i]) for i in tqdm(range(len(sentences)))
        ]

        correct = {
            x: np.sum(np.array(correct)
                      & np.array([t_ in self.attributes.types[x] for t_ in attribute_type]))
            for x in self.attributes.attribute_list
        }
        incorrect = {
            x: np.sum(np.array(incorrect)
                      & np.array([t_ in self.attributes.types[x] for t_ in reverse_type]))
            for x in self.attributes.attribute_list
        }
        # Obtain correct/inocrrect label for SpGender specifically for each sentence

        logging.info(f"Correct assignments:   {correct}\nIncorrect assignments: {incorrect}")
        return correct, incorrect

    def verify_context(self, sentence: str, en_sentence: str, predicted_type: str) -> bool:
        """Verify whether the predicted type matches actual types detected from the given sentences.

        :param sentence: raw sentence in Polish.
        :param en_sentence: raw sentence in English.
        :param predicted_type: type predicted by the model.
        :return: True if type matches actual type, else False.
        """
        attribute = self.attributes.identify_from_type(predicted_type)
        return self.predict_types({"polish": sentence, "english": en_sentence}).get(attribute) == predicted_type

    def predict_types(self, sentence_pair: dict[str, str]) -> dict[str, str]:
        """
        Go through the sentence and identify markers for all phenomena. return them in a list.
        Markers:
        SpGender:   <sp:feminine> or <sp:masculine>
        IlGender:   <il:feminine>, <il:masculine> or <il:mixed>
        IlNumber:   <singular> or <plural>
        Formality:  <formal> or <informal>
        Assumption: if no type is returned for a given phenomenon, then the sentence is ambivalent w.r.t. the phenomenon

        :param sentence_pair:   input data
        :return:            a dictionary of types.
        """
        parsed_polish_sentence = self.parse_sentence(sentence_pair["polish"])
        english_sentence = sentence_pair["english"]
        types = self.initialise_types()

        # 1. Check agreement with speaker's gender.
        types = self.check_speaker_gender(parsed_polish_sentence, types=types)
        # 2. Check formality. If sentence is matched as formal, then return the right features and quit.
        # Lemma suggests formal addressing and no_det makes sure that there are no determinants (e.g. lady vs this lady)
        types, sentence_is_formal = self.check_if_formal(parsed_polish_sentence, english_sentence, types)
        if sentence_is_formal:
            return types

        # 3. If sentence did not match as formal, then keep looking for other interlocutor features.
        # If found, annotate sentence as informal.
        types = self.check_interlocutor(parsed_polish_sentence, types)
        return types

    def annotate(self, data: list[dict]) -> list[str]:
        english = [sample["english"] for sample in data]
        polish = [sample["polish"] for sample in data]
        annotations = []
        for english_, polish_ in tqdm(zip(english, polish)):
            types = self.predict_types({"english": english_, "polish": polish_})
            annotations.append(self.attributes.types_to_str(types))
        return annotations

    def check_speaker_gender(
            self, sentence: spacy.tokens.Doc, types: None | dict[str, str | None] = None
    ) -> dict[str, str | None]:
        if types is None:
            types = self.initialise_types()

        for token in sentence:
            token_feats = token._.feats.split(":")
            head_feats = token.head._.feats.split(":")
            if token.head.pos_ not in ["NOUN", "VERB", "ADJ"]:
                continue

            if "sg" in token_feats and "pri" in token_feats:
                # Past tense and future tense verbs
                if token.head.pos_ == "VERB" and token.dep_ in ["aux:clitic", "aux", "aux:pass"]:
                    types = self.gender_check(token.head, types, "SpGender")

                # Nouns
                if token.head.pos_ == "NOUN" and "inst" in head_feats:
                    if token.dep_ in ["aux:clitic", "cop"]:
                        if self.no_adp(sentence, token.i, token.head.i):
                            if token.head.lemma_.lower() not in self.stopwords:
                                types = self.gender_check(token.head, types, "SpGender")

                # Adjectives
                if token.head.pos_ == "ADJ":
                    if token.dep_ in ["aux:clitic", "aux:pass", "cop", "obl:cmpr", "obl"]:
                        types = self.gender_check(token.head, types, "SpGender")
        return types

    def check_interlocutor(
            self, sentence: spacy.tokens.Doc, types: dict[str, str | None]
    ) -> dict[str, str | None]:
        for token in sentence:
            token_feats = token._.feats.split(":")
            head_feats = token.head._.feats.split(":")
            for number in ("sg", "pl"):
                if number in head_feats and "sec" in head_feats:
                    if token.head.pos_ in ["VERB", "PRON"]:
                        types["IlNumber"] = "<singular>" if number == "sg" else "<plural>"
                        types["Formality"] = "<informal>"
                        if token.pos_ == "ADJ" and number in token_feats:
                            if token.dep_ in ["xcomp:pred", "nsubj", "conj", "nsubj", "iobj", "xcomp",
                                              "amod", "vocative", "obl:cmpr"]:
                                types = self.gender_check(token, types, "IlGender")

                        if token.pos_ == "NOUN":
                            if token.dep_ == "vocative" or (token.dep_ in ["appos", "obj"] and "voc" in token_feats):
                                ner = [a.text for a in sentence.ents]
                                if token.orth_ not in ner:
                                    types = self.gender_check(token, types, "IlGender")
            continue_check = False
            # Your/yours
            if token.lemma_.lower() == "twój":
                types["IlNumber"] = "<singular>"
                types["Formality"] = "<informal>"
            if token.lemma_.lower() == "wasz":
                types["IlNumber"] = "<plural>"
                types["Formality"] = "<informal>"
            for number in ("sg", "pl"):
                if "sec" in token_feats and number in token_feats:
                    if not (token.orth_ == "ś" and sentence[token.i - 1].orth_ in ["czym", "kim"]):
                        types["IlNumber"] = "<singular>" if number == "sg" else "<plural>"
                        types["Formality"] = "<informal>"
                        continue_check = True
            if continue_check:
                # Past tense and future tense verbs
                if token.head.pos_ == "VERB" and token.dep_ in ["aux:clitic", "aux", "aux:pass"]:
                    types = self.gender_check(token.head, types, "IlGender")
                # Nouns
                if token.head.pos_ == "NOUN":
                    if token.dep_ in ["aux:clitic", "cop"]:
                        if self.no_adp(sentence, token.i, token.head.i):
                            if token.head.lemma_.lower() not in self.stopwords:
                                types = self.gender_check(token.head, types, "IlGender")
                # Adjectives
                if token.head.pos_ == "ADJ":
                    # First 3 come from SpGender, obl:cmpr is "takiemu jak ty"
                    if token.dep_ in ["aux:clitic", "aux:pass", "cop", "obl:cmpr", "obl"]:
                        types = self.gender_check(token.head, types, "IlGender")
        return types

    def check_if_formal(
            self, sentence: spacy.tokens.Doc, english_sentence: str, types: dict[str, str | None]
                        ) -> tuple[dict[str, str | None], bool]:
        for token in sentence:
            if token.orth_.lower() == "proszę" and not re.findall(r"please|ask", english_sentence.lower()):
                types["Formality"] = "<formal>"

            if token.lemma_.lower() in ["pan", "pani"] \
                    and self.no_det(sentence, token) \
                    and self.no_appos(sentence, token) \
                    and self.no_title(english_sentence):
                types["Formality"] = "<formal>"
                # Check gender of interlocutor
                types = self.gender_check(token, types, "IlGender")
                # Check number of interlocutor
                number = re.findall(r"sg|pl", token._.feats)[0]
                assert number in ["sg", "pl"]
                types["IlNumber"] = "<singular>" if number == "sg" else "<plural>"
                return types, True

            elif token.lemma_.lower() == "pański":
                types["Formality"] = "<formal>"
                types["IlNumber"] = "<singular>"
                types["IlGender"] = "<il:masculine>"
                return types, True

            if token.lemma_ == "państwo" and self.no_det(sentence, token) and self.no_nation(english_sentence):
                types["Formality"] = "<formal>"
                types["IlNumber"] = "<plural>"
                types["IlGender"] = "<il:mixed>"
                return types, True
        return types, False

    @staticmethod
    def gender_check(token: spacy.tokens.Token, types: dict[str, str | None], attribute: str) -> dict[str, str | None]:
        assert attribute in ["SpGender", "IlGender"]
        prefix = "il" if attribute == "IlGender" else "sp"
        if re.findall(r"m[123]", token._.feats):
            types[attribute] = f"<{prefix}:masculine>"
        if "f" in token._.feats.split(":"):
            types[attribute] = f"<{prefix}:feminine>"
        return types

    @staticmethod
    def no_title(english_sentence: str) -> bool:
        if re.findall(r"lad(ies|y)|gentlem[ea]n|(^| )(sir|mr[ .]|mrs[ .]|ms[ .]|herr)|"
                      r"lord|master|messieurs|dames|monsieur|madam[e ]|ma'am", english_sentence.lower()):
            return False
        return True

    @staticmethod
    def no_det(sentence: spacy.tokens.Doc, token: spacy.tokens.Token) -> bool:
        """'państwo poszli' vs 'ci państwo poszli'. The latter must be recognised as wrong."""
        for t in sentence:
            if t.head == token and t.dep_ == "det":
                return False
        return True

    @staticmethod
    def no_appos(sentence: spacy.tokens.Doc, token: spacy.tokens.Token) -> bool:
        for t in sentence:
            if t.head == token and t.dep_ == "appos" \
                    and "gen" not in t._.feats.split(":"):
                return False
        return True

    @staticmethod
    def no_nation(sentence: str) -> bool:
        if re.findall(
                "(countr|nation|land|state|kingdom|realm|econom|elsewhere|rule)|\b", sentence.lower()
        ):
            return False
        return True

    @staticmethod
    def no_adp(parsed: spacy.tokens.Doc, i: int, j: int) -> bool:
        for x in range(i, j):
            if parsed[x].pos_ == "ADP" and parsed[x].head == parsed[j]:
                return False
        return True

    def test_performance(self):
        logging.info("Running tests.")

        data = read_from_file(os.path.join(os.getcwd(), "data", "detector_testing_sample.json"))
        detector_annotations = self.annotate(data)
        assert calculate_f1(
            detector_annotations, data["manually_labelled_context_csv"]
        ), "F1 score not reached. Make sure that the installation and data paths are correct."

        logging.info("--- Threshold reached. Annotating corpus...")
    

def calculate_f1(predictions, correct_answers):
    stats = {
        "false_positive": 0,
        "false_negative": 0,
        "true_positive": 0,
        "true_negative": 0
    }
    for prediction, correct_answer in zip(predictions, correct_answers):
        prediction = prediction.split(",")
        correct_answer = correct_answer.split(",")
        for i in range(len(prediction)):
            stats["true_positive"] += sum([correct_answer[i] == prediction[i] == ""])
            stats["true_positive"] += sum([correct_answer[i] == prediction[i] and correct_answer[i] != ""])
            stats["false_negative"] += sum([correct_answer[i] != "" and prediction[i] == ""])
            stats["false_positive"] += sum([correct_answer[i] == "" and prediction[i] != ""])

    precision = stats["true_positive"] / (
            stats["true_positive"] + stats["false_positive"]
    ) * 100
    recall = stats["true_positive"] / (
            stats["true_positive"] + stats["false_negative"]
    ) * 100
    f1 = (2 * precision * recall) / (precision + recall)
    logging.info(f"{precision =:.2f}; {recall =:.2f}; {f1 =:.2f}")
    return f1 > 99

def read_from_file(filename):
    with open(filename, "r") as f:
        return json.load(f)

def read_from_file_(filename):
    with open(filename) as f:
        return f.read().splitlines()
        
def main(hyp: str, split: str = "test") -> None:
    detector = Detector()
    attributes = Attributes()
    assert split in ["dev", "test"], "Please choose 'dev' or 'test' as the split"

    # Evaluate on a sample
    detector.test_performance()

    attribute_counts = {
        x: [[], []] for x in attributes.attribute_list
    }
    dataset = read_from_file(os.path.join(os.getcwd(), "data", f"{split}.json"))

    with open(os.path.join(os.getcwd(), hyp)) as f:
        hypotheses = f.read().splitlines()
        dataset["hypothesis"] = hypotheses

    for idx, row in enumerate(dataset):
        attribute = attributes.identify_from_type(row["marking"])
        attribute_counts[attribute][0] += row["polish"]
        attribute_counts[attribute][1] += row["hypothesis"]
    agreement_with_correct_attribute, agreement_with_incorrect_attribute = detector.calculate_type_agreement(
        dataset["hypothesis"], dataset["english"], dataset["marking"]
    )
    results = []
    for att in attributes.attribute_list:
        result = agreement_with_correct_attribute[att] / (
                agreement_with_correct_attribute[att] + agreement_with_incorrect_attribute[att]
        ) * 100
        logging.info(f"Result for attribute {att}: {result:.2f}")
        results.append(result)
    logging.info(f"Average result: {sum(results) / len(results):.2f} %")


if __name__ == "__main__":
    Fire(main)