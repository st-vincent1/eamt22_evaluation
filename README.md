# eamt22_evaluation
Repository containing code and data necessary to evaluate approaches to the EAMT22 English-to-Polish translation task.


### Installation

To run the evaluation, you must first install the necessary packages. It is advisable that you do so in an environment; the instructions below assume creating a Conda environment.

1. Clone this repository:
```bash
git clone https://github.com/st-vincent1/eamt22_evaluation.git
cd eamt22_evaluation
```

2. Create a Conda environment:
```bash
conda create --name eamt22 python=3.8.10
```
3. Install dependencies:
```bash
pip install spacy=2.2.4 tensorflow==2.2.0 keras==2.3.1
```
4. Download the Morfeusz model:
```bash
wget http://zil.ipipan.waw.pl/SpacyPL?action=AttachFile&do=get&target=pl_spacy_model_morfeusz_big-0.1.0.tar.gz
```
5. Install the Morfeusz model:
```bash
python -m pip install pl_spacy_model_morfeusz_big-0.1.0.tar.gz
```

### Evaluation

To run evaluation, given a `[test/dev].hyp` file of hypotheses in Polish for the source file `tester/[test/dev].en`, run:

```bash
python eamt22_evaluate.py --hyp [test/dev].hyp
```

The evaluation takes about 8 minutes on a A100 GPU.


### Citation

If you use the tool contained in this repository, please cite the following paper:


Sebastian T. Vincent, Loïc Barrault, and Carolina Scarton. 2022. 
[Controlling Extra-Textual Attributes about Dialogue Participants: A Case Study of English-to-Polish Neural Machine Translation.](https://aclanthology.org/2022.eamt-1.15/) 
In Proceedings of the 23rd Annual Conference of the European Association for Machine Translation, pages 121–130, Ghent, Belgium. 
European Association for Machine Translation.

Bibkey (ACL Anthology): `vincent-etal-2022-controlling-extra`

