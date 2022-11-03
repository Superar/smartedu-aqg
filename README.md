# SmartEDU - Automatic Question Generation

This repository contains the AQG tool developed within the SmartEDU project.

## Installing dependencies

The python version for this project is Python 3.9.

The first files are related to the python required packages. `Pipfile` and `Pipfile.lock` for `pipenv` and `requirements.txt` for vanilla `pip`.

To install using `pipenv`, run

```shell
pipenv install
```

To install using `pip`, run

```shell
pip install -r requirements.txt
```

## Results and Evaluation

Both folders `evaluation` and `results` relate to the up-to-date experiments with two different similarity measures: BERT and TF-IDF.

## Experiment scripts

The commands to run each experiment are present in the `scripts` folder alongside some python utility scripts needed during the project (e.g. converting excel spreadsheets to plain text files or plotting the results evaluation metrics).

---

## Running the project

The `smartedu-aqg` folder contains the AQG tool. It is implemented in a way that it can be used from command line, through python modules, or through a developed visual interface with streamlit. All this information will be further discussed below.

The task is divided into two separate tasks:

- Sentence selection - As a long text may not be ideal for the generation models, we decided to select a subset of sentences according to some similarity metrics with the given answer. These methods are implemented in the `smartedu-aqg/similarity` folder.
- Question generation - These are the question generation models, which create a question given an input text and an answer. They must be implemented in the `smartedu-aqg/methods` folder as python modules. To do so, they must be in a subfolder containing necessarily a `__init__.py` file and a `__main__.py` file.


### Using command line interface

To run the (currently) only existing method, simply run its python module using the `-m` option.

```shell
python -m smartedu-aqg.methods.bert -f FILE -a ANSWERS -s {tfidf, bert} -o OUTPUT [--tfidf TFIDF] [--model MODEL] [--simmodel SIMMODEL]
```

To get more information about each parameter, please run:

```shell
python -m smartedu-aqg.methods.bert --help
```

### Using GUI

The user interface is implemented in the script `smartedu-aqg/app.py` using [ Streamlit ](https://docs.streamlit.io/library/api-reference). To run it, simply execute:

```shell
streamlit run smartedu-aqg/app.py
```
