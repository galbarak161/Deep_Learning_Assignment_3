from os import path
import spacy
import pandas as pd
import torchtext
import torchtext.data as text_data
from torchtext.legacy.data import Field, BucketIterator, TabularDataset


def tokenize_dataframes(df_train, df_val, df_test):
    df_train.to_json('train.json', orient='records', lines=True)
    df_train.to_csv('train.csv', index=False)
    df_val.to_json('valid.json', orient='records', lines=True)
    df_val.to_csv('valid.csv', index=False)
    df_test.to_json('test.json', orient='records', lines=True)
    df_test.to_csv('test.csv', index=False)

    spacy.load('en_core_web_sm')

    # defines tokenization attributes:
    field_args = dict(tokenize='spacy',
                      init_token='<sos>',
                      eos_token='<eos>',
                      include_lengths=True,
                      lower=True,
                      stop_words=["<", ">", ".", " ", "-"])

    return Field(tokenizer_language="en_core_web_sm", **field_args)


def load_datasets(batch_size: int):
    # Read data txt files
    path_to_dataset = path.join(path.dirname(path.abspath(__file__)), 'data')
    train_path = path.join(path_to_dataset, 'ptb.train.txt')
    validation_path = path.join(path_to_dataset, 'ptb.valid.txt')
    test_path = path.join(path_to_dataset, 'ptb.test.txt')

    train_data = open(train_path, encoding='utf8').read().split('\n')
    val_data = open(validation_path, encoding='utf8').read().split('\n')
    test_data = open(test_path, encoding='utf8').read().split('\n')

    # create dataframes:
    df_train = pd.DataFrame({'data': [line for line in train_data]}, columns=['data'])
    df_val = pd.DataFrame({'data': [line for line in val_data]}, columns=['data'])
    df_test = pd.DataFrame({'data': [line for line in test_data]}, columns=['data'])

    # convert data to json / csv for tokenization
    generic_field = tokenize_dataframes(df_train, df_val, df_test)
    fields = {'data': ('d', generic_field)}

    # create Tabular datasets
    prc_train_data, prc_val_data, prc_test_data = TabularDataset.splits(
        path='', train='train.json', validation='valid.json', test='test.json', format='json', fields=fields
    )

    # create vocabulary
    generic_field.build_vocab(prc_train_data, min_freq=2)

    topn = 30
    print(f'TEXT top{topn}: {generic_field.vocab.itos[0:topn]}\n')

    # create data loaders (bucket iterators):
    dl_train, dl_valid, dl_test = BucketIterator.splits(
        (prc_train_data, prc_val_data, prc_test_data),
        batch_size=batch_size,
        sort=False
    )

    return dl_train, dl_valid, dl_test, generic_field
