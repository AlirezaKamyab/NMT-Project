import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

from tokenizers import Tokenizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset

def load_data(filename:str, 
              seed:int = 123, 
              return_tokenized:bool = False, 
              en_tokenizer:Tokenizer = None, 
              fa_tokenizer:Tokenizer = None):
    
    """
    Loads data from tab-separeted file
    args:
        filename: filename to the dataset which should a tab-separated text file
        seed: random state for doing train-test split
        return_tokenized: tokenizes the English and Persian with their associate tokenizers
        en_tokenizer: tokenizer for English used when "return_tokenized=True"
        fa_tokenizer: tokenizer for Persian used when "return_tokenized=True"

    returns:
        (english_train, persian_train), (english_test, persian_test)
    """

    with open(filename, 'r') as file:
        file = file.read().split('\n')
        file = [x.split('\t') for x in file if x.strip() != '']

    train_file, test_file = train_test_split(file, test_size=0.1, random_state=seed)

    english_train = [x[0].lower() for x in train_file]
    persian_train = [x[1] for x in train_file]

    english_test = [x[0].lower() for x in test_file]
    persian_test = [x[1] for x in test_file]

    if return_tokenized:
        en_tokenizer.no_padding()
        fa_tokenizer.no_padding()
        english_tokenized_train = [x.ids for x in en_tokenizer.encode_batch(english_train)]
        persian_tokenized_train = [x.ids for x in fa_tokenizer.encode_batch(persian_train)]

        english_tokenized_test = [x.ids for x in en_tokenizer.encode_batch(english_test)]
        persian_tokenized_test = [x.ids for x in fa_tokenizer.encode_batch(persian_test)]
        en_tokenizer.enable_padding()
        fa_tokenizer.enable_padding()

        return (english_tokenized_train, persian_tokenized_train), (english_tokenized_test, persian_tokenized_test)
    
    return (english_train, persian_train), (english_test, persian_test)



class TranslationDataset(Dataset):
    def __init__(self, en, fa):
        super(TranslationDataset, self).__init__()
        self.en = en
        self.fa = fa

    def __len__(self):
        return len(self.en)


    def __getitem__(self, idx):
        en = self.en[idx]
        fa = self.fa[idx]

        return en, fa


def get_pipelines(english_train, 
                  persian_train,
                  english_test, 
                  persian_test,
                  english_tokenizer:Tokenizer,
                  persian_tokenizer:Tokenizer,
                  batch_size:int,
                  num_workers:int,
                  prefetch_factor:int,
                  pin_memory:bool = True):
    """
    Builds pipelines for train and test
    args:
        english_train: English data used for train
        persian_train: Persian data used for train
        english_test: English data used for test
        persian_test: Persian data used for test
        english_tokenizer: English tokenizer used to tokenize English text. Note that padding should be enabled
        persian_tokenizer: Persian tokenizer used to tokenize Persian text. Note that padding should be enabled
        batch_size: integer specifying the batch size
        num_workers: integer specifying the number of workers to work in parallel
        prefetch_factor: integer specifying how many batches each worker should prefetch
        pin_memory: boolean, if using cuda devices, it should be set to True

    returns:
        trainloader, testloader
    """

    def collate_fn(batch):
        en, fa = zip(*batch)
        en_batch = [x.ids for x in english_tokenizer.encode_batch(en)]
        fa_batch = [x.ids for x in persian_tokenizer.encode_batch(fa)]
        en_batch = torch.tensor(en_batch, dtype=torch.int32)
        fa_batch = torch.tensor(fa_batch, dtype=torch.int32)
        return en_batch, fa_batch
    
    trainloader = DataLoader(dataset=TranslationDataset(english_train, persian_train), 
                        collate_fn=collate_fn,
                        batch_size=batch_size, 
                        shuffle=True, 
                        num_workers=num_workers, 
                        prefetch_factor=prefetch_factor, 
                        pin_memory=pin_memory)

    testloader = DataLoader(dataset=TranslationDataset(english_test, persian_test), 
                            collate_fn=collate_fn,
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=num_workers, 
                            prefetch_factor=prefetch_factor, 
                            pin_memory=pin_memory)
    
    return trainloader, testloader
        