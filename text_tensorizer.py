import pickle

import torch
import torchtext

torchtext.disable_torchtext_deprecation_warning()

from typing import List
from collections import Counter, OrderedDict
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer

class TextTensorizer:

    tokenizer = get_tokenizer("spacy","en_core_web_sm")
    
    @classmethod
    def text_to_tensor(self, lang_vocab,
                       doc: str | List[str], 
                       tokenize: bool=True ) -> torch.Tensor: 
        
        tokens = doc if not tokenize else self.tokenizer(doc)
        
        text_tensor = [lang_vocab[token] for token in tokens]
        text_tensor = torch.tensor(text_tensor, dtype=torch.long)

        return text_tensor

    @classmethod
    def build_vocab(cls,corpus: List[str], 
                    specials: List[str], 
                    default_token: str=None,
                    output_directory: str=None,
                    min_freq: int=1):
        counter = Counter()
        for text in corpus:
            tokens = cls.tokenizer(text)
            counter.update(tokens)

        sorted_by_freq_tuples = sorted(counter.items(), 
                                       key=lambda x: x[1], 
                                       reverse=True)
        
        ordered_dict = OrderedDict(sorted_by_freq_tuples)    
        result = vocab(ordered_dict, 
                       specials=specials,
                       min_freq=min_freq)

        if default_token is not None and default_token in specials:
            vocab_stoi = result.get_stoi()
            result.set_default_index(vocab_stoi[default_token])

        if output_directory is not None:
            cls.save_vocab(result, output_directory)

        return result.get_itos(), result.get_stoi()
    
    # Saves ITOS and STOI separately (for compatibility)
    @staticmethod
    def save_vocab(lang_vocab, output_directory: str):

        with open(f"{output_directory}vocab_itos.pickle", "wb+") as f:
            pickle.dump(lang_vocab.get_itos(), f)
        with open(f"{output_directory}vocab_stoi.pickle", "wb+") as f:
            pickle.dump(lang_vocab.get_stoi(), f)