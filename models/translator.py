import torch
from tokenizers import Tokenizer

class GreedyTranslator(torch.nn.Module):
    """
    A neural machine translation module that uses a greedy decoding approach to translate
    input text from English to Persian (Farsi) based on a pre-trained transformer model.
    The class generates translations by selecting the highest-probability token at each step.

    Attributes:
        transformer (torch.nn.Module): The transformer model containing encoder, decoder, and classifier layers used for translation.
        en_tokenizer (Tokenizer): A tokenizer for the English language.
        fa_tokenizer (Tokenizer): A tokenizer for the Persian (Farsi) language.
        max_length (int): The maximum length of the generated output sequence. Defaults to 4000.

    Methods:
        forward(source: list):
            Translates a batch of input texts from English to Persian using greedy decoding.
            Returns the translated sentences in Persian.
    """
    def __init__(self, *,
                 transformer, 
                 en_tokenizer:Tokenizer, 
                 fa_tokenizer:Tokenizer, 
                 max_length:int = 4000):
        super(GreedyTranslator, self).__init__()

        transformer.eval()
        self.encoder = transformer.encoder_layer
        self.decoder = transformer.decoder_layer
        self.classifier = transformer.classifier
        self.en_tokenizer = en_tokenizer
        self.fa_tokenizer = fa_tokenizer
        self.en_pad = en_tokenizer.token_to_id('[PAD]')
        self.cls = fa_tokenizer.token_to_id('[CLS]')
        self.sep = fa_tokenizer.token_to_id('[SEP]')
        self.pad = fa_tokenizer.token_to_id('[PAD]')
        self.max_length = max_length


    def forward(self, source:list):
        """
        Performs translation of a batch of input texts from English to Persian using greedy decoding.

        Args:
            source (list): A list of strings where each string is an English sentence to be translated.

        Returns:
            list[str]: A list of translated Persian sentences corresponding to the input sentences.
        """
        source = [x.lower() for x in source]
        device = next(self.parameters()).device
        source = torch.tensor([x.ids for x in self.en_tokenizer.encode_batch(source)], dtype=torch.int32, device=device)
        mask = source == self.en_pad
        target = torch.tensor([[self.cls]] * source.shape[0], dtype=torch.int32, device=device)
        pending = torch.tensor([1] * source.shape[0], dtype=torch.int32, device=device)
        
        with torch.no_grad():
            context = self.encoder(source, mask=mask)

            for _ in range(self.max_length):
                outputs = self.decoder(target, context, mask=mask)
                outputs = self.classifier(outputs)
                outputs = torch.argmax(outputs, dim=-1)[:, -1]
                outputs = outputs * pending + (1 - pending) * self.pad
                pending = torch.logical_not(torch.eq(outputs, self.sep)).to(dtype=torch.int32) * pending
                target = torch.concat([target, outputs.unsqueeze(1)], dim=-1)

                if torch.sum(pending) == 0:
                    break
        return self.fa_tokenizer.decode_batch(target.cpu().numpy(), skip_special_tokens=True)
    


class BeamSearchTranslator(torch.nn.Module):
    """
    A neural machine translation module that uses beam search to generate translations
    from English to Persian (Farsi) based on a pre-trained transformer model. This class
    supports translation with beam search, allowing for the generation of multiple candidate
    translations and selecting the best one based on a scoring mechanism.

    Attributes:
        transformer (torch.nn.Module): The transformer model containing encoder, decoder, and classifier layers used for translation.
        en_tokenizer (Tokenizer): A tokenizer for the English language.
        fa_tokenizer (Tokenizer): A tokenizer for the Persian (Farsi) language.
        max_length (int): The maximum length of the generated output sequence. Defaults to 4000.
        beam_width (int): The number of beams to use in beam search. Defaults to 10.
        alpha (float): The length penalty factor used to adjust scores based on sequence length. Defaults to 0.7.

    Methods:
        forward(text: str, return_all: bool = False):
            Translates the input text from English to Persian using beam search.
            Optionally returns all candidate translations or just the highest-scoring one.
    """
    def __init__(self, *, 
                 transformer, 
                 en_tokenizer:Tokenizer, 
                 fa_tokenizer:Tokenizer,
                 max_length:int = 4000, 
                 beam_width:int = 10, 
                 alpha:float = 0.7):
        super(BeamSearchTranslator, self).__init__()

        transformer.eval()
        self.encoder = transformer.encoder_layer
        self.decoder = transformer.decoder_layer
        self.classifier = transformer.classifier
        self.en_tokenizer = en_tokenizer
        self.fa_tokenizer = fa_tokenizer
        self.en_pad = en_tokenizer.token_to_id('[PAD]')
        self.cls = fa_tokenizer.token_to_id('[CLS]')
        self.sep = fa_tokenizer.token_to_id('[SEP]')
        self.fa_pad = fa_tokenizer.token_to_id('[PAD]')
        self.max_length = max_length
        self.beam_width = beam_width
        self.alpha = alpha

    def forward(self, text:str, return_all:bool = False):
        """
        Performs translation of the input text from English to Persian using beam search.

        Args:
            text (str): The input English text to be translated.
            return_all (bool, optional): If True, returns all candidate translations generated
                by the beam search. If False, returns only the highest-scoring translation. 
                Defaults to False.

        Returns:
            str or List[str]: The translated Persian text. If `return_all` is False, returns a
            single string of the best translation. If `return_all` is True, returns a list of
            all candidate translations.
        """
        device = next(self.parameters()).device
        text = text.lower()
        source = torch.tensor(self.en_tokenizer.encode(text).ids, dtype=torch.int32, device=device)
        source = source.unsqueeze(0)
        mask = source == self.en_pad

        with torch.no_grad():
            context = self.encoder(source, mask=mask)
            
            dec_inp = torch.ones(1, 1, dtype=torch.int32, device=device) * self.cls
            scores = torch.zeros(1, dtype=torch.float32, device=device)
            pending = torch.ones(1, dtype=torch.int32, device=device)
            completed = []
            completed_scores = []
            beam_width = self.beam_width
    
            for _ in range(self.max_length):
                outputs = self.decoder(dec_inp, context, mask=mask)
                outputs = self.classifier(outputs)
                outputs = torch.nn.functional.log_softmax(outputs, dim=-1)
                outputs = outputs[:, -1, :]
                
                outputs = outputs + scores.unsqueeze(1)
    
                scores, indices = torch.topk(outputs.reshape(-1), k=beam_width)
                chosen, indices = torch.unravel_index(indices=indices, shape=outputs.shape)
                
                pending = pending[chosen]
                indices = (indices * pending + ((1 - pending) * self.fa_pad))
                
                dec_inp = dec_inp[chosen]
                dec_inp = torch.concat([dec_inp, indices.unsqueeze(1)], dim=-1)
                pending = 1 - (indices == self.sep).to(dtype=torch.int32)
                
                completed.extend(dec_inp[pending == 0].cpu().numpy().tolist())
                completed_scores.extend(scores[pending == 0].cpu().numpy().tolist())
                dec_inp = dec_inp[pending == 1]
                scores = scores[pending == 1]
                pending = pending[pending == 1]
                beam_width = len(pending)
    
                if len(pending) == 0:
                    break
    
            scores = torch.tensor(completed_scores, dtype=torch.float32, device=device)
            length = torch.tensor([len(x) for x in completed], dtype=torch.int32, device=device)
            scores = scores * (1 / torch.pow(length, self.alpha))
            argscores = torch.argsort(scores, descending=True)
            completed = [completed[x] for x in argscores]
        if return_all:
            return self.fa_tokenizer.decode_batch(completed)
        return self.fa_tokenizer.decode(completed[0])