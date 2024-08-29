import torch

class GreedyTranslator(torch.nn.Module):
    def __init__(self, transformer, en_tokenizer, fa_tokenizer, max_length=4000):
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