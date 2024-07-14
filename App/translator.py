import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


en_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
fa_tokenizer = AutoTokenizer.from_pretrained('bolbolzaban/gpt2-persian')


def main():
    model = tf.keras.models.load_model('./')
    while True:
        source = input('>> ')



class GreedyTranslator(tf.Module):
    def __init__(self, model, fa_tokenizer, *, max_length=50, return_attention_weights=False):
        super(GreedyTranslator, self).__init__()

        self.model = model
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.classifier = model.classifier
        self.fa_tokenizer = fa_tokenizer
        self.max_length = max_length
        self.start_token = fa_tokenizer.cls_token_id
        self.end_token = fa_tokenizer.sep_token_id
        self.pad_token = fa_tokenizer.pad_token_id
        self.return_attention_weights = return_attention_weights

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32)])
    def __call__(self, tokenized_sentence):
        batch_size = tf.shape(tokenized_sentence)[0]
        context = self.encoder(tokenized_sentence)

        decoder_input_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        decoder_input_array = decoder_input_array.write(0, tf.fill([batch_size, ], self.start_token))

        not_finished = tf.ones(batch_size, dtype=tf.float32)

        for i in tf.range(self.max_length):
            decoder_inps = tf.transpose(decoder_input_array.stack())

            outputs = self.decoder(decoder_inps, context, training=False)
            outputs = self.classifier(outputs, training=False)
            outputs = tf.argmax(outputs, axis=-1)
            outputs = tf.cast(outputs[:, -1], tf.float32)

            outputs = outputs * not_finished + (1 - not_finished) * self.pad_token
            not_finished *= tf.cast(tf.logical_not(tf.equal(outputs, self.end_token)), tf.float32)
            outputs = tf.cast(outputs, tf.int32)

            decoder_input_array = decoder_input_array.write(i + 1, outputs)

            if tf.reduce_sum(not_finished) == 0:
                break

        outputs = tf.transpose(decoder_input_array.stack())
        decoder_input_array = decoder_input_array.close()

        if self.return_attention_weights:
            self.model(tokenized_sentence, outputs, training=False)
            attention_weights = self.model.decoder.last_attention_scores
            return outputs, attention_weights

        return outputs


if __name__ == '__main__':
    main()