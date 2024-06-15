#!/usr/bin/env python

# # Neural Machine Translation with Transformers


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '20'

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import re
import unicodedata
import matplotlib.ticker as ticker
from transformers import BertTokenizer, AutoTokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers import TextVectorization
import time

device = tf.config.experimental.list_physical_devices('GPU')[0]
print(device)


en_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
fa_tokenizer = AutoTokenizer.from_pretrained('bolbolzaban/gpt2-persian')


def create_dataset(file_path, num_examples=None):
    en_fa = []
    cnt = 0
    with open(file_path, 'r') as file:
        for line in file.readlines():
            if num_examples is not None and cnt >= num_examples: break
            line = line.split('\t')[:2]
            en = line[0]
            fa = line[1]
            fa = re.sub('\u200c', ' ', fa)
            en_fa.append([en, fa])
            cnt += 1
            if cnt % 100 == 0:
                print(f'\rRead {cnt:>5}', end='')

    print('\nRead', cnt)
    return zip(*en_fa)


def sort_dataset(source, target):
    xy = sorted(zip(source, target), key=lambda x: (len(x[0].split(' ')), len(x[1].split(' ')))) 
    x = [x[0] for x in xy]
    y = [y[1] for y in xy]
    return x, y


### Loading the dataset


NUM_SAMPLES = 2_000_000
FILE_PATH = './Data/combined.txt'
src_lang, targ_lang = create_dataset(FILE_PATH, NUM_SAMPLES)


src_train, src_val, targ_train, targ_val = train_test_split(src_lang, targ_lang, test_size=0.2, shuffle=True)
src_train, targ_train = sort_dataset(src_train, targ_train)
src_val, targ_val = sort_dataset(src_val, targ_val)
print('Source Train examples:', len(src_train))
print('Source Validation examples:', len(src_val))


### Create the pipeline


def en_vectorization(text):
    return en_tokenizer(text, max_length=50, truncation=True, padding='max_length')['input_ids']

def fa_vectorization(text):
    return fa_tokenizer(text, max_length=50, truncation=True, padding='max_length')['input_ids']

start = time.time()
tokenized_src_train = [en_vectorization(x) for x in src_train]
print(f'src train done in {time.time() - start:>4.4f}')
tokenized_targ_train = [fa_vectorization(x) for x in targ_train]
print(f'targ train done in {time.time() - start:>4.4f}')

tokenized_src_val = [en_vectorization(x) for x in src_val]
print(f'src validation done in {time.time() - start:>4.4f}')
tokenized_targ_val = [fa_vectorization(x) for x in targ_val]
print(f'targ validation done in {time.time() - start:>4.4f}')


AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 128
STEPS_PER_EPOCH = len(src_train) // BATCH_SIZE
BUFFER_SIZE = 5000

def preprocess(src, targ):
    src = tf.cast(src, tf.int32)
    src = tf.reshape(src, shape=(-1,))
    targ = tf.cast(targ, tf.int32)
    targ = tf.reshape(targ, shape=(-1,))
    return src, targ

train_ds = tf.data.Dataset.from_tensor_slices((tokenized_src_train, tokenized_targ_train))
train_ds = train_ds.map(preprocess,
                        num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache()
train_ds = train_ds.shuffle(BUFFER_SIZE)
train_ds = train_ds.padded_batch(BATCH_SIZE, 
                                 drop_remainder=True, 
                                 padding_values=(en_tokenizer.pad_token_id, fa_tokenizer.pad_token_id))
train_ds = train_ds.shuffle(STEPS_PER_EPOCH)
train_ds = train_ds.prefetch(AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((tokenized_src_val, tokenized_targ_val))
test_ds = test_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.cache()
test_ds = test_ds.padded_batch(BATCH_SIZE, 
                               drop_remainder=True, 
                               padding_values=(en_tokenizer.pad_token_id, fa_tokenizer.pad_token_id))
test_ds = test_ds.prefetch(AUTOTUNE)

print('Loading cache...')
for i, (x, y) in enumerate(train_ds): 
    print(f'\r{i:>5}', end='')
print()
for i, (x, y) in enumerate(test_ds): 
    print(f'\r{i:>5}', end='')
print()

### Define the model


from keras.layers import Layer, Dropout, Dense, MultiHeadAttention, Add, LayerNormalization, Embedding, Input
from keras.models import Model
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import Metric
from keras.optimizers import Adam


#### Positional Embedding


def positional_encoding(length, depth):
    depth = depth / 2

    positions = tf.range(0, length, dtype='float32')[..., None]
    depths = tf.range(depth)[None, ...] / depth

    angle_rates = 1 / (10000 ** depths)
    angle_rads = positions * angle_rates

    pos_encoding = tf.concat([tf.math.sin(angle_rads), tf.math.cos(angle_rads)], axis=-1)
    
    return pos_encoding

pos_enc = positional_encoding(2048, 512)

plt.pcolormesh(pos_enc.numpy().T, cmap='inferno');
plt.colorbar()


class PositionalEmbedding(Layer):
    def __init__(self, vocab_size, d_model, max_position=2048):
        super(PositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.embedding = Embedding(input_dim=vocab_size, output_dim=d_model, mask_zero=True, name='embedding')
        self.pos_encoding = positional_encoding(max_position, d_model)


    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)
    
    def call(self, inputs):
        length = tf.shape(inputs)[1]
        x = self.embedding(inputs)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


#### Base Attention


class BaseAttention(Layer):
    def __init__(self, **kwargs):
        super(BaseAttention, self).__init__()

        self.mha = MultiHeadAttention(**kwargs)
        self.add = Add(name='add')
        self.layernorm = LayerNormalization(name='layernorm')


#### Cross Attention



class CrossAttention(BaseAttention):
    def __init__(self, **kwargs):
        super(CrossAttention, self).__init__(**kwargs)


    def call(self, query, context, training=False):
        # query has the shape [B, dec_seq_len, d_model]
        # context has the shape [B, enc_seq_len, d_model]
        outputs, attention_scores = self.mha(query=query, 
                           key=context, 
                           value=context, 
                           training=training,
                           return_attention_scores=True)

        # cache attention scores
        self.attention_scores = attention_scores

        outputs = self.add([query, outputs])
        outputs = self.layernorm(outputs)

        return outputs


#### Self Attention



class SelfAttention(BaseAttention):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)


    def call(self, inputs, training=False):
        # inputs has the shape [B, enc_seq_len, d_model]
        outputs = self.mha(query=inputs,
                           key=inputs,
                           value=inputs, 
                           training=training)

        # adds residual connection
        outputs = self.add([inputs, outputs])

        outputs = self.layernorm(outputs)
        return outputs


#### Masked Multi-Head attention


class MaskedMultiHeadAttention(BaseAttention):
    def __init__(self, **kwargs):
        super(MaskedMultiHeadAttention, self).__init__(**kwargs)


    def call(self, inputs, training=False):
        # inputs has the shape [B, dec_seq_len, d_model]
        outputs = self.mha(query=inputs, 
                           key=inputs, 
                           value=inputs, 
                           training=training, 
                           use_causal_mask=True)

        # adds residual connection
        outputs = self.add([inputs, outputs])

        outputs = self.layernorm(outputs)
        return outputs


#### Feed Forward


class FeedForward(Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super(FeedForward, self).__init__()

        self.d_model = d_model
        self.dff = dff
        self.dropout_rate = dropout_rate

        self.W1 = Dense(dff, activation='relu')
        self.W2 = Dense(d_model, activation='linear')

        if dropout_rate > 0.0:
            self.dropout = Dropout(dropout_rate, name='dropout')

        self.add = Add(name='add')
        self.layernorm = LayerNormalization(name='layernorm')


    def call(self, inputs, training=False):
        # inputs has the shape [B, Tq, d_model]
        outputs = self.W1(inputs)
        outputs = self.W2(outputs)

        if self.dropout_rate > 0.0:
            outputs = self.dropout(outputs, training=training)

        # adds residual connection
        outputs = self.add([inputs, outputs])
        outputs = self.layernorm(outputs)
        return outputs


#### Encoder Layer


class EncoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.dff = dff
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.self_attention = SelfAttention(num_heads=num_heads, 
                                            key_dim=d_model, 
                                            dropout=dropout_rate)

        self.ffn = FeedForward(d_model=d_model, 
                               dff=dff, 
                               dropout_rate=dropout_rate)


    def call(self, inputs, training=False):
        # inputs has the shape [B, enc_seq_len, d_model]
        outputs = self.self_attention(inputs, training=training)
        outputs = self.ffn(outputs, training=training)
        return outputs


#### Encoder


class Encoder(Layer):
    def __init__(self, *, d_model, num_heads, dff, N, vocab_size, dropout_rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.N = N
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate

        self.positional_embedding = PositionalEmbedding(vocab_size=vocab_size, 
                                                        d_model=d_model)

        self.stacks = [EncoderLayer(d_model=d_model, 
                                    num_heads=num_heads, 
                                    dff=dff, 
                                    dropout_rate=dropout_rate) for _ in range(N)]

        if dropout_rate > 0.0:
            self.dropout = Dropout(dropout_rate, name='dropout')


    def call(self, inputs, training=False):
        # inputs has the shape [B, seq_len]
        x = self.positional_embedding(inputs)
        if self.dropout_rate > 0.0:
            x = self.dropout(x, training=training)

        for n in range(self.N):
            x = self.stacks[n](x, training=training)

        return x


#### Decoder Layer


class DecoderLayer(Layer):
    def __init__(self, d_model, dff, num_heads, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.num_heads = num_heads
        self.dff = dff
        self.d_model = d_model
        self.dropout_rate = dropout_rate

        self.masked_attention = MaskedMultiHeadAttention(num_heads=num_heads, 
                                                         key_dim=d_model, 
                                                         dropout=dropout_rate)

        self.cross_attention = CrossAttention(num_heads=num_heads, 
                                              key_dim=d_model, 
                                              dropout=dropout_rate)

        self.ffn = FeedForward(d_model=d_model, 
                               dff=dff, 
                               dropout_rate=dropout_rate)


    def call(self, inputs, context, training=False):
        # inputs has the shape [B, dec_seq_len, d_model]
        # context has the shape [B, enc_seq_len, d_model]
        outputs = self.masked_attention(inputs, training=training)
        outputs = self.cross_attention(query=outputs, 
                                       context=context, 
                                       training=training)

        # cache the attention scores
        self.last_attention_scores = self.cross_attention.attention_scores

        outputs = self.ffn(outputs, training=training)
        return outputs


#### Decoder


class Decoder(Layer):
    def __init__(self, *, d_model, num_heads, dff, N, vocab_size, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.N = N
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate

        if dropout_rate > 0.0:
            self.dropout = Dropout(dropout_rate, name='dropout')

        self.positional_embedding = PositionalEmbedding(vocab_size=vocab_size, 
                                                        d_model=d_model)

        self.stacks = [DecoderLayer(d_model=d_model, 
                                    num_heads=num_heads, 
                                    dff=dff, 
                                    dropout_rate=dropout_rate) for _ in range(N)]

    
    def call(self, inputs, context, training=False):
        # inputs has the shape [B, dec_seq_len]
        # context has the shape [B, enc_seq_len, d_model]
        x = self.positional_embedding(inputs)
        if self.dropout_rate > 0.0:
            x = self.dropout(x, training=training)

        for n in range(self.N):
            x = self.stacks[n](x, context=context, training=training)


        # cache attention scores for plotting
        self.last_attention_scores = self.stacks[-1].last_attention_scores

        return x


#### Transformer Architecture


class Transformer(Model):
    def __init__(self, *, d_model, 
                 dff, 
                 num_heads, 
                 N, 
                 source_vocab_size, 
                 target_vocab_size, 
                 dropout_rate=0.1):
        
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.dff = dff
        self.num_heads = num_heads
        self.N = N
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.dropout_rate = dropout_rate

        self.encoder = Encoder(d_model=d_model, 
                               num_heads=num_heads, 
                               dff=dff, 
                               N=N, 
                               vocab_size=source_vocab_size, 
                               dropout_rate=dropout_rate)


        self.decoder = Decoder(d_model=d_model, 
                               num_heads=num_heads, 
                               dff=dff, 
                               N=N, 
                               vocab_size=target_vocab_size, 
                               dropout_rate=dropout_rate)

        self.classifier = Dense(units=target_vocab_size, name='classifier')


    def build(self, input_shapes):
        source_shape, target_shape = input_shapes
        source = Input(shape=source_shape)
        target = Input(shape=target_shape)
        self.call(source, target)
        self.built = True


    def call(self, source, target, training=False):
        context = self.encoder(source, training=training)
        outputs = self.decoder(target, context=context, training=training)
        logits = self.classifier(outputs, training=training)

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            del logits._keras_mask
        except AttributeError:
            pass


        return logits

N = 6
d_model = 256
dff = 1024
num_heads = 8
dropout_rate = 0.1

VOCAB_SRC_SIZE = en_tokenizer.vocab_size
VOCAB_TARG_SIZE = fa_tokenizer.vocab_size

transformer = Transformer(d_model=d_model, 
                          num_heads=num_heads, 
                          dff=dff,
                          N=N,  
                          source_vocab_size=VOCAB_SRC_SIZE, 
                          target_vocab_size=VOCAB_TARG_SIZE,                           
                          dropout_rate=dropout_rate)

transformer.build(input_shapes=[(None, ), (None, )])
transformer.summary()


### Training


loss_object = SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def masked_loss_function(y_true, pred):
    mask = tf.logical_not(tf.equal(y_true, fa_tokenizer.pad_token_id))
    loss = loss_object(y_true, pred)
    mask = tf.cast(mask, loss.dtype)
    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


class MaskedAccuracy(Metric):
    def __init__(self, name='masked_accuracy', **kwargs):
        super(MaskedAccuracy, self).__init__(name=name, **kwargs)

        self.total = self.add_weight(name='total', 
                                     shape=(), 
                                     dtype='float32', 
                                     initializer='zeros')
        
        self.count = self.add_weight(name='count', 
                                     shape=(), 
                                     dtype='float32', 
                                     initializer='zeros')


    def update_state(self, y_true, pred, sample_weights=None):
        # pred = [B, dec_seq_len, target_vocab_size]
        # y_true = [B, dec_seq_len]
        y_pred = tf.argmax(pred, axis=-1)
        y_pred = tf.cast(y_pred, y_true.dtype)
        mask = tf.logical_not(tf.equal(y_true, fa_tokenizer.pad_token_id))

        match = tf.equal(y_pred, y_true)
        match = tf.logical_and(match, mask)

        match = tf.cast(match, tf.float32)
        mask = tf.cast(mask, tf.float32)

        self.total.assign_add(tf.reduce_sum(match))
        self.count.assign_add(tf.reduce_sum(mask))


    def result(self):
        return self.total / self.count if self.count != 0 else 0

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


def get_scheduler(d_model, warmup_steps=4000):
    def get_learning_rate(step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * tf.math.pow(warmup_steps * 1.0, -1.5)

        return tf.math.rsqrt(d_model * 1.0) * tf.minimum(arg1, arg2)

    return get_learning_rate


EPOCHS = 30
scheduler = get_scheduler(d_model, warmup_steps=int(0.15 * STEPS_PER_EPOCH * EPOCHS))
optimizer = Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9)

train_acc = MaskedAccuracy(name='train_acc')
valid_acc = MaskedAccuracy(name='valid_acc')



@tf.function(reduce_retracing=True, input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.int32), 
                                                     tf.TensorSpec(shape=[None, None], dtype=tf.int32)])
def training_step(source, target):
    loss = 0
    y_true = target[:, 1:]
    target = target[:, :-1]

    with tf.GradientTape() as tape:
        pred = transformer(source, target, training=True)
        loss = masked_loss_function(y_true, pred)
        train_acc.update_state(y_true, pred)

    variables = transformer.trainable_variables
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(grads, variables))

    return loss


@tf.function(reduce_retracing=True, input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.int32), 
                                                     tf.TensorSpec(shape=[None, None], dtype=tf.int32)])
def validation_step(source, target):
    loss = 0
    y_true = target[:, 1:]
    target = target[:, :-1]
    pred = transformer(source, target, training=False)
    loss = masked_loss_function(y_true, pred)
    valid_acc.update_state(y_true, pred)
    return loss


train_writer = tf.summary.create_file_writer(logdir='./logs/train')
test_writer = tf.summary.create_file_writer(logdir='./logs/test')


train_mean_losses = []
train_accs = []
valid_mean_losses = []
valid_accs = []
total_steps = 0

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1:>3}')
    train_acc.reset_states()
    valid_acc.reset_states()
    train_mean_loss = 0
    valid_mean_loss = 0
    start = time.time()

    for step, (source, target) in enumerate(train_ds):
        learning_rate = scheduler(total_steps)
        optimizer.learning_rate = learning_rate
        loss = training_step(source, target)
        acc = train_acc.result()
        
        train_mean_loss = train_mean_loss + (1 / (step + 1)) * (loss - train_mean_loss)
        end = time.time()
        total_steps += 1
        print(f'\r{int(end - start):>4} sec | step {step:>4}\tloss {train_mean_loss:>3.4f}\taccuracy {acc:3.2f}', end='')

        with train_writer.as_default():
            tf.summary.scalar('Loss_Per_Step', train_mean_loss, step=total_steps)
            tf.summary.scalar('Accuracy_Per_Step', acc, step=total_steps)
            
    print()
    train_mean_losses.append(train_mean_loss)
    train_accs.append(acc)

    with train_writer.as_default():
        tf.summary.scalar('Loss_Per_Epoch', train_mean_loss, step=epoch)
        tf.summary.scalar('Accuracy_Per_Epoch', acc, step=epoch)
        

    for step, (source, target) in enumerate(test_ds):
        loss = validation_step(source, target)
        acc = valid_acc.result()

        valid_mean_loss = valid_mean_loss + (1 / (step + 1)) * (loss - valid_mean_loss)
        valid_accs.append(acc)
        end = time.time()
        print(f'\r{int(end - start):>4} sec | step {step:>4}\tloss {valid_mean_loss:>3.4f}\taccuracy {acc:3.2f}', end='')
    
    valid_mean_losses.append(train_mean_loss)
    valid_accs.append(acc)

    with test_writer.as_default():
        tf.summary.scalar('Loss_Per_Epoch', valid_mean_loss, step=epoch)
        tf.summary.scalar('Accuracy_Per_Epoch', acc, step=epoch)
    
    transformer.save_weights(f'./{epoch + 1}.h5')
    print('\n')


### Translator Model


class Translator(tf.Module):
    def __init__(self, model, en_tokenizer, fa_tokenizer, max_length=10):
        super(Translator, self).__init__()

        self.en_tokenizer = en_tokenizer
        self.fa_tokenizer = fa_tokenizer
        self.max_length = max_length
        self.encoder = model.layers[0]
        self.decoder = model.layers[1]
        self.classifier = model.layers[2]

    def tokenize_source(self, sentence):
        return self.en_tokenizer.encode(sentence.numpy().decode('utf-8'), add_special_tokens=True)


    #@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)])
    def __call__(self, sentence):
        # sentences [,]
        batch_size = 1
        sentence = self.tokenize_source(sentence)
        sentence = tf.cast(sentence, tf.int32)
        sentence = tf.expand_dims(sentence, axis=0)

        # context has the shape [B, max_seq_len, d_model]
        context = self.encoder(sentence, training=False)

        # dec_inp has the shape [B, 1]
        dec_inp = tf.TensorArray(dtype='int32', dynamic_size=True, size=1)
        dec_inp = dec_inp.write(0, self.fa_tokenizer.cls_token_id)

        for i in tf.range(1, self.max_length):
            dec_inp_model = tf.expand_dims(dec_inp.stack(), 0)
            dec_outputs = self.decoder(dec_inp_model, context, training=False)
            dec_outputs = self.classifier(dec_outputs, training=False)
            dec_outputs = dec_outputs[0, i - 1, :]
            dec_outputs = tf.argmax(dec_outputs, axis=-1)
            dec_outputs = tf.cast(dec_outputs, tf.int32)
            dec_outputs = tf.reshape(dec_outputs, shape=(1, -1))

            dec_inp = dec_inp.write(i, dec_outputs[0, 0])

            if dec_outputs[0, 0] == self.fa_tokenizer.sep_token_id: break

        dec_out = dec_inp.stack()
        dec_out = tf.reshape(dec_out, shape=(-1,))
        return dec_out


#transformer = keras.models.load_model('./models/transformers_v1.tf/')
translator = Translator(model=transformer, en_tokenizer=en_tokenizer, fa_tokenizer=fa_tokenizer, max_length=30)

to_translate = ['this is the best day of my life', 'this model is not accurate', 'i am drowsy I should sleep', 
                'Programming is my only job that I know how to do. I am good at it', 'this model is not good at translating names', 
                'if we train with a larger training set we get a better accuracy', 'we want to test the model accuracy','I love testing alireza\'s code!']
for en in to_translate:
    res = translator(tf.convert_to_tensor(en)).numpy()
    print(fa_tokenizer.decode(res))



# sentence = 'Programming is my only job that I know how to do. I am good at it'
# beam_width = 5
# start, end = fa_tokenizer.cls_token_id, fa_tokenizer.sep_token_id
# vocab_size = fa_tokenizer.vocab_size
# alpha = 0.7


# encoded = en_tokenizer.encode(sentence, add_special_tokens=True)
# encoded = tf.expand_dims(encoded, axis=0)
# encoded = tf.cast(encoded, dtype=tf.int32)

# generated = tf.expand_dims([start], axis=1)
# scores = tf.cast([0.0], dtype='float32')
# lengths = tf.cast([1], dtype='float32')
# not_finished = tf.cast([1], dtype='float32')

# outputs = transformer(tf.repeat(encoded, tf.shape(generated)[0], axis=0), generated)
# outputs = tf.nn.softmax(outputs)
# outputs = outputs[:, -1]

# # flatten outputs
# outputs = tf.reshape(outputs, (-1,))

# # choose out of beam_width * vocab_size
# scores = tf.gather(scores, tf.repeat(tf.range(len(scores)), vocab_size))
# generated = tf.gather(generated, tf.repeat(tf.range(len(generated)), vocab_size))
# lengths = tf.gather(lengths, tf.repeat(tf.range(len(lengths)), vocab_size))
# not_finished = tf.gather(not_finished, tf.repeat(tf.range(len(not_finished)), vocab_size))

# outputs = (tf.math.log(outputs) * not_finished + scores) / tf.pow(lengths + 1, alpha)
# top_scores, top_k = tf.math.top_k(outputs, k=beam_width)

# # update scores and outputs
# scores = top_scores
# generated = tf.gather(generated, top_k)

# lengths = tf.gather(lengths, top_k)
# scores *= tf.pow(lengths + 1, alpha)

# not_finished = tf.gather(not_finished, top_k)


# not_finished = not_finished * tf.cast(top_k % vocab_size != end, tf.float32)
# lengths += not_finished
# # take the newly predicted word; if sentence is already finished add padding
# new_words = tf.cast(top_k % vocab_size, tf.float32)
# new_words = new_words * not_finished + (1 - not_finished) * fa_tokenizer.pad_token_id
# new_words = tf.cast(new_words, tf.int32)
# generated = tf.concat([generated, new_words[..., tf.newaxis]], axis=-1)


# print(not_finished)
# print(scores)
# for i in range(generated.shape[0]):
#     print(fa_tokenizer.decode(generated[i]))


### Save the translator
transformer.save('./models/transformers_v3.tf')





