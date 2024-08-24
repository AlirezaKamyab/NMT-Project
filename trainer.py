import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import sys
import torch
import time
from tokenizers import Tokenizer
from torch.utils.tensorboard import SummaryWriter
from data.dataloader import load_data, get_pipelines
from models.transformer.transformer import Transformer
from models.relativeTransformer.transformer import Transformer as RelTransformer
from torch.optim import Adam
import json

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def main():
    config_file = sys.argv[1]
    with open(config_file, 'r') as file:
        config = json.load(file)

    dataset_path = config['files']['dataset']
    en_tokenizer_path = config['files']['english_tokenizer']
    fa_tokenizer_path = config['files']['persian_tokenizer']

    batch_size = config['pipeline']['batch_size']
    num_workers = config['pipeline']['num_workers']
    prefetch_factor = config['pipeline']['prefetch_factor']
    pin_memory = config['pipeline']['pin_memory']
    seed = config['pipeline']['seed']

    self_attention = config['model']['self_attention']
    d_model = config['model']['d_model']
    num_heads = config['model']['num_heads']
    if self_attention == 'relative':
        k = config['model']['k']
    N = config['model']['N']
    dff = config['model']['dff']
    dropout_rate = config['model']['dropout_rate']
    label_smoothing = config['model']['label_smoothing']

    beta1 = config['optimizer']['beta1']
    beta2 = config['optimizer']['beta2']
    epsilon = config['optimizer']['epsilon']
    warmup_steps = config['optimizer']['warmup_steps']
    epochs = config['optimizer']['epochs']

    torch.manual_seed(seed)
    en_tokenizer = Tokenizer.from_file(en_tokenizer_path)
    fa_tokenizer = Tokenizer.from_file(fa_tokenizer_path)
    en_pad_id = en_tokenizer.padding['pad_id']
    fa_pad_id = fa_tokenizer.padding['pad_id']

    (en_train, fa_train), (en_test, fa_test) = load_data(dataset_path, 
                                                         return_tokenized=False)

    trainloader, testloader = get_pipelines(en_train, 
                                            fa_train,
                                            en_test,
                                            fa_test,
                                            english_tokenizer=en_tokenizer,
                                            persian_tokenizer=fa_tokenizer, 
                                            batch_size=batch_size, 
                                            num_workers=num_workers, 
                                            prefetch_factor=prefetch_factor, 
                                            pin_memory=pin_memory)


    source_vocab_size = en_tokenizer.get_vocab_size()
    target_vocab_size = fa_tokenizer.get_vocab_size()

    if self_attention == 'absolute':
        transformer = Transformer(d_model=d_model, 
                                num_heads=num_heads, 
                                N=N, 
                                dff=dff, 
                                dropout=dropout_rate,
                                source_vocab_size=source_vocab_size, 
                                target_vocab_size=target_vocab_size, 
                                source_padding_idx=en_pad_id, 
                                target_padding_idx=fa_pad_id)
    elif self_attention == 'relative':
        transformer = RelTransformer(d_model=d_model, 
                                     num_heads=num_heads, 
                                     k=k, 
                                     N=N, 
                                     dff=dff, 
                                     dropout=dropout_rate,
                                     source_vocab_size=source_vocab_size, 
                                     target_vocab_size=target_vocab_size, 
                                     source_padding_idx=en_pad_id, 
                                     target_padding_idx=fa_pad_id)

    transformer = transformer.to(device)

    optimizer = Adam(params=transformer.parameters(), betas=[beta1, beta2], eps=epsilon)
    loss_func = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction='none')
    criterion = get_criterion(loss_func=loss_func, target_vocab_size=target_vocab_size, mask_id=fa_pad_id)
    scheduler = get_scheduler(d_model=torch.tensor(d_model), warmup_steps=torch.tensor(warmup_steps))


    writer= SummaryWriter(log_dir='./logs')

    total_steps = 0
    for epoch in range(epochs):
        running_loss = 0.0
        running_acc = 0.0
        loss_metric = {}
        accuracy_metric = {}
        start = time.time()
        print(f'EPOCH {epoch + 1}')
        print('-----------------------------')
        transformer.train()
        for step, (source, target) in enumerate(trainloader):
            learning_rate = scheduler(total_steps)
            change_learning_rate(optimizer, learning_rate)
            writer.add_scalar('learning rate', learning_rate, global_step=total_steps)
            
            optimizer.zero_grad()
            source = source.to(device)
            target = target.to(device)
            y_true = target[:, 1:]
            target = target[:, :-1]

            y_hat = transformer(source, target)
            loss = criterion(y_hat, y_true.to(dtype=torch.long))
            loss.backward()
            optimizer.step()

            acc = masked_accuracy(y_hat, y_true, fa_pad_id)
            end = time.time()
            
            running_loss = running_loss + (1 / (step + 1)) * (loss - running_loss)
            running_acc = running_acc + (1 / (step + 1)) * (acc - running_acc)
            print(f'\r{int(end - start):>4} sec | Step {step:>5}\t Loss {running_loss:>2.4f}\t Accuracy {running_acc:>0.3f}', end='')
            total_steps += 1
        
        print()
        loss_metric['train'] = running_loss
        accuracy_metric['train'] = running_acc
        print('Validation')
        transformer.eval()
        running_loss = 0.0
        running_acc = 0.0

        with torch.no_grad():

            for step, (source, target) in enumerate(testloader):
                source = source.to(device)
                target = target.to(device)
                y_true = target[:, 1:]
                target = target[:, :-1]

                y_hat = transformer(source, target)
                loss = criterion(y_hat, y_true.to(dtype=torch.long))    
                acc = masked_accuracy(y_hat, y_true, fa_pad_id)
                end = time.time()
                
                running_loss = running_loss + (1 / (step + 1)) * (loss - running_loss)
                running_acc = running_acc + (1 / (step + 1)) * (acc - running_acc)
                print(f'\r{int(end - start):>4} sec | Step {step:>5}\t Loss {running_loss:>2.4f}\t Accuracy {running_acc:>0.3f}', end='')

            loss_metric['validation'] = running_loss
            accuracy_metric['validation'] = running_acc

            writer.add_scalars('loss', loss_metric, global_step=total_steps)
            writer.add_scalars('accuracy', accuracy_metric, global_step=total_steps)

            os.makedirs('./checkpoints', exist_ok=True)
            torch.save(transformer.state_dict(), f'./checkpoints/epoch_{epoch + 1}.pt')
            print(f'\n')
    
    torch.save(transformer, 'model.pt')


def change_learning_rate(optimizer, lr):
    for param in optimizer.param_groups:
        param['lr'] = lr
        

def masked_accuracy(y_hat, y_true, mask_id):
    mask = torch.logical_not(y_true == mask_id)
    mask = mask.to(dtype=torch.float32)
    y_pred = torch.argmax(y_hat, dim=-1)
    correct = torch.eq(y_true, y_pred)
    correct = correct.to(dtype=torch.float32) * mask
    return torch.sum(correct) / torch.sum(mask)


def get_scheduler(d_model, warmup_steps=4000):
    def get_learning_rate(step):
        step = torch.tensor(step, dtype=torch.float32)
        arg1 = torch.rsqrt(step)
        arg2 = step * torch.pow(warmup_steps * 1.0, -1.5)

        return torch.rsqrt(d_model * 1.0) * torch.minimum(arg1, arg2)

    return get_learning_rate


def get_criterion(loss_func, target_vocab_size, mask_id):
    def criterion(y_hat, y_true):
        y_hat = torch.reshape(y_hat, [-1, target_vocab_size])
        y_true = torch.reshape(y_true, [-1])
        mask = torch.logical_not(y_true == mask_id)
        mask = mask.to(dtype=torch.float32)
        loss = loss_func(y_hat, y_true) * mask
        return torch.sum(loss) / torch.sum(mask)
    return criterion


if __name__ == '__main__':
    main()