import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import math
import random
import os

from src.model_components import Encoder, Decoder, Seq2Seq
from src.pretrained_utils import PretrainedAssets
from src.utils import save_parameters, load_parameters, count_parameters, format_time
import config


class QADataset(Dataset):
    def __init__(self, src_texts: list[str], trg_texts: list[str], pretrained_assets: PretrainedAssets, max_src_len: int, max_trg_len: int):
        self.src_texts = src_texts
        self.trg_texts = trg_texts
        self.assets = pretrained_assets
        self.max_src_len = max_src_len
        self.max_trg_len = max_trg_len

    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        trg_text = self.trg_texts[idx]

        src_tokenizer = self.assets.tokenizer(
            src_text,
            max_length=self.max_src_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        src_ids = src_tokenizer['input_ids'].squeeze(0)

        trg_tokens_without_specials = self.assets.tokenizer.tokenize(trg_text)

        if len(trg_tokens_without_specials) > self.max_trg_len - 2:
            trg_tokens_without_specials = trg_tokens_without_specials[:self.max_trg_len-2]
        
        trg_ids_list = [self.assets.sos_token_id] + self.assets.tokenizer.convert_tokens_to_ids(trg_tokens_without_specials) + [self.assets.eos_token_id]

        padding_length = self.max_trg_len - len(trg_ids_list)
        trg_ids_list += [self.assets.pad_token_id] * padding_length
        trg_ids = torch.tensor(trg_ids_list, dtype=torch.long)

        return {'src_ids': src_ids, 'trg_ids': trg_ids}
    

def load_qa_data(filepath: str) -> tuple[list[str], list[str]]:
    questions = []
    answers = []
    print(f'Зареждане на данните от {filepath}')

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    questions.append(parts[0])
                    answers.append(parts[1])
    except FileNotFoundError:
        print('Не е намерен файлът с данните')
    
    if not questions or not answers:
        print('Проблем с данните')
    else:
        print(f'Заредени са {len(questions)} двойки Q\A')
    return questions, answers

def train_epoch(model: Seq2Seq,
                dataloader: DataLoader,
                optimizer: optim.Optimizer,
                criterion: nn.Module,
                clip: float,
                device: torch.device,
                epoch_num: int,
                teacher_forcing_ratio: float):
    model.train_mode()
    epoch_loss = 0
    start_time = time.time()

    for i, batch in enumerate(dataloader):
        src_ids = batch['src_ids'].to(device)
        # src_ids = (batch, src_len)
        trg_ids = batch['trg_ids'].to(device)
        # trg_ids.shape = (batch, trg_len)

        optimizer.zero_grad()

        output_logits, _ = model.forward(src_input_ids=src_ids, trg_input_ids=trg_ids, teacher_forcing_ratio=teacher_forcing_ratio)

        # Output.shape = (batch_size * (trg_len - 1), vocab_size)
        # Target.shape = (batch_size * (trg_len - 1))
        output_dim = output_logits.shape[-1]

        predicted_sequence_length = output_logits.shape[1]

        trg_for_loss = trg_ids[:, 1:predicted_sequence_length+1].contiguous()

        loss = criterion(output_logits.view(-1, output_dim), trg_for_loss.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

        if (i + 1) % config.LOG_INTERVAL == 0:
            elapsed_batch_time = time.time() - start_time
            print(f'Epoch: {epoch_num+1:02} | Batch: {i+1:04}/{len(dataloader):04} | '
                  f'Loss: {loss.item():.3f} | Time/batch: {elapsed_batch_time/ (i+1) :.2f}s')

    return epoch_loss / len(dataloader)
    
def evaluate_epoch(model: Seq2Seq,
                   dataloader: DataLoader,
                   criterion: nn.Module,
                   device: torch.device):
    model.eval_mode()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            src_ids = batch['src_ids'].to(device)
            trg_ids = batch['trg_ids'].to(device)

            # For evaluation, teacher_forcing_ratio is 0
            output_logits, _ = model.forward(src_input_ids=src_ids,
                                             trg_input_ids=trg_ids, # Still provide for length matching
                                             teacher_forcing_ratio=0.0)

            output_dim = output_logits.shape[-1]
            predicted_sequence_length = output_logits.shape[1]
            trg_for_loss = trg_ids[:, 1:predicted_sequence_length+1].contiguous()

            loss = criterion(output_logits.view(-1, output_dim),
                             trg_for_loss.view(-1))
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

def run_training():
    print('Начало на тренирането')
    device = config.DEVICE

    assets = PretrainedAssets(config.PRETRAINED_MODEL_NAME, device)
    embedding_dim = assets.embedding_dim
    output_vocab_size = assets.vocab_size

    src_texts, trg_texts = load_qa_data(config.RAW_DATA_PATH)
    if not src_texts:
        print('Не бяха заредени данни')
        return
    
    combined = list(zip(src_texts, trg_texts))
    random.shuffle(combined)
    split_idx = int(len(combined) * 0.8)
    train_data = combined[:split_idx]
    val_data = combined[split_idx:]

    train_src, train_trg = zip(*train_data) if train_data else ([],[])
    val_src, val_trg = zip(*val_data) if val_data else ([],[])

    train_dataset = QADataset(list(train_src), list(train_trg), assets,
                              config.MAX_SRC_SEQ_LENGTH, config.MAX_TRG_SEQ_LENGTH)
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    if val_src:
        val_dataset = QADataset(list(val_src), list(val_trg), assets,
                                config.MAX_SRC_SEQ_LENGTH, config.MAX_TRG_SEQ_LENGTH)
        val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    else:
        val_dataloader = None
    
    encoder = Encoder(
        embedding_dim=embedding_dim,
        encoder_hidden_dim=config.ENCODER_HIDDEN_DIM,
        decoder_hidden_dim=config.DECODER_HIDDEN_DIM, # Ensure compatible
        device=device,
        dtype=torch.float32 # Default dtype
    )
    decoder = Decoder(
        output_vocab_size=output_vocab_size,
        embedding_dim=embedding_dim,
        encoder_hidden_dim=config.ENCODER_HIDDEN_DIM,
        decoder_hidden_dim=config.DECODER_HIDDEN_DIM,
        attention_dim=config.ATTENTION_DIM,
        device=device,
        dtype=torch.float32
    )
    model = Seq2Seq(encoder, decoder, assets, device)
    print(f'Моделът е зареден и готов за трениране. Брой параметри: {count_parameters(model.parameters())}')

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    criterion = nn.CrossEntropyLoss(ignore_index=assets.pad_token_id)

    best_valid_loss = float('inf')
    total_training_start_time = time.time()

    print(f'Начало на трениране с {config.NUM_EPOCHS} епохи.')

    for epoch in range(config.NUM_EPOCHS):
        epoch_start_time = time.time()

        train_loss = train_epoch(model, train_dataloader, optimizer, criterion,
                                 config.CLIP_GRAD_NORM, device, epoch, config.TEACHER_FORCING_RATIO)

        epoch_duration = time.time() - epoch_start_time
        print(f'\nEpoch: {epoch+1:02}/{config.NUM_EPOCHS:02} | Time: {format_time(epoch_duration)}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    
    if val_dataloader:
        valid_loss = evaluate_epoch(model, val_dataloader, criterion, device)
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_parameters(model.parameters(), config.MODEL_SAVE_PATH)
            print(f'\tНов най-добър резултат. Моделът се запазва в {config.MODEL_SAVE_PATH}')
    else:
        save_parameters(model.parameters(), config.MODEL_SAVE_PATH)
        print(f'\tМоделът се запазва в {config.MODEL_SAVE_PATH}')
    print('-' * 80)

    total_training_time = time.time() - total_training_start_time
    print(f'--- Тренирането приключи за {format_time(total_training_time)} ---')

if __name__ == '__main__':
    run_training()
