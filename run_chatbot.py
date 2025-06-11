import torch
import readline
import os

from src.model_components import Encoder, Decoder, Seq2Seq
from src.pretrained_utils import PretrainedAssets
from src.utils import load_parameters, get_device
import config

def generate_answer(question: str,
                    model: Seq2Seq,
                    assets: PretrainedAssets,
                    max_src_len: int,
                    max_trg_len: int,
                    device: torch.device) -> str:
    model.eval_mode()

    src_tokenized = assets.tokenizer(
        question,
        max_length=max_src_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    src_input_ids = src_tokenized['input_ids'].to(device)
    # src_input_ids.shape = (1, src_len)

    with torch.no_grad():
        output_logits, attention_weights = model.forward(
            src_input_ids=src_input_ids,
            trg_input_ids=None,
            teacher_forcing_ratio=0.0,
            max_output_len=max_trg_len
        )
        # output_logits.shape = (1, generated_seq_len, vocab_size)
    predicted_token_ids = output_logits.argmax(dim=2).squeeze(0).tolist()

    answer_tokens = []
    for token_id in predicted_token_ids:
        if token_id == assets.eos_token_id:
            break
        if token_id not in [assets.sos_token_id, assets.pad_token_id]:
            answer_tokens.append(assets.tokenizer.convert_ids_to_tokens(token_id))
    
    answer_text = assets.tokenizer.convert_tokens_to_string(answer_tokens)
    return answer_text.strip(), attention_weights

def main_loop():
    device = get_device()
    print('Зареждане на модела...')

    assets = PretrainedAssets(config.PRETRAINED_MODEL_NAME, device)

    embedding_dim = assets.embedding_dim
    output_vocab_size = assets.vocab_size

    encoder = Encoder(
        embedding_dim=embedding_dim,
        encoder_hidden_dim=config.ENCODER_HIDDEN_DIM,
        decoder_hidden_dim=config.DECODER_HIDDEN_DIM,
        device=device
    )
    decoder = Decoder(
        output_vocab_size=output_vocab_size,
        embedding_dim=embedding_dim,
        encoder_hidden_dim=config.ENCODER_HIDDEN_DIM,
        decoder_hidden_dim=config.DECODER_HIDDEN_DIM,
        attention_dim=config.ATTENTION_DIM,
        device=device
    )
    model = Seq2Seq(encoder, decoder, assets, device)

    model_path = config.MODEL_SAVE_PATH
    if os.path.exists(model_path):
        if load_parameters(list(model.parameters()), model_path, device):
            print('Моделът е зареден')
        else:
            print('Има проблем при зареждането')
    else:
        print(f'Няма запазен модел в {model_path}')
    
    while True:
        try:
            question = input('Въпрос: ')
            if question.lower() in ['exit', 'quit']:
                break
            if not question.strip():
                continue
            answer, attention = generate_answer(
                question, model, assets,
                config.MAX_SRC_SEQ_LENGTH, config.MAX_TRG_SEQ_LENGTH,
                device
            )
            print(f'Отговор: {answer}')
        except KeyboardInterrupt:
            print('\nЗатваряне...')
            break
        except Exception as e:
            print(f'Грешка: {e}')
    print('Край')

if __name__ == '__main__':
    main_loop()
