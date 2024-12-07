import argparse
import time
import os
import json

import torch
import torch.nn.functional as F

from utils import utils
from dataset import problem

# pylint: disable=not-callable


def encode_inputs(inputs, model, beam_size):
    with torch.no_grad():
        src_mask = utils.create_pad_mask(inputs, model.src_pad_idx)
        enc_output = model.encode(inputs, src_mask)
        enc_output = enc_output.repeat(beam_size, 1, 1)
    return enc_output, src_mask


def update_targets(targets, best_indices, idx, vocab_size):
    best_tensor_indices = best_indices // vocab_size
    best_token_indices = torch.fmod(best_indices, vocab_size)
    new_batch = torch.index_select(targets, 0, best_tensor_indices)
    new_batch[:, idx] = best_token_indices
    return new_batch


def get_result_sentence(indices_history, trg_data, vocab_size):
    result = []
    k = 0
    for best_indices in indices_history[::-1]:
        best_idx = best_indices[k]
        # TODO: get this vocab_size from target.pt?
        k = best_idx // vocab_size
        best_token_idx = best_idx % vocab_size
        best_token = trg_data['field'].vocab.itos[best_token_idx]
        result.append(best_token)
    return ' '.join(result[::-1])

def process_bpe(sentence, bpe_token = '@'):
    while bpe_token in sentence:
        split_sent = sentence.split(' ')
        result = []
        i = 0
        while i < len(split_sent):
            s = split_sent[i]
            if bpe_token in s:
                if s.startswith(bpe_token):
                    result[-1] += s[1:]
                elif s.endswith(bpe_token):
                    result.append(s[:-1] + split_sent[i+1])
                    i += 1
                else:
                    s = s.replace(bpe_token, '')
                    result.append(s)
            else:
                result.append(s)
            i += 1
        sentence = ' '.join(result).strip()
    return sentence

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--beam_size', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--share_target_embedding', action='store_true')
    parser.add_argument('--use_bpe', action='store_true')
    parser.add_argument('--average_checkpoints', action='store_true')
    parser.add_argument('--lan', type=str, choices=['fr', 'de', 'cs', 'ig', 'ha'], default='de')
    parser.add_argument('--test_set', type=str, choices=['gnome', 'ubuntu'])
    args = parser.parse_args()
    args_file = os.path.join(args.output_dir, 'test_setting.log')
    with open(args_file, 'w') as fout:
        fout.write(json.dumps(args.__dict__, indent=2))
        
    beam_size = args.beam_size
    
    trg_data = torch.load(args.data_dir + '/target_share.pt') if args.share_target_embedding else torch.load(args.data_dir + '/target.pt')
    vocab = trg_data['field'].vocab.itos
    eos = trg_data['field'].eos_token
    eos_idx = trg_data['field'].vocab.stoi[eos]

    # Load a saved model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = utils.load_checkpoint(args.model_dir, device, model_name= '/average_model.pt' if args.average_checkpoints else '/best_model.pt')
    
    _, test_data, _, _, _ = problem.prepare(args.problem, args.data_dir, args.max_length, args.batch_size, device, args, test_flag=True)
    
    pred_path = os.path.join(args.output_dir, 'pred.txt')
    ans_path = os.path.join(args.output_dir, 'ans.txt')
    
    count = 1
    total = len(test_data)
    with open(pred_path, 'w') as f1, open(ans_path, 'w') as f2:
        for batch in test_data:
            print(f'Processing {count}/{total}')
            inputs = batch.src
            ans = batch.trg
            
            enc_output, src_mask = encode_inputs(inputs, model, beam_size)
            pads = torch.tensor([trg_data['pad_idx']] * beam_size, device=device).unsqueeze(-1)
            targets = pads
            # We'll find a target sequence by beam search.
            scores_history = [torch.zeros((beam_size,), dtype=torch.float,
                                          device=device)]
            indices_history = []
            start_idx = 0
            
            with torch.no_grad():
                for idx in range(start_idx, args.max_length):
                    if idx > start_idx:
                        targets = torch.cat((targets, pads), dim=1)
                    t_self_mask = utils.create_trg_self_mask(targets.size()[1],
                                                             device=targets.device)

                    t_mask = utils.create_pad_mask(targets, trg_data['pad_idx'])
                    pred = model.decode(targets, enc_output, src_mask,
                                        t_self_mask, t_mask)
                    pred = pred[:, idx].squeeze(1)
                    vocab_size = pred.size(1)

                    pred = F.log_softmax(pred, dim=1)
                    if idx == start_idx:
                        scores = pred[0]
                    else:
                        scores = scores_history[-1].unsqueeze(1) + pred
                    length_penalty = pow(((5. + idx + 1.) / 6.), args.alpha)
                    scores = scores / length_penalty
                    scores = scores.view(-1)

                    best_scores, best_indices = scores.topk(beam_size, 0)
                    scores_history.append(best_scores)
                    indices_history.append(best_indices)

                    # Stop searching when the best output of beam is EOS.
                    if best_indices[0].item() % vocab_size == eos_idx:
                        break

                    targets = update_targets(targets, best_indices, idx, vocab_size)
                    
                pred_line = get_result_sentence(indices_history, trg_data, vocab_size).replace(eos, '').strip()
                ans_line = ' '.join(vocab[idx] for idx in ans.view(-1)).replace(eos, '').strip()
                if args.use_bpe:
                    pred_line = process_bpe(pred_line)
                    ans_line = process_bpe(ans_line)
                f1.write(pred_line + '\n')
                f2.write(ans_line + '\n')
                count += 1
    ## use https://www.letsmt.eu/Bleu.aspx to calculate BLEU

if __name__ == '__main__':
    main()
