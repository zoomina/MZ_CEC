from tqdm import tqdm
from tqdm.notebook import tqdm

from transformers import AdamW
from transformers.optimization import WarmupLinearSchedule

from preprocessing import *

def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

def train(train_dataloader, dev_dataloader, model, device, optimizer, loss_fn, epoch):
    for e in range(epoch):
        train_acc = 0.0
        test_acc = 0.0
        model.train()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            loss = loss_fn(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            train_acc += calc_accuracy(out, label)
            if batch_id % log_interval == 0:
                print("epoch {} batch id {} loss {} train acc {}".format(e + 1, batch_id + 1, loss.data.cpu().numpy(),
                                                                         train_acc / (batch_id + 1)))
        print("epoch {} train acc {}".format(e + 1, train_acc / (batch_id + 1)))
        model.eval()  # 모델 평가 부분
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(dev_dataloader)):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            test_acc += calc_accuracy(out, label)
        print("epoch {} test acc {}".format(e + 1, test_acc / (batch_id + 1)))

if __name__ == "__main__":
    max_len = 64
    batch_size = 32
    warmup_ratio = 0.1
    num_epochs = 3
    max_grad_norm = 1
    log_interval = 200
    learning_rate = 5e-5
    num_workers = 2
    model = BERTClassifier().build_model()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    dtls, _ = preprocessing()
    train_dataloader, test_dataloader = data_loader(dtls, max_len, batch_size, num_workers)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    t_total = len(train_dataloader) * num_epochs
    warmup_step = int(t_total * warmup_ratio)

    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_step, t_total=t_total)

    train(train_dataloader, test_dataloader, model, device, optimizer, loss_fn, 3)