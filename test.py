from torch.utils.data import Dataset
from tqdm import tqdm_notebook

from preprocessing import *

def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

def test(test_dataloader, model, device):
    model.eval()
    answer=[]
    train_acc = 0.0
    test_acc = 0.0
    ## TypeError: 'DataLoader' object does not support indexing ##
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        max_vals, max_indices = torch.max(out, 1)
        answer.append(max_indices.cpu().clone().numpy())
        test_acc += calc_accuracy(out, label)
    print(test_acc / (batch_id+1))

if __name__ == "__main__":
    max_len = 64
    batch_size = 24
    num_workers = 1
    model = BERTClassifier().build_model()
    checkpoint = torch.load('result/epoch3_batch24.pt')
    model.load_state_dict(checkpoint)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    _, eval_dtls = preprocessing()

    data_test = test_loader(eval_dtls, max_len, batch_size, num_workers)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=num_workers)

    test(test_dataloader, model, device)
