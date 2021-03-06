import argparse
from tqdm import tqdm
from tqdm.notebook import tqdm

from preprocessing import *

def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    test_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return test_acc

def test(test_dataloader, model, device):
    model.eval()
    answer=[]
    labels=[]
    test_acc = 0.0
    with torch.no_grad():
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length= valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            max_vals, max_indices = torch.max(out, 1)
            answer.extend(max_indices.cpu().clone().numpy())
            test_acc += calc_accuracy(out, label)
            labels.extend(label.cpu().clone().numpy())
    print(test_acc / (batch_id+1))
    labels = np.array(labels).flatten()
    preds = np.array(answer).flatten()
    result = pd.DataFrame({"label": labels, "pred": preds})
    return result

def test_main(checkpoint, data, batch_size, num_workers, num_classes, inner_emotion=-1, test=False):
    max_len = 64
    model = BERTClassifier(num_classes=num_classes).build_model()
    model.load_state_dict(checkpoint)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    eval_dtls = preprocessing(json2csv(data, test=test), inner_emotion=inner_emotion, test=test)

    data_test = test_loader(eval_dtls, max_len, batch_size, num_workers)

    result_df = test(data_test, model, device)

    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("data", help="input_data(.json)_path")
    parser.add_argument("--batch-size", type=int, default=64, help="default=64")
    parser.add_argument("--num-workers", type=int, default=5, help="default=5")
    parser.add_argument("--num-classes", type=int, default=6, help="default=6")
    args = parser.parse_args()
    
    batch_size = args.batch_size
    num_workers = args.num_workers
    checkpoint = torch.load(args.checkpoint)
    '''

    batch_size = 24
    num_workers = 0
    checkpoint = torch.load("result/epoch3_batch24.pt")
    '''
    result_df = test_main(checkpoint, args.data, batch_size, num_workers, args.num_classes)
    result_df.to_csv("result/label_pred_tmp.csv")