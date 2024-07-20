import argparse
import torch
import torch.nn.functional as F
import numpy as np
import os
import tqdm

from swag import data, losses, models, utils
from swag.posteriors import SWAG, KFACLaplace

parser = argparse.ArgumentParser(description="SGD/SWA training")
parser.add_argument("--file", type=str, default=None, required=True, help="checkpoint")

parser.add_argument(
    "--dataset", type=str, default="test", help="dataset name"
)
parser.add_argument(
    "--training_data_path",
    type=str,
    default=None,
    required=True,
    metavar="PATH",
    help="path to datasets location (default: None)",
)
parser.add_argument(
    "--test_data_path",
    type=str,
    default=None,
    required=True,
    metavar="PATH",
    help="path to datasets location (default: None)",
)
parser.add_argument(
    "--use_test",
    dest="use_test",
    action="store_true",
    help="use test dataset instead of validation (default: False)",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size (default: 128)",
)
parser.add_argument("--split_classes", type=int, default=None)
parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    metavar="N",
    help="number of workers (default: 4)",
)
parser.add_argument(
    "--model",
    type=str,
    default="CNN1",
    metavar="MODEL",
    help="model name (default: CNN1)",
)
parser.add_argument(
    "--method",
    type=str,
    default="SWAG",
    choices=["SWAG", "KFACLaplace", "SGD", "HomoNoise", "Dropout", "SWAGDrop"],
    required=True,
)
parser.add_argument(
    "--save_path",
    type=str,
    default=None,
    required=True,
    help="path to npz results file",
)
parser.add_argument("--N", type=int, default=30)
parser.add_argument("--scale", type=float, default=1.0)
parser.add_argument(
    "--cov_mat", action="store_true", help="use sample covariance for swag"
)
parser.add_argument("--use_diag", action="store_true", help="use diag cov for swag")

parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)





args = parser.parse_args()

eps = 1e-12
if args.cov_mat:
    args.cov_mat = True
else:
    args.cov_mat = False

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print("Using model %s" % args.model)
model_cfg = getattr(models, args.model)

print("Loading dataset %s from %s" % (args.dataset, args.test_data_path))
loaders, num_classes = data.loaders(
    args.dataset,
    args.training_data_path,
    args.batch_size,
    args.num_workers,
    model_cfg.transform_train,
    model_cfg.transform_test,
    use_validation=not args.use_test,
    split_classes=args.split_classes,
    shuffle_train=False,
    training_file = args.training_data_path,
    test_file = args.test_data_path

)
"""if args.split_classes is not None:
    num_classes /= 2
    num_classes = int(num_classes)"""

print("Preparing model")
if args.method in ["SWAG", "HomoNoise", "SWAGDrop"]:
    model = SWAG(
        model_cfg.base,
        no_cov_mat=not args.cov_mat,
        max_num_models=20,
        *model_cfg.args,
        num_classes=num_classes,
        **model_cfg.kwargs
    )
elif args.method in ["SGD", "Dropout", "KFACLaplace"]:
    model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
else:
    assert False
model.cuda()


def train_dropout(m):
    if type(m) == torch.nn.modules.dropout.Dropout:
        m.train()


print("Loading model %s" % args.file)
checkpoint = torch.load(args.file)
model.load_state_dict(checkpoint["state_dict"])

if args.method == "KFACLaplace":
    print(len(loaders["train"].dataset))
    model = KFACLaplace(
        model, eps=5e-4, data_size=len(loaders["train"].dataset)
    )  # eps: weight_decay

    t_input, t_target = next(iter(loaders["train"]))
    t_input, t_target = (
        t_input.cuda(non_blocking=True),
        t_target.cuda(non_blocking=True),
    )

if args.method == "HomoNoise":
    std = 0.01
    for module, name in model.params:
        mean = module.__getattr__("%s_mean" % name)
        module.__getattr__("%s_sq_mean" % name).copy_(mean ** 2 + std ** 2)


predict_matrix = []

#N is the no. of realisations.
threshold = 0.5

for i in range(args.N):
    print("%d/%d" % (i + 1, args.N))
    if args.method == "KFACLaplace":
        ## KFAC Laplace needs one forwards pass to load the KFAC model at the beginning
        model.net.load_state_dict(model.mean_state)

        if i == 0:
            model.net.train()

            loss, _ = losses.cross_entropy(model.net, t_input, t_target)
            loss.backward(create_graph=True)
            model.step(update_params=False)

    if args.method not in ["SGD", "Dropout"]:
        sample_with_cov = args.cov_mat and not args.use_diag
        model.sample(scale=args.scale, cov=sample_with_cov)

    # Didn't understood why this step is performed
    if "SWAG" in args.method:
        utils.bn_update(loaders["train"], model)

    model.eval()
    if args.method in ["Dropout", "SWAGDrop"]:
        model.apply(train_dropout)
        # torch.manual_seed(i)
        # utils.bn_update(loaders['train'], model)

    k = 0
    correct = 0
    predictions = np.zeros((len(loaders["test"].dataset), num_classes))
    targets = np.zeros((len(loaders["test"].dataset),num_classes))
    for input, target in tqdm.tqdm(loaders["test"]):
        input = input.cuda(non_blocking=True)
        ##TODO: is this needed?
        # if args.method == 'Dropout':
        #    model.apply(train_dropout)

        if args.method == "KFACLaplace":
            output = model.net(input)
        else:
            output = model(input)

        with torch.no_grad():
            if (num_classes==1):
                yhat=output.cpu().numpy()
            else:
                yhat = F.softmax(output, dim=1).cpu().numpy()
        predictions[k : k + input.size()[0]] += yhat
        targets[k : (k + target.size(0))] += target.numpy()
        k += input.size()[0]
    predict_matrix.append(predictions)
    if args.model =="CNN1":
        pred = torch.where(torch.from_numpy(predictions)>threshold,torch.tensor(1),torch.tensor(0))
        targets_vector = torch.from_numpy(targets)
        correct += pred.eq(targets_vector.data.view_as(pred)).sum().item()
        print("Accuracy:", correct/k)
    else:
        sigmoid_op = torch.sigmoid(torch.from_numpy(predictions))
        pred = torch.where(sigmoid_op>threshold,torch.tensor(1),torch.tensor(0))
        targets_vector = torch.from_numpy(targets)
        correct += pred.eq(targets_vector.data.view_as(pred)).sum().item()
        print("Accuracy:", correct/k)

prediction_matrix = np.concatenate(predict_matrix, axis=1)
print(prediction_matrix)
av_pred = np.mean(prediction_matrix, axis=1)
average_predictions = np.expand_dims(av_pred, axis =1)
# predictions /= args.N
if args.model=="CNN1":
    probabilites = average_predictions
else:
    probabilites = torch.sigmoid(torch.from_numpy(average_predictions)).numpy()


entropies = -np.sum(np.log(average_predictions + eps) * average_predictions, axis=1)
np.savez(args.save_path, entropies=entropies, predictions=prediction_matrix, targets=targets, probabilites = probabilites)
