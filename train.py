import os
import json
import torch
import logging
import configargparse
import torch
import numpy as np
import subprocess

from math import ceil
from tqdm import tqdm
from typing import List, Optional
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.utils.data.dataset import random_split

from utils import setup_tokenizer, set_random_seed, setup_cuda_device,\
    setup_scheduler_optimizer, get_label_type
from TemporalDataSet import temprel_set
from model import prepare_model,prepare_model_eval
from pairwise_ffnn_pytorch import VerbNet
from LibMTL.weighting.abstract_weighting import AbsWeighting

def _get_validated_args(input_args: Optional[List[str]] = None):
    parser = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )

    parser.add_argument("--dataset", type=str, default="TRMF",
                        choices=["TRMF"], required=True,
                        help="Choose a dataset")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Choose a pretrained base model.")
    parser.add_argument("--model_type", type=str, required=True,
                        help="Choose a relation classification model.")
    parser.add_argument("--model_weights", type=str, default="/home/JJJ/MFRV/output",
                        help="The trained model weights.")
    parser.add_argument("--cache_dir", type=str,default="/home/JJJ",
                        help="Cache directory for pretrained models.")
    parser.add_argument("--output_dir", type=str,default="/home/JJJ/MFRV/output",
                        help="Output directory for parameters from trained model.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Not to use CUDA when available.")
    parser.add_argument("--batch_size", type=int, default=4,#4,8
                        help="Batch size for each running.")
    parser.add_argument("--update_batch_size", type=int, default=8,#4,8
                        help="Batch size for each model update.")
    parser.add_argument("--lr", type=float, default=4e-5,#,5e-6,1e-5,#4e-5
                        help="Learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=4,#4
                        help="The number of epochs for pretrained model.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="The maximum gradient norm.")

    parser.add_argument("--do_train", action='store_true',
                        help="Perform training")
    parser.add_argument("--do_eval", action='store_true',
                        help="Perform evaluation on test set")

    parser.add_argument("--beta1", type=float, default=0.9,
                        help="Beta 1 parameters (b1, b2) for optimizer.")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="Beta 1 parameters (b1, b2) for optimizer.")
    parser.add_argument("--eps", type=float, default=1e-8,
                        help="Epsilon for numerical stability for optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Decoupled weight decay to apply on optimizer.")
    parser.add_argument("--num_warmup_ratio", type=float, default=0.1,
                        help="The number of steps for the warmup phase")

    args = parser.parse_args(input_args)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)

    return args


def _get_tensorset(tokenizer):
    logging.info("***** Loading Dataset *****\n")
    trainset = temprel_set("/home/JJJ/MFRV/data/new_data/train_new.xml")
    devset = temprel_set("/home/JJJ/MFRV/data/new_data/dev_new.xml")
    #traindevset = temprel_set("/home/JJJ/MFRV/data/test_augmentation/testoutput1000.xml")
    #traindev_tensorset = traindevset.to_tensor(tokenizer=tokenizer)
    #train_idx = list(range(len(traindev_tensorset)-1852))
    #dev_idx = list(range(len(traindev_tensorset) -
    #              1852, len(traindev_tensorset)))
    #train_tensorset = Subset(traindev_tensorset, train_idx)
    #dev_tensorset = Subset(traindev_tensorset, dev_idx)  # Last 21 docs
    train_tensorset =trainset.to_tensor(tokenizer=tokenizer)
    dev_tensorset = devset.to_tensor(tokenizer=tokenizer)
    logging.info(
        f"All = {len(train_tensorset)}, Train={len(train_tensorset)}, Dev={len(dev_tensorset)}")
        #f"All = {len(traindev_tensorset)}, Train={len(train_tensorset)}, Dev={len(dev_tensorset)}")
    train_tensorset =trainset.to_tensor(tokenizer=tokenizer)
    
    testset = temprel_set("/home/JJJ/MFRV/data/new_data/test_new copy.xml")
    test_tensorset = testset.to_tensor(tokenizer=tokenizer)
    logging.info(f"Test = {len(test_tensorset)}")
    return train_tensorset, dev_tensorset, test_tensorset


def _gather_model_inputs(model_type, batch):
    inputs = {'input_ids': batch[0],
              'attention_mask': batch[1],
              'event_ix': batch[2],
              'labels': batch[3],
              'common_ids': batch[4]}
    if model_type == 'time_anchor':
        return inputs
    else:
        raise ValueError("Invalid model type")


def calc_f1(predicted_labels, all_labels, label_type):
    confusion = np.zeros((len(label_type), len(label_type)))
    predicted_labels = predicted_labels.cpu().numpy()#
    all_labels = all_labels.cpu().numpy()#
    predicted_labels = predicted_labels.astype(int)#
    all_labels = all_labels.astype(int)#
    for i in range(len(predicted_labels)):
        if all_labels[i] < len(label_type) and predicted_labels[i] < len(label_type):
          confusion[all_labels[i]][predicted_labels[i]] += 1
        
        #confusion[all_labels[i]][predicted_labels[i]] += 1

    acc = 1.0 * np.sum([confusion[i][i] for i in range(4)]) / np.sum(confusion)
    true_positive = 0
    for i in range(len(label_type)-1):
        true_positive += confusion[i][i]
    prec = true_positive/(np.sum(confusion)-np.sum(confusion, axis=0)[-1])
    rec = true_positive/(np.sum(confusion)-np.sum(confusion[-1][:]))
    f1 = 2*prec*rec / (rec+prec)

    return acc, prec, rec, f1, confusion


def evaluate(model, model_type, dataloader, device, dataset, threshold=None):
    model.eval()
    label_type = get_label_type(dataset)
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            batch = [x.to(device) for x in batch]
            inputs = _gather_model_inputs(model_type, batch)
            outputs = model(**inputs)
            loss, logits = outputs[0], outputs[1]
            all_logits.append(logits)
            all_labels.append(inputs['labels'])

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    predicted_logits, predicted_labels = torch.max(all_logits, dim=1)

    acc, prec, rec, f1, confusion = calc_f1(
        predicted_labels, all_labels, label_type)
    return acc, prec, rec, f1, confusion, threshold


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


def train(args, model, train_dataloader, dev_dataloader, test_dataloader, device, n_gpu):
    num_training_steps_per_epoch = ceil(
        len(train_dataloader.dataset)/float(args.update_batch_size))
    num_training_steps = args.num_train_epochs * num_training_steps_per_epoch

    scheduler, optimizer = setup_scheduler_optimizer(model=model,
                                                     num_warmup_ratio=args.num_warmup_ratio,
                                                     num_training_steps=num_training_steps,
                                                     lr=args.lr, beta1=args.beta1,
                                                     beta2=args.beta2, eps=args.eps,
                                                     weight_decay=args.weight_decay)

    global_step = 0
    best_acc = 0.
    update_per_batch = args.update_batch_size // args.batch_size
    vat_params = {
    'emb_name': 'roberta_embeddings', 
    'noise_var': 1e-5,#1e-4
    'noise_gamma': 1e-6,#1e-5
    'adv_step_size': 1e-3,#1e-2
    'adv_alpha': 1,
    'norm_type': 'l2'
                    }
    vat = VAT(model=model, **vat_params)
    
    moco_params = {
    'MoCo_beta': 0.5,  #0.5
    'MoCo_beta_sigma': 0.5,#0.5
    'MoCo_gamma': 0.1,
    'MoCo_gamma_sigma': 0.5,
    'MoCo_rho': 0,
                    }

    for epoch in range(1, args.num_train_epochs+1, 1):
        model.train()
        global_loss = 0.
        for i, batch in tqdm(enumerate(train_dataloader),
                             desc=f'Running train for epoch {epoch}',
                             total=len(train_dataloader)):
            batch = [x.to(device) for x in batch]
            #print(batch)
           


            inputs = _gather_model_inputs(args.model_type, batch)
            outputs = model(**inputs)
            loss, logits = outputs[0], outputs[1]

            loss /= update_per_batch
            


            vat_loss = vat.virtual_adversarial_training(inputs, logits)
            vat_loss = 1*vat_loss
            #loss = loss + 10*vat_loss
            losses = []
            losses.append(loss)
            losses.append(vat_loss)
            moco = MoCo()
            moco.init_param(device)
            moco.backward(losses,**moco_params)
            #loss.backward()

            if (i+1) % update_per_batch == 0 or (i+1) == len(train_dataloader):
                # global_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                global_loss = 0
        logging.info(f"Evaluation for epoch {epoch}")
        dev_metrics = evaluate(model, args.model_type,
                               dev_dataloader, device, args.dataset)
        dev_acc, dev_prec, dev_rec, dev_f1, dev_confusion, dev_threshold = dev_metrics
        logging.info(
            f"Acc={dev_acc}, Precision={dev_prec}, Recall={dev_rec}, F1={dev_f1}")
        logging.info(f"Confusion={dev_confusion}")
        if dev_f1 > best_acc:
            logging.info(f"New best, dev_f1={dev_f1} > best_f1={best_acc}")
            best_acc = dev_f1
            if n_gpu > 1:
                model.module.save_pretrained(args.output_dir)
            else:
                model.save_pretrained(args.output_dir)
            logging.info(f"Best model saved in {args.output_dir}")


def main(input_args: Optional[List[str]] = None):
    args = _get_validated_args(input_args)
    try:
        logging.info(f"Current Git Hash: {get_git_revision_hash()}")
    except:
        pass
    device, n_gpu = setup_cuda_device(args.no_cuda)
    set_random_seed(args.seed, n_gpu)

    model_name = args.model_name if args.cache_dir is None else \
        os.path.join(args.cache_dir, args.model_name)
    model = prepare_model(model_name, args.model_type,
                          device, args.model_weights, args.dataset)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    tokenizer = setup_tokenizer(
        model_name=args.model_name, cache_dir=args.cache_dir)
    if args.dataset == "TRMF":
        train_tensorset, dev_tensorset,test_tensorset = _get_tensorset(
            tokenizer)

    

    

    # 数据集文件名列表
    #dataset_files = ["deleted_sentence.xml", "inserted_sentence.xml", "swapped_sentence.xml", "synonym_replacement.xml", "testset-temprel.xml"]
    #dataset_files = ["deleted_sentence.xml","inserted_sentence.xml","testset-temprel.xml"]
    #dataset_files = ["testset-temprel.xml"]
    #dataset_files = ["tcr-temprel.xml"]
    #dataset_files = ["testoutput999.xml"]
    dataset_files = ["test_new copy.xml"]

    
    test_tensorsets = []

    
    for dataset_file in dataset_files:
        dataset_path = os.path.join("/home/JJJ/MFRV/data/new_data", dataset_file)
        testset = temprel_set(dataset_path)
        test_tensorset = testset.to_tensor(tokenizer=tokenizer)
       
        test_tensorsets.append(test_tensorset) 

    ######
    train_dataloader = DataLoader(
        train_tensorset, batch_size=args.batch_size, shuffle=True)
    dev_dataloader = DataLoader(
        dev_tensorset, batch_size=args.batch_size//2, shuffle=False)
    test_dataloader = DataLoader(
        test_tensorsets, batch_size=args.batch_size//2, shuffle=False)

    if args.do_train:
        train(args, model, train_dataloader, dev_dataloader,
              test_dataloader, device, n_gpu)
    #if args.do_eval:
     #   model = prepare_model_eval(model_name, args.model_type,
      #                    device, args.model_weights, args.dataset)
       # test_metrics = evaluate(model, args.model_type, test_dataloader, device, args.dataset)
        #test_acc, test_prec, test_rec, test_f1, test_confusion, test_threshold = test_metrics
        #logging.info(f"Acc={test_acc}, Precision={test_prec}, Recall={test_rec}, F1={test_f1}")
        #logging.info(f"Confusion={test_confusion}")
    
    if args.do_eval:
        model = prepare_model_eval(model_name, args.model_type,
                          device, args.model_weights, args.dataset)
        
        num_augmentations = 1  # 可以根据需要调整次数
        label_type = get_label_type(args.dataset)
       
        all_predictions = []
        test_results = []
        for _ in range(num_augmentations):
            for test_tensorset in test_tensorsets: 
                test_dataloader = DataLoader(
                    test_tensorset, batch_size=args.batch_size//2, shuffle=False)
                #test_metrics = evaluate(model, args.model_type,
                 #                       test_dataloader, device, args.dataset)
                
                #test_acc, test_prec, test_rec, test_f1, test_confusion, test_threshold = test_metrics
                predicted_labels,all_labels,wrong_preds = get_predictions(model, test_dataloader, device, args.model_type)
                all_predictions.append(predicted_labels)
                for index, pred, actual in wrong_preds:
                    test_results.append((index, pred, actual))
                    print(f"Index {index}: Predicted label {pred} does not match actual label {actual}")
        weights = [6, 2, 10, 1, 8]
        
        final_predictions = aggregate_predictions(all_predictions)

       
        
        acc, prec, rec, f1, confusion = calc_f1(
        final_predictions, all_labels, label_type)
        logging.info(
            f"Acc={acc}, Precision={prec}, Recall={rec}, F1={f1}")
        logging.info(f"Confusion={confusion}")
         
        """
        #################################### 
        test_metrics = evaluate(model, args.model_type,
                                test_dataloader, device, args.dataset)
        test_acc, test_prec, test_rec, test_f1, test_confusion, test_threshold = test_metrics
        logging.info(
            f"Acc={test_acc}, Precision={test_prec}, Recall={test_rec}, F1={test_f1}")
        logging.info(f"Confusion={test_confusion}")
        """

def get_predictions(model, dataloader, device, model_type, threshold=None):
    model.eval()
    all_logits, all_labels = [], []
    wrong_preds = []
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            batch = [x.to(device) for x in batch]
            inputs = _gather_model_inputs(model_type, batch)
            outputs = model(**inputs)
            loss, logits = outputs[0], outputs[1]
            all_logits.append(logits)
            all_labels.append(inputs['labels'])

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    predicted_logits, predicted_labels = torch.max(all_logits, dim=1)

    # 可以根据需要应用阈值处理
    if threshold is not None:
        predicted_labels = apply_threshold(predicted_logits, threshold)
    for i in range(len(predicted_labels)):
        if predicted_labels[i] != all_labels[i]:
            wrong_preds.append((i, predicted_labels[i].item(), all_labels[i].item()))
    return predicted_labels,all_labels,wrong_preds



def aggregate_predictions(predictions_list):
    # 这里可以根据需要选择投票或其他汇总方式
    # 这里示例使用投票
    final_predictions = torch.mode(torch.stack(predictions_list), dim=0).values
    return final_predictions
def aggregate_predictions_weighted(predictions_list, weights):
    # 对每个数据集的预测结果进行加权
    weighted_predictions = [w * pred for w, pred in zip(weights, predictions_list)]
    
    # 将加权结果叠加在一起
    weighted_sum = torch.stack(weighted_predictions).sum(dim=0)
    
    # 计算最终的预测结果
    final_predictions = torch.round(weighted_sum / sum(weights))
    
    return final_predictions


def apply_threshold(logits, threshold):
    return (logits >= threshold).long()
# a neural network to extract commonsense knowledge
# The model has been pre-trained
class bigramGetter_fromNN:
    def __init__(self, device, emb_path, mdl_path, ratio=0.3, layer=1, emb_size=200, splitter=','):
        self.verb_i_map = {}
        self.device = device
        f = open(emb_path)
        lines = f.readlines()
        for i, line in enumerate(lines):
            self.verb_i_map[line.split(splitter)[0]] = i
        f.close()
        self.model = VerbNet(
            len(self.verb_i_map), hidden_ratio=ratio, emb_size=emb_size, num_layers=layer)
        self.model.to(self.device)
        checkpoint = torch.load(mdl_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def eval(self, v1, v2):
        return self.model(torch.from_numpy(np.array([[self.verb_i_map[v1], self.verb_i_map[v2]]])).to(self.device))

    def getBigramStatsFromTemprel(self, temprel):
        if type(temprel.lemma) == type((0, 1)):
            v1 = temprel.lemma[0]
            v2 = temprel.lemma[1]
        else:
            v1, v2 = '', ''
            for i, position in enumerate(temprel.position):
                if position == 'E1':
                    v1 = temprel.lemma[i]
                elif position == 'E2':
                    v2 = temprel.lemma[i]
                    break
        if v1 not in self.verb_i_map or v2 not in self.verb_i_map:
            return torch.tensor([0, 0]).view(1, -1).to(self.device)
        return torch.cat((self.eval(v1, v2), self.eval(v2, v1)), 1).view(1, -1)

    def retrieveEmbeddings(self, temprel):
        if type(temprel.lemma) == type((0, 1)):
            v1 = temprel.lemma[0]
            v2 = temprel.lemma[1]
        else:
            v1, v2 = '', ''
            for i, position in enumerate(temprel.position):
                if position == 'E1':
                    v1 = temprel.lemma[i]
                elif position == 'E2':
                    v2 = temprel.lemma[i]
                    break
        if v1 not in self.verb_i_map or v2 not in self.verb_i_map:
            return torch.zeros_like(self.model.retrieveEmbeddings(torch.from_numpy(np.array([[0, 0]])).to(self.device)).view(1, -1))
        return self.model.retrieveEmbeddings(torch.from_numpy(np.array([[self.verb_i_map[v1], self.verb_i_map[v2]]])).to(self.device)).view(1, -1)



class VAT:
    def __init__(self, model, emb_name, noise_var, noise_gamma, adv_step_size, adv_alpha, norm_type):
        self.model = model
        self.emb_name = emb_name
        self.noise_var = noise_var
        self.noise_gamma = noise_gamma
        self.adv_step_size = adv_step_size
        self.adv_alpha = adv_alpha
        self.norm_type = norm_type

    def virtual_adversarial_training(self, inputs, logits):
        # Generate virtual adversarial perturbation
        vat_loss = self.generate_virtual_adversarial_perturbation(inputs, logits)
        
        # Adversarial training
        perturbed_logits = self.model(**inputs)[1]
        adv_loss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(perturbed_logits, dim=-1), torch.nn.functional.softmax(logits, dim=-1), reduction='batchmean')

        return vat_loss + self.adv_alpha * adv_loss

    def generate_virtual_adversarial_perturbation(self, inputs, logits):
        #emb = self.model.roberta_embeddings.weight
        emb = self.model.roberta.embeddings.word_embeddings.weight

        #emb = getattr(self.model, self.emb_name).weight
        emb_dim = emb.size(1)

        # Generate random noise
        d = torch.randn_like(emb)
        d = self.normalize(d)
        noise_gamma_int = int(self.noise_gamma)
        # Iterative perturbation
        for _ in range(noise_gamma_int):
            d.requires_grad_()
            perturbed_inputs = self.perturb_inputs(inputs, d)
            perturbed_logits = self.model(**perturbed_inputs)[1]
            adv_loss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(perturbed_logits, dim=-1), torch.nn.functional.softmax(logits, dim=-1), reduction='batchmean')
            adv_grad = torch.autograd.grad(adv_loss, d, retain_graph=False)[0]

            d = self.project_noise(d, adv_grad)

        # Scale the noise
        d = self.noise_var * self.adv_step_size * self.normalize(d)

        # Perturb the embeddings
        perturbed_emb = emb + d
        setattr(self.model, self.emb_name, torch.nn.Parameter(perturbed_emb))

        # Calculate VAT loss
        perturbed_logits = self.model(**inputs)[1]
        vat_loss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(perturbed_logits, dim=-1), torch.nn.functional.softmax(logits, dim=-1), reduction='batchmean')

        # Reset the embeddings
        setattr(self.model, self.emb_name, torch.nn.Parameter(emb))

        return vat_loss

    def perturb_inputs(self, inputs, perturbation):
        perturbed_inputs = {}
        for key, value in inputs.items():
            perturbed_inputs[key] = value + perturbation
        return perturbed_inputs

    def normalize(self, x):
        norm = x.norm(p=2, dim=-1, keepdim=True)
        return x / norm

    def project_noise(self, d, adv_grad):
        d = d - self.adv_step_size * self.normalize(adv_grad)
        return self.normalize(d)



class ReinforcementAgent:
    def __init__(self, model, reward_scale=1.0):
        self.model = model
        self.reward_scale = reward_scale

    def compute_rewards(self, logits, labels, event_1_time, event_2_time):
        
        causality_reward = compute_temporal_reward(logits, labels, event_1_time, event_2_time)
        alignment_reward = compute_alignment_reward(logits, labels, event_1_time, event_2_time)
        
        return causality_reward, alignment_reward

    def update(self, rewards):
      
        total_reward = sum(rewards)
        scaled_reward = total_reward * self.reward_scale
        
      
        return scaled_reward

def compute_temporal_reward(logits, labels, event_1_time, event_2_time):
   
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits, labels)
    
   
    return -loss.item()

def compute_alignment_reward(logits, labels, event_1_time, event_2_time, weight=1.0):
   
    direction = torch.sign(event_2_time - event_1_time)
    
    
    absolute_difference = torch.abs(event_2_time - event_1_time)
    
   
    weighted_difference = weight * absolute_difference
    
    
    alignment_reward = direction * weighted_difference.mean().item()
    
    return alignment_reward


class MoCo(AbsWeighting):
    r"""MoCo.
    
    This method is proposed in `Mitigating Gradient Bias in Multi-objective Learning: A Provably Convergent Approach (ICLR 2023) <https://openreview.net/forum?id=dLAYGdKTi2>`_ \
    and implemented based on the author' sharing code (Heshan Fernando: fernah@rpi.edu). 

    Args:
        MoCo_beta (float, default=0.5): The learning rate of y.
        MoCo_beta_sigma (float, default=0.5): The decay rate of MoCo_beta.
        MoCo_gamma (float, default=0.1): The learning rate of lambd.
        MoCo_gamma_sigma (float, default=0.5): The decay rate of MoCo_gamma.
        MoCo_rho (float, default=0): The \ell_2 regularization parameter of lambda's update.

    .. warning::
            MoCo is not supported by representation gradients, i.e., ``rep_grad`` must be ``False``.

    """

    def __init__(self):
        super(MoCo, self).__init__()

    def init_param(self,device):
        self.device = device
        self._compute_grad_dim()
        self.task_num = 2
        self.step = 0
        self.y = torch.zeros(self.task_num, self.grad_dim).to(self.device)
        self.lambd = (torch.ones([self.task_num, ]) / self.task_num).to(self.device)
        
    def backward(self, losses, **kwargs):
        self.step += 1
        beta, beta_sigma = kwargs['MoCo_beta'], kwargs['MoCo_beta_sigma']
        gamma, gamma_sigma = kwargs['MoCo_gamma'], kwargs['MoCo_gamma_sigma']
        rho = kwargs['MoCo_rho']

        #if self.rep_grad:
         #   raise ValueError('No support method MoCo with representation gradients (rep_grad=True)')
        #else:
        self._compute_grad_dim()
        grads = self._compute_grad(losses, mode='backward')

        with torch.no_grad():
            for tn in range(self.task_num):
                grads[tn] = grads[tn]/(grads[tn].norm()+1e-8)*losses[tn]
        self.y = self.y - (beta/self.step**beta_sigma) * (self.y - grads)
        self.lambd = torch.nn.functional.softmax(self.lambd - (gamma/self.step**gamma_sigma) * (self.y@self.y.t()+rho*torch.eye(self.task_num).to(self.device))@self.lambd, -1)
        new_grads = self.y.t()@self.lambd

        self._reset_grad(new_grads)
        return self.lambd.detach().cpu().numpy()

    def get_share_params(self):
      
        return self.parameters() 

    def zero_grad_share_params(self):
        for param in self.get_share_params():
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()









if __name__ == "__main__":
    main()
