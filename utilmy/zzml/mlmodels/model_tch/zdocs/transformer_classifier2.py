# -*- coding: utf-8 -*-
"""



"""

from __future__ import absolute_import, division, print_function

import glob
from jsoncomment import JsonComment ; json = JsonComment()
import logging
import math
import os
import random

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import (confusion_matrix, matthews_corrcoef,
                             mean_squared_error)
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              SubsetRandomSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, tqdm_notebook, trange

from pytorch_transformers import (
    WEIGHTS_NAME, AdamW, BertConfig, BertForSequenceClassification,
    BertTokenizer, RobertaConfig, RobertaForSequenceClassification,
    RobertaTokenizer, WarmupLinearSchedule, XLMConfig,
    XLMForSequenceClassification, XLMTokenizer, XLNetConfig,
    XLNetForSequenceClassification, XLNetTokenizer)
from tensorboardX import SummaryWriter
from util_transformer import (convert_examples_to_features, output_modes,
                              processors)

###################################################################################################
MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}



####################################################################################################
# Helper functions
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



from mlmodels.util import os_package_root_path, log, path_norm



####################################################################################################
class Model:
  def __init__(self, model_pars=None, data_pars=None):
        # 4.Define Model    
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args['model_type']]

        self.config = config_class.from_pretrained(args['model_name'], num_labels=2, 
                                                   finetuning_task=args['task_name'])
        self.tokenizer = tokenizer_class.from_pretrained(args['model_name'])

        self.model = model_class.from_pretrained(args['model_name'])
        self.model.to(device)
 







##################################################################################################
def _preprocess_XXXX(df, **kw):
    return df, linear_cols, dnn_cols, train, test, target



def get_dataset(data_pars=None, **kw):
    processor   = processors[task]()
    output_mode = args['output_mode']
    mode        = 'dev' if evaluate else 'train'
    cached_features_file = os.path.join(args['data_dir'], f"cached_{mode}_{args['model_name']}_{args['max_seq_length']}_{task}")
    
    if os.path.exists(cached_features_file) and not args['reprocess_input_data']:
        log("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
               
    else:
        log("Creating features from dataset file at %s", args['data_dir'])
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(args['data_dir']) if evaluate else processor.get_train_examples(args['data_dir'])
        
        if __name__ == "__main__":
            features = convert_examples_to_features(examples, label_list, args['max_seq_length'], tokenizer, output_mode,
                cls_token_at_end=bool(args['model_type'] in ['xlnet']),            # xlnet has a cls token at the end
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=2 if args['model_type'] in ['xlnet'] else 0,
                sep_token=tokenizer.sep_token,
                sep_token_extra=bool(args['model_type'] in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                pad_on_left=bool(args['model_type'] in ['xlnet']),                 # pad on the left for xlnet
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if args['model_type'] in ['xlnet'] else 0)
        
        log("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
        
    all_input_ids    = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask   = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids  = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def fit(model, data_pars=None, model_pars={}, compute_pars=None, out_pars=None, *args, **kw):
    tb_writer        = SummaryWriter()
    torch.manual_seed(1)
    random_indices = torch.randperm(len(train_dataset))[:args['num_samples']]
    # train_sampler    = RandomSampler(train_dataset)
    train_sampler    = SubsetRandomSampler(random_indices)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args['train_batch_size'])
    
    t_total = len(train_dataloader) // args['gradient_accumulation_steps'] * args['num_train_epochs']
    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    
    warmup_steps = math.ceil(t_total * args['warmup_ratio'])
    args['warmup_steps'] = warmup_steps if args['warmup_steps'] == 0 else args['warmup_steps']
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=args['adam_epsilon'])
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args['warmup_steps'], t_total=t_total)
    
    if args['fp16']:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args['fp16_opt_level'])
        
    log("***** Running training *****")
    log("  Num examples = %d", len(train_dataset))
    log("  Num Epochs = %d", args['num_train_epochs'])
    log("  Total train batch size  = %d", args['train_batch_size'])
    log("  Gradient Accumulation steps = %d", args['gradient_accumulation_steps'])
    log("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args['num_train_epochs']), desc="Epoch")
    
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch  = tuple(t.to(device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                      'labels':         batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            print("\r%f" % loss, end='')

            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']

            if args['fp16']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args['max_grad_norm'])
                
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])

            tr_loss += loss.item()
            if (step + 1) % args['gradient_accumulation_steps'] == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1


                if args['logging_steps'] > 0 and global_step % args['logging_steps'] == 0:
                    # Log metrics
                    if args['evaluate_during_training']:  # Only evaluate when single GPU otherwise metrics may not average well
                        results, _ = evaluate(model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args['logging_steps'], global_step)
                    logging_loss = tr_loss


                if args['save_steps'] > 0 and global_step % args['save_steps'] == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args['output_dir'], 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    log("Saving model checkpoint to %s", output_dir)
                    
                


    return global_step, tr_loss / global_step



def predict(model, sess=None, data_pars=None, out_pars=None, compute_pars=None, **kw):
    ##  Model is class
    ## load test dataset
    pass

    """
    data, linear_cols, dnn_cols, train, test, target = get_dataset(**data_pars)
    feature_names = get_feature_names(linear_cols + dnn_cols, )
    test_model_input = {name: test[name] for name in feature_names}

    multiple_value = data_pars.get('multiple_value', None)
    ## predict
    if multiple_value is None:
        pred_ans = model.model.predict(test_model_input, batch_size= compute_pars['batch_size'])
    else:
        pred_ans = None

    return pred_ans
    """




def get_mismatched(labels, preds):
    mismatched = labels != preds
    examples = processor.get_dev_examples(args['data_dir'])
    wrong = [i for (i, v) in zip(examples, mismatched) if v]
    
    return wrong

def get_eval_report(labels, preds):
    mcc = matthews_corrcoef(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    return {
        "mcc": mcc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }, get_mismatched(labels, preds)


def metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    return get_eval_report(labels, preds)


def evaluate(model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args['output_dir']

    results = {}
    EVAL_TASK = args['task_name']

    eval_dataset = load_and_cache_examples(EVAL_TASK, tokenizer, evaluate=True)
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)


    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args['eval_batch_size'])

    # Eval!
    log("***** Running evaluation {} *****".format(prefix))
    log("  Num examples = %d", len(eval_dataset))
    log("  Batch size = %d", args['eval_batch_size'])
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm_notebook(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                      'labels':         batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    if args['output_mode'] == "classification":
        preds = np.argmax(preds, axis=1)

    elif args['output_mode'] == "regression":
        preds = np.squeeze(preds)

    result, wrong = metrics(EVAL_TASK, preds, out_label_ids)
    results.update(result)

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        log("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            log("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return results, wrong



def reset_model():
    pass




########################################################################################################################
class Model_empty(object):
    def __init__(self, model_pars=None, compute_pars=None):
        ## Empty model for Seaialization
        self.model = None


def save(model, out_pars):
    # Save model checkpoint
    output_dir = path  # = os.path.join(args['output_dir'], 'checkpoint-{}'.format(global_step))
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    log("Saving model checkpoint to %s", output_dir)



def load(out_pars=None):
    if not os.path.exists(path):
        print("model file do not exist!")
        return None
    else:
        model = Model_empty()
        model_keras = load_model(path, custom_objects)

        #### Add back the model parameters...
        return model


########################################################################################################################
def path_setup(out_folder="", sublevel=0, data_path="dataset/"):
    #### Relative path
    data_path = os_package_root_path( path_add=data_path)
    out_path = os.getcwd() + "/" + out_folder
    os.makedirs(out_path, exist_ok=True)
    log(data_path, out_path)
    return data_path, out_path


def get_params(choice=0, data_path="dataset/", **kw):
    if choice == 0:
        log("#### Path params   ###################################################")
        data_path, out_path = path_setup(out_folder="/deepctr_test/", data_path=data_path)

        train_data_path = data_path + "criteo_sample.txt"
        data_pars = {"train_data_path": train_data_path, "dataset_type": "criteo", "test_size": 0.2}

        log("#### Model params   #################################################")
        model_pars = {"model_type": "DeepFM", "optimization": "adam", "cost": "binary_crossentropy"}
        compute_pars = {"task": "binary", "batch_size": 256, "epochs": 10, "validation_split": 0.2}
        out_pars = {"path": out_path}

    return model_pars, data_pars, compute_pars, out_pars


def metrics_evaluate():
    log("#### metrics   ####################################################")
    results = {}
    if args['do_eval']:
        checkpoints = [args['output_dir']]
        if args['eval_all_checkpoints']:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args['output_dir'] + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging

        log("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(device)
            result, wrong_preds = evaluate(model, tokenizer, prefix=global_step)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)
    print(results)





########################################################################################################################
########################################################################################################################
def test(data_path="dataset/", pars_choice=0):
    ### Local test
    log("#### Loading params   ##############################################")
    task = args['task_name']

    if task in processors.keys() and task in output_modes.keys():
        processor = processors[task]()
        label_list = processor.get_labels()
        num_labels = len(label_list)
    else:
        raise KeyError(f'{task} not found in processors or in output_modes. Please check utils.py.')

    

    
    log("#### Model init, fit   #############################################")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(args,device)

    log("#### Loading dataset   #############################################")
    if args['do_train']:
        train_dataset = get_dataset(task, model.tokenizer)
        global_step, tr_loss = fit(train_dataset, model.model, model.tokenizer)
        log(" global_step = %s, average loss = %s", global_step, tr_loss)
       


    # log("#### Predict   ####################################################")
    # ypred = predict(model, data_pars, compute_pars, out_pars)
    log("#### Save/Load   ##################################################")
    if args['do_train']:
        if not os.path.exists(args['output_dir']):
                os.makedirs(args['output_dir'])
        log("Saving model checkpoint to %s", args['output_dir'])
        
        model_to_save = model.model.module if hasattr(model.model, 'module') else model.model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args['output_dir'])
        model.tokenizer.save_pretrained(args['output_dir'])
        torch.save(args, os.path.join(args['output_dir'], 'training_args.bin')) 


    log("#### metrics   ####################################################")
    metrics_evaluate()



    log("#### Plot   #######################################################")


    




if __name__ == '__main__':
    VERBOSE = True

    # Constants 
    with open('args.json', 'r') as f:
      args = json.load(f)

    if os.path.exists(args['output_dir']) and os.listdir(args['output_dir']) and args['do_train'] and not args['overwrite_output_dir']:
      raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args['output_dir']))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test(pars_choice=0)
