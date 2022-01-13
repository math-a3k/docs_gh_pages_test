


# coding: utf-8
"""



"""
import glob
from jsoncomment import JsonComment ; json = JsonComment()
import logging
import math
import os
import random

import numpy as np

from tqdm import tqdm, tqdm_notebook, trange
from scipy.stats import pearsonr
from sklearn.metrics import (confusion_matrix, matthews_corrcoef,
                             mean_squared_error)


import torch

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              SubsetRandomSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler

from pytorch_transformers import (
    WEIGHTS_NAME, AdamW, BertConfig, BertForSequenceClassification,
    BertTokenizer, RobertaConfig, RobertaForSequenceClassification,
    RobertaTokenizer, WarmupLinearSchedule, XLMConfig,
    XLMForSequenceClassification, XLMTokenizer, XLNetConfig,
    XLNetForSequenceClassification, XLNetTokenizer)

from tensorboardX import SummaryWriter
from util_transformer import (convert_examples_to_features, output_modes,
                              processors)

####################################################################################################
from mlmodels.util import os_package_root_path, log, path_norm, get_model_uri

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



VERBOSE = False
MODEL_URI = get_model_uri(__file__)


MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}




####################################################################################################





####################################################################################################
class Model:
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None
               ):
        # 4.Define Model    
        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_pars['model_type']]

        self.config = config_class.from_pretrained(model_pars['model_name'], num_labels=2, 
                                                   finetuning_task=model_pars['task_name'])
        self.tokenizer = tokenizer_class.from_pretrained(model_pars['model_name'])
        # downloads the pretrained model and stores it in the cache directory
        self.model = model_class.from_pretrained(model_pars['model_name'],cache_dir=model_pars["cache_dir"] )
        self.model.to(device)
 

##################################################################################################
def _preprocess_XXXX(df, **kw):
    return df, linear_cols, dnn_cols, train, test, target

def load_and_cache_examples(task, tokenizer, evaluate=False):
    processor = processors[task]()
    output_mode = args['output_mode']
    
    mode = 'dev' if evaluate else 'train'
    cached_features_file = os.path.join(data_pars['data_dir'], f"cached_{mode}_{model_pars['model_name']}_{model_pars['max_seq_length']}_{task}")
    
    if os.path.exists(cached_features_file) and not args['reprocess_input_data']:
        log("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
               
    else:
        log("Creating features from dataset file at %s", data_pars['data_dir'])
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(data_pars['data_dir']) if evaluate else processor.get_train_examples(data_pars['data_dir'])
        
        
        features = convert_examples_to_features(examples, label_list, model_pars['max_seq_length'], tokenizer, output_mode,
            cls_token_at_end=bool(model_pars['model_type'] in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if model_pars['model_type'] in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(model_pars['model_type'] in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(model_pars['model_type'] in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if model_pars['model_type'] in ['xlnet'] else 0)
        
        log("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
        
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset

def get_dataset(task, tokenizer, evaluate=False):
    processor   = processors[task]()
    output_mode = model_pars['output_mode']
    mode        = 'dev' if evaluate else 'train'
    cached_features_file = os.path.join(data_pars['data_dir'], f"cached_{mode}_{model_pars['model_name']}_{model_pars['max_seq_length']}_{task}")
    
    if os.path.exists(cached_features_file) and not compute_pars['reprocess_input_data']:
        log("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
               
    else:
        log("Creating features from dataset file at %s", data_pars['data_dir'])
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(data_pars['data_dir']) if evaluate else processor.get_train_examples(data_pars['data_dir'])
        

        features = convert_examples_to_features(examples, label_list, model_pars['max_seq_length'], tokenizer, output_mode,
            cls_token_at_end=bool(model_pars['model_type'] in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if model_pars['model_type'] in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(model_pars['model_type'] in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(model_pars['model_type'] in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if model_pars['model_type'] in ['xlnet'] else 0)
        
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


def fit(train_dataset, model, tokenizer):
    tb_writer        = SummaryWriter()
    torch.manual_seed(1)
    random_indices = torch.randperm(len(train_dataset))[:compute_pars['num_samples']]
    # train_sampler    = RandomSampler(train_dataset)
    train_sampler    = SubsetRandomSampler(random_indices)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=compute_pars['train_batch_size'])
    
    t_total = len(train_dataloader) // compute_pars['gradient_accumulation_steps'] * compute_pars['num_train_epochs']
    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': compute_pars['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    
    warmup_steps = math.ceil(t_total * compute_pars['warmup_ratio'])
    compute_pars['warmup_steps'] = warmup_steps if compute_pars['warmup_steps'] == 0 else compute_pars['warmup_steps']
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=compute_pars['learning_rate'], eps=compute_pars['adam_epsilon'])
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=compute_pars['warmup_steps'], t_total=t_total)
    
    if model_pars['fp16']:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=model_pars['fp16_opt_level'])
        
    log("***** Running training *****")
    log("  Num examples = %d", len(train_dataset))
    log("  Num Epochs = %d", compute_pars['num_train_epochs'])
    log("  Total train batch size  = %d", compute_pars['train_batch_size'])
    log("  Gradient Accumulation steps = %d", compute_pars['gradient_accumulation_steps'])
    log("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(compute_pars['num_train_epochs']), desc="Epoch")
    
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch  = tuple(t.to(device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if model_pars['model_type'] in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                      'labels':         batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            print("\r%f" % loss, end='')

            if compute_pars['gradient_accumulation_steps'] > 1:
                loss = loss / compute_pars['gradient_accumulation_steps']

            if model_pars['fp16']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), compute_pars['max_grad_norm'])
                
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), compute_pars['max_grad_norm'])

            tr_loss += loss.item()
            if (step + 1) % compute_pars['gradient_accumulation_steps'] == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1


                if compute_pars['logging_steps'] > 0 and global_step % compute_pars['logging_steps'] == 0:
                    # Log metrics
                    if compute_pars['evaluate_during_training']:  # Only evaluate when single GPU otherwise metrics may not average well
                        results, _ = evaluate(model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/compute_pars['logging_steps'], global_step)
                    logging_loss = tr_loss


                if compute_pars['save_steps'] > 0 and global_step % compute_pars['save_steps'] == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(out_pars['output_dir'], 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    log("Saving model checkpoint to %s", output_dir)
                    
                


    return global_step, tr_loss / global_step



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


def evaluate(model, tokenizer, model_pars,data_pars, out_pars, compute_pars, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = data_pars['output_dir']

    results = {}
    EVAL_TASK = model_pars['task_name']

    eval_dataset = load_and_cache_examples(EVAL_TASK, tokenizer, evaluate=True)
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)


    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args['eval_batch_size'])

    # Eval!
    log("***** Running evaluation {} *****".format(prefix))
    log("  Num examples = %d", len(eval_dataset))
    log("  Batch size = %d", compute_pars['eval_batch_size'])
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

    

# def predict(model, sess=None, data_pars=None, out_pars=None, compute_pars=None, **kw):
#   ##### Get Data ###############################################
#   data_pars['train'] = False
#   Xpred, ypred = get_dataset(data_pars)

#   #### Do prediction
#   ypred = model.model.predict(Xpred)


#   ### Save Results
  
  
#   ### Return val
#   if compute_pars.get("return_pred_not") is not None :
#     return ypred


  
  
def reset_model():
  pass





def save(model=None, session=None, save_pars={}):
    from mlmodels.util import save_keras
    print(save_pars)
    save_keras(session, save_pars['path'])
     


def load(load_pars={}):
    from mlmodels.util import load_keras
    print(load_pars)
    model0 =  load_keras(load_pars['path'])

    model = Model()
    model.model = model0
    session = None
    return model, session






def get_params(param_pars={}, **kw):
    from jsoncomment import JsonComment ; json = JsonComment()
    choice      = param_pars['choice']
    config_mode = param_pars['config_mode']
    data_path   = param_pars['data_path']


    if choice == "json":
       cf = json.load(open(data_path, mode='r'))
       cf = cf[config_mode]
       return cf['model_pars'], cf['data_pars'], cf['compute_pars'], cf['out_pars']


    if choice == "test01":
        log("#### Path params   ##########################################")
        data_path  = path_norm( "dataset/text/imdb.csv"  )   
        out_path   = path_norm( "/ztest/model_tch/transformer_classifier/" )
        model_path = os.path.join(out_path , "model")


        data_pars    = {"path" : data_path, "train": 1, "maxlen":400, "max_features": 10, }

        model_pars   = {"maxlen":400, "max_features": 10, "embedding_dims":50,

                       }
                       
        compute_pars = {"engine": "adam", "loss": "binary_crossentropy", "metrics": ["accuracy"] ,
                        "batch_size": 32, "epochs":1
                       }

        out_pars     = {"path": out_path,  "model_path": model_path}

        return model_pars, data_pars, compute_pars, out_pars

    else:
        raise Exception(f"Not support choice {choice} yet")




########################################################################################################################
def test(data_path,model_pars, data_pars, compute_pars, out_pars, pars_choice=0):
    ### Local test
    log("#### Loading params   ##############################################")
    task = model_pars['task_name']

    if task in processors.keys() and task in output_modes.keys():
        processor = processors[task]()
        label_list = processor.get_labels()
        num_labels = len(label_list)
    else:
        raise KeyError(f'{task} not found in processors or in output_modes. Please check utils.py.')

    

    
    log("#### Model init, fit   #############################################")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(model_pars, data_pars, compute_pars)

    log("#### Loading dataset   #############################################")
    if model_pars['do_train']:
        train_dataset = get_dataset(task, model.tokenizer)
        global_step, tr_loss = fit(train_dataset, model.model, model.tokenizer)
        log(" global_step = %s, average loss = %s", global_step, tr_loss)
       

    # log("#### Predict   ####################################################")
    # ypred = predict(model, data_pars, compute_pars, out_pars)
    log("#### Save/Load   ##################################################")
    if model_pars['do_train']:
        if not os.path.exists(out_pars['output_dir']):
                os.makedirs(out_pars['output_dir'])
        log("Saving model checkpoint to %s", out_pars['output_dir'])
        
        model_to_save = model.model.module if hasattr(model.model, 'module') else model.model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(out_pars['output_dir'])
        model.tokenizer.save_pretrained(out_pars['output_dir'])
        torch.save(out_pars, os.path.join(out_pars['output_dir'], 'training_args.bin')) 


    log("#### metrics   ####################################################")
    results = {}
    if model_pars['do_eval']:
        checkpoints = [out_pars['output_dir']]
        if compute_pars['eval_all_checkpoints']:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(out_pars['output_dir'] + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging

        log("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model_eval = model.model.from_pretrained(checkpoint)
            model_eval.to(device)
            result, wrong_preds = evaluate(model_eval, model.tokenizer,model_pars, data_pars, compute_pars, out_pars, prefix=global_step)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)
    print(results)




    




if __name__ == '__main__':
    VERBOSE = True

    # Constants 
    param_pars = {'choice': "json", 'config_mode' : 'test', 'data_path' : 'model_tch/transformer_classifier.json' }
    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)
    if os.path.exists(out_pars['output_dir']) and os.listdir(out_pars['output_dir']) and model_pars['do_train'] and not compute_pars['overwrite_output_dir']:
      raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(out_pars['output_dir']))
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # test(pars_choice=0)
    print("Config loaded")
    test("./data/",model_pars, data_pars, compute_pars, out_pars)





