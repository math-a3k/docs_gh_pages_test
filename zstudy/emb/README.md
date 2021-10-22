## Statistics 
---
Using bi-gram from random corpus to train an embedding space, which we then use to generate the co-relation metric for model similariy. 

### Steps to generate statistics 
1. create random word corpus 
```bash
python3 corpus_gen.py
```
2. train the model on the same corpus 
```python
# function in file util_model.py
gensim_model_train_save(
    dirinput="location to data.cor", 
    dirout="./modelout/model.bin", 
    pars={'min_count':2}
)
```
3. create bi-gram embedding space on same corpus, the code will save `similar.pickle` file.
```bash
python3 create_bi_gram.py
```
4. generate similarity metric, this will save the `stats.txt` file with similary metric. 
```bash
python3 check_stats.py
```
   
### Final stats 
after running `check_stats` file, final results will be saved in `stats.txt` file.

```python
# top ten picks from bi-gram for word 'cloud'
{
    "cloud": {  
        'bay': -0.7039359,
        'began': -0.67369014,
        'one': -0.58399326,
        'theory': 0.49223033,
        'growth': -0.4823107,
        'degree': -0.47582757,
        'roughly': 0.45104468,
        'posted': 0.44032887,
        'world': -0.429714,
        'france': -0.41504592
    }
}

# top ten picks from model for word 'cloud'
{
    "cloud": {
        'europe': 0.2469479739665985,
        'people': 0.20017245411872864,
        'theory': 0.18374206125736237,
        'theoretical': 0.1798827052116394,
        'european': 0.14671583473682404,
        'substance': 0.114181287586689,
        'person': 0.11217017471790314,
        'network': 0.10383737087249756,
        'world': 0.09686078876256943,
        'scientific': 0.09196242690086365
    }
}
```

```python
# top ten picks from bi-gram for word 'french'
{
    "french": {
        'theoretical': -0.84520215,
        'different': 0.6960834,
        'growth': -0.68519986,
        'study': 0.67149305,
        'animal': 0.6180694,
        'posted': 0.5687015,
        'american': -0.5278316,
        'one': -0.5231072,
        'person': 0.49240333,
        'rule': 0.48833162
    }
}

# top ten picks from model for word 'french'
{
    "french": {
        'one': 0.266126811504364,
        'rule': 0.21379074454307556,
        'chemistry': 0.21156728267669678,
        'may': 0.16627907752990723,
        'non': 0.1459185779094696,
        'japan': 0.13922397792339325,
        'france': 0.12886042892932892,
        'study': 0.1123766303062439,
        'set': 0.11236829310655594,
        'time': 0.10582320392131805
    }
}
```