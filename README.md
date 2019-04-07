# Demo for fastai tabular + text databunch with end-to-end classification/regression training

Inspired by Wayde Gilliam for his [detailed blog post on Fastai Datablock API](https://blog.usejournal.com/finding-data-block-nirvana-a-journey-through-the-fastai-data-block-api-c38210537fe4).

- Including several new TabularText classes (inherited from ItemLists, LabelLists and DataBunch) to handle both tabular data (continuous and categorical) and textual data (to be converted to numerical ids). All the preprocesses from tabular processor (FillMissing, Categorify, Normalize) and text processor (Tokenizer and Numericalize) are included.

- Combine RNN model (e.g. AWD LSTM) with multi layer perceptron (MLP) head to train both text and tabular data. All good training optimization from fastai learner can be used: fit_one_cycle, learning rate schedule, callbacks, train different groups using freeze_to (differential learning rate)...

The code to build TabularText databunch has been tested using data from [Kaggle PetFinder competition](https://www.kaggle.com/c/petfinder-adoption-prediction/) by comparing the output from Tabular Databunch and Text Databunch. Notebook for that can be found [here](pet-finder-fastai-api-experiment.ipynb)

The code to build TabularText learner has been used to train on data from [Mercari Price Kaggle competition](https://www.kaggle.com/c/mercari-price-suggestion-challenge). The model did train successfully with loss decreaseed after epochs, but the results are just average compared to the leaderboard of that competition (not sure why -> need more testing). [Training notebook](mercari-tabular-text-training-complete.ipynb)


The entire source code is in [fastai_tab_text.py](fastai_tab_text.py). You can create TabularText databuch by following [the same Databunch API from fastai doc](https://docs.fast.ai/tutorial.data.html). You can also look at my [training notebook](mercari-tabular-text-training-complete.ipynb) for both databunch creation and training the learner.

- Example of creating TabularText databunch from pandas Dataframe (using Mercari dataset)

```
cat_names=['category1','category2','category3','brand_name','shipping'] # categorical
cont_names= list(set(train_df.columns) - set(cat_names) - {'price','text'}) # continuous
dep_var = 'price' # label
procs = [FillMissing,Categorify, Normalize]
txt_cols=['text'] # text

def get_tabulartext_databunch(bs=100,val_idxs=val_idxs,path=mercari_path):
    data_lm = load_data(path, 'data_lm.pkl', bs=bs) # data_lm.pkl from mercari-language-model notebook
    collate_fn = partial(mixed_tabular_pad_collate, pad_idx=1, pad_first=True)
    reset_seed()
    return (TabularTextList.from_df(train_df, cat_names, cont_names, txt_cols, vocab=data_lm.vocab, procs=procs, path=path)
                            .split_by_idx(val_idxs)
                            .label_from_df(cols=dep_var)
                            .add_test(TabularTextList.from_df(test_df, cat_names, cont_names, txt_cols,path=path))
                            .databunch(bs=bs,collate_fn=collate_fn, no_check=False))

data = get_tabulartext_databunch(bs=100)
data.show_batch()
```python

- Example of creating TabularText learner and start one-cycle training (note: this is a regression problem)

```python
encoder_name = 'bs60-awdlstm-enc-stage2' # encoder from mercari-language-model notebook
def get_tabulartext_learner(data,params):
    learn= tabtext_learner(data,AWD_LSTM,metrics=[root_mean_squared_error],
                               callback_fns=[partial(SaveModelCallback, monitor='root_mean_squared_error',mode='min',every='improvement',name='best_nn')],
                               **params)
    learn.load_encoder(encoder_name)
    return learn

params={
    'layers':[500,400,200], # neural network at model's head
    'ps': [0.001,0.,0.], # dropout for NN at model's head
    'bptt':70,
    'max_len':20*70,
    'drop_mult': 1., # drop_mult: multiply to different dropouts in AWD LSTM
    'lin_ftrs': [300], # linear layer to AWD_LSTM output, before combining to embeddings
    'ps_lin_ftrs': [0.], # dropout for this linear layer at AWD_LSTM output
    # set 'lin_ftrs': None if you want AWD LSTM output (1200) to be combined straight to embeddings
    'emb_drop': 0., # embeddings dropout
    'y_range': [0,6], # restrict y range for regression problem
    'use_bn': True,    
}


learn = get_tabulartext_learner(data,params,seed=42)
print(learn.model)

learn.fit_one_cycle(3,max_lr=1e-02,pct_start=0.3,moms=(0.8,0.7))
```

Requirement: fastai version 1.0.51 (including pytorch 1.0). Visit [https://github.com/fastai/fastai#installation](https://github.com/fastai/fastai#installation) for more.


Any feedback is welcome! Follow the discussion on fastai forums: [?](?)