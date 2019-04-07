# Demo for fastai tabular + text databunch with end-to-end classification/regression training

Inspired by Wayde Gilliam for his [detailed blog post on Fastai Datablock API](https://blog.usejournal.com/finding-data-block-nirvana-a-journey-through-the-fastai-data-block-api-c38210537fe4).

- Including several new TabularText classes (inherited from ItemLists, LabelLists and DataBunch) to handle both tabular data (continuous and categorical) and textual data (to be converted to numerical ids). All the preprocesses from tabular processor (FillMissing, Categorify, Normalize) and text processor (Tokenizer and Numericalize) are included.

- Combine RNN model (e.g. AWD LSTM) with multi layer perceptron (MLP) head to train both text and tabular data. All good training optimization from fastai learner can be used: fit_one_cycle, learning rate schedule, callbacks, train different groups using freeze_to (differential learning rate)...

The code to build TabularText databunch has been tested using data from [Kaggle PetFinder competition](https://www.kaggle.com/c/petfinder-adoption-prediction/) by comparing the output from Tabular Databunch and Text Databunch. Notebook for that can be found [here](fastai-api-experiment.ipynb)

The code to build TabularText learner has been used to train on data from [Mercari Price Kaggle competition](https://www.kaggle.com/c/mercari-price-suggestion-challenge). The model did train successfully with loss decreaseed after epochs, but the results are just average compared to the leaderboard of that competition (not sure why -> need more testing). [Training notebook](tabular-text-training-complete.ipynb)


The entire source code is in [fastai_tab_text.py](fastai_tab_text.py). You can create TabularText databuch by following [the same Databunch API from fastai doc](https://docs.fast.ai/tutorial.data.html). You can also look at my [training notebook](tabular-text-training-complete.ipynb) for both databunch creation and training the learner.

Requirement: fastai version 1.0.51 (including pytorch 1.0). Visit [https://github.com/fastai/fastai#installation](https://github.com/fastai/fastai#installation) for more.


Any feedback is welcome! Discussion on this can be found on fastai forums: [?](?)