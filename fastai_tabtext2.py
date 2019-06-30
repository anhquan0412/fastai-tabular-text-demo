from fastai.text import *
from fastai.tabular import *


class ConcatDataset(Dataset):
    def __init__(self, x1, x2, y): self.x1,self.x2,self.y = x1,x2,y
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return (self.x1[i], self.x2[i]), self.y[i]
    
def tabtext_collate(batch):
    x,y = list(zip(*batch))
    x1,x2 = list(zip(*x)) # x1 is (cat,cont), x2 is numericalized ids for text
    x1 = to_data(x1)
    x1 = list(zip(*x1))
    x1 = torch.stack(x1[0]), torch.stack(x1[1])
    x2, y = pad_collate(list(zip(x2, y)), pad_idx=1, pad_first=True)
    return (x1, x2), y

class ConcatModel(nn.Module):
    def __init__(self, mod_tab, mod_nlp, layers, drops): 
        super().__init__()
        self.mod_tab = mod_tab
        self.mod_nlp = mod_nlp
        lst_layers = []
        activs = [nn.ReLU(inplace=True),] * (len(layers)-2) + [None]
        for n_in,n_out,p,actn in zip(layers[:-1], layers[1:], drops, activs):
            lst_layers += bn_drop_lin(n_in, n_out, p=p, actn=actn)
        self.layers = nn.Sequential(*lst_layers)

    def forward(self, *x):
        x_tab = self.mod_tab(*x[0])
        x_nlp = self.mod_nlp(x[1])[0]
        x = torch.cat([x_tab, x_nlp], dim=1)
        return self.layers(x)    



def get_tabtext_learner(data,tab_learner,text_learner,lin_layers,ps):
    tab_learner.model.layers = tab_learner.model.layers[:-2] # get rid of related output layers

    text_learner.model[-1].layers =text_learner.model[-1].layers[:-3] # get rid of related output layers
    
    lin_layers = lin_layers+ [tab_learner.data.train_ds.c]
    model = ConcatModel(tab_learner.model, text_learner.model, lin_layers, ps)
    
    loss_func = tab_learner.loss_func

    # assign layer groups for gradual training (unfreezing group)
    layer_groups = [nn.Sequential(*flatten_model(text_learner.layer_groups[0])),
                    nn.Sequential(*flatten_model(text_learner.layer_groups[1])),
                    nn.Sequential(*flatten_model(text_learner.layer_groups[2])),
                    nn.Sequential(*flatten_model(text_learner.layer_groups[3])),
                    nn.Sequential(*(flatten_model(text_learner.layer_groups[4]) + 
                                    flatten_model(model.mod_tab) +
                                    flatten_model(model.layers)))] 
    learner = Learner(data, model, loss_func=loss_func, layer_groups=layer_groups,metrics = tab_learner.metrics)
    return learner

def predict_one_item(learner,item,tab_db,text_db, **kwargs):
    '''
    learner: tabular text learner
    item: pandas series

    Return raw prediction from model and modified prediction (based on y.analyze_pred)
    '''
    tab_oneitem = tab_db.one_item(item,detach=True,cpu=True)
    text_oneitem= text_db.one_item(item,detach=True,cpu=True)
    _batch = [( ([tab_oneitem[0][0][0],tab_oneitem[0][1][0]],text_oneitem[0][0]), tab_oneitem[1][0] )]
    tabtext_onebatch = tabtext_collate(_batch)

    # send to gpu
    tabtext_onebatch = to_device(tabtext_onebatch,None)

    # taken from fastai.basic_train Learner.predict function
    res = learner.pred_batch(batch=tabtext_onebatch)
    raw_pred,x = grab_idx(res,0,batch_first=True),tabtext_onebatch[0]

    ds = learner.data.single_ds
    pred = ds.y.analyze_pred(raw_pred, **kwargs)
    return pred, raw_pred

