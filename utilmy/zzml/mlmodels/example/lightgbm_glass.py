# import library
import os
import mlmodels
from mlmodels.models import module_load
from mlmodels.util import path_norm_dict, path_norm, params_json_load
from mlmodels.optim import optim
from jsoncomment import JsonComment ; json = JsonComment()

print( os.getcwd())






config_path  = 'lightgbm_glass.json'  
model_uri    = "model_sklearn.model_lightgbm"
config_mode  = "test"



pars = json.load(open( config_path , mode='r'))[config_mode]
print(pars)

hypermodel_pars = path_norm_dict( pars.get('hypermodel_pars' ) )
model_pars      = path_norm_dict( pars['model_pars'] )
data_pars       = path_norm_dict( pars['data_pars'] )
compute_pars    = path_norm_dict( pars['compute_pars'] )
out_pars        = path_norm_dict( pars['out_pars'] )




#### Hyper-parameter Search

model_pars_update = optim(
    model_uri       = model_uri,
    hypermodel_pars = hypermodel_pars,
    model_pars      = model_pars,
    data_pars       = data_pars,
    compute_pars    = compute_pars,
    out_pars        = out_pars
)




module         =  module_load( model_uri= model_uri)
model          = module.Model(model_pars, data_pars, compute_pars)
model, session = module.fit(model, data_pars, compute_pars, out_pars)



# predict pipeline
ypred       = module.predict(model,  data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)     
ypred




metrics_val = module.evaluate(model, data_pars, compute_pars, out_pars)
print(metrics_val)


