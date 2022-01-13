# import library
from jsoncomment import JsonComment ; json = JsonComment(), copy
from mlmodels.models import module_load
from mlmodels.util import path_norm_dict, path_norm, params_json_load
print(mlmodels)




#### Model URI and Config JSON
model_uri    = "model_sklearn.model_lightgbm.py"
config_path = path_norm( 'example/hyper_lightgbm_home_retail.json'    )
config_mode = "test"  ### test/prod
print(config_path)



###########################################################################################################
# Model Parameters
# model_pars, data_pars, compute_pars, out_pars
pars = json.load(open(config_path , mode='r'))[config_mode]
for key, pdict in  pars.items() :	
  globals()[key] = path_norm_dict( pdict   )   ###Normalize path
  print(key, globals()[key] )

model_uri = model_pars.get("model_uri", model_uri)





###########################################################################################################
#### Hyper-parameter Search
model_pars_old = copy.deepcopy( model_pars)
model_pars = optim(
    model_uri       = model_uri,
    hypermodel_pars = hypermodel_pars,
    model_pars      = model_pars,
    data_pars       = data_pars,
    compute_pars    = compute_pars,
    out_pars        = out_pars
)



###########################################################################################################
#### Run Best model

#### Setup Model 
module         = module_load( model_uri)
model          = module.Model(model_pars, data_pars, compute_pars) 


#### Fit
model, session = module.fit(model, data_pars, compute_pars, out_pars)           
metrics_val    = module.evaluate(model, data_pars, compute_pars, out_pars)   
print(metrics_val)


#### Inference
ypred          = module.predict(model, session, data_pars, compute_pars, out_pars)   
print(ypred)



#### Save/Load
module.save(model, save_pars ={ 'path': out_pars['path'] +"/model/"})
model2 = module.load(load_pars ={ 'path': out_pars['path'] +"/model/"})



