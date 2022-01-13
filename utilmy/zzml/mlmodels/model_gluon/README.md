# Gluon models 
```  
install/requirements.txt

autogluon
gluonts
pandas
matplotlib
mxnet


autogluon :
  automatic ML using Gluon : RForest, ...


glutonts :
  gluon_deepar.py :
  gluon_ffn.py
  gluon_prophet.py :



```
time series models using GluonTS Toolkit.
This gives an end to end api .For now we have created api for FFN , DeepAR and Prophet.<br/> 
Different functionalities api can do 
* get_dataset(data_params)  -- Load dataset
* fit(model, data_pars=None,...) -- Train the model
* save(model, path) -Save model on specified path
* predict(model, data_pars, compute_pars=None,...) -- Predict result 
* metrics(ypred, data_pars, compute_pars=None,...) --Create metrics from output result. 



