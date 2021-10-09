
import warnings
warnings.filterwarnings('ignore')
import sys, os, json, importlib

####################################################################################################
#### Add path for python import
sys.path.append( os.path.dirname(os.path.abspath(__file__)) + "/")

#### Root folder analysis
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)


####################################################################################################
####################################################################################################
from util_feature import   load, save_list, load_function_uri, save
from run_preprocess import  preprocess, preprocess_load

def log(*s, n=0, m=0):
    sspace = "#" * n
    sjump = "\n" * m
    ### Implement pseudo Logging
    print(sjump, sspace, s, sspace, flush=True)


def save_features(df, name, path):
    if path is not None :
       os.makedirs( f"{path}/{name}", exist_ok=True)
       df.to_parquet( f"{path}/{name}/features.parquet")


def model_dict_load(model_dict, config_path, config_name, verbose=True):
    """
       load the model dict from the python config file.
    :param model_dict:
    :param config_path:
    :param config_name:
    :param verbose:
    :return:
    """
    if model_dict is None :
       log("#### Model Params Dynamic loading  ###############################################")
       model_dict_fun = load_function_uri(uri_name=config_path + "::" + config_name)
       model_dict     = model_dict_fun()   ### params
    if verbose : log( model_dict )
    return model_dict










"""
pip install gamma-facet


Model Inspection
FACET implements several model inspection methods for scikit-learn estimators. FACET enhances model inspection by providing global metrics that complement the local perspective of SHAP. The key global metrics for each pair of features in a model are:

Synergy

The degree to which the model combines information from one feature with another to predict the target. For example, let's assume we are predicting cardiovascular health using age and gender and the fitted model includes a complex interaction between them. This means these two features are synergistic for predicting cardiovascular health. Further, both features are important to the model and removing either one would significantly impact performance. Let's assume age brings more information to the joint contribution than gender. This asymmetric contribution means the synergy for (age, gender) is less than the synergy for (gender, age). To think about it another way, imagine the prediction is a coordinate you are trying to reach. From your starting point, age gets you much closer to this point than gender, however, you need both to get there. Synergy reflects the fact that gender gets more help from age (higher synergy from the perspective of gender) than age does from gender (lower synergy from the perspective of age) to reach the prediction. This leads to an important point: synergy is a naturally asymmetric property of the global information two interacting features contribute to the model predictions. Synergy is expressed as a percentage ranging from 0% (full autonomy) to 100% (full synergy).

Redundancy

The degree to which a feature in a model duplicates the information of a second feature to predict the target. For example, let's assume we had house size and number of bedrooms for predicting house price. These features capture similar information as the more bedrooms the larger the house and likely a higher price on average. The redundancy for (number of bedrooms, house size) will be greater than the redundancy for (house size, number of bedrooms). This is because house size "knows" more of what number of bedrooms does for predicting house price than vice-versa. Hence, there is greater redundancy from the perspective of number of bedrooms. Another way to think about it is removing house size will be more detrimental to model performance than removing number of bedrooms, as house size can better compensate for the absence of number of bedrooms. This also implies that house size would be a more important feature than number of bedrooms in the model. The important point here is that like synergy, redundancy is a naturally asymmetric property of the global information feature pairs have for predicting an outcome. Redundancy is expressed as a percentage ranging from 0% (full uniqueness) to 100% (full redundancy).



"""




# fit the model inspector
from facet.inspection import LearnerInspector
inspector = LearnerInspector()
inspector.fit(crossfit= mymodel )
Synergy

# visualise synergy as a matrix
from pytools.viz.matrix import MatrixDrawer
synergy_matrix = inspector.feature_synergy_matrix(symmetrical=True)
MatrixDrawer(style="matplot%").draw(synergy_matrix, title="Synergy Matrix")







