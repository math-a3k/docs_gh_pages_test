""""
from sklearn.naive_bayes import GaussianNB
from mapie.classification import MapieClassifier
from mapie.metrics import classification_coverage_score, classification_mean_width_score
clf = GaussianNB().fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)
y_pred_proba_max = np.max(y_pred_proba, axis=1)
mapie_score = MapieClassifier(estimator=clf, cv="prefit", method="score")
mapie_score.fit(X_cal, y_cal)
alpha = [0.2, 0.1, 0.05]
y_pred_score, y_ps_score = mapie_score.predict(X_test_mesh, alpha=alpha)


#### Uncertainy interval.
https://mapie.readthedocs.io/en/latest/tutorial_classification.html


"""
def test():
  from sklearn.naive_bayes import GaussianNB
  from mapie.classification import MapieClassifier
  from mapie.metrics import classification_coverage_score, classification_mean_width_score
  clf = GaussianNB().fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  y_pred_proba = clf.predict_proba(X_test)
  y_pred_proba_max = np.max(y_pred_proba, axis=1)
  mapie_score = MapieClassifier(estimator=clf, cv="prefit", method="score")
  mapie_score.fit(X_cal, y_cal)
  alpha = [0.2, 0.1, 0.05]
  y_pred_score, y_ps_score = mapie_score.predict(X_test_mesh, alpha=alpha)


def model_uncertainty_train(clf, Xval, yval, dirout=""):
  from mapie.classification import MapieClassifier
  from mapie.metrics import classification_coverage_score, classification_mean_width_score
  
  mapie_score = MapieClassifier(estimator=clf, cv="prefit", method="score")
  mapie_score.fit(Xval, yval)
  alpha = [0.2, 0.1, 0.05]
  y_pred_score, y_ps_score = mapie_score.predict(X_test_mesh, alpha=alpha)

  from utilmy import save
  save(maple_score, dirout)


  
  
  
