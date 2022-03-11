#!/usr/bin/env python
# coding: utf-8


# attention, prototype, perhaps will merged with util_feature.py


from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import PCA, NMF
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV



def pd_pipeline(bin_cols, text_col, X,y ):
  """function pd_pipeline
  Args:
      bin_cols:   
      text_col:   
      X:   
      y:   
  Returns:
      
  """
  bin_pipe = Pipeline([
    ('select_bin', MySelector(cols=bin_cols)),
    ('binarize', MyBinarizer())
    ])

  text_pipe = Pipeline([
    ('select_text', MySelector(cols=text_cols)),
    ('vectorize', CountVectorizer()),
    ('tfidf', TfidfVectorizer())
    ])

  full_pipeline = Pipeline([
      ('feat_union', FeatureUnion(transformer_list=[
            ('text_pipeline', text_pipe),
            ('bin_pipeline', bin_pipe)
            ])),
      ('reduce_dim', PCA())
      ('classify', LinearSVC())
      ])

  X_train, X_test, y_train, y_test = train_test_split(X, y)

  full_pipeline.fit(X_train, y_train)

  return full_pipeline


pg = [
    {
        'reduce_dim': [PCA(iterated_power=7), NMF()],
        'reduce_dim__n_components': [2, 4, 8],
        'classify__C': [1, 10, 100, 1000]
    },
    {
        'classify': [LinearSVC()],
        'classify__penalty': ['l1', 'l2']
    },
    {
        'classify': [DecisionTreeClassifier()],
        'classify__min_samples_split': [2, 10, 20]
    },
]


def pd_grid_search(full_pipeline,X, y):
  """function pd_grid_search
  Args:
      full_pipeline:   
      X:   
      y:   
  Returns:
      
  """
  X_train, X_test, y_train, y_test = train_test_split(X, y)
  grid_search = GridSearchCV(full_pipeline, param_grid=pg, cv=3)

  y_pred = full_pipeline.predict(X_test)
  print(classification_report(y_test, y_pred))

  print("Best estimator found:")
  print(grid_search.best_estimator_)

  print("Best score:")
  print(grid_search.best_score_)

  print("Best parameters found:")
  print(grid_search.best_params_)
