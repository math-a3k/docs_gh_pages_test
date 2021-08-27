



pytorch_tabular\setup.py
-------------------------functions----------------------
read_requirements(thelibFolder, filename)





pytorch_tabular\examples\adhoc_scaffold.py
-------------------------functions----------------------
make_mixed_classification(n_samples, n_features, n_categories)
print_metrics(y_true, y_pred, tag)





pytorch_tabular\examples\to_test_classification.py




pytorch_tabular\examples\to_test_node.py
-------------------------functions----------------------
regression_data()
classification_data()
test_regression(regression_data, multi_target, embed_categorical, continuous_cols, categorical_cols, continuous_feature_transform, normalize_continuous_features, )
test_classification(classification_data, continuous_cols, categorical_cols, embed_categorical, continuous_feature_transform, normalize_continuous_features, )





pytorch_tabular\examples\to_test_regression.py
-------------------------functions----------------------
fake_metric(y_hat, y)





pytorch_tabular\examples\to_test_regression_custom_models.py




pytorch_tabular\pytorch_tabular\categorical_encoders.py
-------------------------methods----------------------
BaseEncoder.__init__(self, cols, handle_unseen, min_samples, imputed)
BaseEncoder.transform(self, X)
BaseEncoder.fit_transform(self, X, y = None)
BaseEncoder._input_check(self, name, value, options)
BaseEncoder._before_fit_check(self, X, y)
BaseEncoder.save_as_object_file(self, path)
BaseEncoder.load_from_object_file(self, path)
OrdinalEncoder.__init__(self, cols = None, handle_unseen = "impute")
OrdinalEncoder.fit(self, X, y = None)
CategoricalEmbeddingTransformer.__init__(self, tabular_model)
CategoricalEmbeddingTransformer._extract_embedding(self, model)
CategoricalEmbeddingTransformer.fit(self, X, y = None)
CategoricalEmbeddingTransformer.transform(self, X: pd.DataFrame, y = None)
CategoricalEmbeddingTransformer.fit_transform(self, X: pd.DataFrame, y = None)
CategoricalEmbeddingTransformer.save_as_object_file(self, path)
CategoricalEmbeddingTransformer.load_from_object_file(self, path)




pytorch_tabular\pytorch_tabular\feature_extractor.py
-------------------------methods----------------------
DeepFeatureExtractor.__init__(self, tabular_model, extract_keys = ["backbone_features"], drop_original = True)
DeepFeatureExtractor.fit(self, X, y = None)
DeepFeatureExtractor.transform(self, X: pd.DataFrame, y = None)
DeepFeatureExtractor.fit_transform(self, X: pd.DataFrame, y = None)
DeepFeatureExtractor.save_as_object_file(self, path)
DeepFeatureExtractor.load_from_object_file(self, path)




pytorch_tabular\pytorch_tabular\tabular_datamodule.py
-------------------------methods----------------------
TabularDatamodule.__init__(self, train: pd.DataFrame, config: DictConfig, validation: pd.DataFrame  =  None, test: pd.DataFrame  =  None, target_transform: Optional[Union[TransformerMixin, Tuple]]  =  None, train_sampler: Optional[torch.utils.data.Sampler]  =  None, ).__init__())self.validation = validationif target_transform is not None =  validationif target_transform is not None:)
TabularDatamodule.update_config(self)
TabularDatamodule.do_leave_one_out_encoder(self)
TabularDatamodule.preprocess_data(self, data: pd.DataFrame, stage: str  =  "inference")
TabularDatamodule.setup(self, stage: Optional[str]  =  None)
TabularDatamodule.time_features_from_frequency_str(cls, freq_str: str)
TabularDatamodule.val_dataloader(self)
TabularDatamodule.test_dataloader(self)
TabularDatamodule.prepare_inference_dataloader(self, df: pd.DataFrame)
TabularDataset.__init__(self, data: pd.DataFrame, task: str, continuous_cols: List[str]  =  None, categorical_cols: List[str]  =  None, embed_categorical: bool  =  True, target: List[str]  =  None, )
TabularDataset.__len__(self)
TabularDataset.__getitem__(self, idx)




pytorch_tabular\pytorch_tabular\tabular_model.py
-------------------------methods----------------------
TabularModel.__init__(self, config: Optional[DictConfig]  =  None, data_config: Optional[Union[DataConfig, str]]  =  None, model_config: Optional[Union[ModelConfig, str]]  =  None, optimizer_config: Optional[Union[OptimizerConfig, str]]  =  None, trainer_config: Optional[Union[TrainerConfig, str]]  =  None, experiment_config: Optional[Union[ExperimentConfig, str]]  =  None, model_callable: Optional[Callable]  =  None, )
TabularModel._run_validation(self)
TabularModel._read_parse_config(self, config, cls)
TabularModel._get_run_name_uid(self)
TabularModel._setup_experiment_tracking(self)
TabularModel._prepare_callbacks(self)
TabularModel._prepare_dataloader(self, train, validation, test, target_transform = None, train_sampler = None)
TabularModel._prepare_model(self, loss, metrics, optimizer, optimizer_params, reset)
TabularModel._prepare_trainer(self, max_epochs = None, min_epochs = None)
TabularModel.load_best_model(self)
TabularModel._pre_fit(self, train: pd.DataFrame, validation: Optional[pd.DataFrame], test: Optional[pd.DataFrame], loss: Optional[torch.nn.Module], metrics: Optional[List[Callable]], optimizer: Optional[torch.optim.Optimizer], optimizer_params: Dict, train_sampler: Optional[torch.utils.data.Sampler], target_transform: Optional[Union[TransformerMixin, Tuple]], max_epochs: int, min_epochs: int, reset: bool)
TabularModel.fit(self, train: pd.DataFrame, validation: Optional[pd.DataFrame]  =  None, test: Optional[pd.DataFrame]  =  None, loss: Optional[torch.nn.Module]  =  None, metrics: Optional[List[Callable]]  =  None, optimizer: Optional[torch.optim.Optimizer]  =  None, optimizer_params: Dict  =  {}, train_sampler: Optional[torch.utils.data.Sampler]  =  None, target_transform: Optional[Union[TransformerMixin, Tuple]]  =  None, max_epochs: Optional[int]  =  None, min_epochs: Optional[int]  =  None, reset: bool  =  False, seed: Optional[int]  =  None, )
TabularModel.find_learning_rate(self, train: pd.DataFrame, validation: Optional[pd.DataFrame]  =  None, test: Optional[pd.DataFrame]  =  None, loss: Optional[torch.nn.Module]  =  None, metrics: Optional[List[Callable]]  =  None, optimizer: Optional[torch.optim.Optimizer]  =  None, optimizer_params: Dict  =  {}, min_lr: float  =  1e-8, max_lr: float  =  1, num_training: int  =  100, mode: str  =  "exponential", early_stop_threshold: float  =  4.0, plot = True, )
TabularModel.evaluate(self, test: Optional[pd.DataFrame])
TabularModel.predict(self, test: pd.DataFrame, quantiles: Optional[List]  =  [0.25, 0.5, 0.75], n_samples: Optional[int]  =  100, ret_logits = False, )
TabularModel.save_model(self, dir: str)
TabularModel.load_from_checkpoint(cls, dir: str)




pytorch_tabular\pytorch_tabular\utils.py
-------------------------functions----------------------
_make_smooth_weights_for_balanced_classes(y_train, mu = 0.15)
get_class_weighted_cross_entropy(y_train, mu = 0.15)
get_balanced_sampler(y_train)
_initialize_layers(hparams, layer)
_linear_dropout_bn(hparams, in_units, out_units, activation, dropout)
get_gaussian_centers(y, n_components)





pytorch_tabular\pytorch_tabular\__init__.py




pytorch_tabular\tests\conftest.py
-------------------------functions----------------------
load_regression_data()
load_classification_data()
load_timeseries_data()
regression_data()
classification_data()
timeseries_data()





pytorch_tabular\tests\test_autoint.py
-------------------------functions----------------------
test_regression(regression_data, multi_target, continuous_cols, categorical_cols, continuous_feature_transform, normalize_continuous_features, target_range, deep_layers, batch_norm_continuous_input, attention_pooling)
test_classification(classification_data, continuous_cols, categorical_cols, continuous_feature_transform, normalize_continuous_features, deep_layers, batch_norm_continuous_input)





pytorch_tabular\tests\test_categorical_embedding.py
-------------------------functions----------------------
fake_metric(y_hat, y)
test_regression(regression_data, multi_target, continuous_cols, categorical_cols, continuous_feature_transform, normalize_continuous_features, target_range, target_transform, custom_metrics, custom_loss, custom_optimizer, )
test_classification(classification_data, continuous_cols, categorical_cols, continuous_feature_transform, normalize_continuous_features, )
test_embedding_transformer(regression_data)





pytorch_tabular\tests\test_common.py
-------------------------functions----------------------
fake_metric(y_hat, y)
test_save_load(regression_data, model_config_class, continuous_cols, categorical_cols, custom_metrics, custom_loss, custom_optimizer, tmpdir, )
test_feature_extractor(regression_data, model_config_class, continuous_cols, categorical_cols, )





pytorch_tabular\tests\test_datamodule.py
-------------------------functions----------------------
test_dataloader(regression_data, validation_split, multi_target, continuous_cols, categorical_cols, continuous_feature_transform, normalize_continuous_features, target_transform, embedding_dims, )
test_date_encoding(timeseries_data, freq)





pytorch_tabular\tests\test_mdn.py
-------------------------functions----------------------
test_regression(regression_data, multi_target, continuous_cols, categorical_cols, continuous_feature_transform, normalize_continuous_features, variant, num_gaussian)
test_classification(classification_data, continuous_cols, categorical_cols, continuous_feature_transform, normalize_continuous_features, num_gaussian)





pytorch_tabular\tests\test_node.py
-------------------------functions----------------------
test_regression(regression_data, multi_target, embed_categorical, continuous_cols, categorical_cols, continuous_feature_transform, normalize_continuous_features, target_range, )
test_classification(classification_data, continuous_cols, categorical_cols, embed_categorical, continuous_feature_transform, normalize_continuous_features, )
test_embedding_transformer(regression_data)





pytorch_tabular\tests\test_tabnet.py
-------------------------functions----------------------
test_regression(regression_data, multi_target, continuous_cols, categorical_cols, continuous_feature_transform, normalize_continuous_features, target_range)
test_classification(classification_data, continuous_cols, categorical_cols, continuous_feature_transform, normalize_continuous_features, )





pytorch_tabular\tests\__init__.py




pytorch_tabular\pytorch_tabular\config\config.py
-------------------------functions----------------------
_read_yaml(filename)

-------------------------methods----------------------
TrainerConfig.__post_init__(self)
ExperimentConfig.__post_init__(self)
OptimizerConfig.read_from_yaml(filename: str  =  "config/optimizer_config.yml")
ExperimentRunManager.__init__(self, exp_version_manager: str  =  ".tmp/exp_version_manager.yml", )
ExperimentRunManager.update_versions(self, name)
ModelConfig.__post_init__(self)




pytorch_tabular\pytorch_tabular\config\__init__.py




pytorch_tabular\pytorch_tabular\models\base_model.py
-------------------------methods----------------------
BaseModel.__init__(self, config: DictConfig, custom_loss: Optional[torch.nn.Module]  =  None, custom_metrics: Optional[List[Callable]]  =  None, custom_optimizer: Optional[torch.optim.Optimizer]  =  None, custom_optimizer_params: Dict  =  {}, **kwargs)
BaseModel._build_network(self)
BaseModel._setup_loss(self)
BaseModel._setup_metrics(self)
BaseModel.calculate_loss(self, y, y_hat, tag)
BaseModel.calculate_metrics(self, y, y_hat, tag)
BaseModel.data_aware_initialization(self, datamodule)
BaseModel.forward(self, x: Dict)
BaseModel.predict(self, x: Dict, ret_model_output: bool  =  False)
BaseModel.training_step(self, batch, batch_idx)
BaseModel.validation_step(self, batch, batch_idx)
BaseModel.test_step(self, batch, batch_idx)
BaseModel.configure_optimizers(self)
BaseModel.create_plotly_histogram(self, arr, name, bin_dict = None)
BaseModel.validation_epoch_end(self, outputs)




pytorch_tabular\pytorch_tabular\models\__init__.py




pytorch_tabular\pytorch_tabular\models\autoint\autoint.py
-------------------------methods----------------------
AutoIntBackbone.__init__(self, config: DictConfig)
AutoIntBackbone._build_network(self)
AutoIntBackbone.forward(self, x: Dict)
AutoIntModel.__init__(self, config: DictConfig, **kwargs)
AutoIntModel._build_network(self)
AutoIntModel.forward(self, x: Dict)




pytorch_tabular\pytorch_tabular\models\autoint\config.py




pytorch_tabular\pytorch_tabular\models\autoint\__init__.py




pytorch_tabular\pytorch_tabular\models\category_embedding\category_embedding_model.py
-------------------------methods----------------------
FeedForwardBackbone.__init__(self, config: DictConfig, **kwargs)
FeedForwardBackbone._build_network(self)
FeedForwardBackbone.forward(self, x)
CategoryEmbeddingModel.__init__(self, config: DictConfig, **kwargs)
CategoryEmbeddingModel._build_network(self)
CategoryEmbeddingModel.unpack_input(self, x: Dict)
CategoryEmbeddingModel.forward(self, x: Dict)




pytorch_tabular\pytorch_tabular\models\category_embedding\config.py




pytorch_tabular\pytorch_tabular\models\category_embedding\__init__.py




pytorch_tabular\pytorch_tabular\models\mixture_density\config.py




pytorch_tabular\pytorch_tabular\models\mixture_density\mdn.py
-------------------------methods----------------------
MixtureDensityHead.__init__(self, config: DictConfig, **kwargs)
MixtureDensityHead._build_network(self)
MixtureDensityHead.forward(self, x)
MixtureDensityHead.gaussian_probability(self, sigma, mu, target, log = False)
MixtureDensityHead.log_prob(self, pi, sigma, mu, y)
MixtureDensityHead.sample(self, pi, sigma, mu)
MixtureDensityHead.generate_samples(self, pi, sigma, mu, n_samples = None)
MixtureDensityHead.generate_point_predictions(self, pi, sigma, mu, n_samples = None)
BaseMDN.__init__(self, config: DictConfig, **kwargs)
BaseMDN.unpack_input(self, x: Dict)
BaseMDN.forward(self, x: Dict)
BaseMDN.predict(self, x: Dict)
BaseMDN.sample(self, x: Dict, n_samples: Optional[int]  =  None, ret_model_output = False)
BaseMDN.calculate_loss(self, y, pi, sigma, mu, tag = "train")
BaseMDN.training_step(self, batch, batch_idx)
BaseMDN.validation_step(self, batch, batch_idx)
BaseMDN.test_step(self, batch, batch_idx)
BaseMDN.validation_epoch_end(self, outputs)
CategoryEmbeddingMDN.__init__(self, config: DictConfig, **kwargs)
CategoryEmbeddingMDN._build_network(self)
CategoryEmbeddingMDN.unpack_input(self, x: Dict)
NODEMDN.__init__(self, config: DictConfig, **kwargs)
NODEMDN.subset(self, x)
NODEMDN._build_network(self)
NODEMDN.unpack_input(self, x: Dict)
AutoIntMDN.__init__(self, config: DictConfig, **kwargs)
AutoIntMDN._build_network(self)
AutoIntMDN.unpack_input(self, x: Dict)




pytorch_tabular\pytorch_tabular\models\mixture_density\__init__.py




pytorch_tabular\pytorch_tabular\models\node\architecture_blocks.py
-------------------------methods----------------------
DenseODSTBlock.__init__(self, input_dim, num_trees, num_layers, tree_output_dim = 1, max_features = None, input_dropout = 0.0, flatten_output = False, Module = ODST, **kwargs)
DenseODSTBlock.forward(self, x)




pytorch_tabular\pytorch_tabular\models\node\config.py




pytorch_tabular\pytorch_tabular\models\node\node_model.py
-------------------------methods----------------------
NODEBackbone.__init__(self, config: DictConfig, **kwargs)
NODEBackbone._build_network(self)
NODEBackbone.forward(self, x)
NODEModel.__init__(self, config: DictConfig, **kwargs)
NODEModel.subset(self, x)
NODEModel.data_aware_initialization(self, datamodule)
NODEModel._build_network(self)
NODEModel.unpack_input(self, x: Dict)
NODEModel.forward(self, x: Dict)




pytorch_tabular\pytorch_tabular\models\node\odst.py
-------------------------functions----------------------
check_numpy(x)

-------------------------methods----------------------
ODST.__init__(self, in_features, num_trees, depth = 6, tree_output_dim = 1, flatten_output = True, choice_function = sparsemax, bin_function = sparsemoid, initialize_response_ = nn.init.normal_, initialize_selection_logits_ = nn.init.uniform_, threshold_init_beta = 1.0, threshold_init_cutoff = 1.0, )
ODST.forward(self, input)
ODST.initialize(self, input, eps = 1e-6)
ODST.__repr__(self)




pytorch_tabular\pytorch_tabular\models\node\utils.py
-------------------------functions----------------------
to_one_hot(y, depth = None)
sparsemax(input, dim = -1)
sparsemoid(input)
entmax15(input, dim = -1)

-------------------------methods----------------------
Entmax15Function.forward(ctx, input, dim = -1)
Entmax15Function.backward(ctx, grad_output)
Entmax15Function._threshold_and_support(input, dim = -1)
Entmoid15.forward(ctx, input)
Entmoid15._forward(input)
Entmoid15.backward(ctx, grad_output)
Entmoid15._backward(output, grad_output)
Lambda.__init__(self, func)
Lambda.forward(self, *args, **kwargs)
ModuleWithInit.__init__(self)
ModuleWithInit.initialize(self, *args, **kwargs)
ModuleWithInit.__call__(self, *args, **kwargs)




pytorch_tabular\pytorch_tabular\models\node\__init__.py




pytorch_tabular\pytorch_tabular\models\tabnet\config.py




pytorch_tabular\pytorch_tabular\models\tabnet\tabnet_model.py
-------------------------methods----------------------
TabNetBackbone.__init__(self, config: DictConfig, **kwargs)
TabNetBackbone._build_network(self)
TabNetBackbone.forward(self, x: Dict)
TabNetModel.__init__(self, config: DictConfig, **kwargs)
TabNetModel._build_network(self)
TabNetModel.unpack_input(self, x: Dict)
TabNetModel.forward(self, x: Dict)




pytorch_tabular\pytorch_tabular\models\tabnet\__init__.py
