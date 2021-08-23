# All files

<details>
<summary>
<a name='pytorch_tabular/pytorch_tabular/models/mixture_density/config.py' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/config.py'>pytorch_tabular/pytorch_tabular/models/mixture_density/config.py</a>
</summary>
<ul>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/config.py:MixtureDensityHeadConfig' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/config.py#L14'>MixtureDensityHeadConfig</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/config.py:CategoryEmbeddingMDNConfig' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/config.py#L121'>CategoryEmbeddingMDNConfig</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/config.py:NODEMDNConfig' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/config.py#L165'>NODEMDNConfig</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/config.py:AutoIntMDNConfig' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/config.py#L238'>AutoIntMDNConfig</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='pytorch_tabular/pytorch_tabular/models/tabnet/tabnet_model.py' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/tabnet/tabnet_model.py'>pytorch_tabular/pytorch_tabular/models/tabnet/tabnet_model.py</a>
</summary>
<ul>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/models/tabnet/tabnet_model.py:TabNetBackbone' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/tabnet/tabnet_model.py#L19'>TabNetBackbone</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/models/tabnet/tabnet_model.py:TabNetModel' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/tabnet/tabnet_model.py#L50'>TabNetModel</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/tabnet/tabnet_model.py:TabNetBackbone:__init__' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/tabnet/tabnet_model.py#L20'>TabNetBackbone:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>config:  DictConfig,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/tabnet/tabnet_model.py:TabNetBackbone:_build_network' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/tabnet/tabnet_model.py#L25'>TabNetBackbone:_build_network</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/tabnet/tabnet_model.py:TabNetBackbone:forward' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/tabnet/tabnet_model.py#L44'>TabNetBackbone:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x:  Dict,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/tabnet/tabnet_model.py:TabNetModel:__init__' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/tabnet/tabnet_model.py#L20'>TabNetModel:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>config:  DictConfig,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/tabnet/tabnet_model.py:TabNetModel:_build_network' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/tabnet/tabnet_model.py#L25'>TabNetModel:_build_network</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/tabnet/tabnet_model.py:TabNetModel:unpack_input' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/tabnet/tabnet_model.py#L57'>TabNetModel:unpack_input</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x:  Dict,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/tabnet/tabnet_model.py:TabNetModel:forward' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/tabnet/tabnet_model.py#L44'>TabNetModel:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x:  Dict,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='pytorch_tabular/pytorch_tabular/tabular_datamodule.py' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/tabular_datamodule.py'>pytorch_tabular/pytorch_tabular/tabular_datamodule.py</a>
</summary>
<ul>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/tabular_datamodule.py:TabularDatamodule' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/tabular_datamodule.py#L32'>TabularDatamodule</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/tabular_datamodule.py:TabularDataset' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/tabular_datamodule.py#L539'>TabularDataset</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/tabular_datamodule.py:TabularDatamodule:__init__' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/tabular_datamodule.py#L53'>TabularDatamodule:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>train:  pd.DataFrame,<br>config:  DictConfig,<br>validation:  pd.DataFrame  =  None,<br>test:  pd.DataFrame  =  None,<br>target_transform:  Optional[Union[TransformerMixin,<br>Tuple]]  =  None,<br>train_sampler:  Optional[torch.utils.data.Sampler]  =  None,<br>).__init__())self.validation = validationif target_transform is not None:  =  validationif target_transform is not None:,<br></ul>
        <li>Docs:<br>        """The Pytorch Lightning Datamodule for Tabular Data
<br>

<br>
        Args:
<br>
            train (pd.DataFrame): The Training Dataframe
<br>
            config (DictConfig): Merged configuration object from ModelConfig, DataConfig,
<br>
            TrainerConfig, OptimizerConfig & ExperimentConfig
<br>
            validation (pd.DataFrame, optional): Validation Dataframe.
<br>
            If left empty, we use the validation split from DataConfig to split a random sample as validation.
<br>
            Defaults to None.
<br>
            test (pd.DataFrame, optional): Holdout DataFrame to check final performance on.
<br>
            Defaults to None.
<br>
            target_transform (Optional[Union[TransformerMixin, Tuple(Callable)]], optional): If provided, applies the transform to the target before modelling
<br>
            and inverse the transform during prediction. The parameter can either be a sklearn Transformer which has an inverse_transform method, or
<br>
            a tuple of callables (transform_func, inverse_transform_func)
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/tabular_datamodule.py:TabularDatamodule:update_config' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/tabular_datamodule.py#L96'>TabularDatamodule:update_config</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br>        """Calculates and updates a few key information to the config object
<br>

<br>
        Raises:
<br>
            NotImplementedError: [description]
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/tabular_datamodule.py:TabularDatamodule:do_leave_one_out_encoder' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/tabular_datamodule.py#L117'>TabularDatamodule:do_leave_one_out_encoder</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br>        """Checks the special condition for NODE where we use a LeaveOneOutEncoder to encode categorical columns
<br>

<br>
        Returns:
<br>
            bool
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/tabular_datamodule.py:TabularDatamodule:preprocess_data' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/tabular_datamodule.py#L127'>TabularDatamodule:preprocess_data</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>data:  pd.DataFrame,<br>stage:  str  =  "inference",<br></ul>
        <li>Docs:<br>        """The preprocessing, like Categorical Encoding, Normalization, etc. which any dataframe should undergo before feeding into the dataloder
<br>

<br>
        Args:
<br>
            data (pd.DataFrame): A dataframe with the features and target
<br>
            stage (str, optional): Internal parameter. Used to distinguisj between fit and inference. Defaults to "inference".
<br>

<br>
        Returns:
<br>
            tuple[pd.DataFrame, list]: Returns the processed dataframe and the added features(list) as a tuple
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/tabular_datamodule.py:TabularDatamodule:setup' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/tabular_datamodule.py#L245'>TabularDatamodule:setup</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>stage:  Optional[str]  =  None,<br></ul>
        <li>Docs:<br>        """Data Operations you want to perform on all GPUs, like train-test split, transformations, etc.
<br>
        This is called before accessing the dataloaders
<br>

<br>
        Args:
<br>
            stage (Optional[str], optional): Internal parameter to distinguish between fit and inference. Defaults to None.
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/tabular_datamodule.py:TabularDatamodule:time_features_from_frequency_str' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/tabular_datamodule.py#L277'>TabularDatamodule:time_features_from_frequency_str</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>cls,<br>freq_str:  str,<br></ul>
        <li>Docs:<br>        """
<br>
        Returns a list of time features that will be appropriate for the given frequency string.
<br>

<br>
        Parameters
<br>
        ----------
<br>

<br>
        freq_str
<br>
            Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
<br>

<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/tabular_datamodule.py:TabularDatamodule:val_dataloader' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/tabular_datamodule.py#L470'>TabularDatamodule:val_dataloader</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br>        """ Function that loads the validation set. """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/tabular_datamodule.py:TabularDatamodule:test_dataloader' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/tabular_datamodule.py#L484'>TabularDatamodule:test_dataloader</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br>        """ Function that loads the validation set. """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/tabular_datamodule.py:TabularDatamodule:prepare_inference_dataloader' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/tabular_datamodule.py#L502'>TabularDatamodule:prepare_inference_dataloader</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>df:  pd.DataFrame,<br></ul>
        <li>Docs:<br>        """Function that prepares and loads the new data.
<br>

<br>
        Args:
<br>
            df (pd.DataFrame): Dataframe with the features and target
<br>

<br>
        Returns:
<br>
            DataLoader: The dataloader for the passed in dataframe
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/tabular_datamodule.py:TabularDataset:__init__' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/tabular_datamodule.py#L53'>TabularDataset:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>data:  pd.DataFrame,<br>task:  str,<br>continuous_cols:  List[str]  =  None,<br>categorical_cols:  List[str]  =  None,<br>embed_categorical:  bool  =  True,<br>target:  List[str]  =  None,<br>,<br></ul>
        <li>Docs:<br>        """The Pytorch Lightning Datamodule for Tabular Data
<br>

<br>
        Args:
<br>
            train (pd.DataFrame): The Training Dataframe
<br>
            config (DictConfig): Merged configuration object from ModelConfig, DataConfig,
<br>
            TrainerConfig, OptimizerConfig & ExperimentConfig
<br>
            validation (pd.DataFrame, optional): Validation Dataframe.
<br>
            If left empty, we use the validation split from DataConfig to split a random sample as validation.
<br>
            Defaults to None.
<br>
            test (pd.DataFrame, optional): Holdout DataFrame to check final performance on.
<br>
            Defaults to None.
<br>
            target_transform (Optional[Union[TransformerMixin, Tuple(Callable)]], optional): If provided, applies the transform to the target before modelling
<br>
            and inverse the transform during prediction. The parameter can either be a sklearn Transformer which has an inverse_transform method, or
<br>
            a tuple of callables (transform_func, inverse_transform_func)
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/tabular_datamodule.py:TabularDataset:__len__' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/tabular_datamodule.py#L587'>TabularDataset:__len__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br>        """
<br>
        Denotes the total number of samples.
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/tabular_datamodule.py:TabularDataset:__getitem__' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/tabular_datamodule.py#L593'>TabularDataset:__getitem__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>idx,<br></ul>
        <li>Docs:<br>        """
<br>
        Generates one sample of data.
<br>
        """
<br>
</li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='pytorch_tabular/pytorch_tabular/models/base_model.py' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/base_model.py'>pytorch_tabular/pytorch_tabular/models/base_model.py</a>
</summary>
<ul>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/models/base_model.py:BaseModel' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/base_model.py#L27'>BaseModel</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/base_model.py:BaseModel:__init__' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/base_model.py#L28'>BaseModel:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>config:  DictConfig,<br>custom_loss:  Optional[torch.nn.Module]  =  None,<br>custom_metrics:  Optional[List[Callable]]  =  None,<br>custom_optimizer:  Optional[torch.optim.Optimizer]  =  None,<br>custom_optimizer_params:  Dict  =  {},<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/base_model.py:BaseModel:_build_network' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/base_model.py#L59'>BaseModel:_build_network</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/base_model.py:BaseModel:_setup_loss' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/base_model.py#L62'>BaseModel:_setup_loss</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/base_model.py:BaseModel:_setup_metrics' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/base_model.py#L74'>BaseModel:_setup_metrics</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/base_model.py:BaseModel:calculate_loss' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/base_model.py#L90'>BaseModel:calculate_loss</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>y,<br>y_hat,<br>tag,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/base_model.py:BaseModel:calculate_metrics' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/base_model.py#L119'>BaseModel:calculate_metrics</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>y,<br>y_hat,<br>tag,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/base_model.py:BaseModel:data_aware_initialization' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/base_model.py#L163'>BaseModel:data_aware_initialization</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>datamodule,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/base_model.py:BaseModel:forward' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/base_model.py#L167'>BaseModel:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x:  Dict,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/base_model.py:BaseModel:predict' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/base_model.py#L170'>BaseModel:predict</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x:  Dict,<br>ret_model_output:  bool  =  False,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/base_model.py:BaseModel:training_step' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/base_model.py#L177'>BaseModel:training_step</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>batch,<br>batch_idx,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/base_model.py:BaseModel:validation_step' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/base_model.py#L184'>BaseModel:validation_step</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>batch,<br>batch_idx,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/base_model.py:BaseModel:test_step' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/base_model.py#L191'>BaseModel:test_step</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>batch,<br>batch_idx,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/base_model.py:BaseModel:configure_optimizers' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/base_model.py#L198'>BaseModel:configure_optimizers</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/base_model.py:BaseModel:create_plotly_histogram' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/base_model.py#L250'>BaseModel:create_plotly_histogram</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>arr,<br>name,<br>bin_dict = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/base_model.py:BaseModel:validation_epoch_end' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/base_model.py#L272'>BaseModel:validation_epoch_end</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>outputs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='pytorch_tabular/setup.py' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/setup.py'>pytorch_tabular/setup.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='pytorch_tabular/setup.py:read_requirements' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/setup.py#L8'>read_requirements</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>thelibFolder,<br>filename,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='pytorch_tabular/pytorch_tabular/models/node/architecture_blocks.py' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/architecture_blocks.py'>pytorch_tabular/pytorch_tabular/models/node/architecture_blocks.py</a>
</summary>
<ul>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/models/node/architecture_blocks.py:DenseODSTBlock' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/architecture_blocks.py#L13'>DenseODSTBlock</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/node/architecture_blocks.py:DenseODSTBlock:__init__' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/architecture_blocks.py#L14'>DenseODSTBlock:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>input_dim,<br>num_trees,<br>num_layers,<br>tree_output_dim = 1,<br>max_features = None,<br>input_dropout = 0.0,<br>flatten_output = False,<br>Module = ODST,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/node/architecture_blocks.py:DenseODSTBlock:forward' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/architecture_blocks.py#L49'>DenseODSTBlock:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='pytorch_tabular/tests/test_categorical_embedding.py' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/tests/test_categorical_embedding.py'>pytorch_tabular/tests/test_categorical_embedding.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='pytorch_tabular/tests/test_categorical_embedding.py:fake_metric' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/tests/test_categorical_embedding.py#L16'>fake_metric</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>y_hat,<br>y,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='pytorch_tabular/tests/test_categorical_embedding.py:test_regression' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/tests/test_categorical_embedding.py#L46'>test_regression</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>regression_data,<br>multi_target,<br>continuous_cols,<br>categorical_cols,<br>continuous_feature_transform,<br>normalize_continuous_features,<br>target_range,<br>target_transform,<br>custom_metrics,<br>custom_loss,<br>custom_optimizer,<br>,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='pytorch_tabular/tests/test_categorical_embedding.py:test_classification' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/tests/test_categorical_embedding.py#L123'>test_classification</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>classification_data,<br>continuous_cols,<br>categorical_cols,<br>continuous_feature_transform,<br>normalize_continuous_features,<br>,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='pytorch_tabular/tests/test_categorical_embedding.py:test_embedding_transformer' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/tests/test_categorical_embedding.py#L163'>test_embedding_transformer</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>regression_data,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='pytorch_tabular/examples/to_test_regression.py' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/examples/to_test_regression.py'>pytorch_tabular/examples/to_test_regression.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='pytorch_tabular/examples/to_test_regression.py:fake_metric' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/examples/to_test_regression.py#L96'>fake_metric</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>y_hat,<br>y,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='pytorch_tabular/pytorch_tabular/categorical_encoders.py' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/categorical_encoders.py'>pytorch_tabular/pytorch_tabular/categorical_encoders.py</a>
</summary>
<ul>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/categorical_encoders.py:BaseEncoder' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/categorical_encoders.py#L22'>BaseEncoder</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/categorical_encoders.py:OrdinalEncoder' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/categorical_encoders.py#L100'>OrdinalEncoder</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/categorical_encoders.py:CategoricalEmbeddingTransformer' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/categorical_encoders.py#L133'>CategoricalEmbeddingTransformer</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/categorical_encoders.py:BaseEncoder:__init__' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/categorical_encoders.py#L23'>BaseEncoder:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>cols,<br>handle_unseen,<br>min_samples,<br>imputed,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/categorical_encoders.py:BaseEncoder:transform' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/categorical_encoders.py#L32'>BaseEncoder:transform</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>X,<br></ul>
        <li>Docs:<br>        """Transform categorical data based on mapping learnt at fitting time.
<br>
        :param pandas.DataFrame X: DataFrame of features, shape (n_samples, n_features). Must contain columns to encode.
<br>
        :return: encoded DataFrame of shape (n_samples, n_features), initial categorical columns are dropped, and
<br>
            replaced with encoded columns. DataFrame passed in argument is unchanged.
<br>
        :rtype: pandas.DataFrame
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/categorical_encoders.py:BaseEncoder:fit_transform' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/categorical_encoders.py#L59'>BaseEncoder:fit_transform</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>X,<br>y = None,<br></ul>
        <li>Docs:<br>        """Encode given columns of X according to y, and transform X based on the learnt mapping.
<br>
        :param pandas.DataFrame X: DataFrame of features, shape (n_samples, n_features). Must contain columns to encode.
<br>
        :param pandas.Series y: pandas Series of target values, shape (n_samples,).
<br>
            Required only for encoders that need it: TargetEncoder, WeightOfEvidenceEncoder
<br>
        :return: encoded DataFrame of shape (n_samples, n_features), initial categorical columns are dropped, and
<br>
            replaced with encoded columns. DataFrame passed in argument is unchanged.
<br>
        :rtype: pandas.DataFrame
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/categorical_encoders.py:BaseEncoder:_input_check' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/categorical_encoders.py#L71'>BaseEncoder:_input_check</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>name,<br>value,<br>options,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/categorical_encoders.py:BaseEncoder:_before_fit_check' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/categorical_encoders.py#L77'>BaseEncoder:_before_fit_check</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>X,<br>y,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/categorical_encoders.py:BaseEncoder:save_as_object_file' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/categorical_encoders.py#L88'>BaseEncoder:save_as_object_file</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>path,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/categorical_encoders.py:BaseEncoder:load_from_object_file' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/categorical_encoders.py#L95'>BaseEncoder:load_from_object_file</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>path,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/categorical_encoders.py:OrdinalEncoder:__init__' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/categorical_encoders.py#L105'>OrdinalEncoder:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>cols = None,<br>handle_unseen = "impute",<br></ul>
        <li>Docs:<br>        """Instantiation
<br>
        :param [str] cols: list of columns to encode, or None (then all dataset columns will be encoded at fitting time)
<br>
        :param str handle_unseen:
<br>
            'error'  - raise an error if a category unseen at fitting time is found
<br>
            'ignore' - skip unseen categories
<br>
            'impute' - impute new categories to a predefined value, which is same as NAN_CATEGORY
<br>
        :return: None
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/categorical_encoders.py:OrdinalEncoder:fit' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/categorical_encoders.py#L117'>OrdinalEncoder:fit</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>X,<br>y = None,<br></ul>
        <li>Docs:<br>        """Label Encode given columns of X.
<br>
        :param pandas.DataFrame X: DataFrame of features, shape (n_samples, n_features). Must contain columns to encode.
<br>
        :return: None
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/categorical_encoders.py:CategoricalEmbeddingTransformer:__init__' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/categorical_encoders.py#L137'>CategoricalEmbeddingTransformer:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>tabular_model,<br></ul>
        <li>Docs:<br>        """Initializes the Transformer and extracts the neural embeddings
<br>

<br>
        Args:
<br>
            tabular_model (TabularModel): The trained TabularModel object
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/categorical_encoders.py:CategoricalEmbeddingTransformer:_extract_embedding' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/categorical_encoders.py#L150'>CategoricalEmbeddingTransformer:_extract_embedding</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>model,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/categorical_encoders.py:CategoricalEmbeddingTransformer:fit' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/categorical_encoders.py#L117'>CategoricalEmbeddingTransformer:fit</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>X,<br>y = None,<br></ul>
        <li>Docs:<br>        """Label Encode given columns of X.
<br>
        :param pandas.DataFrame X: DataFrame of features, shape (n_samples, n_features). Must contain columns to encode.
<br>
        :return: None
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/categorical_encoders.py:CategoricalEmbeddingTransformer:transform' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/categorical_encoders.py#L179'>CategoricalEmbeddingTransformer:transform</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>X:  pd.DataFrame,<br>y = None,<br></ul>
        <li>Docs:<br>        """Transforms the categorical columns specified to the trained neural embedding from the model
<br>

<br>
        Args:
<br>
            X (pd.DataFrame): DataFrame of features, shape (n_samples, n_features). Must contain columns to encode.
<br>
            y ([type], optional): Only for compatibility. Not used. Defaults to None.
<br>

<br>
        Raises:
<br>
            ValueError: [description]
<br>

<br>
        Returns:
<br>
            pd.DataFrame: The encoded dataframe
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/categorical_encoders.py:CategoricalEmbeddingTransformer:fit_transform' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/categorical_encoders.py#L217'>CategoricalEmbeddingTransformer:fit_transform</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>X:  pd.DataFrame,<br>y = None,<br></ul>
        <li>Docs:<br>        """Encode given columns of X based on the learned embedding.
<br>

<br>
        Args:
<br>
            X (pd.DataFrame): DataFrame of features, shape (n_samples, n_features). Must contain columns to encode.
<br>
            y ([type], optional): Only for compatibility. Not used. Defaults to None.
<br>

<br>
        Returns:
<br>
            pd.DataFrame: The encoded dataframe
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/categorical_encoders.py:CategoricalEmbeddingTransformer:save_as_object_file' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/categorical_encoders.py#L88'>CategoricalEmbeddingTransformer:save_as_object_file</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>path,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/categorical_encoders.py:CategoricalEmbeddingTransformer:load_from_object_file' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/categorical_encoders.py#L95'>CategoricalEmbeddingTransformer:load_from_object_file</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>path,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='pytorch_tabular/pytorch_tabular/utils.py' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/utils.py'>pytorch_tabular/pytorch_tabular/utils.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='pytorch_tabular/pytorch_tabular/utils.py:_make_smooth_weights_for_balanced_classes' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/utils.py#L14'>_make_smooth_weights_for_balanced_classes</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>y_train,<br>mu = 0.15,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='pytorch_tabular/pytorch_tabular/utils.py:get_class_weighted_cross_entropy' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/utils.py#L27'>get_class_weighted_cross_entropy</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>y_train,<br>mu = 0.15,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='pytorch_tabular/pytorch_tabular/utils.py:get_balanced_sampler' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/utils.py#L35'>get_balanced_sampler</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>y_train,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='pytorch_tabular/pytorch_tabular/utils.py:_initialize_layers' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/utils.py#L50'>_initialize_layers</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>hparams,<br>layer,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='pytorch_tabular/pytorch_tabular/utils.py:_linear_dropout_bn' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/utils.py#L77'>_linear_dropout_bn</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>hparams,<br>in_units,<br>out_units,<br>activation,<br>dropout,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='pytorch_tabular/pytorch_tabular/utils.py:get_gaussian_centers' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/utils.py#L89'>get_gaussian_centers</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>y,<br>n_components,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='pytorch_tabular/pytorch_tabular/tabular_model.py' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/tabular_model.py'>pytorch_tabular/pytorch_tabular/tabular_model.py</a>
</summary>
<ul>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/tabular_model.py:TabularModel' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/tabular_model.py#L39'>TabularModel</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/tabular_model.py:TabularModel:__init__' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/tabular_model.py#L40'>TabularModel:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>config:  Optional[DictConfig]  =  None,<br>data_config:  Optional[Union[DataConfig,<br>str]]  =  None,<br>model_config:  Optional[Union[ModelConfig,<br>str]]  =  None,<br>optimizer_config:  Optional[Union[OptimizerConfig,<br>str]]  =  None,<br>trainer_config:  Optional[Union[TrainerConfig,<br>str]]  =  None,<br>experiment_config:  Optional[Union[ExperimentConfig,<br>str]]  =  None,<br>model_callable:  Optional[Callable]  =  None,<br>,<br></ul>
        <li>Docs:<br>        """The core model which orchestrates everything from initializing the datamodule, the model, trainer, etc.
<br>

<br>
        Args:
<br>
            config (Optional[Union[DictConfig, str]], optional): Single OmegaConf DictConfig object or
<br>
                the path to the yaml file holding all the config parameters. Defaults to None.
<br>

<br>
            data_config (Optional[Union[DataConfig, str]], optional): DataConfig object or path to the yaml file. Defaults to None.
<br>

<br>
            model_config (Optional[Union[ModelConfig, str]], optional): A subclass of ModelConfig or path to the yaml file.
<br>
                Determines which model to run from the type of config. Defaults to None.
<br>

<br>
            optimizer_config (Optional[Union[OptimizerConfig, str]], optional): OptimizerConfig object or path to the yaml file.
<br>
                Defaults to None.
<br>

<br>
            trainer_config (Optional[Union[TrainerConfig, str]], optional): TrainerConfig object or path to the yaml file.
<br>
                Defaults to None.
<br>

<br>
            experiment_config (Optional[Union[ExperimentConfig, str]], optional): ExperimentConfig object or path to the yaml file.
<br>
                If Provided configures the experiment tracking. Defaults to None.
<br>

<br>
            model_callable (Optional[Callable], optional): If provided, will override the model callable that will be loaded from the config.
<br>
                Typically used when providing Custom Models
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/tabular_model.py:TabularModel:_run_validation' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/tabular_model.py#L135'>TabularModel:_run_validation</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br>        """Validates the Config params and throws errors if something is wrong
<br>

<br>
        Raises:
<br>
            NotImplementedError: If you provide a multi-target config to a classification task
<br>
            ValueError: If there is a problem with Target Range
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/tabular_model.py:TabularModel:_read_parse_config' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/tabular_model.py#L160'>TabularModel:_read_parse_config</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>config,<br>cls,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/tabular_model.py:TabularModel:_get_run_name_uid' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/tabular_model.py#L181'>TabularModel:_get_run_name_uid</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br>        """Gets the name of the experiment and increments version by 1
<br>

<br>
        Returns:
<br>
            tuple[str, int]: Returns the name and version number
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/tabular_model.py:TabularModel:_setup_experiment_tracking' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/tabular_model.py#L196'>TabularModel:_setup_experiment_tracking</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br>        """Sets up the Experiment Tracking Framework according to the choices made in the Experimentconfig
<br>

<br>
        Raises:
<br>
            NotImplementedError: Raises an Error for invalid choices of log_target
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/tabular_model.py:TabularModel:_prepare_callbacks' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/tabular_model.py#L217'>TabularModel:_prepare_callbacks</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br>        """Prepares the necesary callbacks to the Trainer based on the configuration
<br>

<br>
        Returns:
<br>
            List: A list of callbacks
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/tabular_model.py:TabularModel:_prepare_dataloader' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/tabular_model.py#L250'>TabularModel:_prepare_dataloader</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>train,<br>validation,<br>test,<br>target_transform = None,<br>train_sampler = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/tabular_model.py:TabularModel:_prepare_model' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/tabular_model.py#L275'>TabularModel:_prepare_model</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>loss,<br>metrics,<br>optimizer,<br>optimizer_params,<br>reset,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/tabular_model.py:TabularModel:_prepare_trainer' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/tabular_model.py#L294'>TabularModel:_prepare_trainer</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>max_epochs = None,<br>min_epochs = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/tabular_model.py:TabularModel:load_best_model' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/tabular_model.py#L314'>TabularModel:load_best_model</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br>        """Loads the best model after training is done"""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/tabular_model.py:TabularModel:_pre_fit' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/tabular_model.py#L327'>TabularModel:_pre_fit</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>train:  pd.DataFrame,<br>validation:  Optional[pd.DataFrame],<br>test:  Optional[pd.DataFrame],<br>loss:  Optional[torch.nn.Module],<br>metrics:  Optional[List[Callable]],<br>optimizer:  Optional[torch.optim.Optimizer],<br>optimizer_params:  Dict,<br>train_sampler:  Optional[torch.utils.data.Sampler],<br>target_transform:  Optional[Union[TransformerMixin,<br>Tuple]],<br>max_epochs:  int,<br>min_epochs:  int,<br>reset:  bool,<br></ul>
        <li>Docs:<br>        """Prepares the dataloaders, trainer, and model for the fit process"""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/tabular_model.py:TabularModel:fit' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/tabular_model.py#L372'>TabularModel:fit</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>train:  pd.DataFrame,<br>validation:  Optional[pd.DataFrame]  =  None,<br>test:  Optional[pd.DataFrame]  =  None,<br>loss:  Optional[torch.nn.Module]  =  None,<br>metrics:  Optional[List[Callable]]  =  None,<br>optimizer:  Optional[torch.optim.Optimizer]  =  None,<br>optimizer_params:  Dict  =  {},<br>train_sampler:  Optional[torch.utils.data.Sampler]  =  None,<br>target_transform:  Optional[Union[TransformerMixin,<br>Tuple]]  =  None,<br>max_epochs:  Optional[int]  =  None,<br>min_epochs:  Optional[int]  =  None,<br>reset:  bool  =  False,<br>seed:  Optional[int]  =  None,<br>,<br></ul>
        <li>Docs:<br>        """The fit method which takes in the data and triggers the training
<br>

<br>
        Args:
<br>
            train (pd.DataFrame): Training Dataframe
<br>

<br>
            valid (Optional[pd.DataFrame], optional): If provided, will use this dataframe as the validation while training.
<br>
                Used in Early Stopping and Logging. If left empty, will use 20% of Train data as validation. Defaults to None.
<br>

<br>
            test (Optional[pd.DataFrame], optional): If provided, will use as the hold-out data,
<br>
                which you'll be able to check performance after the model is trained. Defaults to None.
<br>

<br>
            loss (Optional[torch.nn.Module], optional): Custom Loss functions which are not in standard pytorch library
<br>

<br>
            metrics (Optional[List[Callable]], optional): Custom metric functions(Callable) which has the
<br>
                signature metric_fn(y_hat, y) and works on torch tensor inputs
<br>

<br>
            optimizer (Optional[torch.optim.Optimizer], optional): Custom optimizers which are a drop in replacements for standard PyToch optimizers.
<br>
                This should be the Class and not the initialized object
<br>

<br>
            optimizer_params (Optional[Dict], optional): The parmeters to initialize the custom optimizer.
<br>

<br>
            train_sampler (Optional[torch.utils.data.Sampler], optional): Custom PyTorch batch samplers which will be passed to the DataLoaders. Useful for dealing with imbalanced data and other custom batching strategies
<br>

<br>
            target_transform (Optional[Union[TransformerMixin, Tuple(Callable)]], optional): If provided, applies the transform to the target before modelling
<br>
                and inverse the transform during prediction. The parameter can either be a sklearn Transformer which has an inverse_transform method, or
<br>
                a tuple of callables (transform_func, inverse_transform_func)
<br>

<br>
            max_epochs (Optional[int]): Overwrite maximum number of epochs to be run
<br>

<br>
            min_epochs (Optional[int]): Overwrite minimum number of epochs to be run
<br>

<br>
            reset: (bool): Flag to reset the model and train again from scratch
<br>

<br>
            seed: (int): If you have to override the default seed set as part of of ModelConfig
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/tabular_model.py:TabularModel:find_learning_rate' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/tabular_model.py#L449'>TabularModel:find_learning_rate</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>train:  pd.DataFrame,<br>validation:  Optional[pd.DataFrame]  =  None,<br>test:  Optional[pd.DataFrame]  =  None,<br>loss:  Optional[torch.nn.Module]  =  None,<br>metrics:  Optional[List[Callable]]  =  None,<br>optimizer:  Optional[torch.optim.Optimizer]  =  None,<br>optimizer_params:  Dict  =  {},<br>min_lr:  float  =  1e-8,<br>max_lr:  float  =  1,<br>num_training:  int  =  100,<br>mode:  str  =  "exponential",<br>early_stop_threshold:  float  =  4.0,<br>plot = True,<br>,<br></ul>
        <li>Docs:<br>        """Enables the user to do a range test of good initial learning rates, to reduce the amount of guesswork in picking a good starting learning rate.
<br>

<br>
        Args:
<br>
            train (pd.DataFrame): Training Dataframe
<br>

<br>
            valid (Optional[pd.DataFrame], optional): If provided, will use this dataframe as the validation while training.
<br>
                Used in Early Stopping and Logging. If left empty, will use 20% of Train data as validation. Defaults to None.
<br>

<br>
            test (Optional[pd.DataFrame], optional): If provided, will use as the hold-out data,
<br>
                which you'll be able to check performance after the model is trained. Defaults to None.
<br>

<br>
            loss (Optional[torch.nn.Module], optional): Custom Loss functions which are not in standard pytorch library
<br>

<br>
            metrics (Optional[List[Callable]], optional): Custom metric functions(Callable) which has the signature metric_fn(y_hat, y)
<br>

<br>
            optimizer (Optional[torch.optim.Optimizer], optional): Custom optimizers which are a drop in replacements for standard PyToch optimizers.
<br>
                This should be the Class and not the initialized object
<br>

<br>
            optimizer_params (Optional[Dict], optional): The parmeters to initialize the custom optimizer.
<br>

<br>
            min_lr (Optional[float], optional): minimum learning rate to investigate
<br>

<br>
            max_lr (Optional[float], optional): maximum learning rate to investigate
<br>

<br>
            num_training (Optional[int], optional): number of learning rates to test
<br>

<br>
            mode (Optional[str], optional): search strategy, either 'linear' or 'exponential'. If set to
<br>
                'linear' the learning rate will be searched by linearly increasing
<br>
                after each batch. If set to 'exponential', will increase learning
<br>
                rate exponentially.
<br>

<br>
            early_stop_threshold(Optional[float], optional): threshold for stopping the search. If the
<br>
                loss at any point is larger than early_stop_threshold*best_loss
<br>
                then the search is stopped. To disable, set to None.
<br>

<br>
            plot(bool, optional): If true, will plot using matplotlib
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/tabular_model.py:TabularModel:evaluate' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/tabular_model.py#L536'>TabularModel:evaluate</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>test:  Optional[pd.DataFrame],<br></ul>
        <li>Docs:<br>        """Evaluates the dataframe using the loss and metrics already set in config
<br>

<br>
        Args:
<br>
            test (Optional[pd.DataFrame]): The dataframe to be evaluated. If not provided, will try to use the
<br>
                test provided during fit. If that was also not provided will return an empty dictionary
<br>

<br>
        Returns:
<br>
            Union[dict, list]: The final test result dictionary.
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/tabular_model.py:TabularModel:predict' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/tabular_model.py#L558'>TabularModel:predict</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>test:  pd.DataFrame,<br>quantiles:  Optional[List]  =  [0.25,<br>0.5,<br>0.75],<br>n_samples:  Optional[int]  =  100,<br>ret_logits = False,<br>,<br></ul>
        <li>Docs:<br>        """Uses the trained model to predict on new data and return as a dataframe
<br>

<br>
        Args:
<br>
            test (pd.DataFrame): The new dataframe with the features defined during training
<br>
            quantiles (Optional[List]): For probabilistic models like Mixture Density Networks, this specifies
<br>
                the different quantiles to be extracted apart from the `central_tendency` and added to the dataframe.
<br>
                For other models it is ignored. Defaults to [0.25, 0.5, 0.75]
<br>
            n_samples (Optional[int]): Number of samples to draw from the posterior to estimate the quantiles.
<br>
                Ignored for non-probabilistic models. Defaults to 100
<br>
            ret_logits (bool): Flag to return raw model outputs/logits except the backbone features along
<br>
                with the dataframe. Defaults to False
<br>

<br>
        Returns:
<br>
            pd.DataFrame: Returns a dataframe with predictions and features.
<br>
                If classification, it returns probabilities and final prediction
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/tabular_model.py:TabularModel:save_model' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/tabular_model.py#L681'>TabularModel:save_model</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>dir:  str,<br></ul>
        <li>Docs:<br>        """Saves the model and checkpoints in the specified directory
<br>

<br>
        Args:
<br>
            dir (str): The path to the directory to save the model
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/tabular_model.py:TabularModel:load_from_checkpoint' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/tabular_model.py#L712'>TabularModel:load_from_checkpoint</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>cls,<br>dir:  str,<br></ul>
        <li>Docs:<br>        """Loads a saved model from the directory
<br>

<br>
        Args:
<br>
            dir (str): The directory where the model wa saved, along with the checkpoints
<br>

<br>
        Returns:
<br>
            TabularModel: The saved TabularModel
<br>
        """
<br>
</li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='pytorch_tabular/tests/test_mdn.py' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/tests/test_mdn.py'>pytorch_tabular/tests/test_mdn.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='pytorch_tabular/tests/test_mdn.py:test_regression' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/tests/test_mdn.py#L31'>test_regression</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>regression_data,<br>multi_target,<br>continuous_cols,<br>categorical_cols,<br>continuous_feature_transform,<br>normalize_continuous_features,<br>variant,<br>num_gaussian,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='pytorch_tabular/tests/test_mdn.py:test_classification' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/tests/test_mdn.py#L87'>test_classification</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>classification_data,<br>continuous_cols,<br>categorical_cols,<br>continuous_feature_transform,<br>normalize_continuous_features,<br>num_gaussian,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py'>pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py</a>
</summary>
<ul>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py:MixtureDensityHead' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py#L35'>MixtureDensityHead</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py:BaseMDN' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py#L135'>BaseMDN</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py:CategoryEmbeddingMDN' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py#L337'>CategoryEmbeddingMDN</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py:NODEMDN' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py#L379'>NODEMDN</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py:AutoIntMDN' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py#L409'>AutoIntMDN</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py:MixtureDensityHead:__init__' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py#L36'>MixtureDensityHead:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>config:  DictConfig,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py:MixtureDensityHead:_build_network' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py#L41'>MixtureDensityHead:_build_network</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py:MixtureDensityHead:forward' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py#L55'>MixtureDensityHead:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py:MixtureDensityHead:gaussian_probability' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py#L63'>MixtureDensityHead:gaussian_probability</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>sigma,<br>mu,<br>target,<br>log = False,<br></ul>
        <li>Docs:<br>        """Returns the probability of `target` given MoG parameters `sigma` and `mu`.
<br>

<br>
        Arguments:
<br>
            sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
<br>
                size, G is the number of Gaussians, and O is the number of
<br>
                dimensions per Gaussian.
<br>
            mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
<br>
                number of Gaussians, and O is the number of dimensions per Gaussian.
<br>
            target (BxI): A batch of target. B is the batch size and I is the number of
<br>
                input dimensions.
<br>
        Returns:
<br>
            probabilities (BxG): The probability of each point in the probability
<br>
                of the distribution in the corresponding sigma/mu index.
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py:MixtureDensityHead:log_prob' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py#L91'>MixtureDensityHead:log_prob</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>pi,<br>sigma,<br>mu,<br>y,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py:MixtureDensityHead:sample' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py#L101'>MixtureDensityHead:sample</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>pi,<br>sigma,<br>mu,<br></ul>
        <li>Docs:<br>        """Draw samples from a MoG."""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py:MixtureDensityHead:generate_samples' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py#L110'>MixtureDensityHead:generate_samples</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>pi,<br>sigma,<br>mu,<br>n_samples = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py:MixtureDensityHead:generate_point_predictions' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py#L125'>MixtureDensityHead:generate_point_predictions</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>pi,<br>sigma,<br>mu,<br>n_samples = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py:BaseMDN:__init__' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py#L36'>BaseMDN:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>config:  DictConfig,<br>**kwargs,<br></ul>
        <li>Docs:<br>        """Returns the probability of `target` given MoG parameters `sigma` and `mu`.
<br>

<br>
        Arguments:
<br>
            sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
<br>
                size, G is the number of Gaussians, and O is the number of
<br>
                dimensions per Gaussian.
<br>
            mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
<br>
                number of Gaussians, and O is the number of dimensions per Gaussian.
<br>
            target (BxI): A batch of target. B is the batch size and I is the number of
<br>
                input dimensions.
<br>
        Returns:
<br>
            probabilities (BxG): The probability of each point in the probability
<br>
                of the distribution in the corresponding sigma/mu index.
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py:BaseMDN:unpack_input' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py#L144'>BaseMDN:unpack_input</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x:  Dict,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py:BaseMDN:forward' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py#L147'>BaseMDN:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x:  Dict,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py:BaseMDN:predict' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py#L153'>BaseMDN:predict</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x:  Dict,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py:BaseMDN:sample' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py#L159'>BaseMDN:sample</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x:  Dict,<br>n_samples:  Optional[int]  =  None,<br>ret_model_output = False,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py:BaseMDN:calculate_loss' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py#L169'>BaseMDN:calculate_loss</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>y,<br>pi,<br>sigma,<br>mu,<br>tag = "train",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py:BaseMDN:training_step' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py#L208'>BaseMDN:training_step</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>batch,<br>batch_idx,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py:BaseMDN:validation_step' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py#L223'>BaseMDN:validation_step</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>batch,<br>batch_idx,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py:BaseMDN:test_step' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py#L235'>BaseMDN:test_step</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>batch,<br>batch_idx,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py:BaseMDN:validation_epoch_end' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py#L247'>BaseMDN:validation_epoch_end</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>outputs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py:CategoryEmbeddingMDN:__init__' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py#L36'>CategoryEmbeddingMDN:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>config:  DictConfig,<br>**kwargs,<br></ul>
        <li>Docs:<br>        """Returns the probability of `target` given MoG parameters `sigma` and `mu`.
<br>

<br>
        Arguments:
<br>
            sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
<br>
                size, G is the number of Gaussians, and O is the number of
<br>
                dimensions per Gaussian.
<br>
            mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
<br>
                number of Gaussians, and O is the number of dimensions per Gaussian.
<br>
            target (BxI): A batch of target. B is the batch size and I is the number of
<br>
                input dimensions.
<br>
        Returns:
<br>
            probabilities (BxG): The probability of each point in the probability
<br>
                of the distribution in the corresponding sigma/mu index.
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py:CategoryEmbeddingMDN:_build_network' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py#L41'>CategoryEmbeddingMDN:_build_network</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py:CategoryEmbeddingMDN:unpack_input' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py#L144'>CategoryEmbeddingMDN:unpack_input</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x:  Dict,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py:NODEMDN:__init__' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py#L36'>NODEMDN:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>config:  DictConfig,<br>**kwargs,<br></ul>
        <li>Docs:<br>        """Returns the probability of `target` given MoG parameters `sigma` and `mu`.
<br>

<br>
        Arguments:
<br>
            sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
<br>
                size, G is the number of Gaussians, and O is the number of
<br>
                dimensions per Gaussian.
<br>
            mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
<br>
                number of Gaussians, and O is the number of dimensions per Gaussian.
<br>
            target (BxI): A batch of target. B is the batch size and I is the number of
<br>
                input dimensions.
<br>
        Returns:
<br>
            probabilities (BxG): The probability of each point in the probability
<br>
                of the distribution in the corresponding sigma/mu index.
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py:NODEMDN:subset' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py#L383'>NODEMDN:subset</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py:NODEMDN:_build_network' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py#L41'>NODEMDN:_build_network</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py:NODEMDN:unpack_input' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py#L144'>NODEMDN:unpack_input</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x:  Dict,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py:AutoIntMDN:__init__' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py#L36'>AutoIntMDN:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>config:  DictConfig,<br>**kwargs,<br></ul>
        <li>Docs:<br>        """Returns the probability of `target` given MoG parameters `sigma` and `mu`.
<br>

<br>
        Arguments:
<br>
            sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
<br>
                size, G is the number of Gaussians, and O is the number of
<br>
                dimensions per Gaussian.
<br>
            mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
<br>
                number of Gaussians, and O is the number of dimensions per Gaussian.
<br>
            target (BxI): A batch of target. B is the batch size and I is the number of
<br>
                input dimensions.
<br>
        Returns:
<br>
            probabilities (BxG): The probability of each point in the probability
<br>
                of the distribution in the corresponding sigma/mu index.
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py:AutoIntMDN:_build_network' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py#L41'>AutoIntMDN:_build_network</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br>        """Returns the probability of `target` given MoG parameters `sigma` and `mu`.
<br>

<br>
        Arguments:
<br>
            sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
<br>
                size, G is the number of Gaussians, and O is the number of
<br>
                dimensions per Gaussian.
<br>
            mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
<br>
                number of Gaussians, and O is the number of dimensions per Gaussian.
<br>
            target (BxI): A batch of target. B is the batch size and I is the number of
<br>
                input dimensions.
<br>
        Returns:
<br>
            probabilities (BxG): The probability of each point in the probability
<br>
                of the distribution in the corresponding sigma/mu index.
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py:AutoIntMDN:unpack_input' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/mixture_density/mdn.py#L144'>AutoIntMDN:unpack_input</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x:  Dict,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='pytorch_tabular/pytorch_tabular/models/node/node_model.py' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/node_model.py'>pytorch_tabular/pytorch_tabular/models/node/node_model.py</a>
</summary>
<ul>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/models/node/node_model.py:NODEBackbone' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/node_model.py#L22'>NODEBackbone</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/models/node/node_model.py:NODEModel' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/node_model.py#L58'>NODEModel</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/node/node_model.py:NODEBackbone:__init__' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/node_model.py#L23'>NODEBackbone:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>config:  DictConfig,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/node/node_model.py:NODEBackbone:_build_network' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/node_model.py#L28'>NODEBackbone:_build_network</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/node/node_model.py:NODEBackbone:forward' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/node_model.py#L53'>NODEBackbone:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/node/node_model.py:NODEModel:__init__' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/node_model.py#L23'>NODEModel:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>config:  DictConfig,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/node/node_model.py:NODEModel:subset' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/node_model.py#L64'>NODEModel:subset</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/node/node_model.py:NODEModel:data_aware_initialization' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/node_model.py#L67'>NODEModel:data_aware_initialization</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>datamodule,<br></ul>
        <li>Docs:<br>        """Performs data-aware initialization for NODE"""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/node/node_model.py:NODEModel:_build_network' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/node_model.py#L28'>NODEModel:_build_network</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br>        """Performs data-aware initialization for NODE"""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/node/node_model.py:NODEModel:unpack_input' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/node_model.py#L104'>NODEModel:unpack_input</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x:  Dict,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/node/node_model.py:NODEModel:forward' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/node_model.py#L131'>NODEModel:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x:  Dict,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='pytorch_tabular/tests/test_common.py' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/tests/test_common.py'>pytorch_tabular/tests/test_common.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='pytorch_tabular/tests/test_common.py:fake_metric' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/tests/test_common.py#L30'>fake_metric</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>y_hat,<br>y,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='pytorch_tabular/tests/test_common.py:test_save_load' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/tests/test_common.py#L55'>test_save_load</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>regression_data,<br>model_config_class,<br>continuous_cols,<br>categorical_cols,<br>custom_metrics,<br>custom_loss,<br>custom_optimizer,<br>tmpdir,<br>,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='pytorch_tabular/tests/test_common.py:test_feature_extractor' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/tests/test_common.py#L125'>test_feature_extractor</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>regression_data,<br>model_config_class,<br>continuous_cols,<br>categorical_cols,<br>,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='pytorch_tabular/examples/to_test_node.py' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/examples/to_test_node.py'>pytorch_tabular/examples/to_test_node.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='pytorch_tabular/examples/to_test_node.py:regression_data' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/examples/to_test_node.py#L14'>regression_data</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='pytorch_tabular/examples/to_test_node.py:classification_data' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/examples/to_test_node.py#L25'>classification_data</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='pytorch_tabular/examples/to_test_node.py:test_regression' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/examples/to_test_node.py#L39'>test_regression</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>regression_data,<br>multi_target,<br>embed_categorical,<br>continuous_cols,<br>categorical_cols,<br>continuous_feature_transform,<br>normalize_continuous_features,<br>,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='pytorch_tabular/examples/to_test_node.py:test_classification' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/examples/to_test_node.py#L84'>test_classification</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>classification_data,<br>continuous_cols,<br>categorical_cols,<br>embed_categorical,<br>continuous_feature_transform,<br>normalize_continuous_features,<br>,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='pytorch_tabular/tests/test_autoint.py' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/tests/test_autoint.py'>pytorch_tabular/tests/test_autoint.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='pytorch_tabular/tests/test_autoint.py:test_regression' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/tests/test_autoint.py#L33'>test_regression</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>regression_data,<br>multi_target,<br>continuous_cols,<br>categorical_cols,<br>continuous_feature_transform,<br>normalize_continuous_features,<br>target_range,<br>deep_layers,<br>batch_norm_continuous_input,<br>attention_pooling,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='pytorch_tabular/tests/test_autoint.py:test_classification' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/tests/test_autoint.py#L102'>test_classification</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>classification_data,<br>continuous_cols,<br>categorical_cols,<br>continuous_feature_transform,<br>normalize_continuous_features,<br>deep_layers,<br>batch_norm_continuous_input,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='pytorch_tabular/tests/test_node.py' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/tests/test_node.py'>pytorch_tabular/tests/test_node.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='pytorch_tabular/tests/test_node.py:test_regression' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/tests/test_node.py#L30'>test_regression</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>regression_data,<br>multi_target,<br>embed_categorical,<br>continuous_cols,<br>categorical_cols,<br>continuous_feature_transform,<br>normalize_continuous_features,<br>target_range,<br>,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='pytorch_tabular/tests/test_node.py:test_classification' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/tests/test_node.py#L97'>test_classification</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>classification_data,<br>continuous_cols,<br>categorical_cols,<br>embed_categorical,<br>continuous_feature_transform,<br>normalize_continuous_features,<br>,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='pytorch_tabular/tests/test_node.py:test_embedding_transformer' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/tests/test_node.py#L142'>test_embedding_transformer</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>regression_data,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='pytorch_tabular/examples/adhoc_scaffold.py' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/examples/adhoc_scaffold.py'>pytorch_tabular/examples/adhoc_scaffold.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='pytorch_tabular/examples/adhoc_scaffold.py:make_mixed_classification' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/examples/adhoc_scaffold.py#L11'>make_mixed_classification</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>n_samples,<br>n_features,<br>n_categories,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='pytorch_tabular/examples/adhoc_scaffold.py:print_metrics' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/examples/adhoc_scaffold.py#L32'>print_metrics</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>y_true,<br>y_pred,<br>tag,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='pytorch_tabular/pytorch_tabular/models/category_embedding/config.py' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/category_embedding/config.py'>pytorch_tabular/pytorch_tabular/models/category_embedding/config.py</a>
</summary>
<ul>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/models/category_embedding/config.py:CategoryEmbeddingModelConfig' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/category_embedding/config.py#L12'>CategoryEmbeddingModelConfig</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='pytorch_tabular/tests/test_tabnet.py' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/tests/test_tabnet.py'>pytorch_tabular/tests/test_tabnet.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='pytorch_tabular/tests/test_tabnet.py:test_regression' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/tests/test_tabnet.py#L29'>test_regression</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>regression_data,<br>multi_target,<br>continuous_cols,<br>categorical_cols,<br>continuous_feature_transform,<br>normalize_continuous_features,<br>target_range,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='pytorch_tabular/tests/test_tabnet.py:test_classification' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/tests/test_tabnet.py#L87'>test_classification</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>classification_data,<br>continuous_cols,<br>categorical_cols,<br>continuous_feature_transform,<br>normalize_continuous_features,<br>,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='pytorch_tabular/pytorch_tabular/models/node/odst.py' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/odst.py'>pytorch_tabular/pytorch_tabular/models/node/odst.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='pytorch_tabular/pytorch_tabular/models/node/odst.py:check_numpy' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/odst.py#L15'>check_numpy</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br></ul>
        <li>Docs:<br>    """ Makes sure x is a numpy array """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/models/node/odst.py:ODST' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/odst.py#L24'>ODST</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/node/odst.py:ODST:__init__' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/odst.py#L25'>ODST:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>in_features,<br>num_trees,<br>depth = 6,<br>tree_output_dim = 1,<br>flatten_output = True,<br>choice_function = sparsemax,<br>bin_function = sparsemoid,<br>initialize_response_ = nn.init.normal_,<br>initialize_selection_logits_ = nn.init.uniform_,<br>threshold_init_beta = 1.0,<br>threshold_init_cutoff = 1.0,<br>,<br></ul>
        <li>Docs:<br>        """
<br>
        Oblivious Differentiable Sparsemax Trees. http://tinyurl.com/odst-readmore
<br>
        One can drop (sic!) this module anywhere instead of nn.Linear
<br>
        :param in_features: number of features in the input tensor
<br>
        :param num_trees: number of trees in this layer
<br>
        :param tree_dim: number of response channels in the response of individual tree
<br>
        :param depth: number of splits in every tree
<br>
        :param flatten_output: if False, returns [..., num_trees, tree_dim],
<br>
            by default returns [..., num_trees * tree_dim]
<br>
        :param choice_function: f(tensor, dim) -> R_simplex computes feature weights s.t. f(tensor, dim).sum(dim) == 1
<br>
        :param bin_function: f(tensor) -> R[0, 1], computes tree leaf weights
<br>

<br>
        :param initialize_response_: in-place initializer for tree output tensor
<br>
        :param initialize_selection_logits_: in-place initializer for logits that select features for the tree
<br>
        both thresholds and scales are initialized with data-aware init (or .load_state_dict)
<br>
        :param threshold_init_beta: initializes threshold to a q-th quantile of data points
<br>
            where q ~ Beta(:threshold_init_beta:, :threshold_init_beta:)
<br>
            If this param is set to 1, initial thresholds will have the same distribution as data points
<br>
            If greater than 1 (e.g. 10), thresholds will be closer to median data value
<br>
            If less than 1 (e.g. 0.1), thresholds will approach min/max data values.
<br>

<br>
        :param threshold_init_cutoff: threshold log-temperatures initializer, in (0, inf)
<br>
            By default(1.0), log-remperatures are initialized in such a way that all bin selectors
<br>
            end up in the linear region of sparse-sigmoid. The temperatures are then scaled by this parameter.
<br>
            Setting this value > 1.0 will result in some margin between data points and sparse-sigmoid cutoff value
<br>
            Setting this value < 1.0 will cause (1 - value) part of data points to end up in flat sparse-sigmoid region
<br>
            For instance, threshold_init_cutoff = 0.9 will set 10% points equal to 0.0 or 1.0
<br>
            Setting this value > 1.0 will result in a margin between data points and sparse-sigmoid cutoff value
<br>
            All points will be between (0.5 - 0.5 / threshold_init_cutoff) and (0.5 + 0.5 / threshold_init_cutoff)
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/node/odst.py:ODST:forward' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/odst.py#L112'>ODST:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>input,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/node/odst.py:ODST:initialize' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/odst.py#L148'>ODST:initialize</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>input,<br>eps = 1e-6,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/node/odst.py:ODST:__repr__' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/odst.py#L197'>ODST:__repr__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='pytorch_tabular/pytorch_tabular/models/autoint/autoint.py' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/autoint/autoint.py'>pytorch_tabular/pytorch_tabular/models/autoint/autoint.py</a>
</summary>
<ul>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/models/autoint/autoint.py:AutoIntBackbone' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/autoint/autoint.py#L21'>AutoIntBackbone</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/models/autoint/autoint.py:AutoIntModel' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/autoint/autoint.py#L141'>AutoIntModel</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/autoint/autoint.py:AutoIntBackbone:__init__' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/autoint/autoint.py#L22'>AutoIntBackbone:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>config:  DictConfig,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/autoint/autoint.py:AutoIntBackbone:_build_network' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/autoint/autoint.py#L29'>AutoIntBackbone:_build_network</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/autoint/autoint.py:AutoIntBackbone:forward' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/autoint/autoint.py#L91'>AutoIntBackbone:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x:  Dict,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/autoint/autoint.py:AutoIntModel:__init__' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/autoint/autoint.py#L142'>AutoIntModel:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>config:  DictConfig,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/autoint/autoint.py:AutoIntModel:_build_network' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/autoint/autoint.py#L29'>AutoIntModel:_build_network</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/autoint/autoint.py:AutoIntModel:forward' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/autoint/autoint.py#L91'>AutoIntModel:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x:  Dict,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='pytorch_tabular/tests/conftest.py' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/tests/conftest.py'>pytorch_tabular/tests/conftest.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='pytorch_tabular/tests/conftest.py:load_regression_data' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/tests/conftest.py#L11'>load_regression_data</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='pytorch_tabular/tests/conftest.py:load_classification_data' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/tests/conftest.py#L22'>load_classification_data</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='pytorch_tabular/tests/conftest.py:load_timeseries_data' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/tests/conftest.py#L36'>load_timeseries_data</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='pytorch_tabular/tests/conftest.py:regression_data' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/tests/conftest.py#L47'>regression_data</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='pytorch_tabular/tests/conftest.py:classification_data' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/tests/conftest.py#L52'>classification_data</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='pytorch_tabular/tests/conftest.py:timeseries_data' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/tests/conftest.py#L57'>timeseries_data</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='pytorch_tabular/pytorch_tabular/models/node/utils.py' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/utils.py'>pytorch_tabular/pytorch_tabular/models/node/utils.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='pytorch_tabular/pytorch_tabular/models/node/utils.py:to_one_hot' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/utils.py#L13'>to_one_hot</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>y,<br>depth = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='pytorch_tabular/pytorch_tabular/models/node/utils.py:sparsemax' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/utils.py#L100'>sparsemax</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>input,<br>dim = -1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='pytorch_tabular/pytorch_tabular/models/node/utils.py:sparsemoid' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/utils.py#L104'>sparsemoid</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>input,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='pytorch_tabular/pytorch_tabular/models/node/utils.py:entmax15' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/utils.py#L193'>entmax15</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>input,<br>dim = -1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/models/node/utils.py:Entmax15Function' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/utils.py#L110'>Entmax15Function</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/models/node/utils.py:Entmoid15' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/utils.py#L161'>Entmoid15</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/models/node/utils.py:Lambda' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/utils.py#L199'>Lambda</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/models/node/utils.py:ModuleWithInit' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/utils.py#L208'>ModuleWithInit</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/node/utils.py:Entmax15Function:forward' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/utils.py#L47'>Entmax15Function:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>ctx,<br>input,<br>dim = -1,<br></ul>
        <li>Docs:<br>        """sparsemax: normalizing sparse transform (a la softmax)
<br>

<br>
        Parameters:
<br>
            input (Tensor): any shape
<br>
            dim: dimension along which to apply sparsemax
<br>

<br>
        Returns:
<br>
            output (Tensor): same shape as input
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/node/utils.py:Entmax15Function:backward' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/utils.py#L66'>Entmax15Function:backward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>ctx,<br>grad_output,<br></ul>
        <li>Docs:<br>        """Sparsemax building block: compute the threshold
<br>

<br>
        Args:
<br>
            input: any dimension
<br>
            dim: dimension along which to apply the sparsemax
<br>

<br>
        Returns:
<br>
            the threshold value
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/node/utils.py:Entmax15Function:_threshold_and_support' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/utils.py#L78'>Entmax15Function:_threshold_and_support</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>input,<br>dim = -1,<br></ul>
        <li>Docs:<br>        """Sparsemax building block: compute the threshold
<br>

<br>
        Args:
<br>
            input: any dimension
<br>
            dim: dimension along which to apply the sparsemax
<br>

<br>
        Returns:
<br>
            the threshold value
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/node/utils.py:Entmoid15:forward' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/utils.py#L165'>Entmoid15:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>ctx,<br>input,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/node/utils.py:Entmoid15:_forward' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/utils.py#L172'>Entmoid15:_forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>input,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/node/utils.py:Entmoid15:backward' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/utils.py#L66'>Entmoid15:backward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>ctx,<br>grad_output,<br></ul>
        <li>Docs:<br>        """Sparsemax building block: compute the threshold
<br>

<br>
        Args:
<br>
            input: any dimension
<br>
            dim: dimension along which to apply the sparsemax
<br>

<br>
        Returns:
<br>
            the threshold value
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/node/utils.py:Entmoid15:_backward' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/utils.py#L185'>Entmoid15:_backward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>output,<br>grad_output,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/node/utils.py:Lambda:__init__' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/utils.py#L200'>Lambda:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>func,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/node/utils.py:Lambda:forward' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/utils.py#L204'>Lambda:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>*args,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/node/utils.py:ModuleWithInit:__init__' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/utils.py#L211'>ModuleWithInit:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/node/utils.py:ModuleWithInit:initialize' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/utils.py#L222'>ModuleWithInit:initialize</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>*args,<br>**kwargs,<br></ul>
        <li>Docs:<br>        """ initialize module tensors using first batch of data """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/node/utils.py:ModuleWithInit:__call__' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/utils.py#L226'>ModuleWithInit:__call__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>*args,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='pytorch_tabular/pytorch_tabular/models/tabnet/config.py' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/tabnet/config.py'>pytorch_tabular/pytorch_tabular/models/tabnet/config.py</a>
</summary>
<ul>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/models/tabnet/config.py:TabNetModelConfig' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/tabnet/config.py#L12'>TabNetModelConfig</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='pytorch_tabular/tests/test_datamodule.py' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/tests/test_datamodule.py'>pytorch_tabular/tests/test_datamodule.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='pytorch_tabular/tests/test_datamodule.py:test_dataloader' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/tests/test_datamodule.py#L43'>test_dataloader</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>regression_data,<br>validation_split,<br>multi_target,<br>continuous_cols,<br>categorical_cols,<br>continuous_feature_transform,<br>normalize_continuous_features,<br>target_transform,<br>embedding_dims,<br>,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='pytorch_tabular/tests/test_datamodule.py:test_date_encoding' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/tests/test_datamodule.py#L113'>test_date_encoding</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>timeseries_data,<br>freq,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='pytorch_tabular/examples/to_test_regression_custom_models.py' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/examples/to_test_regression_custom_models.py'>pytorch_tabular/examples/to_test_regression_custom_models.py</a>
</summary>
<ul>
    <details>
        <summary>
        class | <a name='pytorch_tabular/examples/to_test_regression_custom_models.py:MultiStageModelConfig' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/examples/to_test_regression_custom_models.py#L51'>MultiStageModelConfig</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='pytorch_tabular/pytorch_tabular/models/autoint/config.py' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/autoint/config.py'>pytorch_tabular/pytorch_tabular/models/autoint/config.py</a>
</summary>
<ul>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/models/autoint/config.py:AutoIntConfig' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/autoint/config.py#L12'>AutoIntConfig</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='pytorch_tabular/pytorch_tabular/models/category_embedding/category_embedding_model.py' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/category_embedding/category_embedding_model.py'>pytorch_tabular/pytorch_tabular/models/category_embedding/category_embedding_model.py</a>
</summary>
<ul>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/models/category_embedding/category_embedding_model.py:FeedForwardBackbone' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/category_embedding/category_embedding_model.py#L19'>FeedForwardBackbone</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/models/category_embedding/category_embedding_model.py:CategoryEmbeddingModel' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/category_embedding/category_embedding_model.py#L52'>CategoryEmbeddingModel</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/category_embedding/category_embedding_model.py:FeedForwardBackbone:__init__' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/category_embedding/category_embedding_model.py#L20'>FeedForwardBackbone:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>config:  DictConfig,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/category_embedding/category_embedding_model.py:FeedForwardBackbone:_build_network' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/category_embedding/category_embedding_model.py#L26'>FeedForwardBackbone:_build_network</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/category_embedding/category_embedding_model.py:FeedForwardBackbone:forward' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/category_embedding/category_embedding_model.py#L47'>FeedForwardBackbone:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/category_embedding/category_embedding_model.py:CategoryEmbeddingModel:__init__' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/category_embedding/category_embedding_model.py#L20'>CategoryEmbeddingModel:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>config:  DictConfig,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/category_embedding/category_embedding_model.py:CategoryEmbeddingModel:_build_network' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/category_embedding/category_embedding_model.py#L26'>CategoryEmbeddingModel:_build_network</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/category_embedding/category_embedding_model.py:CategoryEmbeddingModel:unpack_input' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/category_embedding/category_embedding_model.py#L74'>CategoryEmbeddingModel:unpack_input</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x:  Dict,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/models/category_embedding/category_embedding_model.py:CategoryEmbeddingModel:forward' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/category_embedding/category_embedding_model.py#L96'>CategoryEmbeddingModel:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x:  Dict,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='pytorch_tabular/pytorch_tabular/config/config.py' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/config/config.py'>pytorch_tabular/pytorch_tabular/config/config.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='pytorch_tabular/pytorch_tabular/config/config.py:_read_yaml' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/config/config.py#L13'>_read_yaml</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>filename,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/config/config.py:TrainerConfig' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/config/config.py#L162'>TrainerConfig</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/config/config.py:ExperimentConfig' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/config/config.py#L353'>ExperimentConfig</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/config/config.py:OptimizerConfig' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/config/config.py#L425'>OptimizerConfig</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/config/config.py:ExperimentRunManager' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/config/config.py#L479'>ExperimentRunManager</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/config/config.py:ModelConfig' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/config/config.py#L513'>ModelConfig</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/config/config.py:TrainerConfig:__post_init__' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/config/config.py#L145'>TrainerConfig:__post_init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/config/config.py:ExperimentConfig:__post_init__' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/config/config.py#L145'>ExperimentConfig:__post_init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/config/config.py:OptimizerConfig:read_from_yaml' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/config/config.py#L472'>OptimizerConfig:read_from_yaml</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>filename:  str  =  "config/optimizer_config.yml",<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/config/config.py:ExperimentRunManager:__init__' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/config/config.py#L480'>ExperimentRunManager:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>exp_version_manager:  str  =  ".tmp/exp_version_manager.yml",<br>,<br></ul>
        <li>Docs:<br>        """The manages the versions of the experiments based on the name. It is a simple dictionary(yaml) based lookup.
<br>
        Primary purpose is to avoid overwriting of saved models while runing the training without changing the experiment name.
<br>

<br>
        Args:
<br>
            exp_version_manager (str, optional): The path of the yml file which acts as version control.
<br>
            Defaults to ".tmp/exp_version_manager.yml".
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/config/config.py:ExperimentRunManager:update_versions' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/config/config.py#L501'>ExperimentRunManager:update_versions</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>name,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/config/config.py:ModelConfig:__post_init__' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/config/config.py#L145'>ModelConfig:__post_init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='pytorch_tabular/pytorch_tabular/feature_extractor.py' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/feature_extractor.py'>pytorch_tabular/pytorch_tabular/feature_extractor.py</a>
</summary>
<ul>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/feature_extractor.py:DeepFeatureExtractor' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/feature_extractor.py#L22'>DeepFeatureExtractor</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/feature_extractor.py:DeepFeatureExtractor:__init__' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/feature_extractor.py#L23'>DeepFeatureExtractor:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>tabular_model,<br>extract_keys = ["backbone_features"],<br>drop_original = True,<br></ul>
        <li>Docs:<br>        """Initializes the Transformer and extracts the neural features
<br>

<br>
        Args:
<br>
            tabular_model (TabularModel): The trained TabularModel object
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/feature_extractor.py:DeepFeatureExtractor:fit' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/feature_extractor.py#L40'>DeepFeatureExtractor:fit</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>X,<br>y = None,<br></ul>
        <li>Docs:<br>        """Just for compatibility. Does not do anything"""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/feature_extractor.py:DeepFeatureExtractor:transform' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/feature_extractor.py#L44'>DeepFeatureExtractor:transform</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>X:  pd.DataFrame,<br>y = None,<br></ul>
        <li>Docs:<br>        """Transforms the categorical columns specified to the trained neural features from the model
<br>

<br>
        Args:
<br>
            X (pd.DataFrame): DataFrame of features, shape (n_samples, n_features). Must contain columns to encode.
<br>
            y ([type], optional): Only for compatibility. Not used. Defaults to None.
<br>

<br>
        Raises:
<br>
            ValueError: [description]
<br>

<br>
        Returns:
<br>
            pd.DataFrame: The encoded dataframe
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/feature_extractor.py:DeepFeatureExtractor:fit_transform' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/feature_extractor.py#L92'>DeepFeatureExtractor:fit_transform</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>X:  pd.DataFrame,<br>y = None,<br></ul>
        <li>Docs:<br>        """Encode given columns of X based on the learned features.
<br>

<br>
        Args:
<br>
            X (pd.DataFrame): DataFrame of features, shape (n_samples, n_features). Must contain columns to encode.
<br>
            y ([type], optional): Only for compatibility. Not used. Defaults to None.
<br>

<br>
        Returns:
<br>
            pd.DataFrame: The encoded dataframe
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/feature_extractor.py:DeepFeatureExtractor:save_as_object_file' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/feature_extractor.py#L105'>DeepFeatureExtractor:save_as_object_file</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>path,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='pytorch_tabular/pytorch_tabular/feature_extractor.py:DeepFeatureExtractor:load_from_object_file' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/feature_extractor.py#L112'>DeepFeatureExtractor:load_from_object_file</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>path,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='pytorch_tabular/pytorch_tabular/models/node/config.py' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/config.py'>pytorch_tabular/pytorch_tabular/models/node/config.py</a>
</summary>
<ul>
    <details>
        <summary>
        class | <a name='pytorch_tabular/pytorch_tabular/models/node/config.py:NodeConfig' href='https://github.com/manujosephv/pytorch_tabular/pytorch_tabular/pytorch_tabular/models/node/config.py#L8'>NodeConfig</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>
