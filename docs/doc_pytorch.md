# All files

<details>
<summary>
<a name='train_vae.py' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/train_vae.py'>train_vae.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='train_vae.py:save_model' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/train_vae.py#L196'>save_model</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>path,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='dalle_pytorch/tokenizer.py' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/tokenizer.py'>dalle_pytorch/tokenizer.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='dalle_pytorch/tokenizer.py:default_bpe' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/tokenizer.py#L21'>default_bpe</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='dalle_pytorch/tokenizer.py:bytes_to_unicode' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/tokenizer.py#L25'>bytes_to_unicode</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='dalle_pytorch/tokenizer.py:get_pairs' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/tokenizer.py#L37'>get_pairs</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>word,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='dalle_pytorch/tokenizer.py:basic_clean' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/tokenizer.py#L45'>basic_clean</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>text,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='dalle_pytorch/tokenizer.py:whitespace_clean' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/tokenizer.py#L50'>whitespace_clean</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>text,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='dalle_pytorch/tokenizer.py:SimpleTokenizer' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/tokenizer.py#L55'>SimpleTokenizer</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='dalle_pytorch/tokenizer.py:HugTokenizer' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/tokenizer.py#L158'>HugTokenizer</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='dalle_pytorch/tokenizer.py:ChineseTokenizer' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/tokenizer.py#L196'>ChineseTokenizer</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='dalle_pytorch/tokenizer.py:YttmTokenizer' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/tokenizer.py#L232'>YttmTokenizer</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/tokenizer.py:SimpleTokenizer:__init__' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/tokenizer.py#L56'>SimpleTokenizer:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>bpe_path  =  default_bpe(,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/tokenizer.py:SimpleTokenizer:bpe' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/tokenizer.py#L78'>SimpleTokenizer:bpe</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>token,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/tokenizer.py:SimpleTokenizer:encode' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/tokenizer.py#L119'>SimpleTokenizer:encode</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>text,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/tokenizer.py:SimpleTokenizer:decode' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/tokenizer.py#L127'>SimpleTokenizer:decode</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>tokens,<br>remove_start_end  =  True,<br>pad_tokens  =  {},<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/tokenizer.py:SimpleTokenizer:tokenize' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/tokenizer.py#L137'>SimpleTokenizer:tokenize</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>texts,<br>context_length  =  256,<br>truncate_text  =  False,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/tokenizer.py:HugTokenizer:__init__' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/tokenizer.py#L159'>HugTokenizer:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>bpe_path  =  None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/tokenizer.py:HugTokenizer:decode' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/tokenizer.py#L167'>HugTokenizer:decode</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>tokens,<br>pad_tokens  =  {},<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/tokenizer.py:HugTokenizer:encode' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/tokenizer.py#L119'>HugTokenizer:encode</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>text,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/tokenizer.py:HugTokenizer:tokenize' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/tokenizer.py#L137'>HugTokenizer:tokenize</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>texts,<br>context_length  =  256,<br>truncate_text  =  False,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/tokenizer.py:ChineseTokenizer:__init__' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/tokenizer.py#L197'>ChineseTokenizer:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/tokenizer.py:ChineseTokenizer:decode' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/tokenizer.py#L167'>ChineseTokenizer:decode</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>tokens,<br>pad_tokens  =  {},<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/tokenizer.py:ChineseTokenizer:encode' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/tokenizer.py#L119'>ChineseTokenizer:encode</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>text,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/tokenizer.py:ChineseTokenizer:tokenize' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/tokenizer.py#L137'>ChineseTokenizer:tokenize</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>texts,<br>context_length  =  256,<br>truncate_text  =  False,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/tokenizer.py:YttmTokenizer:__init__' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/tokenizer.py#L159'>YttmTokenizer:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>bpe_path  =  None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/tokenizer.py:YttmTokenizer:decode' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/tokenizer.py#L167'>YttmTokenizer:decode</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>tokens,<br>pad_tokens  =  {},<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/tokenizer.py:YttmTokenizer:encode' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/tokenizer.py#L247'>YttmTokenizer:encode</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>texts,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/tokenizer.py:YttmTokenizer:tokenize' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/tokenizer.py#L137'>YttmTokenizer:tokenize</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>texts,<br>context_length  =  256,<br>truncate_text  =  False,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='train_dalle.py' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/train_dalle.py'>train_dalle.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='train_dalle.py:exists' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/train_dalle.py#L141'>exists</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>val,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='train_dalle.py:get_trainable_params' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/train_dalle.py#L144'>get_trainable_params</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>model,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='train_dalle.py:cp_path_to_dir' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/train_dalle.py#L147'>cp_path_to_dir</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>cp_path,<br>tag,<br></ul>
        <li>Docs:<br>    """Convert a checkpoint path to a directory with `tag` inserted.
<br>
    If `cp_path` is already a directory, return it unchanged.
<br>
    """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='train_dalle.py:group_weight' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/train_dalle.py#L317'>group_weight</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>model,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='train_dalle.py:imagetransform' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/train_dalle.py#L344'>imagetransform</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>b,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='train_dalle.py:tokenize' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/train_dalle.py#L347'>tokenize</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>s,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='train_dalle.py:save_model' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/train_dalle.py#L510'>save_model</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>path,<br>epoch = 0,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='dalle_pytorch/reversible.py' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/reversible.py'>dalle_pytorch/reversible.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='dalle_pytorch/reversible.py:route_args' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/reversible.py#L8'>route_args</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>router,<br>args,<br>depth,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='dalle_pytorch/reversible.py:Deterministic' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/reversible.py#L20'>Deterministic</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='dalle_pytorch/reversible.py:ReversibleBlock' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/reversible.py#L54'>ReversibleBlock</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='dalle_pytorch/reversible.py:_ReversibleFunction' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/reversible.py#L108'>_ReversibleFunction</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='dalle_pytorch/reversible.py:SequentialSequence' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/reversible.py#L126'>SequentialSequence</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='dalle_pytorch/reversible.py:ReversibleSequence' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/reversible.py#L143'>ReversibleSequence</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/reversible.py:Deterministic:__init__' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/reversible.py#L21'>Deterministic:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>net,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/reversible.py:Deterministic:record_rng' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/reversible.py#L29'>Deterministic:record_rng</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>*args,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/reversible.py:Deterministic:forward' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/reversible.py#L35'>Deterministic:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>*args,<br>record_rng  =  False,<br>set_rng  =  False,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/reversible.py:ReversibleBlock:__init__' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/reversible.py#L55'>ReversibleBlock:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>f,<br>g,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/reversible.py:ReversibleBlock:forward' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/reversible.py#L60'>ReversibleBlock:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br>f_args  =  {},<br>g_args  =  {},<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/reversible.py:ReversibleBlock:backward_pass' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/reversible.py#L70'>ReversibleBlock:backward_pass</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>y,<br>dy,<br>f_args  =  {},<br>g_args  =  {},<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/reversible.py:_ReversibleFunction:forward' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/reversible.py#L110'>_ReversibleFunction:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>ctx,<br>x,<br>blocks,<br>args,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/reversible.py:_ReversibleFunction:backward' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/reversible.py#L119'>_ReversibleFunction:backward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>ctx,<br>dy,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/reversible.py:SequentialSequence:__init__' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/reversible.py#L127'>SequentialSequence:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>layers,<br>args_route  =  {},<br>layer_dropout  =  0.,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/reversible.py:SequentialSequence:forward' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/reversible.py#L134'>SequentialSequence:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/reversible.py:ReversibleSequence:__init__' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/reversible.py#L144'>ReversibleSequence:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>blocks,<br>args_route  =  {},<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/reversible.py:ReversibleSequence:forward' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/reversible.py#L134'>ReversibleSequence:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='dalle_pytorch/distributed_backends/deepspeed_backend.py' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/deepspeed_backend.py'>dalle_pytorch/distributed_backends/deepspeed_backend.py</a>
</summary>
<ul>
    <details>
        <summary>
        class | <a name='dalle_pytorch/distributed_backends/deepspeed_backend.py:DeepSpeedBackend' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/deepspeed_backend.py#L9'>DeepSpeedBackend</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/deepspeed_backend.py:DeepSpeedBackend:wrap_arg_parser' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/deepspeed_backend.py#L15'>DeepSpeedBackend:wrap_arg_parser</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>parser,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/deepspeed_backend.py:DeepSpeedBackend:_initialize' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/deepspeed_backend.py#L36'>DeepSpeedBackend:_initialize</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/deepspeed_backend.py:DeepSpeedBackend:_require_torch_distributed_init' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/deepspeed_backend.py#L42'>DeepSpeedBackend:_require_torch_distributed_init</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br>        """Raise an error when `torch.distributed` has not been
<br>
        initialized yet.
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/deepspeed_backend.py:DeepSpeedBackend:_get_world_size' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/deepspeed_backend.py#L50'>DeepSpeedBackend:_get_world_size</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/deepspeed_backend.py:DeepSpeedBackend:_get_rank' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/deepspeed_backend.py#L54'>DeepSpeedBackend:_get_rank</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/deepspeed_backend.py:DeepSpeedBackend:_get_local_rank' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/deepspeed_backend.py#L58'>DeepSpeedBackend:_get_local_rank</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/deepspeed_backend.py:DeepSpeedBackend:_local_barrier' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/deepspeed_backend.py#L62'>DeepSpeedBackend:_local_barrier</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/deepspeed_backend.py:DeepSpeedBackend:_check_args' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/deepspeed_backend.py#L66'>DeepSpeedBackend:_check_args</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>args,<br>optimizer,<br>lr_scheduler,<br>kwargs,<br></ul>
        <li>Docs:<br>        """Return an appropriate optimizer and learning rate scheduler
<br>
        after checking the values passed to `distribute`.
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/deepspeed_backend.py:DeepSpeedBackend:_check_argvs' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/deepspeed_backend.py#L75'>DeepSpeedBackend:_check_argvs</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>args,<br>optimizer,<br>lr_scheduler,<br>kwargs,<br></ul>
        <li>Docs:<br>        """Apply several sanity checks to the given command
<br>
        line arguments.
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/deepspeed_backend.py:DeepSpeedBackend:_check_config' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/deepspeed_backend.py#L105'>DeepSpeedBackend:_check_config</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>args,<br>optimizer,<br>lr_scheduler,<br>kwargs,<br></ul>
        <li>Docs:<br>        """Return an appropriate optimizer and learning rate scheduler
<br>
        for the DeepSpeed configuration.
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/deepspeed_backend.py:DeepSpeedBackend:_distribute' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/deepspeed_backend.py#L135'>DeepSpeedBackend:_distribute</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>args = None,<br>model = None,<br>optimizer = None,<br>model_parameters = None,<br>training_data = None,<br>lr_scheduler = None,<br>**kwargs,<br>,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/deepspeed_backend.py:DeepSpeedBackend:_average_all' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/deepspeed_backend.py#L165'>DeepSpeedBackend:_average_all</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>tensor,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='dalle_pytorch/distributed_backends/distributed_backend.py' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/distributed_backend.py'>dalle_pytorch/distributed_backends/distributed_backend.py</a>
</summary>
<ul>
    <details>
        <summary>
        class | <a name='dalle_pytorch/distributed_backends/distributed_backend.py:DistributedBackend' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/distributed_backend.py#L12'>DistributedBackend</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/distributed_backend.py:DistributedBackend:__init__' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/distributed_backend.py#L42'>DistributedBackend:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/distributed_backend.py:DistributedBackend:has_backend' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/distributed_backend.py#L48'>DistributedBackend:has_backend</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br>        """Return whether the backend module is now imported."""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/distributed_backend.py:DistributedBackend:check_batch_size' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/distributed_backend.py#L56'>DistributedBackend:check_batch_size</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>batch_size,<br></ul>
        <li>Docs:<br>        """Check whether the batch size makes sense for distribution."""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/distributed_backend.py:DistributedBackend:wrap_arg_parser' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/distributed_backend.py#L62'>DistributedBackend:wrap_arg_parser</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>parser,<br></ul>
        <li>Docs:<br>        """Add arguments to support optional distributed backend usage."""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/distributed_backend.py:DistributedBackend:initialize' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/distributed_backend.py#L66'>DistributedBackend:initialize</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br>        """Initialize the distributed backend."""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/distributed_backend.py:DistributedBackend:_initialize' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/distributed_backend.py#L71'>DistributedBackend:_initialize</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br>        """Initialize the distributed backend."""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/distributed_backend.py:DistributedBackend:require_init' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/distributed_backend.py#L75'>DistributedBackend:require_init</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br>        """Raise an error when the backend has not been initialized yet."""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/distributed_backend.py:DistributedBackend:get_world_size' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/distributed_backend.py#L82'>DistributedBackend:get_world_size</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br>        """Return the amount of distributed processes."""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/distributed_backend.py:DistributedBackend:_get_world_size' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/distributed_backend.py#L87'>DistributedBackend:_get_world_size</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br>        """Return the amount of distributed processes."""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/distributed_backend.py:DistributedBackend:get_rank' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/distributed_backend.py#L91'>DistributedBackend:get_rank</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br>        """Return the global rank of the calling worker process."""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/distributed_backend.py:DistributedBackend:_get_rank' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/distributed_backend.py#L96'>DistributedBackend:_get_rank</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br>        """Return the global rank of the calling worker process."""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/distributed_backend.py:DistributedBackend:get_local_rank' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/distributed_backend.py#L100'>DistributedBackend:get_local_rank</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br>        """Return the local rank of the calling worker process.
<br>
        The local rank is the rank based on a single node's processes.
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/distributed_backend.py:DistributedBackend:_get_local_rank' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/distributed_backend.py#L107'>DistributedBackend:_get_local_rank</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br>        """Return the local rank of the calling worker process.
<br>
        The local rank is the rank based on a single node's processes.
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/distributed_backend.py:DistributedBackend:is_root_worker' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/distributed_backend.py#L113'>DistributedBackend:is_root_worker</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br>        """Return whether the calling worker has the root rank."""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/distributed_backend.py:DistributedBackend:is_local_root_worker' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/distributed_backend.py#L117'>DistributedBackend:is_local_root_worker</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br>        """Return whether the calling worker has the root rank on this node."""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/distributed_backend.py:DistributedBackend:local_barrier' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/distributed_backend.py#L121'>DistributedBackend:local_barrier</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br>        """Wait until all processes on this node have called this function."""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/distributed_backend.py:DistributedBackend:_local_barrier' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/distributed_backend.py#L126'>DistributedBackend:_local_barrier</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br>        """Wait until all processes on this node have called this function."""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/distributed_backend.py:DistributedBackend:distribute' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/distributed_backend.py#L130'>DistributedBackend:distribute</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>args = None,<br>model = None,<br>optimizer = None,<br>model_parameters = None,<br>training_data = None,<br>lr_scheduler = None,<br>**kwargs,<br>,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/distributed_backend.py:DistributedBackend:_distribute' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/distributed_backend.py#L155'>DistributedBackend:_distribute</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>args = None,<br>model = None,<br>optimizer = None,<br>model_parameters = None,<br>training_data = None,<br>lr_scheduler = None,<br>**kwargs,<br>,<br></ul>
        <li>Docs:<br>        """Return a distributed model engine, optimizer, dataloader, and
<br>
        learning rate scheduler. These are obtained by wrapping the
<br>
        given values with the backend.
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/distributed_backend.py:DistributedBackend:average_all' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/distributed_backend.py#L171'>DistributedBackend:average_all</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>tensor,<br></ul>
        <li>Docs:<br>        """Return the average of `tensor` over all workers."""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/distributed_backend.py:DistributedBackend:_average_all' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/distributed_backend.py#L176'>DistributedBackend:_average_all</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>tensor,<br></ul>
        <li>Docs:<br>        """Return the average of `tensor` over all workers."""
<br>
</li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='dalle_pytorch/dalle_pytorch.py' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/dalle_pytorch.py'>dalle_pytorch/dalle_pytorch.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='dalle_pytorch/dalle_pytorch.py:exists' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/dalle_pytorch.py#L16'>exists</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>val,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='dalle_pytorch/dalle_pytorch.py:default' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/dalle_pytorch.py#L19'>default</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>val,<br>d,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='dalle_pytorch/dalle_pytorch.py:is_empty' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/dalle_pytorch.py#L28'>is_empty</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>t,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='dalle_pytorch/dalle_pytorch.py:masked_mean' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/dalle_pytorch.py#L31'>masked_mean</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>t,<br>mask,<br>dim  =  1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='dalle_pytorch/dalle_pytorch.py:set_requires_grad' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/dalle_pytorch.py#L35'>set_requires_grad</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>model,<br>value,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='dalle_pytorch/dalle_pytorch.py:eval_decorator' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/dalle_pytorch.py#L39'>eval_decorator</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>fn,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='dalle_pytorch/dalle_pytorch.py:top_k' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/dalle_pytorch.py#L50'>top_k</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>logits,<br>thres  =  0.5,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='dalle_pytorch/dalle_pytorch.py:always' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/dalle_pytorch.py#L22'>always</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='dalle_pytorch/dalle_pytorch.py:ResBlock' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/dalle_pytorch.py#L60'>ResBlock</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='dalle_pytorch/dalle_pytorch.py:DiscreteVAE' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/dalle_pytorch.py#L74'>DiscreteVAE</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='dalle_pytorch/dalle_pytorch.py:CLIP' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/dalle_pytorch.py#L229'>CLIP</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='dalle_pytorch/dalle_pytorch.py:DALLE' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/dalle_pytorch.py#L309'>DALLE</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/dalle_pytorch.py:always:__init__' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/dalle_pytorch.py#L23'>always:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>val,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/dalle_pytorch.py:always:__call__' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/dalle_pytorch.py#L25'>always:__call__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br>*args,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/dalle_pytorch.py:ResBlock:__init__' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/dalle_pytorch.py#L61'>ResBlock:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>chan,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/dalle_pytorch.py:ResBlock:forward' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/dalle_pytorch.py#L71'>ResBlock:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/dalle_pytorch.py:DiscreteVAE:__init__' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/dalle_pytorch.py#L75'>DiscreteVAE:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>image_size  =  256,<br>num_tokens  =  512,<br>codebook_dim  =  512,<br>num_layers  =  3,<br>num_resnet_blocks  =  0,<br>hidden_dim  =  64,<br>channels  =  3,<br>smooth_l1_loss  =  False,<br>temperature  =  0.9,<br>straight_through  =  False,<br>kl_div_loss_weight  =  0.,<br>(0.5,<br>) * 3,<br>(0.5,<br>) * 3),<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/dalle_pytorch.py:DiscreteVAE:_register_external_parameters' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/dalle_pytorch.py#L142'>DiscreteVAE:_register_external_parameters</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br>        """Register external parameters for DeepSpeed partitioning."""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/dalle_pytorch.py:DiscreteVAE:norm' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/dalle_pytorch.py#L154'>DiscreteVAE:norm</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>images,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/dalle_pytorch.py:DiscreteVAE:get_codebook_indices' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/dalle_pytorch.py#L166'>DiscreteVAE:get_codebook_indices</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>images,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/dalle_pytorch.py:DiscreteVAE:decode' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/dalle_pytorch.py#L171'>DiscreteVAE:decode</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>img_seq,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/dalle_pytorch.py:DiscreteVAE:forward' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/dalle_pytorch.py#L183'>DiscreteVAE:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>img,<br>return_loss  =  False,<br>return_recons  =  False,<br>return_logits  =  False,<br>temp  =  None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/dalle_pytorch.py:CLIP:__init__' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/dalle_pytorch.py#L75'>CLIP:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>*,<br>dim_text  =  512,<br>dim_image  =  512,<br>dim_latent  =  512,<br>num_text_tokens  =  10000,<br>text_enc_depth  =  6,<br>text_seq_len  =  256,<br>text_heads  =  8,<br>num_visual_tokens  =  512,<br>visual_enc_depth  =  6,<br>visual_heads  =  8,<br>visual_image_size  =  256,<br>visual_patch_size  =  32,<br>channels  =  3,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/dalle_pytorch.py:CLIP:forward' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/dalle_pytorch.py#L183'>CLIP:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>text,<br>image,<br>text_mask  =  None,<br>return_loss  =  False,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/dalle_pytorch.py:DALLE:__init__' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/dalle_pytorch.py#L75'>DALLE:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>*,<br>dim,<br>vae,<br>num_text_tokens  =  10000,<br>text_seq_len  =  256,<br>depth,<br>heads  =  8,<br>dim_head  =  64,<br>reversible  =  False,<br>attn_dropout  =  0.,<br>ff_dropout  =  0,<br>sparse_attn  =  False,<br>attn_types  =  None,<br>loss_img_weight  =  7,<br>stable  =  False,<br>shift_tokens  =  True,<br>rotary_emb  =  True,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/dalle_pytorch.py:DALLE:generate_texts' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/dalle_pytorch.py#L405'>DALLE:generate_texts</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>tokenizer,<br>text  =  None,<br>*,<br>filter_thres  =  0.5,<br>temperature  =  1.,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/dalle_pytorch.py:DALLE:generate_images' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/dalle_pytorch.py#L453'>DALLE:generate_images</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>text,<br>*,<br>clip  =  None,<br>mask  =  None,<br>filter_thres  =  0.5,<br>temperature  =  1.,<br>img  =  None,<br>num_init_img_tokens  =  None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/dalle_pytorch.py:DALLE:forward' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/dalle_pytorch.py#L183'>DALLE:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>text,<br>image  =  None,<br>mask  =  None,<br>return_loss  =  False,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='generate.py' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/generate.py'>generate.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='generate.py:exists' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/generate.py#L64'>exists</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>val,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='dalle_pytorch/distributed_backends/horovod_backend.py' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/horovod_backend.py'>dalle_pytorch/distributed_backends/horovod_backend.py</a>
</summary>
<ul>
    <details>
        <summary>
        class | <a name='dalle_pytorch/distributed_backends/horovod_backend.py:HorovodBackend' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/horovod_backend.py#L6'>HorovodBackend</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/horovod_backend.py:HorovodBackend:wrap_arg_parser' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/horovod_backend.py#L12'>HorovodBackend:wrap_arg_parser</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>parser,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/horovod_backend.py:HorovodBackend:check_batch_size' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/horovod_backend.py#L15'>HorovodBackend:check_batch_size</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>batch_size,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/horovod_backend.py:HorovodBackend:_initialize' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/horovod_backend.py#L20'>HorovodBackend:_initialize</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/horovod_backend.py:HorovodBackend:_get_world_size' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/horovod_backend.py#L25'>HorovodBackend:_get_world_size</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/horovod_backend.py:HorovodBackend:_get_rank' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/horovod_backend.py#L28'>HorovodBackend:_get_rank</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/horovod_backend.py:HorovodBackend:_get_local_rank' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/horovod_backend.py#L31'>HorovodBackend:_get_local_rank</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/horovod_backend.py:HorovodBackend:_local_barrier' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/horovod_backend.py#L34'>HorovodBackend:_local_barrier</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/horovod_backend.py:HorovodBackend:_distribute' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/horovod_backend.py#L38'>HorovodBackend:_distribute</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>_args = None,<br>model = None,<br>optimizer = None,<br>_model_parameters = None,<br>training_data = None,<br>lr_scheduler = None,<br>**_kwargs,<br>,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/horovod_backend.py:HorovodBackend:_average_all' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/horovod_backend.py#L55'>HorovodBackend:_average_all</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>tensor,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='dalle_pytorch/transformer.py' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/transformer.py'>dalle_pytorch/transformer.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='dalle_pytorch/transformer.py:exists' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/transformer.py#L17'>exists</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>val,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='dalle_pytorch/transformer.py:default' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/transformer.py#L20'>default</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>val,<br>d,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='dalle_pytorch/transformer.py:cast_tuple' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/transformer.py#L23'>cast_tuple</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>val,<br>depth  =  1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='dalle_pytorch/transformer.py:DivideMax' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/transformer.py#L30'>DivideMax</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='dalle_pytorch/transformer.py:LayerScale' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/transformer.py#L40'>LayerScale</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='dalle_pytorch/transformer.py:PreNorm' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/transformer.py#L58'>PreNorm</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='dalle_pytorch/transformer.py:GEGLU' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/transformer.py#L69'>GEGLU</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='dalle_pytorch/transformer.py:FeedForward' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/transformer.py#L74'>FeedForward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='dalle_pytorch/transformer.py:PreShiftToken' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/transformer.py#L89'>PreShiftToken</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='dalle_pytorch/transformer.py:Transformer' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/transformer.py#L130'>Transformer</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/transformer.py:DivideMax:__init__' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/transformer.py#L31'>DivideMax:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>dim,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/transformer.py:DivideMax:forward' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/transformer.py#L35'>DivideMax:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/transformer.py:LayerScale:__init__' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/transformer.py#L41'>LayerScale:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>dim,<br>depth,<br>fn,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/transformer.py:LayerScale:forward' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/transformer.py#L53'>LayerScale:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/transformer.py:PreNorm:__init__' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/transformer.py#L59'>PreNorm:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>dim,<br>fn,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/transformer.py:PreNorm:forward' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/transformer.py#L53'>PreNorm:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/transformer.py:GEGLU:forward' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/transformer.py#L35'>GEGLU:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/transformer.py:FeedForward:__init__' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/transformer.py#L75'>FeedForward:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>dim,<br>dropout  =  0.,<br>mult  =  4.,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/transformer.py:FeedForward:forward' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/transformer.py#L35'>FeedForward:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/transformer.py:PreShiftToken:__init__' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/transformer.py#L90'>PreShiftToken:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>fn,<br>image_size,<br>seq_len,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/transformer.py:PreShiftToken:forward' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/transformer.py#L53'>PreShiftToken:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/transformer.py:Transformer:__init__' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/transformer.py#L131'>Transformer:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>*,<br>dim,<br>depth,<br>seq_len,<br>reversible  =  False,<br>causal  =  True,<br>heads  =  8,<br>dim_head  =  64,<br>ff_mult  =  4,<br>attn_dropout  =  0.,<br>ff_dropout  =  0.,<br>attn_types  =  None,<br>image_fmap_size  =  None,<br>sparse_attn  =  False,<br>stable  =  False,<br>shift_tokens  =  True,<br>rotary_emb  =  True,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/transformer.py:Transformer:forward' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/transformer.py#L53'>Transformer:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='dalle_pytorch/loader.py' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/loader.py'>dalle_pytorch/loader.py</a>
</summary>
<ul>
    <details>
        <summary>
        class | <a name='dalle_pytorch/loader.py:TextImageDataset' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/loader.py#L10'>TextImageDataset</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/loader.py:TextImageDataset:__init__' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/loader.py#L11'>TextImageDataset:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>folder,<br>text_len = 256,<br>image_size = 128,<br>truncate_captions = False,<br>resize_ratio = 0.75,<br>tokenizer = None,<br>shuffle = False,<br></ul>
        <li>Docs:<br>        """
<br>
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
<br>
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/loader.py:TextImageDataset:__len__' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/loader.py#L55'>TextImageDataset:__len__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/loader.py:TextImageDataset:random_sample' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/loader.py#L58'>TextImageDataset:random_sample</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/loader.py:TextImageDataset:sequential_sample' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/loader.py#L61'>TextImageDataset:sequential_sample</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>ind,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/loader.py:TextImageDataset:skip_sample' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/loader.py#L66'>TextImageDataset:skip_sample</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>ind,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/loader.py:TextImageDataset:__getitem__' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/loader.py#L71'>TextImageDataset:__getitem__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>ind,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='dalle_pytorch/vae.py' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/vae.py'>dalle_pytorch/vae.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='dalle_pytorch/vae.py:exists' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/vae.py#L37'>exists</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>val,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='dalle_pytorch/vae.py:default' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/vae.py#L40'>default</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>val,<br>d,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='dalle_pytorch/vae.py:load_model' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/vae.py#L43'>load_model</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>path,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='dalle_pytorch/vae.py:map_pixels' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/vae.py#L47'>map_pixels</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br>eps  =  0.1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='dalle_pytorch/vae.py:unmap_pixels' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/vae.py#L50'>unmap_pixels</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>x,<br>eps  =  0.1,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='dalle_pytorch/vae.py:download' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/vae.py#L53'>download</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>url,<br>filename  =  None,<br>root  =  CACHE_PATH,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='dalle_pytorch/vae.py:make_contiguous' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/vae.py#L96'>make_contiguous</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>module,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='dalle_pytorch/vae.py:get_obj_from_str' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/vae.py#L138'>get_obj_from_str</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>string,<br>reload = False,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='dalle_pytorch/vae.py:instantiate_from_config' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/vae.py#L145'>instantiate_from_config</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>config,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='dalle_pytorch/vae.py:OpenAIDiscreteVAE' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/vae.py#L103'>OpenAIDiscreteVAE</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='dalle_pytorch/vae.py:VQGanVAE' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/vae.py#L150'>VQGanVAE</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/vae.py:OpenAIDiscreteVAE:__init__' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/vae.py#L104'>OpenAIDiscreteVAE:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/vae.py:OpenAIDiscreteVAE:get_codebook_indices' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/vae.py#L116'>OpenAIDiscreteVAE:get_codebook_indices</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>img,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/vae.py:OpenAIDiscreteVAE:decode' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/vae.py#L122'>OpenAIDiscreteVAE:decode</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>img_seq,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/vae.py:OpenAIDiscreteVAE:forward' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/vae.py#L132'>OpenAIDiscreteVAE:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>img,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/vae.py:VQGanVAE:__init__' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/vae.py#L151'>VQGanVAE:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>vqgan_model_path = None,<br>vqgan_config_path = None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/vae.py:VQGanVAE:_register_external_parameters' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/vae.py#L185'>VQGanVAE:_register_external_parameters</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br>        """Register external parameters for DeepSpeed partitioning."""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/vae.py:VQGanVAE:get_codebook_indices' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/vae.py#L116'>VQGanVAE:get_codebook_indices</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>img,<br></ul>
        <li>Docs:<br>        """Register external parameters for DeepSpeed partitioning."""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/vae.py:VQGanVAE:decode' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/vae.py#L122'>VQGanVAE:decode</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>img_seq,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/vae.py:VQGanVAE:forward' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/vae.py#L132'>VQGanVAE:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>img,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='dalle_pytorch/attention.py' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/attention.py'>dalle_pytorch/attention.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='dalle_pytorch/attention.py:exists' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/attention.py#L13'>exists</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>val,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='dalle_pytorch/attention.py:uniq' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/attention.py#L16'>uniq</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>arr,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='dalle_pytorch/attention.py:default' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/attention.py#L19'>default</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>val,<br>d,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='dalle_pytorch/attention.py:max_neg_value' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/attention.py#L24'>max_neg_value</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>t,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='dalle_pytorch/attention.py:stable_softmax' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/attention.py#L27'>stable_softmax</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>t,<br>dim  =  -1,<br>alpha  =  32 ** 2,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='dalle_pytorch/attention.py:apply_pos_emb' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/attention.py#L32'>apply_pos_emb</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>pos_emb,<br>qkv,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='dalle_pytorch/attention.py:Attention' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/attention.py#L39'>Attention</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='dalle_pytorch/attention.py:SparseConvCausalAttention' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/attention.py#L90'>SparseConvCausalAttention</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='dalle_pytorch/attention.py:SparseAxialCausalAttention' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/attention.py#L211'>SparseAxialCausalAttention</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        class | <a name='dalle_pytorch/attention.py:SparseAttention' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/attention.py#L325'>SparseAttention</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/attention.py:Attention:__init__' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/attention.py#L40'>Attention:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>dim,<br>seq_len,<br>causal  =  True,<br>heads  =  8,<br>dim_head  =  64,<br>dropout  =  0.,<br>stable  =  False,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/attention.py:Attention:forward' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/attention.py#L56'>Attention:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br>mask  =  None,<br>rotary_pos_emb  =  None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/attention.py:SparseConvCausalAttention:__init__' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/attention.py#L91'>SparseConvCausalAttention:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>dim,<br>seq_len,<br>image_size  =  32,<br>kernel_size  =  5,<br>dilation  =  1,<br>heads  =  8,<br>dim_head  =  64,<br>dropout  =  0.,<br>stable  =  False,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/attention.py:SparseConvCausalAttention:forward' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/attention.py#L56'>SparseConvCausalAttention:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br>mask  =  None,<br>rotary_pos_emb  =  None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/attention.py:SparseAxialCausalAttention:__init__' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/attention.py#L212'>SparseAxialCausalAttention:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>dim,<br>seq_len,<br>image_size  =  32,<br>axis  =  0,<br>heads  =  8,<br>dim_head  =  64,<br>dropout  =  0.,<br>stable  =  False,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/attention.py:SparseAxialCausalAttention:forward' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/attention.py#L56'>SparseAxialCausalAttention:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br>mask  =  None,<br>rotary_pos_emb  =  None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/attention.py:SparseAttention:__init__' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/attention.py#L326'>SparseAttention:__init__</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>*args,<br>block_size  =  16,<br>text_seq_len  =  256,<br>num_random_blocks  =  None,<br>**kwargs,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/attention.py:SparseAttention:forward' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/attention.py#L56'>SparseAttention:forward</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>x,<br>mask  =  None,<br>rotary_pos_emb  =  None,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='dalle_pytorch/distributed_utils.py' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_utils.py'>dalle_pytorch/distributed_utils.py</a>
</summary>
<ul>
    <details>
        <summary>
        function | <a name='dalle_pytorch/distributed_utils.py:wrap_arg_parser' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_utils.py#L34'>wrap_arg_parser</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>parser,<br></ul>
        <li>Docs:<br>    """Add arguments to support optional distributed backend usage."""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='dalle_pytorch/distributed_utils.py:set_backend_from_args' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_utils.py#L48'>set_backend_from_args</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>args,<br></ul>
        <li>Docs:<br>    """Set and return the backend based on the given `args`."""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='dalle_pytorch/distributed_utils.py:require_set_backend' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_utils.py#L79'>require_set_backend</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br>    """Raise an `AssertionError` when the backend has not been set."""
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        function | <a name='dalle_pytorch/distributed_utils.py:using_backend' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_utils.py#L87'>using_backend</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>test_backend,<br></ul>
        <li>Docs:<br>    """Return whether the backend is set to `test_backend`.
<br>

<br>
    `test_backend` may be a string of the name of the backend or
<br>
    its class.
<br>
    """
<br>
</li>
        </ul>
    </details></ul>
</details>

<details>
<summary>
<a name='dalle_pytorch/distributed_backends/dummy_backend.py' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/dummy_backend.py'>dalle_pytorch/distributed_backends/dummy_backend.py</a>
</summary>
<ul>
    <details>
        <summary>
        class | <a name='dalle_pytorch/distributed_backends/dummy_backend.py:DummyBackend' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/dummy_backend.py#L4'>DummyBackend</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/dummy_backend.py:DummyBackend:has_backend' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/dummy_backend.py#L15'>DummyBackend:has_backend</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/dummy_backend.py:DummyBackend:wrap_arg_parser' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/dummy_backend.py#L18'>DummyBackend:wrap_arg_parser</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>parser,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/dummy_backend.py:DummyBackend:_initialize' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/dummy_backend.py#L21'>DummyBackend:_initialize</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/dummy_backend.py:DummyBackend:_get_world_size' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/dummy_backend.py#L24'>DummyBackend:_get_world_size</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/dummy_backend.py:DummyBackend:_get_rank' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/dummy_backend.py#L27'>DummyBackend:_get_rank</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/dummy_backend.py:DummyBackend:_get_local_rank' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/dummy_backend.py#L30'>DummyBackend:_get_local_rank</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/dummy_backend.py:DummyBackend:_local_barrier' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/dummy_backend.py#L33'>DummyBackend:_local_barrier</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/dummy_backend.py:DummyBackend:_distribute' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/dummy_backend.py#L36'>DummyBackend:_distribute</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>_args = None,<br>model = None,<br>optimizer = None,<br>_model_parameters = None,<br>training_data = None,<br>lr_scheduler = None,<br>**_kwargs,<br>,<br></ul>
        <li>Docs:<br>        """Return the model, optimizer, dataloader, and learning rate scheduler
<br>
        as is.
<br>
        """
<br>
</li>
        </ul>
    </details>
    <details>
        <summary>
        method | <a name='dalle_pytorch/distributed_backends/dummy_backend.py:DummyBackend:_average_all' href='https://github.com/lucidrains/DALLE-pytorch/tree/main/dalle_pytorch/distributed_backends/dummy_backend.py#L51'>DummyBackend:_average_all</a>
        </summary>
        <ul>
        <li>Args:</li>
        <ul>self,<br>tensor,<br></ul>
        <li>Docs:<br></li>
        </ul>
    </details></ul>
</details>
