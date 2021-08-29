

DALLE-pytorch\generate.py
-------------------------functions----------------------
exists(val)



DALLE-pytorch\setup.py


DALLE-pytorch\train_dalle.py
-------------------------functions----------------------
exists(val)
get_trainable_params(model)
cp_path_to_dir(cp_path, tag)
group_weight(model)
imagetransform(b)
tokenize(s)
save_model(path, epoch = 0)



DALLE-pytorch\train_vae.py
-------------------------functions----------------------
save_model(path)



DALLE-pytorch\dalle_pytorch\attention.py
-------------------------functions----------------------
exists(val)
uniq(arr)
default(val, d)
max_neg_value(t)
stable_softmax(t, dim  =  -1, alpha  =  32 ** 2)
apply_pos_emb(pos_emb, qkv)

-------------------------methods----------------------
Attention.__init__(self, dim, seq_len, causal  =  True, heads  =  8, dim_head  =  64, dropout  =  0., stable  =  False)
Attention.forward(self, x, mask  =  None, rotary_pos_emb  =  None)
SparseConvCausalAttention.__init__(self, dim, seq_len, image_size  =  32, kernel_size  =  5, dilation  =  1, heads  =  8, dim_head  =  64, dropout  =  0., stable  =  False, **kwargs)
SparseConvCausalAttention.forward(self, x, mask  =  None, rotary_pos_emb  =  None)
SparseAxialCausalAttention.__init__(self, dim, seq_len, image_size  =  32, axis  =  0, heads  =  8, dim_head  =  64, dropout  =  0., stable  =  False, **kwargs)
SparseAxialCausalAttention.forward(self, x, mask  =  None, rotary_pos_emb  =  None)
SparseAttention.__init__(self, *args, block_size  =  16, text_seq_len  =  256, num_random_blocks  =  None, **kwargs)
SparseAttention.forward(self, x, mask  =  None, rotary_pos_emb  =  None)


DALLE-pytorch\dalle_pytorch\dalle_pytorch.py
-------------------------functions----------------------
exists(val)
default(val, d)
is_empty(t)
masked_mean(t, mask, dim  =  1)
set_requires_grad(model, value)
eval_decorator(fn)
top_k(logits, thres  =  0.5)

-------------------------methods----------------------
always.__init__(self, val)
always.__call__(self, x, *args, **kwargs)
ResBlock.__init__(self, chan)
ResBlock.forward(self, x)
DiscreteVAE.__init__(self, image_size  =  256, num_tokens  =  512, codebook_dim  =  512, num_layers  =  3, num_resnet_blocks  =  0, hidden_dim  =  64, channels  =  3, smooth_l1_loss  =  False, temperature  =  0.9, straight_through  =  False, kl_div_loss_weight  =  0., (0.5, ) * 3, (0.5, ) * 3))
DiscreteVAE._register_external_parameters(self)
DiscreteVAE.norm(self, images)
DiscreteVAE.get_codebook_indices(self, images)
DiscreteVAE.decode(self, img_seq)
DiscreteVAE.forward(self, img, return_loss  =  False, return_recons  =  False, return_logits  =  False, temp  =  None)
CLIP.__init__(self, *, dim_text  =  512, dim_image  =  512, dim_latent  =  512, num_text_tokens  =  10000, text_enc_depth  =  6, text_seq_len  =  256, text_heads  =  8, num_visual_tokens  =  512, visual_enc_depth  =  6, visual_heads  =  8, visual_image_size  =  256, visual_patch_size  =  32, channels  =  3)
CLIP.forward(self, text, image, text_mask  =  None, return_loss  =  False)
DALLE.__init__(self, *, dim, vae, num_text_tokens  =  10000, text_seq_len  =  256, depth, heads  =  8, dim_head  =  64, reversible  =  False, attn_dropout  =  0., ff_dropout  =  0, sparse_attn  =  False, attn_types  =  None, loss_img_weight  =  7, stable  =  False, shift_tokens  =  True, rotary_emb  =  True)
DALLE.generate_texts(self, tokenizer, text  =  None, *, filter_thres  =  0.5, temperature  =  1.)
DALLE.generate_images(self, text, *, clip  =  None, mask  =  None, filter_thres  =  0.5, temperature  =  1., img  =  None, num_init_img_tokens  =  None)
DALLE.forward(self, text, image  =  None, mask  =  None, return_loss  =  False)


DALLE-pytorch\dalle_pytorch\distributed_utils.py
-------------------------functions----------------------
wrap_arg_parser(parser)
set_backend_from_args(args)
require_set_backend()
using_backend(test_backend)



DALLE-pytorch\dalle_pytorch\loader.py
-------------------------methods----------------------
TextImageDataset.__init__(self, folder, text_len = 256, image_size = 128, truncate_captions = False, resize_ratio = 0.75, tokenizer = None, shuffle = False)
TextImageDataset.__len__(self)
TextImageDataset.random_sample(self)
TextImageDataset.sequential_sample(self, ind)
TextImageDataset.skip_sample(self, ind)
TextImageDataset.__getitem__(self, ind)


DALLE-pytorch\dalle_pytorch\reversible.py
-------------------------functions----------------------
route_args(router, args, depth)

-------------------------methods----------------------
Deterministic.__init__(self, net)
Deterministic.record_rng(self, *args)
Deterministic.forward(self, *args, record_rng  =  False, set_rng  =  False, **kwargs)
ReversibleBlock.__init__(self, f, g)
ReversibleBlock.forward(self, x, f_args  =  {}, g_args  =  {})
ReversibleBlock.backward_pass(self, y, dy, f_args  =  {}, g_args  =  {})
_ReversibleFunction.forward(ctx, x, blocks, args)
_ReversibleFunction.backward(ctx, dy)
SequentialSequence.__init__(self, layers, args_route  =  {}, layer_dropout  =  0.)
SequentialSequence.forward(self, x, **kwargs)
ReversibleSequence.__init__(self, blocks, args_route  =  {})
ReversibleSequence.forward(self, x, **kwargs)


DALLE-pytorch\dalle_pytorch\tokenizer.py
-------------------------functions----------------------
default_bpe()
bytes_to_unicode()
get_pairs(word)
basic_clean(text)
whitespace_clean(text)

-------------------------methods----------------------
SimpleTokenizer.__init__(self, bpe_path  =  default_bpe()
SimpleTokenizer.bpe(self, token)
SimpleTokenizer.encode(self, text)
SimpleTokenizer.decode(self, tokens, remove_start_end  =  True, pad_tokens  =  {})
SimpleTokenizer.tokenize(self, texts, context_length  =  256, truncate_text  =  False)
HugTokenizer.__init__(self, bpe_path  =  None)
HugTokenizer.decode(self, tokens, pad_tokens  =  {})
HugTokenizer.encode(self, text)
HugTokenizer.tokenize(self, texts, context_length  =  256, truncate_text  =  False)
ChineseTokenizer.__init__(self)
ChineseTokenizer.decode(self, tokens, pad_tokens  =  {})
ChineseTokenizer.encode(self, text)
ChineseTokenizer.tokenize(self, texts, context_length  =  256, truncate_text  =  False)
YttmTokenizer.__init__(self, bpe_path  =  None)
YttmTokenizer.decode(self, tokens, pad_tokens  =  {})
YttmTokenizer.encode(self, texts)
YttmTokenizer.tokenize(self, texts, context_length  =  256, truncate_text  =  False)


DALLE-pytorch\dalle_pytorch\transformer.py
-------------------------functions----------------------
exists(val)
default(val, d)
cast_tuple(val, depth  =  1)

-------------------------methods----------------------
DivideMax.__init__(self, dim)
DivideMax.forward(self, x)
LayerScale.__init__(self, dim, depth, fn)
LayerScale.forward(self, x, **kwargs)
PreNorm.__init__(self, dim, fn)
PreNorm.forward(self, x, **kwargs)
GEGLU.forward(self, x)
FeedForward.__init__(self, dim, dropout  =  0., mult  =  4.)
FeedForward.forward(self, x)
PreShiftToken.__init__(self, fn, image_size, seq_len)
PreShiftToken.forward(self, x, **kwargs)
Transformer.__init__(self, *, dim, depth, seq_len, reversible  =  False, causal  =  True, heads  =  8, dim_head  =  64, ff_mult  =  4, attn_dropout  =  0., ff_dropout  =  0., attn_types  =  None, image_fmap_size  =  None, sparse_attn  =  False, stable  =  False, shift_tokens  =  False, rotary_emb  =  True)
Transformer.forward(self, x, **kwargs)


DALLE-pytorch\dalle_pytorch\vae.py
-------------------------functions----------------------
exists(val)
default(val, d)
load_model(path)
map_pixels(x, eps  =  0.1)
unmap_pixels(x, eps  =  0.1)
download(url, filename  =  None, root  =  CACHE_PATH)
make_contiguous(module)
get_obj_from_str(string, reload = False)
instantiate_from_config(config)

-------------------------methods----------------------
OpenAIDiscreteVAE.__init__(self)
OpenAIDiscreteVAE.get_codebook_indices(self, img)
OpenAIDiscreteVAE.decode(self, img_seq)
OpenAIDiscreteVAE.forward(self, img)
VQGanVAE.__init__(self, vqgan_model_path = None, vqgan_config_path = None)
VQGanVAE._register_external_parameters(self)
VQGanVAE.get_codebook_indices(self, img)
VQGanVAE.decode(self, img_seq)
VQGanVAE.forward(self, img)


DALLE-pytorch\dalle_pytorch\__init__.py


DALLE-pytorch\dalle_pytorch\distributed_backends\deepspeed_backend.py
-------------------------methods----------------------
DeepSpeedBackend.wrap_arg_parser(self, parser)
DeepSpeedBackend._initialize(self)
DeepSpeedBackend._require_torch_distributed_init()
DeepSpeedBackend._get_world_size(self)
DeepSpeedBackend._get_rank(self)
DeepSpeedBackend._get_local_rank(self)
DeepSpeedBackend._local_barrier(self)
DeepSpeedBackend._check_args(self, args, optimizer, lr_scheduler, kwargs)
DeepSpeedBackend._check_argvs(self, args, optimizer, lr_scheduler, kwargs)
DeepSpeedBackend._check_config(self, args, optimizer, lr_scheduler, kwargs)
DeepSpeedBackend._distribute(self, args = None, model = None, optimizer = None, model_parameters = None, training_data = None, lr_scheduler = None, **kwargs, )
DeepSpeedBackend._average_all(self, tensor)


DALLE-pytorch\dalle_pytorch\distributed_backends\distributed_backend.py
-------------------------methods----------------------
DistributedBackend.__init__(self)
DistributedBackend.has_backend(self)
DistributedBackend.check_batch_size(self, batch_size)
DistributedBackend.wrap_arg_parser(self, parser)
DistributedBackend.initialize(self)
DistributedBackend._initialize(self)
DistributedBackend.require_init(self)
DistributedBackend.get_world_size(self)
DistributedBackend._get_world_size(self)
DistributedBackend.get_rank(self)
DistributedBackend._get_rank(self)
DistributedBackend.get_local_rank(self)
DistributedBackend._get_local_rank(self)
DistributedBackend.is_root_worker(self)
DistributedBackend.is_local_root_worker(self)
DistributedBackend.local_barrier(self)
DistributedBackend._local_barrier(self)
DistributedBackend.distribute(self, args = None, model = None, optimizer = None, model_parameters = None, training_data = None, lr_scheduler = None, **kwargs, )
DistributedBackend._distribute(self, args = None, model = None, optimizer = None, model_parameters = None, training_data = None, lr_scheduler = None, **kwargs, )
DistributedBackend.average_all(self, tensor)
DistributedBackend._average_all(self, tensor)


DALLE-pytorch\dalle_pytorch\distributed_backends\dummy_backend.py
-------------------------methods----------------------
DummyBackend.has_backend(self)
DummyBackend.wrap_arg_parser(self, parser)
DummyBackend._initialize(self)
DummyBackend._get_world_size(self)
DummyBackend._get_rank(self)
DummyBackend._get_local_rank(self)
DummyBackend._local_barrier(self)
DummyBackend._distribute(self, _args = None, model = None, optimizer = None, _model_parameters = None, training_data = None, lr_scheduler = None, **_kwargs, )
DummyBackend._average_all(self, tensor)


DALLE-pytorch\dalle_pytorch\distributed_backends\horovod_backend.py
-------------------------methods----------------------
HorovodBackend.wrap_arg_parser(self, parser)
HorovodBackend.check_batch_size(self, batch_size)
HorovodBackend._initialize(self)
HorovodBackend._get_world_size(self)
HorovodBackend._get_rank(self)
HorovodBackend._get_local_rank(self)
HorovodBackend._local_barrier(self)
HorovodBackend._distribute(self, _args = None, model = None, optimizer = None, _model_parameters = None, training_data = None, lr_scheduler = None, **_kwargs, )
HorovodBackend._average_all(self, tensor)


DALLE-pytorch\dalle_pytorch\distributed_backends\__init__.py
