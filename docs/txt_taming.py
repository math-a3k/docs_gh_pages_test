



taming-transformers\main.py
-------------------------functions----------------------
get_obj_from_str(string, reload = False)
get_parser(**parser_kwargs)
nondefault_trainer_args(opt)
instantiate_from_config(config)

-------------------------methods----------------------
WrappedDataset.__init__(self, dataset)
WrappedDataset.__len__(self)
WrappedDataset.__getitem__(self, idx)
DataModuleFromConfig.__init__(self, batch_size, train = None, validation = None, test = None, wrap = False, num_workers = None)
DataModuleFromConfig.prepare_data(self)
DataModuleFromConfig.setup(self, stage = None)
DataModuleFromConfig._train_dataloader(self)
DataModuleFromConfig._val_dataloader(self)
DataModuleFromConfig._test_dataloader(self)
SetupCallback.__init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config)
SetupCallback.on_pretrain_routine_start(self, trainer, pl_module)
ImageLogger.__init__(self, batch_frequency, max_images, clamp = True, increase_log_steps = True)
ImageLogger._wandb(self, pl_module, images, batch_idx, split)
ImageLogger._testtube(self, pl_module, images, batch_idx, split)
ImageLogger.log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx)
ImageLogger.log_img(self, pl_module, batch, batch_idx, split = "train")
ImageLogger.check_frequency(self, batch_idx)
ImageLogger.on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
ImageLogger.on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)




taming-transformers\setup.py




taming-transformers\scripts\extract_depth.py
-------------------------functions----------------------
get_state(gpu)
depth_to_rgba(x)
rgba_to_depth(x)
run(x, state)
get_filename(relpath, level = -2)
save_depth(dataset, path, debug = False)





taming-transformers\scripts\extract_segmentation.py
-------------------------functions----------------------
rescale_bgr(x)
run_model(img, model)
get_input(batch, k)
save_segmentation(segmentation, path)
iterate_dataset(dataloader, destpath, model)

-------------------------methods----------------------
COCOStuffSegmenter.__init__(self, config)
COCOStuffSegmenter.forward(self, x, upsample = None)
COCOStuffSegmenter._pre_process(self, x)
COCOStuffSegmenter.mean(self)
COCOStuffSegmenter.std(self)
COCOStuffSegmenter.input_size(self)




taming-transformers\scripts\extract_submodel.py




taming-transformers\scripts\make_samples.py
-------------------------functions----------------------
save_image(x, path)
run_conditional(model, dsets, outdir, top_k, temperature, batch_size = 1)
get_parser()
load_model_from_config(config, sd, gpu = True, eval_mode = True)
get_data(config)
load_model_and_dset(config, ckpt, gpu, eval_mode)





taming-transformers\scripts\sample_conditional.py
-------------------------functions----------------------
bchw_to_st(x)
save_img(xstart, fname)
get_interactive_image(resize = False)
single_image_to_torch(x, permute = True)
pad_to_M(x, M)
run_conditional(model, dsets)
get_parser()
load_model_from_config(config, sd, gpu = True, eval_mode = True)
get_data(config)
load_model_and_dset(config, ckpt, gpu, eval_mode)





taming-transformers\scripts\sample_fast.py
-------------------------functions----------------------
chw_to_pillow(x)
sample_classconditional(model, batch_size, class_label, steps = 256, temperature = None, top_k = None, callback = None, dim_z = 256, h = 16, w = 16, verbose_time = False, top_p = None)
sample_unconditional(model, batch_size, steps = 256, temperature = None, top_k = None, top_p = None, callback = None, dim_z = 256, h = 16, w = 16, verbose_time = False)
run(logdir, model, batch_size, temperature, top_k, unconditional = True, num_samples = 50000, given_classes = None, top_p = None)
save_from_logs(logs, logdir, base_count, key = "samples", cond_key = None)
get_parser()
load_model_from_config(config, sd, gpu = True, eval_mode = True)
load_model(config, sd, gpu = True, eval_mode = True)





taming-transformers\taming\lr_scheduler.py
-------------------------methods----------------------
LambdaWarmUpCosineScheduler.__init__(self, warm_up_steps, lr_min, lr_max, lr_start, max_decay_steps, verbosity_interval = 0)
LambdaWarmUpCosineScheduler.schedule(self, n)
LambdaWarmUpCosineScheduler.__call__(self, n)




taming-transformers\taming\util.py
-------------------------functions----------------------
download(url, local_path, chunk_size = 1024)
md5_hash(path)
get_ckpt_path(name, root, check = False)
retrieve(list_or_dict, key, splitval = "/", default = None, expand = True, pass_success = False)

-------------------------methods----------------------
KeyNotFoundError.__init__(self, cause, keys = None, visited = None)




taming-transformers\taming\data\ade20k.py
-------------------------methods----------------------
Examples.__init__(self, size = 256, random_crop = False, interpolation = "bicubic")
ADE20kBase.__init__(self, config = None, size = None, random_crop = False, interpolation = "bicubic", crop_size = None)
ADE20kBase.__len__(self)
ADE20kBase.__getitem__(self, i)
ADE20kTrain.__init__(self, config = None, size = None, random_crop = True, interpolation = "bicubic", crop_size = None)
ADE20kTrain.get_split(self)
ADE20kValidation.get_split(self)




taming-transformers\taming\data\base.py
-------------------------methods----------------------
ConcatDatasetWithIndex.__getitem__(self, idx)
ImagePaths.__init__(self, paths, size = None, random_crop = False, labels = None)
ImagePaths.__len__(self)
ImagePaths.preprocess_image(self, image_path)
ImagePaths.__getitem__(self, i)
NumpyPaths.preprocess_image(self, image_path)




taming-transformers\taming\data\coco.py
-------------------------methods----------------------
Examples.__init__(self, size = 256, random_crop = False, interpolation = "bicubic")
CocoBase.__init__(self, size = None, dataroot = "", datajson = "", onehot_segmentation = False, use_stuffthing = False, crop_size = None, force_no_crop = False, given_files = None)
CocoBase.__len__(self)
CocoBase.preprocess_image(self, image_path, segmentation_path)
CocoBase.__getitem__(self, i)
CocoImagesAndCaptionsTrain.__init__(self, size, onehot_segmentation = False, use_stuffthing = False, crop_size = None, force_no_crop = False)
CocoImagesAndCaptionsTrain.get_split(self)
CocoImagesAndCaptionsValidation.__init__(self, size, onehot_segmentation = False, use_stuffthing = False, crop_size = None, force_no_crop = False, given_files = None)
CocoImagesAndCaptionsValidation.get_split(self)




taming-transformers\taming\data\custom.py
-------------------------methods----------------------
CustomBase.__init__(self, *args, **kwargs)
CustomBase.__len__(self)
CustomBase.__getitem__(self, i)
CustomTrain.__init__(self, size, training_images_list_file)
CustomTest.__init__(self, size, test_images_list_file)




taming-transformers\taming\data\faceshq.py
-------------------------methods----------------------
FacesBase.__init__(self, *args, **kwargs)
FacesBase.__len__(self)
FacesBase.__getitem__(self, i)
CelebAHQTrain.__init__(self, size, keys = None)
CelebAHQValidation.__init__(self, size, keys = None)
FFHQTrain.__init__(self, size, keys = None)
FFHQValidation.__init__(self, size, keys = None)
FacesHQTrain.__init__(self, size, keys = None, crop_size = None, coord = False)
FacesHQTrain.__len__(self)
FacesHQTrain.__getitem__(self, i)
FacesHQValidation.__init__(self, size, keys = None, crop_size = None, coord = False)
FacesHQValidation.__len__(self)
FacesHQValidation.__getitem__(self, i)




taming-transformers\taming\data\imagenet.py
-------------------------functions----------------------
give_synsets_from_indices(indices, path_to_yaml = "data/imagenet_idx_to_synset.yaml")
str_to_indices(string)
get_preprocessor(size = None, random_crop = False, additional_targets = None, crop_size = None)
rgba_to_depth(x)
imscale(x, factor, keepshapes = False, keepmode = "bicubic")

-------------------------methods----------------------
ImageNetBase.__init__(self, config = None)
ImageNetBase.__len__(self)
ImageNetBase.__getitem__(self, i)
ImageNetBase._prepare(self)
ImageNetBase._filter_relpaths(self, relpaths)
ImageNetBase._prepare_synset_to_human(self)
ImageNetBase._prepare_idx_to_synset(self)
ImageNetBase._load(self)
ImageNetTrain._prepare(self)
ImageNetValidation._prepare(self)
BaseWithDepth.__init__(self, config = None, size = None, random_crop = False, crop_size = None, root = None)
BaseWithDepth.__len__(self)
BaseWithDepth.preprocess_depth(self, path)
BaseWithDepth.__getitem__(self, i)
ImageNetTrainWithDepth.__init__(self, random_crop = True, sub_indices = None, **kwargs)
ImageNetTrainWithDepth.get_base_dset(self)
ImageNetTrainWithDepth.get_depth_path(self, e)
ImageNetValidationWithDepth.__init__(self, sub_indices = None, **kwargs)
ImageNetValidationWithDepth.get_base_dset(self)
ImageNetValidationWithDepth.get_depth_path(self, e)
RINTrainWithDepth.__init__(self, config = None, size = None, random_crop = True, crop_size = None)
RINValidationWithDepth.__init__(self, config = None, size = None, random_crop = False, crop_size = None)
DRINExamples.__init__(self)
DRINExamples.__len__(self)
DRINExamples.preprocess_image(self, image_path)
DRINExamples.preprocess_depth(self, path)
DRINExamples.__getitem__(self, i)
ImageNetScale.__init__(self, size = None, crop_size = None, random_crop = False, up_factor = None, hr_factor = None, keep_mode = "bicubic")
ImageNetScale.__len__(self)
ImageNetScale.__getitem__(self, i)
ImageNetScaleTrain.__init__(self, random_crop = True, **kwargs)
ImageNetScaleTrain.get_base(self)
ImageNetScaleValidation.get_base(self)
ImageNetEdges.__init__(self, up_factor = 1, **kwargs)
ImageNetEdges.__getitem__(self, i)
ImageNetEdgesTrain.__init__(self, random_crop = True, **kwargs)
ImageNetEdgesTrain.get_base(self)
ImageNetEdgesValidation.get_base(self)




taming-transformers\taming\data\sflckr.py
-------------------------methods----------------------
SegmentationBase.__init__(self, data_csv, data_root, segmentation_root, size = None, random_crop = False, interpolation = "bicubic", n_labels = 182, shift_segmentation = False, )
SegmentationBase.__len__(self)
SegmentationBase.__getitem__(self, i)
Examples.__init__(self, size = None, random_crop = False, interpolation = "bicubic")




taming-transformers\taming\data\utils.py
-------------------------functions----------------------
unpack(path)
reporthook(bar)
get_root(name)
is_prepared(root)
mark_prepared(root)
prompt_download(file_, source, target_dir, content_dir = None)
download_url(file_, url, target_dir)
download_urls(urls, target_dir)
quadratic_crop(x, bbox, alpha = 1.0)





taming-transformers\taming\models\cond_transformer.py
-------------------------functions----------------------
disabled_train(self, mode = True)

-------------------------methods----------------------
Net2NetTransformer.__init__(self, transformer_config, first_stage_config, cond_stage_config, permuter_config = None, ckpt_path = None, ignore_keys = [], first_stage_key = "image", cond_stage_key = "depth", downsample_cond_size = -1, pkeep = 1.0, sos_token = 0, unconditional = False, )
Net2NetTransformer.init_from_ckpt(self, path, ignore_keys = list()
Net2NetTransformer.init_first_stage_from_ckpt(self, config)
Net2NetTransformer.init_cond_stage_from_ckpt(self, config)
Net2NetTransformer.forward(self, x, c)
Net2NetTransformer.top_k_logits(self, logits, k)
Net2NetTransformer.sample(self, x, c, steps, temperature = 1.0, sample = False, top_k = None, callback=lambda k = lambda k: None)
Net2NetTransformer.encode_to_z(self, x)
Net2NetTransformer.encode_to_c(self, c)
Net2NetTransformer.decode_to_img(self, index, zshape)
Net2NetTransformer.log_images(self, batch, temperature = None, top_k = None, callback = None, lr_interface = False, **kwargs)
Net2NetTransformer.get_input(self, key, batch)
Net2NetTransformer.get_xc(self, batch, N = None)
Net2NetTransformer.shared_step(self, batch, batch_idx)
Net2NetTransformer.training_step(self, batch, batch_idx)
Net2NetTransformer.validation_step(self, batch, batch_idx)
Net2NetTransformer.configure_optimizers(self)




taming-transformers\taming\models\vqgan.py
-------------------------methods----------------------
VQModel.__init__(self, ddconfig, lossconfig, n_embed, embed_dim, ckpt_path = None, ignore_keys = [], image_key = "image", colorize_nlabels = None, monitor = None, remap = None, sane_index_shape = False, # tell vector quantizer to return indices as bhw)
VQModel.init_from_ckpt(self, path, ignore_keys = list()
VQModel.encode(self, x)
VQModel.decode(self, quant)
VQModel.decode_code(self, code_b)
VQModel.forward(self, input)
VQModel.get_input(self, batch, k)
VQModel.training_step(self, batch, batch_idx, optimizer_idx)
VQModel.validation_step(self, batch, batch_idx)
VQModel.configure_optimizers(self)
VQModel.get_last_layer(self)
VQModel.log_images(self, batch, **kwargs)
VQModel.to_rgb(self, x)
VQSegmentationModel.__init__(self, n_labels, *args, **kwargs)
VQSegmentationModel.configure_optimizers(self)
VQSegmentationModel.training_step(self, batch, batch_idx)
VQSegmentationModel.validation_step(self, batch, batch_idx)
VQSegmentationModel.log_images(self, batch, **kwargs)
VQNoDiscModel.__init__(self, ddconfig, lossconfig, n_embed, embed_dim, ckpt_path = None, ignore_keys = [], image_key = "image", colorize_nlabels = None)
VQNoDiscModel.training_step(self, batch, batch_idx)
VQNoDiscModel.validation_step(self, batch, batch_idx)
VQNoDiscModel.configure_optimizers(self)
GumbelVQ.__init__(self, ddconfig, lossconfig, n_embed, embed_dim, temperature_scheduler_config, ckpt_path = None, ignore_keys = [], image_key = "image", colorize_nlabels = None, monitor = None, kl_weight = 1e-8, remap = None, )
GumbelVQ.temperature_scheduling(self)
GumbelVQ.encode_to_prequant(self, x)
GumbelVQ.decode_code(self, code_b)
GumbelVQ.training_step(self, batch, batch_idx, optimizer_idx)
GumbelVQ.validation_step(self, batch, batch_idx)
GumbelVQ.log_images(self, batch, **kwargs)




taming-transformers\taming\modules\util.py
-------------------------functions----------------------
count_params(model)

-------------------------methods----------------------
ActNorm.__init__(self, num_features, logdet = False, affine = True, allow_reverse_init = False)
ActNorm.initialize(self, input)
ActNorm.forward(self, input, reverse = False)
ActNorm.reverse(self, output)
AbstractEncoder.__init__(self)
AbstractEncoder.encode(self, *args, **kwargs)
Labelator.__init__(self, n_classes, quantize_interface = True)
Labelator.encode(self, c)
SOSProvider.__init__(self, sos_token, quantize_interface = True)
SOSProvider.encode(self, x)




taming-transformers\taming\modules\diffusionmodules\model.py
-------------------------functions----------------------
get_timestep_embedding(timesteps, embedding_dim)
nonlinearity(x)
Normalize(in_channels)

-------------------------methods----------------------
Upsample.__init__(self, in_channels, with_conv)
Upsample.forward(self, x)
Downsample.__init__(self, in_channels, with_conv)
Downsample.forward(self, x)
ResnetBlock.__init__(self, *, in_channels, out_channels = None, conv_shortcut = False, dropout, temb_channels = 512)
ResnetBlock.forward(self, x, temb)
AttnBlock.__init__(self, in_channels)
AttnBlock.forward(self, x)
Model.__init__(self, *, ch, out_ch, ch_mult = (1, 2, 4, 8), num_res_blocks, attn_resolutions, dropout = 0.0, resamp_with_conv = True, in_channels, resolution, use_timestep = True)
Model.forward(self, x, t = None)
Encoder.__init__(self, *, ch, out_ch, ch_mult = (1, 2, 4, 8), num_res_blocks, attn_resolutions, dropout = 0.0, resamp_with_conv = True, in_channels, resolution, z_channels, double_z = True, **ignore_kwargs)
Encoder.forward(self, x)
Decoder.__init__(self, *, ch, out_ch, ch_mult = (1, 2, 4, 8), num_res_blocks, attn_resolutions, dropout = 0.0, resamp_with_conv = True, in_channels, resolution, z_channels, give_pre_end = False, **ignorekwargs)
Decoder.forward(self, z)
VUNet.__init__(self, *, ch, out_ch, ch_mult = (1, 2, 4, 8), num_res_blocks, attn_resolutions, dropout = 0.0, resamp_with_conv = True, in_channels, c_channels, resolution, z_channels, use_timestep = False, **ignore_kwargs)
VUNet.forward(self, x, z)
SimpleDecoder.__init__(self, in_channels, out_channels, *args, **kwargs)
SimpleDecoder.forward(self, x)
UpsampleDecoder.__init__(self, in_channels, out_channels, ch, num_res_blocks, resolution, 2, 2), dropout=0.0) = 0.0):)
UpsampleDecoder.forward(self, x)




taming-transformers\taming\modules\discriminator\model.py
-------------------------functions----------------------
weights_init(m)

-------------------------methods----------------------
NLayerDiscriminator.__init__(self, input_nc = 3, ndf = 64, n_layers = 3, use_actnorm = False)
NLayerDiscriminator.forward(self, input)




taming-transformers\taming\modules\losses\lpips.py
-------------------------functions----------------------
normalize_tensor(x, eps = 1e-10)
spatial_average(x, keepdim = True)

-------------------------methods----------------------
LPIPS.__init__(self, use_dropout = True)
LPIPS.load_from_pretrained(self, name = "vgg_lpips")
LPIPS.from_pretrained(cls, name = "vgg_lpips")
LPIPS.forward(self, input, target)
ScalingLayer.__init__(self)
ScalingLayer.forward(self, inp)
NetLinLayer.__init__(self, chn_in, chn_out = 1, use_dropout = False)
vgg16.__init__(self, requires_grad = False, pretrained = True)
vgg16.forward(self, X)




taming-transformers\taming\modules\losses\segmentation.py
-------------------------methods----------------------
BCELoss.forward(self, prediction, target)
BCELossWithQuant.__init__(self, codebook_weight = 1.)
BCELossWithQuant.forward(self, qloss, target, prediction, split)




taming-transformers\taming\modules\losses\vqperceptual.py
-------------------------functions----------------------
adopt_weight(weight, global_step, threshold = 0, value = 0.)
hinge_d_loss(logits_real, logits_fake)
vanilla_d_loss(logits_real, logits_fake)

-------------------------methods----------------------
DummyLoss.__init__(self)
VQLPIPSWithDiscriminator.__init__(self, disc_start, codebook_weight = 1.0, pixelloss_weight = 1.0, disc_num_layers = 3, disc_in_channels = 3, disc_factor = 1.0, disc_weight = 1.0, perceptual_weight = 1.0, use_actnorm = False, disc_conditional = False, disc_ndf = 64, disc_loss = "hinge")
VQLPIPSWithDiscriminator.calculate_adaptive_weight(self, nll_loss, g_loss, last_layer = None)
VQLPIPSWithDiscriminator.forward(self, codebook_loss, inputs, reconstructions, optimizer_idx, global_step, last_layer = None, cond = None, split = "train")




taming-transformers\taming\modules\losses\__init__.py




taming-transformers\taming\modules\misc\coord.py
-------------------------methods----------------------
CoordStage.__init__(self, n_embed, down_factor)
CoordStage.eval(self)
CoordStage.encode(self, c)
CoordStage.decode(self, c)




taming-transformers\taming\modules\transformer\mingpt.py
-------------------------functions----------------------
top_k_logits(logits, k)
sample(model, x, steps, temperature = 1.0, sample = False, top_k = None)
sample_with_past(x, model, steps, temperature = 1., sample_logits = True, top_k = None, top_p = None, callback = None)

-------------------------methods----------------------
GPTConfig.__init__(self, vocab_size, block_size, **kwargs)
CausalSelfAttention.__init__(self, config)
CausalSelfAttention.forward(self, x, layer_past = None)
Block.__init__(self, config)
Block.forward(self, x, layer_past = None, return_present = False)
GPT.__init__(self, vocab_size, block_size, n_layer = 12, n_head = 8, n_embd = 256, embd_pdrop = 0., resid_pdrop = 0., attn_pdrop = 0., n_unmasked = 0)
GPT.get_block_size(self)
GPT._init_weights(self, module)
GPT.forward(self, idx, embeddings = None, targets = None)
GPT.forward_with_past(self, idx, embeddings = None, targets = None, past = None, past_length = None)
DummyGPT.__init__(self, add_value = 1)
DummyGPT.forward(self, idx)
CodeGPT.__init__(self, vocab_size, block_size, in_channels, n_layer = 12, n_head = 8, n_embd = 256, embd_pdrop = 0., resid_pdrop = 0., attn_pdrop = 0., n_unmasked = 0)
CodeGPT.get_block_size(self)
CodeGPT._init_weights(self, module)
CodeGPT.forward(self, idx, embeddings = None, targets = None)
KMeans.__init__(self, ncluster = 512, nc = 3, niter = 10)
KMeans.is_initialized(self)
KMeans.initialize(self, x)
KMeans.forward(self, x, reverse = False, shape = None)




taming-transformers\taming\modules\transformer\permuter.py
-------------------------functions----------------------
mortonify(i, j)

-------------------------methods----------------------
AbstractPermuter.__init__(self, *args, **kwargs)
AbstractPermuter.forward(self, x, reverse = False)
Identity.__init__(self)
Identity.forward(self, x, reverse = False)
Subsample.__init__(self, H, W)
Subsample.forward(self, x, reverse = False)
ZCurve.__init__(self, H, W)
ZCurve.forward(self, x, reverse = False)
SpiralOut.__init__(self, H, W)
SpiralOut.forward(self, x, reverse = False)
SpiralIn.__init__(self, H, W)
SpiralIn.forward(self, x, reverse = False)
Random.__init__(self, H, W)
Random.forward(self, x, reverse = False)
AlternateParsing.__init__(self, H, W)
AlternateParsing.forward(self, x, reverse = False)




taming-transformers\taming\modules\vqvae\quantize.py
-------------------------methods----------------------
VectorQuantizer.__init__(self, n_e, e_dim, beta)
VectorQuantizer.forward(self, z)
VectorQuantizer.get_codebook_entry(self, indices, shape)
GumbelQuantize.__init__(self, num_hiddens, embedding_dim, n_embed, straight_through = True, kl_weight = 5e-4, temp_init = 1.0, use_vqinterface = True, remap = None, unknown_index = "random")
GumbelQuantize.remap_to_used(self, inds)
GumbelQuantize.unmap_to_all(self, inds)
GumbelQuantize.forward(self, z, temp = None, return_logits = False)
GumbelQuantize.get_codebook_entry(self, indices, shape)
VectorQuantizer2.__init__(self, n_e, e_dim, beta, remap = None, unknown_index = "random", sane_index_shape = False, legacy = True)
VectorQuantizer2.remap_to_used(self, inds)
VectorQuantizer2.unmap_to_all(self, inds)
VectorQuantizer2.forward(self, z, temp = None, rescale_logits = False, return_logits = False)
VectorQuantizer2.get_codebook_entry(self, indices, shape)
