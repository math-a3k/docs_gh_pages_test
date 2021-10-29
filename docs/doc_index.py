

utilmy/__init__.py


utilmy/adatasets.py
-------------------------functions----------------------
dataset_classifier_XXXXX(nrows = 500, **kw)
dataset_classifier_pmlb(name = '', return_X_y = False)
fetch_dataset(url_dataset, path_target = None, file_target = None)
log(*s)
log2(*s)
pd_train_test_split(df, coly = None)
pd_train_test_split2(df, coly)
test0()
test1()
test_all()
test_dataset_classification_fake(nrows = 500)
test_dataset_classification_petfinder(nrows = 1000)
test_dataset_classifier_covtype(nrows = 500)
test_dataset_regression_fake(nrows = 500, n_features = 17)



utilmy/configs/__init__.py


utilmy/configs/test.py
-------------------------functions----------------------
create_fixtures_data(tmp_path)
test_validate_yaml_failed_silent(tmp_path)
test_validate_yaml_types(tmp_path)
test_validate_yaml_types_failed(tmp_path)



utilmy/configs/util_config.py
-------------------------functions----------------------
config_isvalid_pydantic(config_dict: dict, pydanctic_schema: str  =  'config_py.yaml', silent: bool  =  False)
config_isvalid_yamlschema(config_dict: dict, schema_path: str  =  'config_val.yaml', silent: bool  =  False)
config_load(config_path:    str   =  None, path_default:   str   =  None, config_default: dict  =  None, save_default:   bool  =  False, to_dataclass:   bool  =  True, )
convert_dict_to_pydantic(config_dict: dict, schema_name: str)
convert_yaml_to_box(yaml_path: str)
global_verbosity(cur_path, path_relative = "/../../config.json", default = 5, key = 'verbosity', )
log(*s)
loge(*s)
pydantic_model_generator(input_file: Union[Path, str], input_file_type, output_file: Path, **kwargs, )
test4()
test_example()
test_pydanticgenrator()
test_yamlschema()
zzz_config_load_validate(config_path: str, schema_path: str, silent: bool  =  False)



utilmy/data.py
-------------------------functions----------------------
help()
log(*s)



utilmy/dates.py
-------------------------functions----------------------
date_generate(start = '2018-01-01', ndays = 100)
date_is_holiday(array)
date_now(fmt="%Y-%m-%d %H = "%Y-%m-%d %H:%M:%S %Z%z", add_days = 0, timezone = 'Asia/Tokyo')
date_weekday_excel(x)
date_weekmonth(d)
date_weekmonth2(d)
date_weekyear2(dt)
date_weekyear_excel(x)
log(*s)
pd_date_split(df, coldate  =   'time_key', prefix_col  = "", sep = "/", verbose = False)
random_dates(start, end, size)
random_genders(size, p = None)
test_all()



utilmy/debug.py
-------------------------functions----------------------
help()
log(*s)
log10(*s, nmax = 60)
log5(*s)
log_trace(msg = "", dump_path = "", globs = None)
print_everywhere()
profiler_start()
profiler_stop()



utilmy/decorators.py
-------------------------functions----------------------
dummy_func()
profiled_sum()
profiler_context()
profiler_decorator(func)
profiler_decorator_base(fnc)
profiler_decorator_base_test()
test0()
test_all()
test_decorators()
test_decorators2()
thread_decorator(func)
thread_decorator_test()
timeout_decorator(seconds = 10, error_message = os.strerror(errno.ETIME)
timeout_decorator_test()
timer_decorator(func)



utilmy/deeplearning/__init__.py


utilmy/deeplearning/keras/loss_graph.py
-------------------------functions----------------------
create_fake_neighbor(x, max_neighbors)
create_graph_loss(max_neighbors = 2)
help()
map_func(x_batch, y_batch, neighbors, neighbor_weights)
test_adversarial()
test_graph_loss()
test_step(x, y, model, loss_fn, nbr_features_layer = None, ### Graphregularizer = None, #### Graph)
train_step(x, y, model, loss_fn, optimizer, nbr_features_layer = None, ### Graphregularizer = None, ## Graph) as tape_w)



utilmy/deeplearning/keras/loss_vq_vae2.py
-------------------------functions----------------------
encoder_Base(latent_dim)
get_vqvae_layer_hierarchical(latent_dim = 16, num_embeddings = 64)
plot_original_reconst_img(orig, rec)
test_vqvae2()

-------------------------methods----------------------
PixelConvLayer.__init__(self, mask_type, **kwargs)
PixelConvLayer.build(self, input_shape)
PixelConvLayer.call(self, inputs)
Quantizer.__init__(self, number_of_embeddings, embedding_dimensions, beta = 0.25, **kwargs)
Quantizer.call(self, x)
Quantizer.get_code_indices(self, flattened_inputs)
ResidualBlock.__init__(self, filters, **kwargs)
ResidualBlock.call(self, inputs)
VQ_VAE_Trainer_2.__init__(self, train_variance, latent_dim = 16, number_of_embeddings = 128, **kwargs)
VQ_VAE_Trainer_2.metrics(self)
VQ_VAE_Trainer_2.train_step(self, x)


utilmy/deeplearning/keras/template_train.py
-------------------------functions----------------------
label_get_data()
param_set()
params_set2()
pd_get_dummies(df, cols_cat, cat_dict:dict, only_onehot = True)
train_step(x, model, y_label_list = None)
validation_step(x, model, y_label_list = None)



utilmy/deeplearning/keras/train_graph_loss.py
-------------------------functions----------------------
cal_loss_macro_soft_f1(y, y_hat)
log(*s)
make_classifier(class_dict)
make_decoder()
make_encoder(n_outputs = 1)
metric_accuracy(y_val, y_pred_head, class_dict)
metric_accuracy_2(y_test, y_pred, dd)
plot_grid(images, title = '')
plot_original_images(test_sample)
plot_reconstructed_images(model, test_sample)
save_best(model, model_dir2, valid_loss, best_loss, counter)
save_model_state(model, model_dir2)
test_step(x, y, model, loss_fn)
train_step(x, model, y_label_list = None)
train_step(x, model, y_label_list = None)
train_step(x, model, y_label_list = None)
train_step_2(x, model, y_label_list = None)
train_stop(counter, patience)
valid_image_check(img_list, path = "", tag = "", y_labels = "", n_sample = 3, renorm = True)
validation_step(x, model)
validation_step(x, model)
validation_step(x, model)
visualize_imgs(img_list, path, tag, y_labels, n_sample = None)

-------------------------methods----------------------
GraphDataGenerator.__getitem__(self, idx)
GraphDataGenerator.__init__(self, data_iter, graph_dict)
GraphDataGenerator.__len__(self)
GraphDataGenerator._map_func(self, index, x_batch, *y_batch)
LearningRateDecay.plot(self, epochs, title = "Learning Rate Schedule")
PolynomialDecay.__call__(self, epoch)
PolynomialDecay.__init__(self, max_epochs = 100, init_lr = 0.01, power = 1.0)
StepDecay.__call__(self, epoch)
StepDecay.__init__(self, init_lr = 0.01, factor = 0.25, drop_every = 10)


utilmy/deeplearning/keras/train_vqvae_loss.py
-------------------------functions----------------------
apply_func(s, values)
build_model_2(input_shape, num_classes)
clf_loss_crossentropy(y_true, y_pred)
custom_loss(y_true, y_pred)
decoder_base(latent_dim, shape)
encoder_base(input_shape, latent_dim)
make_decoder()
make_encoder(n_outputs = 1)
make_vqvae_classifier(class_dict)
make_vqvae_decoder(input_shape, latent_dim)
make_vqvae_encoder(input_shape, latent_dim)
metric_accuracy(y_val, y_pred_head, class_dict)
perceptual_loss_function(x, x_recon, z_mean, z_logsigma, kl_weight = 0.00005, y_label_heads = None, y_pred_heads = None, clf_loss_fn = None)
plot_grid(images, title = '')
print_log(*s)
train_step(x, model, train_variance, y_label_list = None)
train_step(x, model, train_variance, y_label_list = None)
validation_step(x, model, train_variance)
validation_step(x, model, train_variance)
visualize_imgs(img_list, path, tag, y_labels, n_sample = None)

-------------------------methods----------------------
CustomDataGenerator0.__getitem__(self, idx)
CustomDataGenerator0.__init__(self, image_dir, label_path, class_dict, split = 'train', batch_size = 8, transforms = None)
CustomDataGenerator0.__len__(self)
CustomDataGenerator0.on_epoch_end(self)
CustomDataGenerator.__getitem__(self, idx)
CustomDataGenerator.__init__(self, x, y, batch_size = 32, augmentations = None)
CustomDataGenerator.__len__(self)
LearningRateDecay.plot(self, epochs, title = "Learning Rate Schedule")
PolynomialDecay.__call__(self, epoch)
PolynomialDecay.__init__(self, max_epochs = 100, init_lr = 0.01, power = 1.0)
Quantizer.__init__(self, number_of_embeddings, embedding_dimensions, beta = 0.25, **kwargs)
Quantizer.call(self, x)
Quantizer.get_code_indices(self, flattened_inputs)
SprinklesTransform.__init__(self, num_holes = 100, side_length = 10, always_apply = False, p = 1.0)
SprinklesTransform.apply(self, image, **params)
StepDecay.__call__(self, epoch)
StepDecay.__init__(self, init_lr = 0.01, factor = 0.25, drop_every = 10)
VQ_VAE.__init__(self, latent_dim, class_dict, num_embeddings = 64, image_size = 64)
VQ_VAE.__init__(self, latent_dim, class_dict, num_embeddings = 64, image_size = 64)
VQ_VAE.call(self, x, training = True, mask = None)
VQ_VAE.call(self, x, training = True, mask = None)
VQ_VAE.decode(self, encoder_A_outputs, encoder_B_outputs, apply_sigmoid = False)
VQ_VAE.decode(self, encoder_A_outputs, encoder_B_outputs, apply_sigmoid = False)
VQ_VAE.encode(self, x)
VQ_VAE.encode(self, x)
VQ_VAE.reparameterize(self, z_mean, z_logsigma)
VQ_VAE.reparameterize(self, z_mean, z_logsigma)


utilmy/deeplearning/keras/util_dataloader.py
-------------------------functions----------------------
_byte_feature(value)
_float_feature(value)
_int64_feature(value)
build_tfrecord(x, tfrecord_out_path, max_records)
data_add_onehot(dfref, img_dir, labels_col)
get_data_sample(batch_size, x_train, labels_val)
help()
test()
to_OneHot(df, dfref, labels_col)

-------------------------methods----------------------
CustomDataGenerator.__getitem__(self, idx)
CustomDataGenerator.__init__(self, x, y, batch_size = 32, augmentations = None)
CustomDataGenerator.__len__(self)
CustomDataGenerator_img.__getitem__(self, idx)
CustomDataGenerator_img.__getitem__(self, idx)
CustomDataGenerator_img.__init__(self, img_dir, label_path, class_list, split = 'train', batch_size = 8, transforms = None)
CustomDataGenerator_img.__init__(self, img_dir, label_path, class_list, split = 'train', batch_size = 8, transforms = None)
CustomDataGenerator_img.__len__(self)
CustomDataGenerator_img.__len__(self)
CustomDataGenerator_img.on_epoch_end(self)
CustomDataGenerator_img.on_epoch_end(self)
RealCustomDataGenerator.__getitem__(self, idx)
RealCustomDataGenerator.__init__(self, image_dir, label_path, class_dict, split = 'train', batch_size = 8, transforms = None, shuffle = True)
RealCustomDataGenerator.__len__(self)
RealCustomDataGenerator._load_data(self, label_path)
RealCustomDataGenerator.on_epoch_end(self)
SprinklesTransform.__init__(self, num_holes = 30, side_length = 5, always_apply = False, p = 1.0)
SprinklesTransform.apply(self, image, **params)


utilmy/deeplearning/keras/util_layers.py
-------------------------functions----------------------
help()
make_classifier(label_name_ncount:dict = None, layers_dim = [128, 1024], tag = '1')
make_classifier_2(latent_dim, class_dict)
make_classifier_multihead(label_name_ncount:dict = None, layers_dim = [128, 1024], tag = '1')
make_decoder(xdim, ydim, latent_dim)
make_encoder(xdim = 256, ydim = 256, latent_dim = 10)
test_all()
test_cdfvae()
test_resnetlayer()
vae_loss(x, output)

-------------------------methods----------------------
CNNBlock.__init__(self, filters, kernels, strides = 1, padding = 'valid', activation = None)
CNNBlock.call(self, input_tensor, training = True)
DFC_VAE.__init__(self, latent_dim, class_dict)
DFC_VAE.call(self, x, training = True, mask = None, y_label_list = None)
DFC_VAE.decode(self, z, apply_sigmoid = False)
DFC_VAE.encode(self, x)
DFC_VAE.reparameterize(self, z_mean, z_logsigma)
DepthConvBlock.__init__(self, filters)
DepthConvBlock.call(self, inputs)
ResBlock.__init__(self, filters, kernels)
ResBlock.call(self, input_tensor, training = False)


utilmy/deeplearning/keras/util_loss.py
-------------------------functions----------------------
learning_rate_schedule(mode = "step", epoch = 1, cc = None)
loss_clf_macro_soft_f1(y, y_hat)
loss_perceptual_function(x, x_recon, z_mean, z_logsigma, kl_weight = 0.00005, y_label_heads = None, y_pred_heads = None, clf_loss_fn = None)
loss_schedule(mode = "step", epoch = 1)
metric_accuracy(y_test, y_pred, dd)
test_all()
test_loss1()

-------------------------methods----------------------
LearningRateDecay.plot(self, epochs, title = "Learning Rate Schedule", path = None)
StepDecay.__call__(self, epoch)
StepDecay.__init__(self, init_lr = 0.01, factor = 0.25, drop_every = 5)


utilmy/deeplearning/keras/util_models.py
-------------------------functions----------------------
get_final_image(file_path, model_path, target_size)
test_all()
test_classactivation()

-------------------------methods----------------------
GradCAM.__init__(self, model, classIdx, layerName = None)
GradCAM.compute_heatmap(self, image, eps = 1e-8)
GradCAM.find_target_layer(self)
GradCAM.overlay_heatmap(self, heatmap, image, alpha = 0.5, colormap = cv2.COLORMAP_JET)


utilmy/deeplearning/keras/util_similarity.py
-------------------------functions----------------------
__cast_left_and_right_to_tensors(left, right)
__get_rows_counts(left, right)
__get_tensor_reshaped_norm(tensor, reshape_shape)
__get_tensor_sqr(tensor, reshape_shape, tile_shape)
help()
test_all()
test_tf_cdist()
tf_cdist(left, right, metric = 'euclidean')
tf_cdist_cos(left, right)
tf_cdist_euclidean(left, right)



utilmy/deeplearning/keras/util_train.py
-------------------------functions----------------------
check_valid_image(img_list, path = "", tag = "", y_labels = "", n_sample = 3, renorm = True)
clean_duplicates(ll)
config_save(cc, path)
get_custom_label_data()
image_check(name, img, renorm = False)
log(*s)
model_reload(model_reload_name, cc, )
np_remove_duplicates(seq)
os_path_copy(in_dir, path, ext = "*.py")
pd_category_filter(df, category_map)
print_debug_info(*s)
print_log_info(*s)
save_best_model(model, model_dir2, curr_loss, best_loss, counter, epoch, dd)
save_model_state(model, model_dir2)
test_step(x, y, model, loss_fn)
tf_compute_set(cc:dict)
train_step_2(x, y, model, loss_fn, optimizer)
train_stop(counter, patience)



utilmy/deeplearning/torch/util_train.py


utilmy/deeplearning/util_dl.py
-------------------------functions----------------------
create_train_npz()
create_train_parquet()
data_mnist_get_train_test(batch = 32)
down_page(query, out_dir = "query1", genre_en = '', id0 = "", cat = "", npage = 1)
gpu_available()
gpu_usage()
model_deletes(dry = 0)
tensorboard_log(pars_dict:dict = None, writer = None, verbose = True)
test_all()
tf_check()



utilmy/deeplearning/util_embedding.py
-------------------------functions----------------------
convert_txt_to_vector_parquet(dirin = None, dirout = None, skip = 0, nmax = 10**8)
data_add_onehot(dfref, img_dir, labels_col)
embedding_load_parquet(dirin = "df.parquet", nmax  =  500)
embedding_table_comparison(embeddings_1:list, embeddings_2:list, labels_1:list, labels_2:list, plot_title, plot_width = 1200, plot_height = 600, xaxis_font_size = '12pt', yaxis_font_size = '12pt')
embedding_to_parquet(dirin = None, dirout = None, skip = 0, nmax = 10**8, is_linevalid_fun=Nonedirout) ; os_makedirs(dirout)  ; time.sleep(4)if is_linevalid_fun is None  = Nonedirout) ; os_makedirs(dirout)  ; time.sleep(4)if is_linevalid_fun is None : #### Validate linew):)
faiss_create_index(df_or_path = None, col = 'emb', dir_out = "", db_type  =  "IVF4096,Flat", nfile = 1000, emb_dim = 200)
faiss_topk(df = None, root = None, colid = 'id', colemb = 'emb', faiss_index = None, topk = 200, npool = 1, nrows = 10**7, nfile = 1000)
np_matrix_to_str(m, map_dict)
np_matrix_to_str2(m, map_dict)
np_matrix_to_str_sim(m)
np_str_to_array(vv, l2_norm = True, mdim  =  200)
np_str_to_array(vv, l2_norm = True, mdim  =  200)
sim_score2(path = "")
simscore_cosinus_calc(embs, words)
test()
topk(topk = 100, dname = None, pattern = "df_*1000*.parquet", filter1 = None)
topk(topk = 100, dname = None, pattern = "df_*1000*.parquet", filter1 = None)
topk_export()
topk_nearest_vector(x0, vector_list, topk = 3)
topk_nearest_vector(x0, vector_list, topk = 3)
topk_predict()
unzip(in_dir, out_dir)
viz_run(dirin = "in/model.vec", dirout = "ztmp/", nmax = 100)

-------------------------methods----------------------
TopToolbar.__init__(self)
vizEmbedding.__init__(self, path = "myembed.parquet", num_clusters = 5, sep = ";", config:dict = None)
vizEmbedding.create_clusters(self, after_dim_reduction = True)
vizEmbedding.create_visualization(self, dir_out = "ztmp/", mode = 'd3', cols_label = None, show_server = False, **kw)
vizEmbedding.dim_reduction(self, mode = "mds", col_embed = 'embed', ndim = 2, nmax =  5000, dir_out = None, ntest = 10000, npool = 2)
vizEmbedding.draw_hiearchy(self)
vizEmbedding.run_all(self, mode = "mds", col_embed = 'embed', ndim = 2, nmax =  5000, dir_out = "ztmp/", ntest = 10000)


utilmy/deeplearning/util_yolo.py
-------------------------functions----------------------
convert_to_yolov5(info_dict:Dict, names:Dict, output:str)
test_all()
test_convert_to_yolov5()
test_yolov5_from_xml()
yolo_extract_info_from_xml(xml_file:str)
yolov5_from_xml(xml_file_path:str  =  "None", xml_folder:str =  "None", output:str = "None")



utilmy/deeplearning/zz_util_topk.py
-------------------------functions----------------------
convert_txt_to_vector_parquet(dirin = None, dirout = None, skip = 0, nmax = 10**8)
data_add_onehot(dfref, img_dir, labels_col)
folder_size()
gzip()
np_matrix_to_str(m, map_dict)
np_matrix_to_str2(m, map_dict)
np_matrix_to_str_sim(m)
np_str_to_array(vv, l2_norm = True, mdim  =  200)
predict(name = None)
test()
topk()
topk_export()
topk_nearest_vector(x0, vector_list, topk = 3)
topk_predict()
unzip(in_dir, out_dir)



utilmy/distributed.py
-------------------------functions----------------------
date_now(fmt = "%Y-%m-%d %H =  "%Y-%m-%d %H:%M:%S %Z%z")
help()
load(to_file = "")
load_serialize(name)
log_mem(*s)
os_lock_acquireLock(plock:str = "tmp/plock.lock")
os_lock_execute(fun_run, fun_args = None, ntry = 5, plock = "tmp/plock.lock", sleep = 5)
os_lock_releaseLock(locked_file_descriptor)
save(dd, to_file = "", verbose = False)
save_serialize(name, value)
test1_functions()
test2_funtions_thread()
test3_index()
test_all()
test_tofilesafe()
time_sleep_random(nmax = 5)
to_file_safe(msg:str, fpath:str)

-------------------------methods----------------------
FileWriter.__init__(self, fpath)
FileWriter.write(self, msg)
IndexLock.__init__(self, findex, plock)
IndexLock.get(self)
IndexLock.put(self, val = "", ntry = 100, plock = "tmp/plock.lock")


utilmy/docs/__init__.py


utilmy/docs/cli.py
-------------------------functions----------------------
os_remove(filepath)
run_cli()



utilmy/docs/code_parser.py
-------------------------functions----------------------
_clean_data(array)
_get_all_line(file_path)
_get_all_lines_define_function(function_name, array, indentMethod = '')
_get_all_lines_in_class(class_name, array)
_get_all_lines_in_function(function_name, array, indentMethod = '')
_get_and_clean_all_lines(file_path)
_get_avg_char_per_word(row)
_get_define_function_stats(array)
_get_docs(all_lines, index_1, func_lines)
_get_function_stats(array, indent)
_get_functions(row)
_get_words(row)
_remmove_commemt_line(line)
_remove_empty_line(line)
_validate_file(file_path)
export_call_graph(repo_link: str, out_path:str = None)
export_call_graph_url(repo_link: str, out_path:str = None)
export_stats_perfile(in_path:str = None, out_path:str = None)
export_stats_perrepo(in_path:str = None, out_path:str = None, repo_name:str = None)
export_stats_perrepo_txt(in_path:str = None, out_path:str = None, repo_name:str = None)
export_stats_pertype(in_path:str = None, type:str = None, out_path:str = None)
export_stats_repolink(repo_link: str, out_path:str = None)
export_stats_repolink_txt(repo_link: str, out_path:str = None)
get_file_stats(file_path)
get_list_class_info(file_path)
get_list_class_methods(file_path)
get_list_class_name(file_path)
get_list_class_stats(file_path)
get_list_function_info(file_path)
get_list_function_name(file_path)
get_list_function_stats(file_path)
get_list_import_class_as(file_path: str)
get_list_imported_func(file_path: str)
get_list_method_info(file_path)
get_list_method_stats(file_path)
get_list_variable_global(file_path)
get_stats(df:pd.DataFrame, file_path:str)
test_example()
write_to_file(uri, type, list_functions, list_classes, list_imported, dict_functions, list_class_as, out_path)



utilmy/docs/generate_doc.py
-------------------------functions----------------------
markdown_create_file(list_info, prefix = '')
markdown_create_function(uri, name, type, args_name, args_type, args_value, start_line, list_docs, prefix = "")
markdown_createall(dfi, prefix = "")
run_markdown(repo_stat_file, output = 'docs/doc_main.md', prefix="https = "https://github.com/user/repo/tree/a")
run_table(repo_stat_file, output = 'docs/doc_table.md', prefix="https = "https://github.com/user/repo/tree/a")
table_all_row(list_rows)
table_create(uri, name, type, start_line, list_funtions, prefix)
table_create_row(uri, name, type, start_line, list_funtions, prefix)
test()



utilmy/docs/test.py
-------------------------functions----------------------
calculateBuyPrice(enter, profit)
calculateSellPrice(enter, profit)
get_list_price()
list_buy_price(start, bottom, delta)
list_sell_price(start, top, delta)
log(data)
trading_down()
trading_up()
update_price()



utilmy/excel/xlvba.py
-------------------------functions----------------------
invokenumpy()
invokesklearn()
load_csv(csvfile)
loaddf()



utilmy/graph.py


utilmy/iio.py


utilmy/images/util_image.py
-------------------------functions----------------------
help()
image_cache_check(db_path:str = "db_images.cache", dirout:str = "tmp/", tag = "cache1")
image_cache_create()
image_cache_save(image_path_list:str = "db_images.cache", db_dir:str = "tmp/", tag = "cache1")
image_center_crop(img, dim)
image_check(path_npz, keys = ['train'], path = "", tag = "", n_sample = 3, renorm = True)
image_check_npz(path_npz, keys = ['train'], path = "", tag = "", n_sample = 3, renorm = True)
image_face_blank(in_dir = "", level  =  "/*", out_dir = f"", npool = 30)
image_merge(image_list, n_dim, padding_size, max_height, total_width)
image_padding_generate(paddings_number: int  =  1, min_padding: int  =  1, max_padding: int  =  1)
image_read(filepath_or_buffer: Union[str, io.BytesIO])
image_remove_bg(in_dir = "", out_dir = "", level = 1)
image_remove_extra_padding(img, inverse = False, removedot = True)
image_resize(image, width = None, height = None, inter = cv2.INTER_AREA)
image_resize2(image, width = None, height = None, inter = cv2.INTER_AREA)
image_resize_pad(img, size = (256, 256)
image_resize_ratio(image, width = None, height = None, inter = cv2.INTER_AREA)
image_save()
image_show_in_row(image_list:dict = None)
image_text_blank(in_dir, out_dir, level = "/*")
log(*s)
log2(*s)
os_path_check(path, n = 5)
prep_image(image_paths, nmax = 10000000)
prep_images(image_paths, nmax = 10000000)
prep_images2(image_paths, nmax = 10000000)
prep_images_multi(image_path_list:list, prepro_image_fun = None, npool = 1)
run_multiprocess(myfun, list_args, npool = 10, **kwargs)
test()



utilmy/keyvalue.py
-------------------------functions----------------------
db_create_dict_pandas(df = None, cols = None, colsu = None)
db_flush(db_dir)
db_init(db_dir:str = "path", globs = None)
db_load_dict(df, colkey, colval, verbose = True)
db_merge()
db_size(db_dir =  None)
diskcache_config(db_path = None, task = 'commit')
diskcache_get(cache)
diskcache_getall(cache, limit = 1000000000)
diskcache_getkeys(cache)
diskcache_keycount(cache)
diskcache_load(db_path_or_object = "", size_limit = 100000000000, verbose = True)
diskcache_save(df, colkey, colvalue, db_path = "./dbcache.db", size_limit = 100000000000, timeout = 10, shards = 1, tbreak = 1, ## Break during insert to prevent big WAL file**kw)
diskcache_save2(df, colkey, colvalue, db_path = "./dbcache.db", size_limit = 100000000000, timeout = 10, shards = 1, npool = 10, sqlmode =  'fast', verbose = True)
os_environ_set(name, value)
os_path_size(folder = None)

-------------------------methods----------------------
DBlist.__init__(self, config_dict = None, config_path = None)
DBlist.add(self, db_path)
DBlist.check(self, db_path = None)
DBlist.clean(self, )
DBlist.info(self, )
DBlist.list(self, show = True)
DBlist.remove(self, db_path)
DBlist.show(self, db_path = None, n = 4)


utilmy/logs/__init__.py


utilmy/logs/test_log.py
-------------------------functions----------------------
test1()
test2()
test_launch_server()
test_server()

-------------------------methods----------------------
LoggingStreamHandler.handle(self)


utilmy/logs/util_log.py
-------------------------functions----------------------
log(log_config_path: str  =  None, log_template: str  =  "default", **kwargs)
log2(*s)
log3(*s)
logc(*s)
loge(*s)
logger_setup(log_config_path: str  =  None, log_template: str  =  "default", **kwargs)
logr(*s)
logw(*s)
test()
z_logger_custom_1()
z_logger_stdout_override()



utilmy/nlp/text.py
-------------------------functions----------------------
help()
help_get_codesource(func)
log(*s)
pd_text_getcluster(df:pd.DataFrame, col:str = 'col', threshold = 0.5, num_perm:int = 5, npool = 1, chunk  =  100000)
pd_text_hash_create_lsh(df, col, sep = " ", threshold = 0.7, num_perm = 10, npool = 1, chunk  =  20000)
pd_text_similarity(df: pd.DataFrame, cols = [], algo = '')
test()
test()
test_all()
test_lsh()



utilmy/nlp/util_cocount.py
-------------------------functions----------------------
calc_comparison_stats(model, ccount_name_dict, ccount_score_dict, corpus_file = "data.cor", top = 20, output_dir = "./no_ss_test")
cocount_calc_matrix(dirin = "gen_text_dist3.txt", dense = True)
cocount_get_topk(matrix, w_to_id)
cocount_matrix_to_dict(matrix, w_to_id)
cocount_norm(matrix)
corpus_add_prefix(dirin = "gen_text_dist3.txt", dirout = "gen_text_dist4.txt")
corpus_generate(outfile = "data.cor", unique_words_needed = 1000)
corpus_generate_from_cocount(dirin = "./data.cor", dirout = "gen_text_dist3.txt", unique_words = 100, sentences_count = 1000)
create_1gram_stats(dirin, w_to_id)
get_top_k(w, ccount_name_dict, ccount_score_dict, top = 5)
load_model(dirin = "./modelout/model.bin")
run_all()
train_model(dirinput = "./data.cor", dirout = "./modelout/model.bin", **params)



utilmy/nlp/util_model.py
-------------------------functions----------------------
bigram_get_list(ranid, mode = 'name, proba')
bigram_get_seq3(ranid, itemtag, lname, pnorm)
bigram_load_convert(path)
bigram_write_seq(rr = 0, dirin = None, dirout = None, tag = "")
ccount_get_sample(lname, lproba = None, pnorm = None, k = 5)
embedding_load_parquet(dirin = "df.parquet", nmax = 500)
embedding_model_to_parquet(model_vector_path = "model.vec", nmax = 500)
embedding_to_parquet(dirin = None, dirout = None, skip = 0, nmax = 10 ** 8, is_linevalid_fun=Nonedirout);dirout);4)if is_linevalid_fun is None = Nonedirout);dirout);4)if is_linevalid_fun is None:  #### Validate linew):)
generate_random_bigrams(n_words = 100, word_length = 4, bigrams_length = 5000)
gensim_model_check(model_path)
gensim_model_load(dirin, modeltype = 'fastext', **kw)
gensim_model_train_save(model_or_path = None, dirinput = 'lee_background.cor', dirout = "./modelout/model", epochs = 1, pars: dict  =  None, **kw)
help()
np_intersec(va, vb)
np_str_to_array(vv, l2_norm = True, mdim = 200)
test_all()
test_gensim1()
text_generate_random_sentences(dirout = None, n_sentences = 5, )
text_preprocess(sentence, lemmatizer, stop_words)
write_random_sentences_from_bigrams_to_file(dirout, n_sentences = 14000)



utilmy/nlp/util_nlp.py
-------------------------functions----------------------
add_detect_lang(data, column)
add_encode_variable(dtf, column)
add_ner_spacy(data, column, ner = None, lst_tag_filter = None, grams_join = "_", create_features = True)
add_preprocessed_text(data, column, lst_regex = None, punkt = False, lower = False, slang = False, lst_stopwords = None, stemm = False, lemm = False, remove_na = True)
add_sentiment(data, column, algo = "vader", sentiment_range = (-1, 1)
add_text_length(data, column)
add_word_freq(data, column, lst_words, freq = "count")
bart(corpus, ratio = 0.2)
create_ngrams_detectors(corpus, grams_join = " ", lst_common_terms = [], min_count = 5, top = 10, figsize = (10, 7)
create_stopwords(lst_langs = ["english"], lst_add_words = [], lst_keep_words = [])
display_string_matching(a, b, both = True, sentences = True, titles = [])
dtf_partitioning(dtf, y, test_size = 0.3, shuffle = False)
embedding_bert(x, tokenizer = None, nlp = None, log = False)
embedding_w2v(x, nlp = None, value_na = 0)
evaluate_multi_classif(y_test, predicted, predicted_prob, figsize = (15, 5)
evaluate_summary(y_test, predicted)
explainer_attention(model, tokenizer, txt_instance, lst_ngrams_detectors = [], top = 5, figsize = (5, 3)
explainer_lime(model, y_train, txt_instance, top = 10)
explainer_shap(model, X_train, X_instance, dic_vocabulary, class_names, top = 10)
explainer_similarity_classif(tokenizer, nlp, dic_clusters, txt_instance, token_level = False, top = 5, figsize = (20, 10)
features_selection(X, y, X_names, top = None, print_top = 10)
fit_bert_classif(X_train, y_train, X_test, encode_y = False, dic_y_mapping = None, model = None, epochs = 100, batch_size = 64)
fit_bow(corpus, vectorizer = None, vocabulary = None)
fit_dl_classif(X_train, y_train, X_test, encode_y = False, dic_y_mapping = None, model = None, weights = None, epochs = 100, batch_size = 256)
fit_lda(corpus, ngrams = 1, grams_join = " ", lst_ngrams_detectors = [], n_topics = 3, figsize = (10, 7)
fit_ml_classif(X_train, y_train, X_test, vectorizer = None, classifier = None)
fit_seq2seq(X_train, y_train, X_embeddings, y_embeddings, model = None, build_encoder_decoder = True, epochs = 100, batch_size = 64)
fit_w2v(corpus, ngrams = 1, grams_join = " ", lst_ngrams_detectors = [], min_count = 1, size = 300, window = 20, sg = 1, epochs = 100)
get_similar_words(lst_words, top, nlp = None)
ner_displacy(txt, ner = None, lst_tag_filter = None, title = None, serve = False)
plot_distributions(dtf, x, max_cat = 20, top = None, y = None, bins = None, figsize = (10, 5)
plot_w2v(lst_words = None, nlp = None, plot_type = "2d", top = 20, annotate = True, figsize = (10, 5)
plot_w2v_cluster(dic_words = None, nlp = None, plot_type = "2d", annotate = True, figsize = (10, 5)
plot_wordcloud(corpus, max_words = 150, max_font_size = 35, figsize = (10, 10)
predict_seq2seq(X_test, encoder_model, decoder_model, fitted_tokenizer, special_tokens = ("<START>", "<END>")
predict_similarity_classif(X, dic_y)
retrain_ner_spacy(train_data, output_dir, model = "blank", n_iter = 100)
sparse2dtf(X, dic_vocabulary, X_names, prefix = "")
tags_freq(tags, top = 30, figsize = (10, 5)
text2seq(corpus, ngrams = 1, grams_join = " ", lst_ngrams_detectors = [], fitted_tokenizer = None, top = None, oov = None, maxlen = None)
textrank(corpus, ratio = 0.2)
tokenize_bert(corpus, tokenizer = None, maxlen = None)
utils_bert_embedding(txt, tokenizer, nlp, log = False)
utils_cosine_sim(a, b, nlp = None)
utils_lst_count(lst, top = None)
utils_ner_features(lst_dics_tuples, tag)
utils_ner_text(txt, ner = None, lst_tag_filter = None, grams_join = "_")
utils_plot_keras_training(training)
utils_preprocess_ngrams(corpus, ngrams = 1, grams_join = " ", lst_ngrams_detectors = [])
utils_preprocess_text(txt, lst_regex = None, punkt = True, lower = True, slang = True, lst_stopwords = None, stemm = False, lemm = True)
utils_string_matching(a, lst_b, threshold = None, top = None)
vlookup(lst_left, lst_right, threshold = 0.7, top = 1)
vocabulary_embeddings(dic_vocabulary, nlp = None)
word_clustering(corpus, nlp = None, ngrams = 1, grams_join = " ", lst_ngrams_detectors = [], n_clusters = 3)
word_freq(corpus, ngrams = [1, 2, 3], top = 10, figsize = (10, 7)



utilmy/nlp/util_rank.py
-------------------------functions----------------------
rank_biased_overlap(list1, list2, p = 0.9)
rank_topk_kendall(a:list, b:list, topk = 5, p = 0)
rbo_find_p()

-------------------------methods----------------------
RankingSimilarity.__init__(self, S: Union[List, np.ndarray], T: Union[List, np.ndarray], verbose = False)
RankingSimilarity._bound_range(self, value: float)
RankingSimilarity.assert_p(self, p: float)
RankingSimilarity.rbo(self, k: Optional[float]  =  None, p: float  =  1.0, ext: bool  =  False)
RankingSimilarity.rbo_ext(self, p = 0.98)
RankingSimilarity.top_weightness(self, p: Optional[float]  =  None, d: Optional[int]  =  None)


utilmy/nnumpy.py
-------------------------functions----------------------
is_float(x)
is_int(x)
np_add_remove(set_, to_remove, to_add)
np_list_intersection(l1, l2)
test0()
test1()
to_datetime(x)
to_dict(**kw)
to_float(x, valdef = -1)
to_int(x, valdef = -1)
to_timeunix(datex = "2018-01-16")

-------------------------methods----------------------
dict_to_namespace.__init__(self, d)


utilmy/oos.py
-------------------------functions----------------------
help()
is_float(x)
is_int(x)
log10(*s, nmax = 60)
log5(*s)
log_trace(msg = "", dump_path = "", globs = None)
np_add_remove(set_, to_remove, to_add)
np_list_intersection(l1, l2)
os_clean_memory(varlist, globx)
os_copy_safe(dirin = None, dirout = None, nlevel = 5, nfile = 5000, logdir = "./", pattern = "*", exclude = "", force = False, sleep = 0.5, cmd_fallback = "", verbose = Trueimport shutil, time, os, globflist = [] ; dirinj = dirinnlevel) =  [] ; dirinj = dirinnlevel):)
os_cpu()
os_file_check(fp)
os_file_replacestring(findstr, replacestr, some_dir, pattern = "*.*", dirlevel = 1)
os_get_function_name()
os_getcwd()
os_import(mod_name = "myfile.config.model", globs = None, verbose = True)
os_makedirs(dir_or_file)
os_memory()
os_merge_safe(dirin_list = None, dirout = None, nlevel = 5, nfile = 5000, nrows = 10**8, cmd_fallback  =  "umount /mydrive/  && mount /mydrive/  ", sleep = 0.3)
os_path_size(path  =  '.')
os_path_split(fpath:str = "")
os_platform_ip()
os_platform_os()
os_removedirs(path, verbose = False)
os_search_content(srch_pattern = None, mode = "str", dir1 = "", file_pattern = "*.*", dirlevel = 1)
os_sizeof(o, ids, hint = " deep_getsizeof(df_pd, set()
os_sleep_cpu(cpu_min = 30, sleep = 10, interval = 5, msg =  "", verbose = True)
os_system(ll, logfile = None, sleep_sec = 10)
os_system_list(ll, logfile = None, sleep_sec = 10)
os_to_file(txt = "", filename = "ztmp.txt", mode = 'a')
os_variable_check(ll, globs = None, do_terminate = True)
os_variable_exist(x, globs, msg = "")
os_variable_init(ll, globs)
os_walk(path, pattern = "*", dirlevel = 50)
print_everywhere()
profiler_start()
profiler_stop()
test0()
test1()
test2()
test4()
test5()
test_all()
to_datetime(x)
to_dict(**kw)
to_float(x)
to_int(x)
to_timeunix(datex = "2018-01-16")
z_os_search_fast(fname, texts = None, mode = "regex/str")

-------------------------methods----------------------
dict_to_namespace.__init__(self, d)
toFileSafe.__init__(self, fpath)
toFileSafe.log(self, msg)
toFileSafe.w(self, msg)
toFileSafe.write(self, msg)


utilmy/parallel.py
-------------------------functions----------------------
help()
log(*s, **kw)
log2(*s, **kw)
multiproc_run(fun_async, input_list: list, n_pool = 5, start_delay = 0.1, verbose = True, input_fixed:dict = None, npool = None, **kw)
multiproc_tochunk(flist, npool = 2)
multithread_run(fun_async, input_list: list, n_pool = 5, start_delay = 0.1, verbose = True, input_fixed:dict = None, npool = None, **kw)
multithread_run_list(**kwargs)
pd_apply_parallel(df, fun_apply = None, npool = 5, verbose = True)
pd_groupby_parallel(df, colsgroup = None, fun_apply = None, npool: int  =  1, **kw, )
pd_groupby_parallel2(df, colsgroup = None, fun_apply = None, npool: int  =  1, **kw, )
pd_random(nrows = 1000, ncols =  5)
pd_read_file(path_glob = "*.pkl", ignore_index = True, cols = None, verbose = False, nrows = -1, nfile = 1000000, concat_sort = True, n_pool = 1, npool = None, drop_duplicates = None, col_filter = None, col_filter_val = None, dtype_reduce = None, fun_apply = None, **kw)
pd_read_file2(path_glob = "*.pkl", ignore_index = True, cols = None, verbose = False, nrows = -1, nfile = 1000000, concat_sort = True, n_pool = 1, npool = None, drop_duplicates = None, col_filter = None, col_filter_val = None, dtype_reduce = None, fun_apply = None, **kw)
test0()
test_fun_run(list_vars, const = 1, const2 = 1)
test_fun_sum(group, name = None)
test_fun_sum2(list_vars, const = 1, const2 = 1)
test_fun_sum_inv(group, name = None)
test_pdreadfile()
test_run_multithread(thread_name, num, string)
test_run_multithread2(thread_name, arg)
test_sum(x)
z_pd_read_file3(path_glob = "*.pkl", ignore_index = True, cols = None, verbose = False, nrows = -1, concat_sort = True, n_pool = 1, npool = None, drop_duplicates = None, col_filter = None, col_filter_val = None, dtype_reduce = None, **kw)
ztest1()
ztest2()
zz_pd_groupby_parallel5(df, colsgroup = None, fun_apply = None, npool = 5, verbose = False, **kw)
zz_pd_read_file3(path_glob = "*.pkl", ignore_index = True, cols = None, nrows = -1, concat_sort = True, n_pool = 1, npool = None, drop_duplicates = None, col_filter = None, col_filter_val = None, dtype_reduce = None, fun_apply = None, max_file = -1, #### apply function for each subverbose = False, **kw)



utilmy/ppandas.py
-------------------------functions----------------------
is_float(x)
is_int(x)
np_add_remove(set_, to_remove, to_add)
np_list_intersection(l1, l2)
pd_add_noise(df, level = 0.05, cols_exclude:list = [])
pd_cartesian(df1, df2)
pd_col_bins(df, col, nbins = 5)
pd_cols_unique_count(df, cols_exclude:list = [], nsample = -1)
pd_del(df, cols:list)
pd_dtype_count_unique(df, col_continuous = [])
pd_dtype_getcontinuous(df, cols_exclude:list = [], nsample = -1)
pd_dtype_reduce(dfm, int0  = 'int32', float0  =  'float32')
pd_dtype_to_category(df, col_exclude, treshold = 0.5)
pd_filter(df, filter_dict = "shop_id=11, l1_genre_id>600, l2_genre_id<80311,", verbose = False)
pd_merge(df1, df2, on = None, colkeep = None)
pd_plot_histogram(dfi, path_save = None, nbin = 20.0, q5 = 0.005, q95 = 0.995, nsample =  -1, show = False, clear = True)
pd_plot_multi(df, plot_type = None, cols_axe1:list = [], cols_axe2:list = [], figsize = (8, 4)
pd_random(nrows = 100)
pd_sample_strat(df, col, n)
pd_show(df, nrows = 100, reader = 'notepad.exe', **kw)
pd_to_file(df, filei, check = 0, verbose = True, show = 'shape', **kw)
pd_to_hiveparquet(dirin, dirout = "/ztmp_hive_parquet/df.parquet", verbose = False)
pd_to_mapdict(df, colkey = 'ranid', colval = 'item_tag', naval = '0', colkey_type = 'str', colval_type = 'str', npool = 5, nrows = 900900900, verbose = True)
to_datetime(x)
to_dict(**kw)
to_float(x)
to_int(x)
to_timeunix(datex = "2018-01-16")

-------------------------methods----------------------
dict_to_namespace.__init__(self, d)


utilmy/prepro/__init__.py


utilmy/prepro/prepro.py
-------------------------functions----------------------
_pd_colnum(df, col, pars)
_pd_colnum_fill_na_median(df, col, pars)
log(*s)
log2(*s)
log3(*s)
log4(*s, n = 0, m = 1)
log4_pd(name, df, *s)
os_convert_topython_code(txt)
pd_col_atemplate(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_col_genetic_transform(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_colcat_bin(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_colcat_encoder_generic(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_colcat_minhash(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_colcat_to_onehot(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_colcross(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_coldate(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_colnum_bin(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_colnum_binto_onehot(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_colnum_normalize(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_colnum_quantile_norm(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_coly(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_coly_clean(df: pd.DataFrame, col: list = None, pars: dict = None)
prepro_load(prefix, pars)
prepro_save(prefix, pars, df_new, cols_new, prepro)
save_json(js, pfile, mode = 'a')
test()



utilmy/prepro/prepro_rec.py
-------------------------functions----------------------
_preprocess_criteo(df, **kw)
_preprocess_movielens(df, **kw)



utilmy/prepro/prepro_text.py
-------------------------functions----------------------
log(*s, n = 0, m = 1)
log_pd(df, *s, n = 0, m = 1)
logs(*s)
nlp_get_stopwords()
pd_coltext(df, col, stopwords =  None, pars = None)
pd_coltext_clean(df, col, stopwords =  None, pars = None)
pd_coltext_universal_google(df, col, pars = {})
pd_coltext_wordfreq(df, col, stopwords, ntoken = 100)



utilmy/prepro/prepro_tseries.py
-------------------------functions----------------------
log(*s)
log2(*s)
log3(*s)
logd(*s, n = 0, m = 0)
m5_dataset()
pd_prepro_custom(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_prepro_custom2(df: pd.DataFrame, cols: list = None, pars: dict = None)
pd_ts_autoregressive(df: pd.DataFrame, cols: list = None, pars: dict = None)
pd_ts_date(df: pd.DataFrame, cols: list = None, pars: dict = None)
pd_ts_deltapy_generic(df: pd.DataFrame, cols: list = None, pars: dict = None)
pd_ts_difference(df: pd.DataFrame, cols: list = None, pars: dict = None)
pd_ts_groupby(df: pd.DataFrame, cols: list = None, pars: dict = None)
pd_ts_lag(df: pd.DataFrame, cols: list = None, pars: dict = None)
pd_ts_onehot(df: pd.DataFrame, cols: list = None, pars: dict = None)
pd_ts_rolling(df: pd.DataFrame, cols: list = None, pars: dict = None)
pd_ts_tsfresh_features(df: pd.DataFrame, cols: list = None, pars: dict = None)
test_deltapy_all()
test_deltapy_all2()
test_deltapy_get_method(df)
test_get_sampledata(url="https = "https://github.com/firmai/random-assets-two/raw/master/numpy/tsla.csv")
test_prepro_v1()



utilmy/prepro/run_feature_profile.py
-------------------------functions----------------------
log(*s, n = 0, m = 0)
run_profile(path_data = None, path_output = "data/out/ztmp/", n_sample = 5000)



utilmy/prepro/util_feature.py
-------------------------functions----------------------
col_extractname(col_onehot)
col_remove(cols, colsremove, mode = "exact")
estimator_boostrap_bayes(err, alpha = 0.05, )
estimator_bootstrap(err, custom_stat = None, alpha = 0.05, n_iter = 10000)
estimator_std_normal(err, alpha = 0.05, )
feature_correlation_cat(df, colused)
feature_importance_perm(clf, Xtrain, ytrain, cols, n_repeats = 8, scoring = 'neg_root_mean_squared_error', show_graph = 1)
feature_selection_multicolinear(df, threshold = 1.0)
fetch_dataset(url_dataset, path_target = None, file_target = None)
fetch_spark_koalas(path_data_x, path_data_y = '', colid = "jobId", n_sample = -1)
load(name, path)
load_dataset(path_data_x, path_data_y = '', colid = "jobId", n_sample = -1)
load_features(name, path)
load_function_uri(uri_name="myfolder/myfile.py = "myfolder/myfile.py::myFunction")
log(*s, n = 0, m = 1, **kw)
log2(*s, **kw)
log3(*s, **kw)
metrics_eval(metric_list = ["mean_squared_error"], ytrue = None, ypred = None, ypred_proba = None, return_dict = False)
np_conv_to_one_col(np_array, sep_char = "_")
os_get_function_name()
os_getcwd()
pa_read_file(path =   'folder_parquet/', cols = None, n_rows = 1000, file_start = 0, file_end = 100000, verbose = 1, )
pa_write_file(df, path =   'folder_parquet/', cols = None, n_rows = 1000, partition_cols = None, overwrite = True, verbose = 1, filesystem  =  'hdfs')
params_check(pars, check_list, name = "")
pd_col_fillna(dfref, colname = None, method = "frequent", value = None, colgroupby = None, return_val = "dataframe,param", )
pd_col_filter(df, filter_val = None, iscol = 1)
pd_col_merge_onehot(df, colname)
pd_col_to_num(df, colname = None, default = np.nan)
pd_col_to_onehot(dfref, colname = None, colonehot = None, return_val = "dataframe,column")
pd_colcat_mapping(df, colname)
pd_colcat_mergecol(df, col_list, x0, colid = "easy_id")
pd_colcat_toint(dfref, colname, colcat_map = None, suffix = None)
pd_colcat_tonum(df, colcat = "all", drop_single_label = False, drop_fact_dict = True)
pd_colnum_normalize(df0, colname, pars, suffix = "_norm", return_val = 'dataframe,param')
pd_colnum_tocat(df, colname = None, colexclude = None, colbinmap = None, bins = 5, suffix = "_bin", method = "uniform", na_value = -1, return_val = "dataframe,param", params={"KMeans_n_clusters" = {"KMeans_n_clusters": 8, "KMeans_init": 'k-means++', "KMeans_n_init": 10,"KMeans_max_iter": 300, "KMeans_tol": 0.0001, "KMeans_precompute_distances": 'auto',"KMeans_verbose": 0, "KMeans_random_state": None,"KMeans_copy_x": True, "KMeans_n_jobs": None, "KMeans_algorithm": 'auto'})
pd_colnum_tocat_stat(df, feature, target_col, bins, cuts = 0)
pd_feature_generate_cross(df, cols, cols_cross_input = None, pct_threshold = 0.2, m_combination = 2)
pd_pipeline_apply(df, pipeline)
pd_read_file(path_glob = "*.pkl", ignore_index = True, cols = None, verbose = False, nrows = -1, concat_sort = True, n_pool = 1, drop_duplicates = None, col_filter = None, col_filter_val = None, **kw)
pd_stat_correl_pair(df, coltarget = None, colname = None)
pd_stat_dataset_shift(dftrain, dftest, colused, nsample = 10000, buckets = 5, axis = 0)
pd_stat_datashift_psi(expected, actual, buckettype = 'bins', buckets = 10, axis = 0)
pd_stat_distribution_colnum(df, nrows = 2000, verbose = False)
pd_stat_histogram(df, bins = 50, coltarget = "diff")
pd_stat_pandas_profile(df, savefile = "report.html", title = "Pandas Profile")
pd_stat_shift_changes(df, target_col, features_list = 0, bins = 10, df_test = 0)
pd_stat_shift_trend_changes(df, feature, target_col, threshold = 0.03)
pd_stat_shift_trend_correlation(df, df_test, colname, target_col)
save(df, name, path = None)
save_features(df, name, path = None)
save_list(path, name_list, glob)
test_get_classification_data(name = None)
test_heteroscedacity(y, y_pred, pred_value_only = 1)
test_mutualinfo(error, Xtest, colname = None, bins = 5)
test_normality(error, distribution = "norm", test_size_limit = 5000)

-------------------------methods----------------------
dict2.__init__(self, d)


utilmy/spark/conda/script.py


utilmy/spark/main.py
-------------------------functions----------------------
config_default()
config_getdefault()
main()
pd_to_spark_hive_format(df, dirout)
spark_init(config:dict = None, appname = 'app1', local = "local[*]")
test()



utilmy/spark/script/hadoopVersion.py


utilmy/spark/script/pysparkTest.py
-------------------------functions----------------------
inside(p)



utilmy/spark/setup.py


utilmy/spark/src/__init__.py


utilmy/spark/src/afpgrowth/main.py


utilmy/spark/src/functions/GetFamiliesFromUserAgent.py
-------------------------functions----------------------
getall_families_from_useragent(ua_string)



utilmy/spark/src/tables/table_predict_session_length.py
-------------------------functions----------------------
preprocess(spark, conf, check = True)
run(spark:SparkSession, config_path: str = 'config.yaml', mode:str = 'train,pred')



utilmy/spark/src/tables/table_predict_url_unique.py
-------------------------functions----------------------
preprocess(spark, conf, check = True)
run(spark:SparkSession, config_path: str = 'config.yaml', mode:str = 'train,pred')



utilmy/spark/src/tables/table_predict_volume.py
-------------------------functions----------------------
model_predict(df:pd.DataFrame, conf_model:dict, verbose:bool = True)
model_train(df:object, conf_model:dict, verbose:bool = True)
preprocess(spark, conf, check = True)
run(spark:SparkSession, config_path: str = 'config.yaml')



utilmy/spark/src/tables/table_user_log.py
-------------------------functions----------------------
create_userid(userlogDF:pyspark.sql.DataFrame)
run(spark:SparkSession, config_name:str)



utilmy/spark/src/tables/table_user_session_log.py
-------------------------functions----------------------
run(spark:SparkSession, config_name = 'config.yaml')



utilmy/spark/src/tables/table_user_session_stats.py
-------------------------functions----------------------
run(spark:SparkSession, config_name: str = 'config.yaml')



utilmy/spark/src/util_hadoop.py
-------------------------functions----------------------
hdfs_down(from_dir = "", to_dir = "", verbose = False, n_pool = 1, **kw)



utilmy/spark/src/util_models.py
-------------------------functions----------------------
Predict(spark, df_m:pyspark.sql.DataFrame, features:list, regressor:str, path:str = None, conf_model:dict = None)
TimeSeriesSplit(df_m:pyspark.sql.DataFrame, splitRatio:float, sparksession:object)
Train(spark, df_m:pyspark.sql.DataFrame, features:list, regressor:str, path:str = None, conf_model:dict = None)
os_makedirs(path:str)



utilmy/spark/src/util_spark.py


utilmy/spark/src/utils.py
-------------------------functions----------------------
config_load(config_path:str)
log()
log()
log2(*s)
log3(*s)
log_sample(*s)
logger_setdefault()
spark_check(df:pyspark.sql.DataFrame, conf:dict = None, path:str = "", nsample:int = 10, save = True, verbose = True, returnval = False)

-------------------------methods----------------------
to_namespace.__init__(self, d)


utilmy/spark/tests/__init__.py


utilmy/spark/tests/conftest.py
-------------------------functions----------------------
config()
spark_session(config: dict)



utilmy/spark/tests/test_common.py
-------------------------functions----------------------
assert_equal_spark_df(expected_sorted_df: DataFrame, actual_sorted_df: DataFrame, df_name: str)
assert_equal_spark_df_schema(expected_schema: [tuple], actual_schema: [tuple], df_name: str)
assert_equal_spark_df_sorted(expected_sorted_df: DataFrame, actual_sorted_df: DataFrame, df_name: str)



utilmy/spark/tests/test_functions.py
-------------------------functions----------------------
test_getall_families_from_useragent(spark_session: SparkSession)



utilmy/spark/tests/test_table_user_log.py
-------------------------functions----------------------
test_table_user_log_run(spark_session: SparkSession, config: dict)



utilmy/spark/tests/test_table_user_session_log.py
-------------------------functions----------------------
test_table_user_session_log(spark_session: SparkSession)
test_table_user_session_log_run(spark_session: SparkSession)
test_table_usersession_log_stats(spark_session: SparkSession, config: dict)



utilmy/spark/tests/test_table_user_session_stats.py
-------------------------functions----------------------
test_table_user_session_stats(spark_session: SparkSession)
test_table_user_session_stats_ip(spark_session: SparkSession, config: dict)
test_table_user_session_stats_run(spark_session: SparkSession)



utilmy/spark/tests/test_table_volume_predict.py
-------------------------functions----------------------
test_preprocess(spark_session: SparkSession, config: dict)



utilmy/spark/tests/test_utils.py
-------------------------functions----------------------
test_spark_check(spark_session: SparkSession, config: dict)



utilmy/tabular.py
-------------------------functions----------------------
estimator_boostrap_bayes(err, alpha = 0.05, )
estimator_bootstrap(err, custom_stat = None, alpha = 0.05, n_iter = 10000)
estimator_std_normal(err, alpha = 0.05, )
help()
log(*s)
np_col_extractname(col_onehot)
np_conv_to_one_col(np_array, sep_char = "_")
np_list_remove(cols, colsremove, mode = "exact")
pd_data_drift_detect_alibi(df:pd.DataFrame, ### Reference datasetdf_new:pd.DataFrame, ### Test dataset to be checkedmethod:str = "'regressoruncertaintydrift','classifieruncertaintydrift','ksdrift','mmddrift','learnedkerneldrift','chisquaredrift','tabulardrift', 'classifierdrift','spotthediffdrift'", backend:str = 'tensorflow,pytorch', model = None, ### Pre-trained modelp_val = 0.05, **kwargs)
pd_stat_correl_pair(df, coltarget = None, colname = None)
pd_stat_distribution_colnum(df, nrows = 2000, verbose = False)
pd_stat_histogram(df, bins = 50, coltarget = "diff")
pd_stat_pandas_profile(df, savefile = "report.html", title = "Pandas Profile")
pd_stat_shift_changes(df, target_col, features_list = 0, bins = 10, df_test = 0)
pd_stat_shift_trend_changes(df, feature, target_col, threshold = 0.03)
pd_stat_shift_trend_correlation(df, df_test, colname, target_col)
pd_to_scipy_sparse_matrix(df)
pd_train_test_split_time(df, test_period  =  40, cols = None, coltime  = "time_key", sort = True, minsize = 5, n_sample = 5, verbose = False)
test0()
test1()
test3()
test_all()
test_anova(df, col1, col2)
test_heteroscedacity(y, y_pred, pred_value_only = 1)
test_hypothesis(df_obs, df_ref, method = '', **kw)
test_multiple_comparisons(data: pd.DataFrame, label = 'y', adjuster = True)
test_mutualinfo(error, Xtest, colname = None, bins = 5)
test_normality(df, column, test_type)
test_normality2(df, column, test_type)
test_plot_qqplot(df, col_name)
y_adjuster_log(y_true, y_pred_log, error_func, **kwargs)



utilmy/tabular/util_drift.py


utilmy/templates/__init__.py


utilmy/templates/cli.py
-------------------------functions----------------------
run_cli()
template_copy(name, out_dir)
template_show()



utilmy/templates/templist/pypi_package/mygenerator/__init__.py


utilmy/templates/templist/pypi_package/mygenerator/dataset.py
-------------------------functions----------------------
dataset_build_meta_mnist(path: Optional[Union[str, pathlib.Path]]  =  None, get_image_fn = None, meta = None, image_suffix = "*.png", **kwargs, )

-------------------------methods----------------------
ImageDataset.__init__(self, path: Optional[Union[str, pathlib.Path]]  =  None, get_image_fn = None, meta = None, image_suffix = "*.png", **kwargs, )
ImageDataset.__len__(self)
ImageDataset.get_image_only(self, idx: int)
ImageDataset.get_label_list(self, label: Any)
ImageDataset.get_sample(self, idx: int)
ImageDataset.read_image(self, filepath_or_buffer: Union[str, io.BytesIO])
ImageDataset.save(self, path: str, prefix: str  =  "img", suffix: str  =  "png", nrows: int  =  -1)
NlpDataset.__init__(self, meta: pd.DataFrame)
NlpDataset.__len__(self)
NlpDataset.get_sample(self, idx: int)
NlpDataset.get_text_only(self, idx: int)
PhoneNlpDataset.__init__(self, size: int  =  1)
PhoneNlpDataset.__len__(self)
PhoneNlpDataset.get_phone_number(self, idx, islocal = False)


utilmy/templates/templist/pypi_package/mygenerator/pipeline.py
-------------------------functions----------------------
run_generate_numbers_sequence(sequence: str, min_spacing: int  =  1, max_spacing: int  =  10, image_width: int  =  280, ### image_widthoutput_path: str  =  "./", config_file: str  =  "config/config.yaml", )
run_generate_phone_numbers(num_images: int  =  10, min_spacing: int  =  1, max_spacing: int  =  10, image_width: int  =  280, output_path: str  =  "./", config_file: str  =  "config/config.yaml", )



utilmy/templates/templist/pypi_package/mygenerator/transform.py
-------------------------methods----------------------
CharToImages.__init__(self, font: dataset.ImageDataset)
CharToImages.fit(self, ds: dataset.NlpDataset)
CharToImages.fit_transform(self, ds: dataset.NlpDataset)
CharToImages.transform(self, ds: dataset.NlpDataset)
CombineImagesHorizontally.__init__(self, padding_range: Tuple[int, int], combined_width: int)
CombineImagesHorizontally.transform(self, ds: dataset.ImageDataset)
CombineImagesHorizontally.transform_sample(self, image_list: List[np.ndarray], 1, 1), combined_width = 10, min_image_width = 2, validate = True, )
ImageTransform.__init__(self)
ImageTransform.fit(self, ds: dataset.ImageDataset)
ImageTransform.fit_transform(self, ds: dataset.ImageDataset)
ImageTransform.transform(self, ds: dataset.ImageDataset)
RemoveWhitePadding.transform(self, ds: dataset.ImageDataset)
RemoveWhitePadding.transform_sample(self, image: np.ndarray)
ScaleImage.__init__(self, width: Optional[int]  =  None, height: Optional[int]  =  None, inter = cv2.INTER_AREA)
ScaleImage.transform(self, ds: dataset.ImageDataset)
ScaleImage.transform_sample(self, image, width = None, height = None, inter = cv2.INTER_AREA)
TextToImage.__init__(self, font_dir: Union[str, pathlib.Path], spacing_range: Tuple[int, int], image_width: int)
TextToImage.fit(self, ds: dataset.NlpDataset)
TextToImage.fit_transform(self, ds: dataset.NlpDataset)
TextToImage.transform(self, ds: dataset.NlpDataset)


utilmy/templates/templist/pypi_package/mygenerator/util_exceptions.py
-------------------------functions----------------------
config_load(config_path: Optional[Union[str, pathlib.Path]]  =  None)
dataset_donwload(url, path_target)
dataset_get_path(cfg: dict)
log(*s)
log2(*s)
loge(*s)
logger_setup()
logw(*s)
os_extract_archive(file_path, path = ".", archive_format = "auto")
to_file(s, filep)



utilmy/templates/templist/pypi_package/mygenerator/util_image.py
-------------------------functions----------------------
image_merge(image_list, n_dim, padding_size, max_height, total_width)
image_read(filepath_or_buffer: Union[str, io.BytesIO])
image_remove_extra_padding(img, inverse = False, removedot = True)
image_resize(image, width = None, height = None, inter = cv2.INTER_AREA)
padding_generate(paddings_number: int  =  1, min_padding: int  =  1, max_padding: int  =  1)



utilmy/templates/templist/pypi_package/mygenerator/utils.py
-------------------------functions----------------------
config_load(config_path: Optional[Union[str, pathlib.Path]]  =  None)
dataset_donwload(url, path_target)
dataset_get_path(cfg: dict)
log(*s)
log2(*s)
loge(*s)
logger_setup()
logw(*s)
os_extract_archive(file_path, path = ".", archive_format = "auto")
to_file(s, filep)



utilmy/templates/templist/pypi_package/mygenerator/validate.py
-------------------------functions----------------------
image_padding_get(img, threshold = 0, inverse = True)
image_padding_load(img_path, threshold = 15)
image_padding_validate(final_image, min_padding, max_padding)
run_image_padding_validate(min_spacing: int  =  1, max_spacing: int  =  1, image_width: int  =  5, input_path: str  =  "", inverse_image: bool  =  True, config_file: str  =  "default", **kwargs, )



utilmy/templates/templist/pypi_package/run_pipy.py
-------------------------functions----------------------
ask(question, ans = 'yes')
get_current_githash()
git_commit(message)
main(*args)
pypi_upload()
update_version(path, n = 1)

-------------------------methods----------------------
Version.__init__(self, major, minor, patch)
Version.__repr__(self)
Version.__str__(self)
Version.new_version(self, orig)
Version.parse(cls, string)
Version.stringify(self)


utilmy/templates/templist/pypi_package/setup.py
-------------------------functions----------------------
get_current_githash()



utilmy/templates/templist/pypi_package/tests/__init__.py


utilmy/templates/templist/pypi_package/tests/conftest.py


utilmy/templates/templist/pypi_package/tests/test_common.py


utilmy/templates/templist/pypi_package/tests/test_dataset.py
-------------------------functions----------------------
test_image_dataset_get_image_only()
test_image_dataset_get_label_list()
test_image_dataset_get_sampe()
test_image_dataset_len()
test_nlp_dataset_len()



utilmy/templates/templist/pypi_package/tests/test_import.py
-------------------------functions----------------------
test_import()



utilmy/templates/templist/pypi_package/tests/test_pipeline.py
-------------------------functions----------------------
test_generate_phone_numbers(tmp_path)



utilmy/templates/templist/pypi_package/tests/test_transform.py
-------------------------functions----------------------
create_font_files(font_dir)
test_chars_to_images_transform()
test_combine_images_horizontally_transform()
test_scale_image_transform()
test_text_to_image_transform(tmp_path)



utilmy/templates/templist/pypi_package/tests/test_util_image.py
-------------------------functions----------------------
create_blank_image(width, height, rgb_color = (0, 0, 0)
test_image_merge()
test_image_read(tmp_path)
test_image_remove_extra_padding()
test_image_resize()



utilmy/templates/templist/pypi_package/tests/test_validate.py
-------------------------functions----------------------
test_image_padding_get()



utilmy/text.py
-------------------------functions----------------------
help()
help_get_codesource(func)
log(*s)
pd_text_getcluster(df:pd.DataFrame, col:str = 'col', threshold = 0.5, num_perm:int = 5, npool = 1, chunk  =  100000)
pd_text_hash_create_lsh(df, col, sep = " ", threshold = 0.7, num_perm = 10, npool = 1, chunk  =  20000)
pd_text_similarity(df: pd.DataFrame, cols = [], algo = '')
test()
test()
test_all()
test_lsh()



utilmy/tseries/util_tseries.py


utilmy/util_download.py


utilmy/util_sampling.py
-------------------------functions----------------------
reservoir_sampling(src, nsample, temp_fac = 1.5, rs = None)
test()



utilmy/util_zip.py
-------------------------functions----------------------
dataset_donwload(url, path_target)
dataset_get_path(cfg: dict)
dir_size(dirin = "mypath", dirout = "./save.txt")
gzip(dirin = '/mydir', dirout = "./")
os_extract_archive(file_path, path = ".", archive_format = "auto")
to_file(s, filep)
unzip(in_dir, out_dir)



utilmy/utilmy.py
-------------------------functions----------------------
git_current_hash(mode = 'full')
git_repo_root()
glob_glob(dirin, nfile = 1000)
help_create(modulename = 'utilmy.nnumpy', prefixs = None)
import_function(fun_name = None, module_name = None)
load(to_file = "")
pd_generate_data(ncols = 7, nrows = 100)
pd_getdata(verbose = True)
pd_random(ncols = 7, nrows = 100)
save(dd, to_file = "", verbose = False)
test_all()

-------------------------methods----------------------
Session.__init__(self, dir_session = "ztmp/session/", )
Session.load(self, name, glob:dict = None, tag = "")
Session.load_session(self, folder, globs = None)
Session.save(self, name, glob = None, tag = "")
Session.save_session(self, folder, globs, tag = "")
Session.show(self)


utilmy/utils.py
-------------------------functions----------------------
config_load(config_path: Optional[Union[str, pathlib.Path]]  =  None)
dataset_donwload(url, path_target)
log(*s)
log2(*s)
loge(*s)
logger_setup()
logw(*s)
os_extract_archive(file_path, path = ".", archive_format = "auto")
test0()
test1()
test_all()
to_file(s, filep)



utilmy/viz/__init__.py


utilmy/viz/embedding.py
-------------------------functions----------------------
embedding_load_parquet(path = "df.parquet", nmax  =  500)
embedding_load_word2vec(model_vector_path = "model.vec", nmax  =  500)
log(*s)
run(dir_in = "in/model.vec", dir_out = "ztmp/", nmax = 100)
tokenize_text(text)

-------------------------methods----------------------
vizEmbedding.__init__(self, path = "myembed.parquet", num_clusters = 5, sep = ";", config:dict = None)
vizEmbedding.create_clusters(self, after_dim_reduction = True)
vizEmbedding.create_visualization(self, dir_out = "ztmp/", mode = 'd3', cols_label = None, show_server = False, **kw)
vizEmbedding.dim_reduction(self, mode = "mds", col_embed = 'embed', ndim = 2, nmax =  5000, dir_out = None)
vizEmbedding.draw_hiearchy(self)
vizEmbedding.run_all(self, mode = "mds", col_embed = 'embed', ndim = 2, nmax =  5000, dir_out = "ztmp/")


utilmy/viz/template1.py


utilmy/viz/util_map.py


utilmy/viz/vizhtml.py
-------------------------functions----------------------
colormap_get_names()
help_get_codesource(func)
html_show(html_code, verbose = True)
html_show_chart_highchart(html_code, verbose = True)
images_to_html(dir_input = "*.png", title = "", verbose = False)
mlpd3_add_tooltip(fig, points, labels)
mpld3_server_start()
pd_plot_highcharts(df)
pd_plot_histogram_highcharts(df:pd.DataFrame, colname:str = None, binsNumber = None, binWidth = None, title:str = "", xaxis_label:str =  "x-axis", yaxis_label:str = "y-axis", cfg:dict = {}, mode = 'd3', save_img = "", show = False, verbose = True, **kw)
pd_plot_histogram_matplot(df:pd.DataFrame, col: str = '', colormap:str = 'RdYlBu', title: str = '', nbin = 20.0, q5 = 0.005, q95 = 0.995, nsample = -1, save_img: str = "", xlabel: str = None, ylabel: str = None, verbose = True, **kw)
pd_plot_network(df:pd.DataFrame, cola: str = 'col_node1', colb: str = 'col_node2', coledge: str = 'col_edge', colweight: str = "weight", html_code:bool  =  True)
pd_plot_scatter_get_data(df0:pd.DataFrame, colx: str = None, coly: str = None, collabel: str = None, colclass1: str = None, colclass2: str = None, nmax: int = 20000, **kw)
pd_plot_scatter_highcharts(df0:pd.DataFrame, colx:str = None, coly:str = None, collabel: str = None, colclass1: str = None, colclass2: str = None, colclass3: str = None, nsample = 10000, cfg:dict = {}, mode = 'd3', save_img = '', verbose = True, **kw)
pd_plot_scatter_matplot(df:pd.DataFrame, colx: str = None, coly: str = None, collabel: str = None, colclass1: str = None, colclass2: str = None, cfg: dict  =  {}, mode = 'd3', save_path: str = '', verbose = True, **kw)
pd_plot_tseries_highcharts(df, coldate:str = None, date_format:str = '%m/%d/%Y', coly1:list  = [], coly2:list  = [], figsize:tuple  =   None, title:str = None, xlabel:str = None, y1label:str = None, y2label:str = None, cfg:dict = {}, mode = 'd3', save_img = "", verbose = True, **kw)
pd_plot_tseries_matplot(df:pd.DataFrame, plot_type: str = None, coly1: list  =  [], coly2: list  =  [], 8, 4), spacing = 0.1, verbose = True, **kw))
test1(verbose = False)
test2(verbose = False)
zz_css_get_template(css_name:str =  "A4_size")
zz_pd_plot_histogram_highcharts_old(df, col, figsize = None, title = None, cfg:dict = {}, mode = 'd3', save_img = '')
zz_test_get_random_data(n = 100)

-------------------------methods----------------------
mpld3_TopToolbar.__init__(self)


utilmy/viz/zarchive/__init__.py


utilmy/viz/zarchive/toptoolbar.py
-------------------------methods----------------------
TopToolbar.__init__(self)


utilmy/zml/__init__.py


utilmy/zml/core_deploy.py
-------------------------functions----------------------
load_arguments()



utilmy/zml/core_run.py
-------------------------functions----------------------
check(config='outlier_predict.py = 'outlier_predict.py::titanic_lightgbm')
data_profile(config = '')
data_profile2(config = '')
deploy()
get_config_path(config = '')
get_global_pars(config_uri = "")
hyperparam_wrapper(config_full = "", ntrials = 2, n_sample = 5000, debug = 1, path_output          =  "data/output/titanic1/", path_optuna_storage  =  'data/output/optuna_hyper/optunadb.db', metric_name = 'accuracy_score', mdict_range = None)
log(*s)
predict(config = '', nsample = None)
preprocess(config = '', nsample = None)
train(config = '', nsample = None)
train_sampler(config = '', nsample = None)
transform(config = '', nsample = None)



utilmy/zml/core_test.py
-------------------------functions----------------------
json_load(path)
log_info_repo(arg = None)
log_remote_push(name = None)
log_remote_start(arg = None)
log_separator(space = 140)
os_bash(cmd)
os_file_current_path()
os_system(cmd, dolog = 1, prefix = "", dateformat='+%Y-%m-%d_%H = '+%Y-%m-%d_%H:%M:%S,%3N')
to_logfile(prefix = "", dateformat='+%Y-%m-%d_%H = '+%Y-%m-%d_%H:%M:%S,%3N')



utilmy/zml/datasketch_hashing.py
-------------------------functions----------------------
create_hash(df, column_name, threshold, num_perm)
find_clusters(df, column_name, threshold, num_perm)



utilmy/zml/example/classifier/classifier_adfraud.py
-------------------------functions----------------------
check()
global_pars_update(model_dict, data_name, config_name)



utilmy/zml/example/classifier/classifier_airline.py
-------------------------functions----------------------
airline_lightgbm(path_model_out = "")
check()
global_pars_update(model_dict, data_name, config_name)



utilmy/zml/example/classifier/classifier_bankloan.py
-------------------------functions----------------------
bank_lightgbm()
check()
global_pars_update(model_dict, data_name, config_name)



utilmy/zml/example/classifier/classifier_cardiff.py
-------------------------functions----------------------
cardif_lightgbm(path_model_out = "")
check()
global_pars_update(model_dict, data_name, config_name)



utilmy/zml/example/classifier/classifier_income.py
-------------------------functions----------------------
check()
global_pars_update(model_dict, data_name, config_name)
income_status_lightgbm(path_model_out = "")



utilmy/zml/example/classifier/classifier_multi.py
-------------------------functions----------------------
check()
global_pars_update(model_dict, data_name, config_name)
multi_lightgbm()



utilmy/zml/example/classifier/classifier_optuna.py
-------------------------functions----------------------
check()
global_pars_update(model_dict, data_name, config_name)
titanic_lightoptuna()



utilmy/zml/example/classifier/classifier_sentiment.py
-------------------------functions----------------------
check()
data_profile(path_data_train = "", path_model = "", n_sample =  5000)
global_pars_update(model_dict, data_name, config_name)
os_get_function_name()
predict(config = None, nsample = None)
preprocess(config = None, nsample = None)
run_all()
sentiment_bayesian_pyro(path_model_out = "")
sentiment_elasticnetcv(path_model_out = "")
sentiment_lightgbm(path_model_out = "")
train(config = None, nsample = None)



utilmy/zml/example/classifier_mlflow.py
-------------------------functions----------------------
check()
global_pars_update(model_dict, data_name, config_name)
pd_col_myfun(df = None, col = None, pars = {})
titanic_lightgbm()



utilmy/zml/example/click/online_shopping.py
-------------------------functions----------------------
check()
global_pars_update(model_dict, data_name, config_name)
online_lightgbm()



utilmy/zml/example/click/outlier_predict.py
-------------------------functions----------------------
check()
data_profile(path_data_train = "", path_model = "", n_sample =  5000)
global_pars_update(model_dict, data_name, config_name)
os_get_function_name()
titanic_pyod(path_model_out = "")



utilmy/zml/example/click/test_online_shopping.py
-------------------------functions----------------------
check()
global_pars_update(model_dict, data_name, config_name)
online_lightgbm()
pd_col_myfun(df = None, col = None, pars = {})



utilmy/zml/example/regress/regress_airbnb.py
-------------------------functions----------------------
airbnb_elasticnetcv(path_model_out = "")
airbnb_lightgbm(path_model_out = "")
global_pars_update(model_dict, data_name, config_name)
y_norm(y, inverse = True, mode = 'boxcox')



utilmy/zml/example/regress/regress_boston.py
-------------------------functions----------------------
boston_causalnex(path_model_out = "")
boston_lightgbm(path_model_out = "")
global_pars_update(model_dict, data_name, config_name)
y_norm(y, inverse = True, mode = 'boxcox')



utilmy/zml/example/regress/regress_house.py
-------------------------functions----------------------
check()
data_profile()
global_pars_update(model_dict, data_name, config_name)
house_price_elasticnetcv(path_model_out = "")
house_price_lightgbm(path_model_out = "")
predict()
preprocess()
run_all()
train()
y_norm(y, inverse = True, mode = 'boxcox')



utilmy/zml/example/regress/regress_salary.py
-------------------------functions----------------------
check()
global_pars_update(model_dict, data_name, config_name)
salary_bayesian_pyro(path_model_out = "")
salary_elasticnetcv(path_model_out = "")
salary_glm(path_model_out = "")
salary_lightgbm(path_model_out = "")
y_norm(y, inverse = True, mode = 'boxcox')



utilmy/zml/example/svd/benchmark_mf.py
-------------------------functions----------------------
daal4py_als(A, k)
daal4py_svd(A, k)
factorize(S, num_factors, lambda_reg = 1e-5, num_iterations = 20, init_std = 0.01, verbose = False, dtype = 'float32', recompute_factors = recompute_factors, *args, **kwargs)
gensim_svd(A, k)
implicit_mf(A, k)
iter_rows(S)
linear_surplus_confidence_matrix(B, alpha)
log_surplus_confidence_matrix(B, alpha, epsilon)
nmf_1(A, k)
nmf_2(A, k)
nmf_3(A, k)
nmf_4(A, k)
nmf_5(A, k)
recompute_factors(Y, S, lambda_reg, dtype = 'float32')
recompute_factors_bias(Y, S, lambda_reg, dtype = 'float32')
scipy_svd(A, K)
sklearn_randomized_svd(A, k)
sklearn_truncated_arpack_svd(A, k)
sklearn_truncated_randomized_svd(A, k)
sparsesvd_svd(A, k)
time_ns()
time_reps(func, params, reps)
wmf(A, k)

-------------------------methods----------------------
ImplicitMF.__init__(self, counts, num_factors = 40, num_iterations = 30, reg_param = 0.8)
ImplicitMF.iteration(self, user, fixed_vecs)
ImplicitMF.train_model(self)


utilmy/zml/example/svd/benchmark_mf0.py
-------------------------functions----------------------
factorize(S, num_factors, lambda_reg = 1e-5, num_iterations = 20, init_std = 0.01, verbose = False, dtype = 'float32', recompute_factors = recompute_factors, *args, **kwargs)
gensim_svd(A, k)
implicit_mf(A, k)
iter_rows(S)
linear_surplus_confidence_matrix(B, alpha)
log_surplus_confidence_matrix(B, alpha, epsilon)
nmf_1(A, k)
nmf_2(A, k)
nmf_3(A, k)
nmf_4(A, k)
nmf_5(A, k)
recompute_factors(Y, S, lambda_reg, dtype = 'float32')
recompute_factors_bias(Y, S, lambda_reg, dtype = 'float32')
scipy_svd(A, K)
sklearn_randomized_svd(A, k)
sklearn_truncated_arpack_svd(A, k)
sklearn_truncated_randomized_svd(A, k)
sparsesvd_svd(A, k)
time_ns()
time_reps(func, params, reps)
wmf(A, k)

-------------------------methods----------------------
ImplicitMF.__init__(self, counts, num_factors = 40, num_iterations = 30, reg_param = 0.8)
ImplicitMF.iteration(self, user, fixed_vecs)
ImplicitMF.train_model(self)


utilmy/zml/example/test.py
-------------------------functions----------------------
check()
data_profile(path_data_train = "", path_model = "", n_sample =  5000)
global_pars_update(model_dict, data_name, config_name)
os_get_function_name()
predict(config = None, nsample = None)
preprocess(config = None, nsample = None)
run_all()
titanic1(path_model_out = "")
train(config = None, nsample = None)



utilmy/zml/example/test_automl.py
-------------------------functions----------------------
config1()
global_pars_update(model_dict, data_name, config_name)



utilmy/zml/example/test_features.py
-------------------------functions----------------------
config1(path_model_out = "")
config2(path_model_out = "")
config3(path_model_out = "")
config4(path_model_out = "")
config9(path_model_out = "")
global_pars_update(model_dict, data_name, config_name)
pd_col_amyfun(df: pd.DataFrame, col: list = None, pars: dict = None)



utilmy/zml/example/test_hyperopt.py
-------------------------functions----------------------
global_pars_update(model_dict, data_name, config_name)
hyperparam(config_full = "", ntrials = 2, n_sample = 5000, debug = 1, path_output          =  "data/output/titanic1/", path_optuna_storage  =  'data/output/optuna_hyper/optunadb.db')
hyperparam_wrapper(config_full = "", ntrials = 2, n_sample = 5000, debug = 1, path_output          =  "data/output/titanic1/", path_optuna_storage  =  'data/output/optuna_hyper/optunadb.db', metric_name = 'accuracy_score', mdict_range = None)
post_process_fun(y)
pre_process_fun(y)
titanic1(path_model_out = "")



utilmy/zml/example/test_keras_vaemdn.py
-------------------------functions----------------------
config_sampler()
global_pars_update(model_dict, data_name, config_name)



utilmy/zml/example/test_keras_vaemdn2.py
-------------------------functions----------------------
config1()
global_pars_update(model_dict, data_name, config_name)



utilmy/zml/example/test_mkeras.py
-------------------------functions----------------------
config1()
global_pars_update(model_dict, data_name, config_name)



utilmy/zml/example/test_mkeras_dense.py
-------------------------functions----------------------
config1()
global_pars_update(model_dict, data_name, config_name)



utilmy/zml/example/titanic_gefs.py
-------------------------functions----------------------
config1()
global_pars_update(model_dict, data_name, config_name)



utilmy/zml/example/tseries/tseries_m5sales.py
-------------------------functions----------------------
custom_generate_feature_all(input_path  =  data_path, out_path = ".", input_raw_path  = ".", auxiliary_csv_path  =  None, coldrop  =  None, colindex  =  None, merge_cols_mapping  =  None, coldate  =  None, colcat  =  None, colid  =  None, coly  =  None, max_rows  =  10)
custom_get_colsname(colid, coly)
custom_rawdata_merge(out_path = 'out/', max_rows = 10)
featurestore_filter_features(mode  = "random", colid  =  None, coly  =  None)
featurestore_generate_feature(dir_in, dir_out, my_fun_features, features_group_name, input_raw_path  =  None, auxiliary_csv_path  =  None, coldrop  =  None, index_cols  =  None, merge_cols_mapping  =  None, colcat  =  None, colid = None, coly  =  None, coldate  =  None, max_rows  =  5, step_wise_saving  =  False)
featurestore_get_feature_fromcolname(path, selected_cols, colid)
featurestore_get_filelist_fromcolname(selected_cols, colid)
featurestore_get_filename(file_name, path)
featurestore_meta_update(featnames, filename, colcat)
pd_col_tocat(df, nan_cols, colcat)
pd_merge(df_list, cols_join)
pd_ts_tsfresh(df, input_raw_path, dir_out, features_group_name, auxiliary_csv_path, drop_cols, index_cols, merge_cols_mapping, cat_cols  =  None, id_cols  =  None, dep_col  =  None, coldate  =  None, max_rows  =  10)
pd_tsfresh_m5data(df_sales, dir_out, features_group_name, drop_cols, df_calendar, index_cols, merge_cols_mapping, id_cols)
pd_tsfresh_m5data_sales(df_sales, dir_out, features_group_name, drop_cols, df_calendar, index_cols, merge_cols_mapping, id_cols)
run_train(input_path  = "data/input/tseries/tseries_m5/raw", out_path = data_path, do_generate_raw = True, do_generate_feature = True, do_train = False, max_rows  =  10)
train(input_path, n_experiments  =  3, colid  =  None, coly  =  None)

-------------------------methods----------------------
FeatureStore.__init__(self)


utilmy/zml/example/tseries/tseries_retail.py
-------------------------functions----------------------
custom_generate_feature_all(input_path  =  data_path, out_path = ".", input_raw_path  = ".", auxiliary_csv_path  =  None, coldrop  =  None, colindex  =  None, merge_cols_mapping  =  None, coldate  =  None, colcat  =  None, colid  =  None, coly  =  None, max_rows  =  10)
custom_get_colsname(colid, coly)
custom_rawdata_merge(out_path = 'out/', max_rows = 10)
featurestore_filter_features(mode  = "random", colid  =  None, coly  =  None)
featurestore_generate_feature(dir_in, dir_out, my_fun_features, features_group_name, input_raw_path  =  None, auxiliary_csv_path  =  None, coldrop  =  None, index_cols  =  None, merge_cols_mapping  =  None, colcat  =  None, colid = None, coly  =  None, coldate  =  None, max_rows  =  5, step_wise_saving  =  False)
featurestore_get_feature_fromcolname(path, selected_cols, colid)
featurestore_get_filelist_fromcolname(selected_cols, colid)
featurestore_get_filename(file_name, path)
featurestore_meta_update(featnames, filename, colcat)
pd_col_tocat(df, nan_cols, colcat)
pd_merge(df_list, cols_join)
run_train(input_path  = "data/input/tseries/retail/raw", out_path = data_path, do_generate_raw = True, do_generate_feature = True, do_train = False, max_rows  =  10)
train(input_path, n_experiments  =  3, colid  =  None, coly  =  None)

-------------------------methods----------------------
FeatureStore.__init__(self)


utilmy/zml/example/tseries/tseries_sales.py
-------------------------functions----------------------
custom_generate_feature_all(input_path  =  data_path, out_path = ".", input_raw_path  = ".", auxiliary_csv_path  =  None, coldrop  =  None, colindex  =  None, merge_cols_mapping  =  None, coldate  =  None, colcat  =  None, colid  =  None, coly  =  None, max_rows  =  10)
custom_get_colsname(colid, coly)
custom_rawdata_merge(out_path = 'out/', max_rows = 10)
featurestore_filter_features(mode  = "random", colid  =  None, coly  =  None)
featurestore_generate_feature(dir_in, dir_out, my_fun_features, features_group_name, input_raw_path  =  None, auxiliary_csv_path  =  None, coldrop  =  None, index_cols  =  None, merge_cols_mapping  =  None, colcat  =  None, colid = None, coly  =  None, coldate  =  None, max_rows  =  5, step_wise_saving  =  False)
featurestore_get_feature_fromcolname(path, selected_cols, colid)
featurestore_get_filelist_fromcolname(selected_cols, colid)
featurestore_get_filename(file_name, path)
featurestore_meta_update(featnames, filename, colcat)
pd_col_tocat(df, nan_cols, colcat)
pd_merge(df_list, cols_join)
run_generate_train_data(input_path  = "data/input/tseries/retail/raw", out_path = data_path, do_generate_raw = True, do_generate_feature = True, do_train = False, max_rows  =  10)

-------------------------methods----------------------
FeatureStore.__init__(self)


utilmy/zml/example/zfraud.py
-------------------------functions----------------------
global_pars_update(model_dict, data_name, config_name)



utilmy/zml/source/bin/__init__.py


utilmy/zml/source/bin/auto_feature_AFEM/AFE.py
-------------------------functions----------------------
timer(func)

-------------------------methods----------------------
BasePath.__init__(self, pathstype, name = None)
BasePath._inversepathstype(self)
BasePath.getinversepathstype(self)
BasePath.getlastentityid(self)
BasePath.getpathentities(self)
BasePath.getpathname(self)
BasePath.getpathstype(self)
Entity.__init__(self, entity_id, dataframe, index, time_index = None, variable_types = None)
Entity.getcolumns(self, columns)
Entity.getfeatname(self)
Entity.getfeattype(self, featname)
Entity.merge(self, features, path, how = 'right')
EntitySet.__init__(self, name)
EntitySet._pathstype(self, paths)
EntitySet._search_path(self, shortpaths, targetnode, maxdepth, max_famous_son)
EntitySet.addrelationship(self, entityA, entityB, keyA, keyB)
EntitySet.collectiontransform(self, path, target)
EntitySet.draw(self)
EntitySet.entity_from_dataframe(self, entity_id, dataframe, index, time_index = None, variable_types = None)
EntitySet.getentity(self, entityid)
EntitySet.search_path(self, targetnode, maxdepth, max_famous_son)
Function.__init__(self, arg)
Generator.__init__(self, es)
Generator._layer(self, path, start_part = None, start_part_id = None)
Generator.add_compute_series(self, compute_series, start_part = None)
Generator.aggregate(self, path, function, iftimeuse  =  True, winsize = 'all', lagsize = 'last')
Generator.collect_agg(self, inputs)
Generator.defaultfunc(self, path)
Generator.layer(self, path, start_part = None, start_part_id = None)
Generator.layer_sequencal_agg(self, path, es, ngroups  =  None, njobs = 1)
Generator.layers(self, paths, start_part = None, start_part_id = None)
Generator.pathcompunation(self, pathsfunc)
Generator.pathcompute(self, cs, ngroups = 'auto', njobs = 1)
Generator.pathfilter(self, path, function, start_part = None, start_part_id = None)
Generator.reload_data(self, es)
Generator.singlepathcompunation(self, pathstype, targetfeatures, functionset)
Generator.transform(self, path, featurenames, function)
Path.__init__(self, pathstype, df, firstindex, start_time_index, lastindex, last_time_index, name = None, start_part_id = None)
Path.getfirstkey(self)
Path.getlastkey(self)
Path.getlasttimeindex(self)
Path.getpathdetail(self)
Path.getstartpartname(self)
Path.getstarttimeindex(self)


utilmy/zml/source/bin/auto_feature_AFEM/__init__.py


utilmy/zml/source/bin/column_encoder.py
-------------------------methods----------------------
MinHashEncoder.__init__(self, n_components, ngram_range = (2, 4)
MinHashEncoder.fit(self, X, y = None)
MinHashEncoder.get_unique_ngrams(self, string, ngram_range)
MinHashEncoder.minhash(self, string, n_components, ngram_range)
MinHashEncoder.transform(self, X)
OneHotEncoderRemoveOne.__init__(self, n_values = None, categorical_features = None, categories = "auto", sparse = True, dtype = np.float64, handle_unknown = "error", )
OneHotEncoderRemoveOne.transform(self, X, y = None)
PasstroughEncoder.__init__(self, passthrough = True)
PasstroughEncoder.fit(self, X, y = None)
PasstroughEncoder.transform(self, X)


utilmy/zml/source/bin/deltapy/__init__.py


utilmy/zml/source/bin/deltapy/extract.py
-------------------------functions----------------------
_embed_seq(X, Tau, D)
_embed_seq(X, Tau, D)
_embed_seq(X, Tau, D)
_estimate_friedrich_coefficients(x, m, r)
_hjorth_mobility(epochs)
_roll(a, shift)
abs_energy(x)
ar_coefficient(x, param=[{"coeff" = [{"coeff": 5, "k": 5}])
augmented_dickey_fuller(x, param=[{"attr" = [{"attr": "teststat"}])
binned_entropy(x, max_bins = 10)
c3(x, lag = 3)
cad_prob(cads, param = cad_param)
cid_ce(x, normalize)
count_above_mean(x)
detrended_fluctuation_analysis(epochs)
fft_coefficient(x, param = [{"coeff" =  [{"coeff": 10, "attr": "real"}])
find_freq(serie, param = freq_param)
fisher_information(epochs, param = fisher_param)
flux_perc(magnitude)
get_length_sequences_where(x)
gskew(x)
has_duplicate_max(x)
higuchi_fractal_dimension(epochs, param = hig_param)
hjorth_complexity(epochs)
hurst_exponent(epochs)
index_mass_quantile(x, param=[{"q" = [{"q": 0.3}])
kurtosis(x)
largest_lyauponov_exponent(epochs, param = lyaup_param)
last_location_of_maximum(x)
length(x)
linear_trend_timewise(x, param= [{"attr" =  [{"attr": "pvalue"}])
longest_strike_below_mean(x)
max_langevin_fixed_point(x, r = 3, m = 30)
mean_abs_change(x)
mean_second_derivative_central(x)
number_cwt_peaks(x, param = cwt_param)
partial_autocorrelation(x, param=[{"lag" = [{"lag": 1}])
percent_amplitude(x, param  = perc_param)
petrosian_fractal_dimension(epochs)
range_cum_s(magnitude)
set_property(key, value)
spkt_welch_density(x, param=[{"coeff" = [{"coeff": 5}])
stetson_k(x)
stetson_mean(x, param = stestson_param)
structure_func(time, param = struct_param)
svd_entropy(epochs, param = svd_param)
symmetry_looking(x, param=[{"r" = [{"r": 0.2}])
var_index(time, param = var_index_param)
variance_larger_than_standard_deviation(x)
whelch_method(data, param = whelch_param)
willison_amplitude(X, param = will_param)
wozniak(magnitude, param = woz_param)
zero_crossing_derivative(epochs, param = zero_param)



utilmy/zml/source/bin/deltapy/interact.py
-------------------------functions----------------------
autoregression(df, drop = None, settings={"autoreg_lag" = {"autoreg_lag":4})
decision_tree_disc(df, cols, depth = 4)
genetic_feat(df, num_gen = 20, num_comp = 10)
haversine_distance(row, lon = "Open", lat = "Close")
lowess(df, cols, y, f = 2. / 3., iter = 3)
muldiv(df, feature_list)
quantile_normalize(df, drop)
tech(df)



utilmy/zml/source/bin/deltapy/mapper.py
-------------------------functions----------------------
a_chi(df, drop = None, lags = 1, sample_steps = 2)
cross_lag(df, drop = None, lags = 1, components = 4)
encoder_dataset(df, drop = None, dimesions = 20)
feature_agg(df, drop = None, components = 4)
lle_feat(df, drop = None, components = 4)
neigh_feat(df, drop, neighbors = 6)
pca_feature(df, memory_issues = False, mem_iss_component = False, variance_or_components = 0.80, n_components = 5, drop_cols = None, non_linear = True)



utilmy/zml/source/bin/deltapy/transform.py
-------------------------functions----------------------
bkb(df, cols)
butter_lowpass(cutoff, fs = 20, order = 5)
butter_lowpass_filter(df, cols, cutoff, fs = 20, order = 5)
fast_fracdiff(x, cols, d)
fft_feat(df, cols)
harmonicradar_cw(df, cols, fs, fc)
infer_seasonality(train, index = 0)
initial_seasonal_components(series, slen)
initial_trend(series, slen)
instantaneous_phases(df, cols)
kalman_feat(df, cols)
modify(df, cols)
multiple_lags(df, start = 1, end = 3, columns = None)
multiple_rolling(df, windows  =  [1, 2], functions = ["mean", "std"], columns = None)
naive_dec(df, columns, freq = 2)
operations(df, features)
outlier_detect(data, col, threshold = 1, method = "IQR")
perd_feat(df, cols)
prophet_feat(df, cols, date, freq, train_size = 150)
robust_scaler(df, drop = None, quantile_range = (25, 75)
saw(df, cols)
standard_scaler(df, drop)
triple_exponential_smoothing(df, cols, slen, alpha, beta, gamma, n_preds)
windsorization(data, col, para, strategy = 'both')



utilmy/zml/source/bin/hunga_bunga/__init__.py
-------------------------methods----------------------
HungaBungaZeroKnowledge.__init__(self, brain = False, test_size  =  0.2, n_splits  =  5, random_state = None, upsample = True, scoring = None, verbose = True, normalize_x  =  True, n_jobs  = cpu_count()
HungaBungaZeroKnowledge.fit(self, X, y)
HungaBungaZeroKnowledge.predict(self, x)


utilmy/zml/source/bin/hunga_bunga/classification.py
-------------------------functions----------------------
run_all_classifiers(x, y, small  =  True, normalize_x  =  True, n_jobs = cpu_count()

-------------------------methods----------------------
HungaBungaClassifier.__init__(self, brain = False, test_size  =  0.2, n_splits  =  5, random_state = None, upsample = True, scoring = None, verbose = False, normalize_x  =  True, n_jobs  = cpu_count()
HungaBungaClassifier.fit(self, x, y)
HungaBungaClassifier.predict(self, x)


utilmy/zml/source/bin/hunga_bunga/core.py
-------------------------functions----------------------
cv_clf(x, y, test_size  =  0.2, n_splits  =  5, random_state = None, doesUpsample  =  True)
cv_reg(x, test_size  =  0.2, n_splits  =  5, random_state = None)
main_loop(models_n_params, x, y, isClassification, test_size  =  0.2, n_splits  =  5, random_state = None, upsample = True, scoring = None, verbose = True, n_jobs  = cpu_count()
timeit(klass, params, x, y)
upsample_indices_clf(inds, y)

-------------------------methods----------------------
GridSearchCVProgressBar._get_param_iterator(self)
RandomizedSearchCVProgressBar._get_param_iterator(self)


utilmy/zml/source/bin/hunga_bunga/params.py


utilmy/zml/source/bin/hunga_bunga/regression.py
-------------------------functions----------------------
gen_reg_data(x_mu = 10., x_sigma = 1., num_samples = 100, num_features = 3, y_formula = sum, y_sigma = 1.)
run_all_regressors(x, y, small  =  True, normalize_x  =  True, n_jobs = cpu_count()

-------------------------methods----------------------
HungaBungaRegressor.__init__(self, brain = False, test_size  =  0.2, n_splits  =  5, random_state = None, upsample = True, scoring = None, verbose = False, normalize_x  =  True, n_jobs  = cpu_count()
HungaBungaRegressor.fit(self, x, y)
HungaBungaRegressor.predict(self, x)


utilmy/zml/source/models/akeras/Autokeras.py
-------------------------functions----------------------
evaluate(model, data_pars = None, compute_pars = None, out_pars = None)
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kwargs)
get_config_file()
get_dataset(data_pars)
get_dataset_imbd(data_pars)
get_dataset_titanic(data_pars)
get_params(param_pars = None, **kw)
load(load_pars, config_mode = "test")
predict(model, session = None, data_pars = None, compute_pars = None, out_pars = None)
save(model, session = None, save_pars = None, config_mode = "test")
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")
test_single(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, out_pars = None)


utilmy/zml/source/models/akeras/__init__.py


utilmy/zml/source/models/akeras/armdn.py


utilmy/zml/source/models/akeras/charcnn.py
-------------------------functions----------------------
evaluate(model, session = None, data_pars = None, compute_pars = None, out_pars = None, **kw)
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
get_dataset(data_pars = None, **kw)
get_params(param_pars = {}, **kw)
load(load_pars = None)
predict(model, session = None, data_pars = None, out_pars = None, compute_pars = None, **kw)
reset_model()
save(model = None, save_pars = None, session = None)
str_to_indexes(s)
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")
tokenize(data, num_of_classes = 4)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy/zml/source/models/akeras/charcnn_zhang.py
-------------------------functions----------------------
evaluate(model, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)
fit(model, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)
get_dataset(data_pars = None, **kw)
get_params(param_pars = {}, **kw)
load(load_pars = {})
predict(model, sess = None, data_pars = {}, out_pars = {}, compute_pars = {}, **kw)
reset_model()
save(model = None, session = None, save_pars = {})
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy/zml/source/models/akeras/deepctr.py
-------------------------functions----------------------
_config_process(config)
_preprocess_criteo(df, **kw)
_preprocess_movielens(df, **kw)
config_load(data_path, file_default, config_mode)
fit(model, session = None, compute_pars = None, data_pars = None, out_pars = None, **kwargs)
get_dataset(data_pars = None, **kw)
get_params(choice = "", data_path = "dataset/", config_mode = "test", **kwargs)
metrics(ypred, ytrue = None, session = None, compute_pars = None, data_pars = None, out_pars = None, **kwargs)
path_setup(out_folder = "", sublevel = 0, data_path = "dataset/")
predict(model, session = None, compute_pars = None, data_pars = None, out_pars = None, **kwargs)
reset_model()
test(data_path = "dataset/", pars_choice = 0, **kwargs)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, **kwargs)


utilmy/zml/source/models/akeras/namentity_crm_bilstm.py
-------------------------functions----------------------
evaluate(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
get_dataset(data_pars)
get_params(param_pars = {}, **kw)
load(load_pars)
predict(model, sess = None, data_pars = None, out_pars = None, compute_pars = None, **kw)
reset_model()
save(model = None, session = None, save_pars = None)
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, **kwargs)


utilmy/zml/source/models/akeras/preprocess.py
-------------------------functions----------------------
_preprocess_criteo(df, **kw)
_preprocess_movielens(df, **kw)
_preprocess_none(df, **kw)
get_dataset(**kw)
log(*s, n = 0, m = 1)
os_package_root_path(filepath, sublevel = 0, path_add = "")
test(data_path = "dataset/", pars_choice = 0)



utilmy/zml/source/models/akeras/textcnn.py
-------------------------functions----------------------
evaluate(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
get_dataset(data_pars = None, **kw)
get_params(param_pars = {}, **kw)
load(load_pars = {})
predict(model, sess = None, data_pars = None, out_pars = None, compute_pars = None, **kw)
reset_model()
save(model = None, session = None, save_pars = {})
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy/zml/source/models/akeras/util.py
-------------------------functions----------------------
_config_process(data_path, config_mode = "test")
fit(model, data_pars = None, model_pars = None, compute_pars = None, out_pars = None, session = None, **kwargs)
get_dataset(**kw)
load(path)
log(*s, n = 0, m = 1)
metrics(ypred, data_pars, compute_pars = None, out_pars = None, **kwargs)
os_package_root_path(filepath, sublevel = 0, path_add = "")
predict(model, data_pars, compute_pars = None, out_pars = None, **kwargs)
save(model, path)

-------------------------methods----------------------
Model_empty.__init__(self, model_pars = None, compute_pars = None)


utilmy/zml/source/models/atorch/__init__.py


utilmy/zml/source/models/atorch/matchZoo.py
-------------------------functions----------------------
evaluate(model, data_pars = None, compute_pars = None, out_pars = None)
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kwargs)
get_config_file()
get_data_loader(model_name, preprocessor, preprocess_pars, raw_data)
get_dataset(_model, preprocessor, _preprocessor_pars, data_pars)
get_glove_embedding_matrix(term_index, dimension)
get_params(param_pars = None, **kw)
get_raw_dataset(data_info, **args)
get_task(model_pars, task)
load(load_pars)
predict(model, session = None, data_pars = None, compute_pars = None, out_pars = None)
save(model, session = None, save_pars = None)
test_train(data_path, pars_choice, model_name)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, out_pars = None)


utilmy/zml/source/models/atorch/textcnn.py
-------------------------functions----------------------
_get_device()
_train(m, device, train_itr, optimizer, epoch, max_epoch)
_valid(m, device, test_itr)
analyze_datainfo_paths(data_info)
clean_str(string)
create_data_iterator(batch_size, tabular_train, tabular_valid, d)
create_tabular_dataset(data_info, **args)
evaluate(model, session = None, data_pars = None, compute_pars = None, out_pars = None, **kwargs)
fit(model, sess = None, data_pars = None, compute_pars = None, out_pars = None, **kwargs)
get_config_file()
get_data_file()
get_dataset(data_pars = None, out_pars = None, **kwargs)
get_params(param_pars = None, **kw)
load(load_pars =  None)
predict(model, session = None, data_pars = None, compute_pars = None, out_pars = None, return_ytrue = 1)
save(model, session = None, save_pars = None)
split_train_valid(data_info, **args)
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)
TextCNN.__init__(self, model_pars = None, data_pars = None, compute_pars = None, **kwargs)
TextCNN.forward(self, x)
TextCNN.rebuild_embed(self, vocab_built)


utilmy/zml/source/models/atorch/torch_ctr.py
-------------------------functions----------------------
customModel()
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
get_dataset(data_pars = None, task_type = "train", **kw)
get_params(param_pars = {}, **kw)
init(*kw, **kwargs)
load_info(path = "")
load_model(path = "")
log(*s)
predict(Xpred = None, data_pars = {}, compute_pars = None, out_pars = {}, **kw)
preprocess(prepro_pars)
reset()
save(path = None, info = None)
test(config = '')
test2(config = '')

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy/zml/source/models/atorch/torchhub.py
-------------------------functions----------------------
_get_device()
_train(m, device, train_itr, criterion, optimizer, epoch, max_epoch, imax = 1)
_valid(m, device, test_itr, criterion, imax = 1)
evaluate(model, data_pars = None, compute_pars = None, out_pars = None)
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kwargs)
get_config_file()
get_dataset(data_pars = None, **kw)
get_params(param_pars = None, **kw)
load(load_pars)
predict(model, session = None, data_pars = None, compute_pars = None, out_pars = None, imax  =  1, return_ytrue = 1)
save(model, session = None, save_pars = None)
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")
test2(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, out_pars = None)


utilmy/zml/source/models/atorch/transformer_sentence.py
-------------------------functions----------------------
evaluate(model, session = None, data_pars = None, compute_pars = None, out_pars = None, **kw)
fit(model, data_pars = None, model_pars = None, compute_pars = None, out_pars = None, *args, **kw)
fit2(model, data_pars = None, model_pars = None, compute_pars = None, out_pars = None, *args, **kw)
get_dataset(data_pars = None, **kw)
get_dataset2(data_pars = None, model = None, **kw)
get_params(param_pars, **kw)
load(load_pars = None)
predict(model, session = None, data_pars = None, out_pars = None, compute_pars = None, **kw)
predict2(model, session = None, data_pars = None, out_pars = None, compute_pars = None, **kw)
reset_model()
save(model, session = None, save_pars = None)
test(data_path = "dataset/", pars_choice = "test01", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, **kwargs)


utilmy/zml/source/models/atorch/util_data.py


utilmy/zml/source/models/atorch/util_transformer.py
-------------------------functions----------------------
_truncate_seq_pair(tokens_a, tokens_b, max_length)
convert_example_to_feature(example_row, pad_token = 0, sequence_a_segment_id = 0, sequence_b_segment_id = 1, cls_token_segment_id = 1, pad_token_segment_id = 0, mask_padding_with_zero = True, sep_token_extra = False)
convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_mode, cls_token_at_end = False, sep_token_extra = False, pad_on_left = False, cls_token = '[CLS]', sep_token = '[SEP]', pad_token = 0, sequence_a_segment_id = 0, sequence_b_segment_id = 1, cls_token_segment_id = 1, pad_token_segment_id = 0, mask_padding_with_zero = True, ) - 2))

-------------------------methods----------------------
BinaryProcessor._create_examples(self, lines, set_type)
BinaryProcessor.get_dev_examples(self, data_dir)
BinaryProcessor.get_labels(self)
BinaryProcessor.get_train_examples(self, data_dir)
DataProcessor._read_tsv(cls, input_file, quotechar = None)
DataProcessor.get_dev_examples(self, data_dir)
DataProcessor.get_labels(self)
DataProcessor.get_train_examples(self, data_dir)
InputExample.__init__(self, guid, text_a, text_b = None, label = None)
InputFeatures.__init__(self, input_ids, input_mask, segment_ids, label_id)
TransformerDataReader.__init__(self, **args)
TransformerDataReader.compute(self, input_tmp)
TransformerDataReader.get_data(self)


utilmy/zml/source/models/dataset.py
-------------------------functions----------------------
eval_dict(src, dst = {})
fIt_(dataset_url, training_iterations, batch_size, evaluation_interval)
get_dataset_split_for_model_petastorm(Xtrain, ytrain = None, pars:dict = None)
log(*s)
main()
pack_features_vector(features, labels)
pack_features_vector(features, labels)
pack_features_vector(features, labels)
python_hello_world(dataset_url='file = 'file:///tmp/external_dataset')
pytorch_hello_world(dataset_url='file = 'file:///tmp/external_dataset')
tensorflow_hello_world(dataset_url='file = 'file:///tmp/external_dataset')
test1()
train_and_test(dataset_url, training_iterations, batch_size, evaluation_interval)

-------------------------methods----------------------
dictEval.__init__(self)
dictEval.eval_dict(self, src, dst = {})
dictEval.pandas_create(self, key2, path, )
dictEval.reset(self)
dictEval.tf_dataset_create(self, key2, path_pattern, batch_size = 32, **kw)


utilmy/zml/source/models/keras_deepctr.py
-------------------------functions----------------------
eval(data_pars = None, compute_pars = None, out_pars = None, **kw)
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
get_dataset(data_pars = None, task_type = "train", **kw)
get_xy_dataset(data_sample = None)
get_xy_fd(use_neg = False, hash_flag = False, use_session = False)
get_xy_random(X, y, cols_family = {})
get_xy_random2(X, y, cols_family = {})
init(*kw, **kwargs)
load_info(path = "")
load_model(path = "", load_weight = False)
log(*s)
log2(*s)
log3(*s)
predict(Xpred = None, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)
preprocess(prepro_pars)
reset()
save(path = None, save_weight = False)
test(config = '')
test_helper(model_name, model_pars, data_pars, compute_pars)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, **kwargs)


utilmy/zml/source/models/keras_widedeep.py
-------------------------functions----------------------
ModelCustom2()
WideDeep_sparse(model_pars2)
fit(data_pars = None, compute_pars = None, out_pars = None)
get_dataset_split(data_pars = None, task_type = "train", **kw)
get_dataset_split_for_model_pandastuple(Xtrain, ytrain = None, data_pars = None, )
get_dataset_split_for_model_petastorm(Xtrain, ytrain = None, pars:dict = None)
get_dataset_split_for_model_tfsparse(Xtrain, ytrain = None, pars:dict = None)
init(*kw, **kwargs)
load_model(path = "")
log(*s)
log2(*s)
log3(*s)
model_summary(path = "ztmp/")
predict(Xpred = None, data_pars = None, compute_pars = None, out_pars = None)
reset()
save(path = None, info = None)
test(config = '', n_sample  =  100)
test2(config = '')
test_helper(model_pars, data_pars, compute_pars)
zz_Modelsparse2()
zz_WideDeep_dense(model_pars2)
zz_get_dataset(data_pars = None, task_type = "train", **kw)
zz_get_dataset2(data_pars = None, task_type = "train", **kw)
zz_get_dataset_tuple_keras(pattern, batch_size, mode = tf.estimator.ModeKeys.TRAIN, truncate = None)
zz_input_template_feed_keras_model(Xtrain, cols_type_received, cols_ref, **kw)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, )
tf_FeatureColumns.__init__(self, dataframe = None)
tf_FeatureColumns.bucketized_columns(self, columnsBoundaries)
tf_FeatureColumns.categorical_columns(self, indicator_column_names, colcat_nunique = None, output = False)
tf_FeatureColumns.crossed_feature_columns(self, columns_crossed, nameOfLayer, bucket_size = 10)
tf_FeatureColumns.data_to_tensorflow(self, df, target, model = 'sparse', shuffle_train = False, shuffle_test = False, shuffle_val = False, batch_size = 32, test_split = 0.2, colnum = [], colcat = [])
tf_FeatureColumns.data_to_tensorflow_split(self, df, target, model = 'sparse', shuffle_train = False, shuffle_test = False, shuffle_val = False, batch_size = 32, test_split = 0.2, colnum = [], colcat = [])
tf_FeatureColumns.df_to_dataset(self, dataframe, target, shuffle = True, batch_size = 32)
tf_FeatureColumns.df_to_dataset_dense(self, dataframe, target, shuffle = True, batch_size = 32)
tf_FeatureColumns.embeddings_columns(self, coldim_dict)
tf_FeatureColumns.get_features(self)
tf_FeatureColumns.hashed_columns(self, hashed_columns_dict)
tf_FeatureColumns.numeric_columns(self, columnsName)
tf_FeatureColumns.split_sparse_data(self, df, shuffle_train = False, shuffle_test = False, shuffle_val = False, batch_size = 32, test_split = 0.2, colnum = [], colcat = [])
tf_FeatureColumns.transform_output(self, featureColumn)


utilmy/zml/source/models/keras_widedeep_dense.py
-------------------------functions----------------------
Modelcustom(n_wide_cross, n_wide, n_deep, n_feat = 8, m_EMBEDDING = 10, loss = 'mse', metric  =  'mean_squared_error')
evaluate(Xy_pred = None, data_pars = None, compute_pars = {}, out_pars = {}, **kw)
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
get_dataset(Xtrain, cols_type_received, cols_ref)
get_dataset2(data_pars = None, task_type = "train", **kw)
get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
get_dataset_tuple_keras(Xtrain, cols_type_received, cols_ref, **kw)
init(*kw, **kwargs)
load_info(path = "")
load_model(path = "")
log(*s)
log2(*s)
log3(*s)
predict(Xpred = None, data_pars = None, compute_pars = {}, out_pars = {}, **kw)
reset()
save(path = None, info = None)
test(config = '')
test_helper(model_pars, data_pars, compute_pars)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy/zml/source/models/model_bayesian_numpyro.py
-------------------------functions----------------------
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
get_dataset(data_pars = None, task_type = "train", **kw)
get_params(param_pars = {}, **kw)
init(*kw, **kwargs)
init(*kw, **kwargs)
load_info(path = "")
load_model(path = "")
log(*s)
log(*s)
log2(*s)
log3(*s)
predict(Xpred = None, data_pars = {}, compute_pars = None, out_pars = {}, **kw)
preprocess(prepro_pars)
reset()
reset()
save(path = None, info = None)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy/zml/source/models/model_bayesian_pyro.py
-------------------------functions----------------------
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
get_dataset(data_pars = None, task_type = "train", **kw)
init(*kw, **kwargs)
load_info(path = "")
load_model(path = "")
log(*s)
log2(*s)
log3(*s)
model_class_loader(m_name = 'BayesianRegression', class_list:list = None)
predict(Xpred = None, data_pars = {}, compute_pars = None, out_pars = {}, **kw)
reset()
save(path = None, info = None)
test(nrows = 500)
test_dataset_regress_fake(nrows = 500)
y_norm(y, inverse = True, mode = 'boxcox')

-------------------------methods----------------------
BayesianRegression.__init__(self, X_dim:int = 17, y_dim:int = 1)
BayesianRegression.forward(self, x, y = None)
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy/zml/source/models/model_encoder.py
-------------------------functions----------------------
decode(Xpred = None, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)
fit(data_pars: dict = None, compute_pars: dict = None, out_pars: dict = None, **kw)
get_dataset(Xtrain, cols_type_received, cols_ref, split = False)
get_dataset_tuple(Xtrain, cols_type_received, cols_ref, split = False)
init(*kw, **kwargs)
init(*kw, **kwargs)
load_info(path = "")
load_model(path = "")
log(*s)
log(*s)
log2(*s)
log2(*s)
log3(*s)
log3(*s)
pd_autoencoder(df, col, pars)
pd_covariate_shift_adjustment()
pd_export(df, col, pars)
predict(Xpred = None, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)
reset()
reset()
save(path = None, info = None)
test(nrows = 500)
test_dataset_classi_fake(nrows = 500)
test_helper(model_pars, data_pars, compute_pars)
transform(Xpred = None, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy/zml/source/models/model_gefs.py
-------------------------functions----------------------
adult(data)
australia(data)
bank(data)
cmc(data)
credit(data)
electricity(data)
eval(data_pars = None, compute_pars = None, out_pars = None, **kw)
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
gef_get_stats(data, ncat = None)
gef_is_continuous(data)
gef_normalize_data(data, maxv, minv)
gef_standardize_data(data, mean, std)
german(data)
get_data(data_pars = None, task_type = "train", **kw)
get_dataset(data_pars = None, task_type = "train", **kw)
get_dummies(data)
init(*kw, **kwargs)
is_continuous(v_array)
learncats(data, classcol = None, continuous_ids = [])
load_info(path = "")
load_model(path = "")
log(*s)
log2(*s)
log3(*s)
pd_colcat_get_catcount(df, colcat, coly, continuous_ids = None)
predict(Xpred = None, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)
reset()
save(path = None, info = None)
segment(data)
test(n_sample  =  100)
test2()
test_converion()
test_helper(model_pars, data_pars, compute_pars)
train_test_split(data, ncat, train_ratio = 0.7, prep = 'std')
train_test_split2(data, ncat, train_ratio = 0.7, prep = 'std')
vowel(data)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy/zml/source/models/model_numpyro.py
-------------------------functions----------------------
init(*kw, **kwargs)
log(*s)
log2(*s)
log3(*s)
metrics(y: pd.Series, yhat: pd.Series)
require_fitted(f)
reset()

-------------------------methods----------------------
AlreadyFittedError.__init__(self, model)
BaseModel.__init__(self, rng_seed: int  =  None)
BaseModel.__repr__(self)
BaseModel.fit(self, df: pd.DataFrame, sampler: str  =  "NUTS", rng_key: np.ndarray  =  None, sampler_kwargs: typing.Dict[str, typing.Any]  =  None, **mcmc_kwargs, )
BaseModel.formula(self)
BaseModel.from_dict(cls, data: typing.Dict[str, typing.Any], **model_kw)
BaseModel.grouped_metrics(self, df: pd.DataFrame, groupby: typing.Union[str, typing.List[str]], aggfunc: typing.Callable  =  onp.sum, aggerrs: bool  =  True, )
BaseModel.likelihood_func(self, yhat)
BaseModel.link(x)
BaseModel.metrics(self, df: pd.DataFrame, aggerrs: bool  =  True)
BaseModel.model(self, df: pd.DataFrame)
BaseModel.num_chains(self)
BaseModel.num_samples(self)
BaseModel.predict(self, df: pd.DataFrame, ci: bool  =  False, ci_interval: float  =  0.9, aggfunc: typing.Union[str, typing.Callable]  =  "mean", )
BaseModel.preprocess_config_dict(cls, config: dict)
BaseModel.sample_posterior_predictive(self, df: pd.DataFrame, hdpi: bool  =  False, hdpi_interval: float  =  0.9, rng_key: np.ndarray  =  None, )
BaseModel.samples_df(self)
BaseModel.samples_flat(self)
BaseModel.split_rand_key(self, n: int  =  1)
BaseModel.to_json(self)
BaseModel.transform(cls, df: pd.DataFrame)
Bernoulli.likelihood_func(self, probs)
Bernoulli.link(x)
IncompleteFeature.__init__(self, name, key)
IncompleteModel.__init__(self, model, attribute)
IncompleteSamples.__init__(self, name)
Normal.likelihood_func(self, yhat)
Normal.link(x)
NotFittedError.__init__(self, func = None)
NumpyEncoder.default(self, obj)
Poisson.likelihood_func(self, yhat)
Poisson.link(x)
ShabadooException.__str__(self)


utilmy/zml/source/models/model_outlier.py
-------------------------functions----------------------
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
get_dataset(Xtrain, ytrain = None, data_pars = None, )
get_dataset2(data_pars = None, task_type = "train", **kw)
get_dataset_split_for_model_pandastuple(Xtrain, ytrain = None, data_pars = None, )
init(*kw, **kwargs)
load_info(path = "")
load_model(path = "")
log(*s)
log2(*s)
log3(*s)
predict(Xpred = None, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)
reset()
save(path = None, info = None)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy/zml/source/models/model_sampler.py
-------------------------functions----------------------
eval(data_pars = None, compute_pars = None, out_pars = None, **kw)
fit(data_pars: dict = None, compute_pars: dict = None, out_pars: dict = None, **kw)
get_dataset(Xtrain, cols_type_received, cols_ref, split = False)
get_dataset_tuple(Xtrain, cols_type_received, cols_ref, split = False)
init(*kw, **kwargs)
load_info(path = "")
load_model(path = "")
log(*s)
log2(*s)
log3(*s)
predict(Xpred = None, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)
reset()
save(path = None, info = None)
test()
test2(n_sample  =  1000)
test_helper(model_pars, data_pars, compute_pars)
transform(Xpred = None, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)
zz_pd_augmentation_sdv(df, col = None, pars = {})
zz_pd_covariate_shift_adjustment()
zz_pd_sample_imblearn(df = None, col = None, pars = None)
zz_test()

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy/zml/source/models/model_sklearn.py
-------------------------functions----------------------
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
get_dataset(Xtrain, ytrain = None, data_pars = None, )
get_dataset2(data_pars = None, task_type = "train", **kw)
get_dataset_split_for_model_pandastuple(Xtrain, ytrain = None, data_pars = None, )
get_params(deep = False)
get_params_sklearn(deep = False)
init(*kw, **kwargs)
load_info(path = "")
load_model(path = "")
log(*s)
log2(*s)
log3(*s)
model_automl()
predict(Xpred = None, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)
reset()
save(path = None, info = None)
test(n_sample           =  1000)
zz_eval(data_pars = None, compute_pars = None, out_pars = None, **kw)
zz_preprocess(prepro_pars)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy/zml/source/models/model_tseries.py
-------------------------functions----------------------
LighGBM_recursive(lightgbm_pars= {'objective' =  {'objective':'quantile', 'alpha': 0.5}, forecaster_pars = {'window_length' =  {'window_length': 4})
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
get_dataset(data_pars = None, task_type = "train", **kw)
init(*kw, **kwargs)
load_info(path = "")
load_model(path = "")
log(*s)
log2(*s)
log3(*s)
predict(Xpred = None, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)
predict_forward(Xpred = None, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)
reset()
save(path = None, info = None)
test(nrows = 10000, coly = None, coldate = None, colcat = None)
test2(nrows = 1000, file_path = None, coly = None, coldate = None, colcat = None)
test_dataset_tseries(nrows = 10000, coly = None, coldate = None, colcat = None)
time_train_test_split(df, test_size  =  0.4, cols = None, coltime  = "time_key", sort = True, minsize = 5, n_sample = 5, verbose = False)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy/zml/source/models/model_vaem.py
-------------------------functions----------------------
decode2(data_decode, scaling_factor, list_discrete, records_d, plot = False)
encode2(data_decode, list_discrete, records_d, fast_plot)
init(*kw, **kwargs)
load_data(filePath, categories, cat_col, num_cols, discrete_cols, targetCol, nsample, delimiter)
load_info(path = "")
load_model(path = "")
log(*s)
log2(*s)
log3(*s)
p_vae_active_learning(Data_train_compressed, Data_train, mask_train, Data_test, mask_test_compressed, mask_test, cat_dims, dim_flt, dic_var_type, args, list_discrete, records_d, estimation_method = 1)
reset()
save(path = '', info = None)
save_model2(model, output_dir)
test()
train_p_vae(stage, x_train, Data_train, mask_train, epochs, latent_dim, cat_dims, dim_flt, batch_size, p, K, iteration, list_discrete, records_d, args)

-------------------------methods----------------------
Model_custom.__init__(self)
Model_custom.decode(self)
Model_custom.encode(self)
Model_custom.fit(self,filePath, categories,cat_cols,num_cols,discrete_cols,targetCol,nsample  =  -1,delimiter=',',plot=False)


utilmy/zml/source/models/model_vaemdn.py
-------------------------functions----------------------
AUTOENCODER_BASIC(X_input_dim, loss_type = "CosineSimilarity", lr = 0.01, epsilon = 1e-3, decay = 1e-4, optimizer = 'adam', encodingdim  =  50, dim_list = "50,25,10")
AUTOENCODER_MULTIMODAL(input_shapes = [10], hidden_dims = [128, 64, 8], output_activations = ['sigmoid', 'relu'], loss  =  ['bernoulli_divergence', 'poisson_divergence'], optimizer = 'adam')
VAEMDN(model_pars)
benchmark(config = '', dmin = 5, dmax = 6)
decode(Xpred = None, data_pars = None, compute_pars = {}, out_pars = {}, index  =  0, **kw)
encode(Xpred = None, data_pars = None, compute_pars = {}, out_pars = {}, model_class = 'VAEMDN', **kw)
fit(data_pars = None, compute_pars = None, out_pars = None, model_class = 'VAEMDN', **kw)
get_dataset(Xtrain, cols_type_received, cols_ref)
get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
get_label(encoder, x_train, dummy_train, class_num = 5, batch_size = 256)
init(*kw, **kwargs)
load_info(path = "")
load_model(path = "", model_class = 'VAEMDN')
log(*s)
log2(*s)
log3(*s)
predict(Xpred = None, data_pars = None, compute_pars = {}, out_pars = {}, model_class = 'VAEMDN', **kw)
reset()
sampling(args)
save(path = None, info = None)
test(n_rows = 100)
test2(n_sample           =  1000)
test3(n_sample  =  1000)
test4()
test_dataset_correlation(n_rows = 100)
test_helper(model_pars, data_pars, compute_pars)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy/zml/source/models/optuna_lightgbm.py
-------------------------functions----------------------
benchmark()
benchmark_helper(train_df, test_df)
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
get_dataset(data_pars = None, task_type = "train", **kw)
init(*kw, **kwargs)
load_info(path = "")
load_model(path = "")
log(*s)
log2(*s)
log3(*s)
os_makedirs(dir_or_file)
predict(Xpred = None, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)
reset()
save(path = None, info = None)
test(nrows = 500)
test_dataset_classi_fake(nrows = 500)
test_helper(model_pars, data_pars, compute_pars)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy/zml/source/models/repo/functions.py
-------------------------functions----------------------
fit(vae, x_train, epochs = 1, batch_size = 256)
get_dataset(state_num = 10, time_len = 50000, signal_dimension = 15, CNR = 1, window_len = 11, half_window_len = 5)
get_model(original_dim, class_num = 5, intermediate_dim = 64, intermediate_dim_2 = 16, latent_dim = 3, batch_size = 256, Lambda1 = 1, Lambda2 = 200, Alpha = 0.075)
load(model, path)
sampling(args)
save(model)
test(self, encoder, x_train, dummy_train, class_num = 5, batch_size = 256)



utilmy/zml/source/models/repo/model_rec.py
-------------------------functions----------------------
eval(Xpred = None, data_pars: dict = {}, compute_pars: dict = {}, out_pars: dict = {}, **kw)
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
get_dataset(data_pars, task_type = "train")
get_dataset2(data_pars = None, task_type = "train", **kw)
get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
init(*kw, **kwargs)
load_info(path = "")
log(*s)
log2(*s)
log3(*s)
os_makedirs(dir_or_file)
predict(Xpred = None, data_pars = None, compute_pars = {}, out_pars = {}, **kw)
reset()
save(path = None, info = None)
test(n_sample           =  1000)
test_helper(model_pars, data_pars, compute_pars)
train_test_split2(df, coly)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, global_pars = None)


utilmy/zml/source/models/repo/model_rec_ease.py
-------------------------functions----------------------
eval(Xpred = None, data_pars: dict = {}, compute_pars: dict = {}, out_pars: dict = {}, **kw)
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
get_dataset(data_pars, task_type = "train")
get_dataset2(data_pars = None, task_type = "train", **kw)
get_dataset_sampler(data_pars)
get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
init(*kw, **kwargs)
init_dataset(data_pars)
load_info(path = "")
log(*s)
log2(*s)
log3(*s)
os_makedirs(dir_or_file)
predict(Xpred = None, data_pars = None, compute_pars = {}, out_pars = {}, **kw)
reset()
save(path = None, info = None)
test(n_sample           =  1000)
test_helper(model_pars, data_pars, compute_pars)
train_test_split2(df, coly)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, global_pars = None)


utilmy/zml/source/models/torch_ease.py
-------------------------functions----------------------
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
get_dataset(data_pars = None, task_type = "train", **kwargs)
init(*kw, **kwargs)
load_info(path = "")
log(*s)
log2(*s)
log3(*s)
os_makedirs(dir_or_file)
predict(Xpred = None, data_pars = None, compute_pars = {}, out_pars = {}, **kw)
reset()
save(path = None, info = None)
test(nrows = 1000)
test_dataset_goodbooks(nrows = 1000)
test_helper(model_pars, data_pars, compute_pars)
train_test_split2(df, coly)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy/zml/source/models/torch_rectorch.py
-------------------------functions----------------------
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
get_dataset(data_pars = None, task_type = "train")
init(*kw, **kwargs)
load_info(path = "")
log(*s)
log2(*s)
log3(*s)
make_rand_sparse_dataset(n_rows = 1000, )
predict(Xpred = None, data_pars = None, compute_pars = {}, out_pars = {}, **kw)
reset()
save(path = None, info = None)
test(n_sample           =  1000)
test_helper(model_pars, data_pars, compute_pars)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy/zml/source/models/torch_rvae.py
-------------------------functions----------------------
compute_metrics(model, X, dataset_obj, args, epoch, losses_save, logit_pi_prev, X_clean, target_errors, mode)
decode(Xpred = None, data_pars: dict = {}, compute_pars: dict = {}, out_pars: dict = {}, **kw)
encode(Xpred = None, data_pars: dict = {}, compute_pars: dict = {}, out_pars: dict = {}, **kw)
eval(Xpred = None, data_pars: dict = {}, compute_pars: dict = {}, out_pars: dict = {}, **kw)
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
get_dataset(data_pars, task_type = "train")
init(*kw, **kwargs)
load_info(path = "")
log(*s)
log2(*s)
log3(*s)
predict(Xpred = None, data_pars = None, compute_pars = {}, out_pars = {}, **kw)
reset()
save(path = None, info = None)
test(nrows = 1000)
test_helper(m)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, global_pars = None)
RVAE.__init__(self, args)
RVAE._get_dataset_obj(self)
RVAE._save_to_csv(self, X_data, X_data_clean, target_errors, attributes, losses_save, dataset_obj, path_output, args, epoch, mode = 'train')
RVAE.decode(self, z)
RVAE.encode(self, x_data, one_hot_categ = False, masking = False, drop_mask = [], in_aux_samples = [])
RVAE.fit(self)
RVAE.forward(self, x_data, n_epoch = None, one_hot_categ = False, masking = False, drop_mask = [], in_aux_samples = [])
RVAE.get_inputs(self, x_data, one_hot_categ = False, masking = False, drop_mask = [], in_aux_samples = [])
RVAE.loss_function(self, input_data, p_params, q_params, q_samples, clean_comp_only = False, data_eval_clean = False)
RVAE.predict(self, x_data, n_epoch = None, one_hot_categ = False, masking = False, drop_mask = [], in_aux_samples = [])
RVAE.reparameterize(self, q_params, eps_samples = None)
RVAE.sample_normal(self, q_params_z, eps = None)
RVAE.save(self)


utilmy/zml/source/models/torch_tabular.py
-------------------------functions----------------------
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
get_dataset(Xtrain, cols_type_received, cols_ref = None)
get_dataset_tuple(Xtrain, cols_type_received, cols_ref = None)
init(*kw, **kwargs)
load_info(path = "")
load_model(path = "")
log(*s)
log2(*s)
log3(*s)
predict(Xpred = None, data_pars: dict = {}, compute_pars: dict = {}, out_pars: dict = {}, **kw)
reset()
save(path = None, info = None)
test(n_sample  =  100)
test2(nrows = 10000)
test3(n_sample  =  100)
test_helper(m, X_valid)
train_test_split2(df, coly)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy/zml/source/models/util_models.py
-------------------------functions----------------------
log(*s)
test_dataset_classi_fake(nrows = 500)
test_dataset_classifier_covtype(nrows = 500)
test_dataset_petfinder(nrows = 1000)
test_dataset_regress_fake(nrows = 500)
tf_data_create_sparse(cols_type_received:dict =  {'cols_sparse' : ['col1', 'col2'], 'cols_num'    : ['cola', 'colb']}, cols_ref:list =   [ 'col_sparse', 'col_num'  ], Xtrain:pd.DataFrame = None, **kw)
tf_data_file_to_dataset(pattern, batch_size, mode = tf.estimator.ModeKeys.TRAIN, truncate = None)
tf_data_pandas_to_dataset(training_df, colsX, coly)



utilmy/zml/source/models/ztmp2/keras_widedeep_2.py
-------------------------functions----------------------
ModelCustom2()
Modelcustom(n_wide_cross, n_wide, n_deep, n_feat = 8, m_EMBEDDING = 10, loss = 'mse', metric  =  'mean_squared_error')
Modelsparse2()
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
get_dataset(Xtrain, cols_type_received, cols_ref)
get_dataset2(data_pars = None, task_type = "train", **kw)
get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
get_dataset_tuple_keras(pattern, batch_size, mode = tf.estimator.ModeKeys.TRAIN, truncate = None)
init(*kw, **kwargs)
input_template_feed_keras(Xtrain, cols_type_received, cols_ref, **kw)
load_info(path = "")
load_model(path = "")
log(*s)
log2(*s)
predict(Xpred = None, data_pars = None, compute_pars = {}, out_pars = {}, **kw)
reset()
save(path = None, info = None)
test(config = '')
test_helper(model_pars, data_pars, compute_pars)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy/zml/source/models/ztmp2/keras_widedeep_old.py
-------------------------functions----------------------
Modelcustom(n_wide_cross, n_wide, n_feat = 8, m_EMBEDDING = 10, loss = 'mse', metric  =  'mean_squared_error')
eval(data_pars = None, compute_pars = None, out_pars = None, **kw)
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
get_dataset(data_pars = None, task_type = "train", **kw)
get_dataset2(data_pars = None, task_type = "train", **kw)
get_params(deep = False)
get_params_sklearn(deep = False)
init(*kw, **kwargs)
load_info(path = "")
load_model(path = "")
log(*s)
predict(Xpred = None, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)
preprocess(prepro_pars)
reset()
save(path = None)
test(config = '')
test2(config = '')
test_helper(model_pars, data_pars, compute_pars)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy/zml/source/models/ztmp2/model_vaem.py
-------------------------functions----------------------
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
get_dataset(Xtrain, cols_type_received, cols_ref)
get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
init(*kw, **kwargs)
load_dataset()seed  =  3000"./data/bank/bankmarketing_train.csv")bank_raw.info())label_column="y")matrix1, ["job"])matrix1, ["marital"])matrix1, ["education"])matrix1, ["default"])matrix1, ["housing"])matrix1, ["loan"])matrix1, ["contact"])matrix1, ["month"])matrix1, ["day_of_week"])matrix1, ["poutcome"])matrix1, ["y"])(matrix1.values).astype(float))[0, :]max_Data  =  0.7min_Data = 0.3[0, 1, 2, 3, 4, 5, 6, 7])[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])[8, 9])np.in1d(list_flt, list_discrete).nonzero()[0])list_cat)list_flt)>0 and len(list_cat)>0)
load_info(path = "")
load_model(path = "")
log(*s)
log2(*s)
predict(Xpred = None, data_pars = None, compute_pars = {}, out_pars = {}, **kw)
reset()
save(path = None, info = None)
test(config = '')
test_helper(model_pars, data_pars, compute_pars)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy/zml/source/models/ztmp2/model_vaem3.py
-------------------------functions----------------------
decode2(data_decode, scaling_factor, list_discrete, records_d, plot = False, args = None)
encode2(data_decode, list_discrete, records_d, fast_plot)
init(*kw, **kwargs)
load_data(filePath, categories, cat_col, num_cols, discrete_cols, targetCol, nsample, delimiter)
load_info(path = "")
load_model(path = "")
log(*s)
log2(*s)
p_vae_active_learning(Data_train_comp, Data_train, mask_train, Data_test, mask_test_comp, mask_test, cat_dims, dim_flt, dic_var_type, args, list_discrete, records_d, estimation_method = 1)
reset()
save(model, output_dir)
save_model2(model, output_dir)
train_p_vae(stage, x_train, Data_train, mask_train, epochs, latent_dim, cat_dims, dim_flt, batch_size, p, K, iteration, list_discrete, records_d, args)

-------------------------methods----------------------
Model.__init__(self)
Model.decode(self, plot = False, args = None)
Model.encode(self, plot = False, args = None)
Model.fit(self, p)


utilmy/zml/source/models/ztmp2/modelsVaem.py
-------------------------functions----------------------
decode2(data_decode, scaling_factor, list_discrete, records_d, plot = False)
encode2(data_decode, list_discrete, records_d, fast_plot)
init(*kw, **kwargs)
load_data(filePath, categories, cat_col, num_cols, discrete_cols, targetCol, nsample, delimiter)
load_info(path = "")
load_model(path = "")
log(*s)
log2(*s)
p_vae_active_learning(Data_train_compressed, Data_train, mask_train, Data_test, mask_test_compressed, mask_test, cat_dims, dim_flt, dic_var_type, args, list_discrete, records_d, estimation_method = 1)
reset()
save(model, output_dir)
save_model2(model, output_dir)
train_p_vae(stage, x_train, Data_train, mask_train, epochs, latent_dim, cat_dims, dim_flt, batch_size, p, K, iteration, list_discrete, records_d, args)

-------------------------methods----------------------
Model.__init__(self)
Model.decode(self)
Model.encode(self)
Model.fit(self,filePath, categories,cat_cols,num_cols,discrete_cols,targetCol,nsample  =  -1,delimiter=',',plot=False)


utilmy/zml/source/models/ztmp2/torch_rvae2.py
-------------------------functions----------------------
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
get_dataset(Xtrain, cols_type_received, cols_ref)
get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
init(*kw, **kwargs)
load_info(path = "")
load_model(path = "")
log(*s)
log2(*s)
predict(Xpred = None, data_pars: dict = {}, compute_pars: dict = {}, out_pars: dict = {}, **kw)
reset()
save(path = None, info = None)
test(nrows = 1000)
test2(nrow = 10000)
test_dataset_1(nrows = 1000)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy/zml/source/models/ztmp2/torch_tabular2.py
-------------------------functions----------------------
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
get_dataset(data_pars = None, task_type = "train", **kw)
get_dataset2(data_pars = None, task_type = "train", **kw)
get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
init(*kw, **kwargs)
load_info(path = "")
load_model(path = "")
log(*s)
log2(*s)
predict(Xpred = None, data_pars: dict = {}, compute_pars: dict = {}, out_pars: dict = {}, **kw)
reset()
save(path = None, info = None)
test(nrows = 1000)
test2(nrows = 10000)
test3()
test_dataset_covtype(nrows = 1000)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy/zml/source/prepro.py
-------------------------functions----------------------
_pd_colnum(df, col, pars)
_pd_colnum_fill_na_median(df, col, pars)
log(*s)
log2(*s)
log3(*s)
log4(*s, n = 0, m = 1)
log4_pd(name, df, *s)
os_convert_topython_code(txt)
pd_col_atemplate(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_col_genetic_transform(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_colcat_bin(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_colcat_encoder_generic(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_colcat_minhash(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_colcat_to_onehot(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_colcross(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_coldate(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_colnum_bin(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_colnum_binto_onehot(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_colnum_normalize(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_colnum_quantile_norm(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_coly(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_coly_clean(df: pd.DataFrame, col: list = None, pars: dict = None)
prepro_load(prefix, pars)
prepro_save(prefix, pars, df_new, cols_new, prepro)
save_json(js, pfile, mode = 'a')
test()



utilmy/zml/source/prepro_rec.py
-------------------------functions----------------------
_preprocess_criteo(df, **kw)
_preprocess_movielens(df, **kw)



utilmy/zml/source/prepro_text.py
-------------------------functions----------------------
log(*s, n = 0, m = 1)
log_pd(df, *s, n = 0, m = 1)
logs(*s)
nlp_get_stopwords()
pd_coltext(df, col, stopwords =  None, pars = None)
pd_coltext_clean(df, col, stopwords =  None, pars = None)
pd_coltext_universal_google(df, col, pars = {})
pd_coltext_wordfreq(df, col, stopwords, ntoken = 100)



utilmy/zml/source/prepro_tseries.py
-------------------------functions----------------------
log(*s)
log2(*s)
log3(*s)
logd(*s, n = 0, m = 0)
m5_dataset()
pd_prepro_custom(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_prepro_custom2(df: pd.DataFrame, cols: list = None, pars: dict = None)
pd_ts_autoregressive(df: pd.DataFrame, cols: list = None, pars: dict = None)
pd_ts_date(df: pd.DataFrame, cols: list = None, pars: dict = None)
pd_ts_deltapy_generic(df: pd.DataFrame, cols: list = None, pars: dict = None)
pd_ts_difference(df: pd.DataFrame, cols: list = None, pars: dict = None)
pd_ts_groupby(df: pd.DataFrame, cols: list = None, pars: dict = None)
pd_ts_lag(df: pd.DataFrame, cols: list = None, pars: dict = None)
pd_ts_onehot(df: pd.DataFrame, cols: list = None, pars: dict = None)
pd_ts_rolling(df: pd.DataFrame, cols: list = None, pars: dict = None)
pd_ts_tsfresh_features(df: pd.DataFrame, cols: list = None, pars: dict = None)
test_deltapy_all()
test_deltapy_all2()
test_deltapy_get_method(df)
test_get_sampledata(url="https = "https://github.com/firmai/random-assets-two/raw/master/numpy/tsla.csv")
test_prepro_v1()



utilmy/zml/source/run_feature_profile.py
-------------------------functions----------------------
log(*s, n = 0, m = 0)
run_profile(path_data = None, path_output = "data/out/ztmp/", n_sample = 5000)



utilmy/zml/source/run_hyperopt.py
-------------------------functions----------------------
eval_dict(src, dst = {})
log(*s)
run_hyper_optuna(obj_fun, pars_dict_init, pars_dict_range, engine_pars, ntrials = 3)
test_hyper()
test_hyper2()
test_hyper3()



utilmy/zml/source/run_inference.py
-------------------------functions----------------------
log(*s)
log2(*s)
log3(*s)
map_model(model_name="model_sklearn = "model_sklearn:MyClassModel")
model_dict_load(model_dict, config_path, config_name, verbose = True)
predict(model_dict, dfX, cols_family, post_process_fun = None)
run_data_check(path_data, path_data_ref, path_model, path_output, sample_ratio = 0.5)
run_predict(config_name, config_path, n_sample = -1, path_data = None, path_output = None, pars = {}, model_dict = None, return_mode = 'file')
run_predict_batch(config_name, config_path, n_sample = -1, path_data = None, path_output = None, pars = {}, model_dict = None, return_mode = 'file')



utilmy/zml/source/run_inpection.py
-------------------------functions----------------------
log(*s, n = 0, m = 0)
model_dict_load(model_dict, config_path, config_name, verbose = True)
save_features(df, name, path)



utilmy/zml/source/run_mlflow.py
-------------------------functions----------------------
register(run_name, params, metrics, signature, model_class, tracking_uri= "sqlite =  "sqlite:///local.db")



utilmy/zml/source/run_preprocess.py
-------------------------functions----------------------
load_features(name, path)
log(*s)
log2(*s)
log3(*s)
log_pd(df, *s, n = 0, m = 1)
model_dict_load(model_dict, config_path, config_name, verbose = True)
preprocess(path_train_X = "", path_train_y = "", path_pipeline_export = "", cols_group = None, n_sample = 5000, preprocess_pars = {}, path_features_store = None)
preprocess_batch(path_train_X = "", path_train_y = "", path_pipeline_export = "", cols_group = None, n_sample = 5000, preprocess_pars = {}, path_features_store = None)
preprocess_inference(df, path_pipeline = "data/pipeline/pipe_01/", preprocess_pars = {}, cols_group = None)
preprocess_load(path_train_X = "", path_train_y = "", path_pipeline_export = "", cols_group = None, n_sample = 5000, preprocess_pars = {}, path_features_store = None)
run_preprocess(config_name, config_path, n_sample = 5000, mode = 'run_preprocess', model_dict = Nonemodel_dict, config_path, config_name, verbose = True)m = model_dict['global_pars']path_data         = m['path_data_preprocess']'path_data_prepro_X', path_data + "/features.zip") # ### Can be a list of zip or parquet files'path_data_prepro_y', path_data + "/target.zip")   # ### Can be a list of zip or parquet filespath_output          =  m['path_train_output']'path_pipeline', path_output + "/pipeline/" )'path_features_store', path_output + '/features_store/' )  #path_data_train replaced with path_output, because preprocessed files are stored there'path_check_out', path_output + "/check/" )path_output)"#### load input column family  ###################################################")cols_group = model_dict['data_pars']['cols_input_type']  ### the model config file"#### Preprocess  #################################################################")preprocess_pars = model_dict['model_pars']['pre_process_pars']if mode == "run_preprocess"  =  model_dict['data_pars']['cols_input_type']  ### the model config file"#### Preprocess  #################################################################")preprocess_pars = model_dict['model_pars']['pre_process_pars']if mode == "run_preprocess" :)
save_features(df, name, path = None)



utilmy/zml/source/run_sampler.py
-------------------------functions----------------------
log(*s)
log2(*s)
log3(*s)
map_model(model_name)
model_dict_load(model_dict, config_path, config_name, verbose = True)
run_train(config_name, config_path = "source/config_model.py", n_sample = 5000, mode = "run_preprocess", model_dict = None, return_mode = 'file', **kw)
run_transform(config_name, config_path, n_sample = 1, path_data = None, path_output = None, pars = {}, model_dict = None, return_mode = "")
save_features(df, name, path)
train(model_dict, dfX, cols_family, post_process_fun)
transform(model_name, path_model, dfX, cols_family, model_dict)



utilmy/zml/source/run_train.py
-------------------------functions----------------------
cols_validate(model_dict)
data_split(dfX, data_pars, model_path, colsX, coly)
log(*s)
log2(*s)
log3(*s)
map_model(model_name)
mlflow_register(dfXy, model_dict: dict, stats: dict, mlflow_pars:dict)
model_dict_load(model_dict, config_path, config_name, verbose = True)
run_model_check(path_output, scoring)
run_train(config_name, config_path = "source/config_model.py", n_sample = 5000, mode = "run_preprocess", model_dict = None, return_mode = 'file', **kw)
save_features(df, name, path)
train(model_dict, dfX, cols_family, post_process_fun)



utilmy/zml/source/util.py
-------------------------functions----------------------
create_appid(filename)
create_logfilename(filename)
create_uniqueid()
download_dtopbox(data_pars)
download_googledrive(file_list=[ {  "fileid" = [ {  "fileid": "1-K72L8aQPsl2qt_uBF-kzbai3TYG6Qg4", "path_target":  "data/input/download/test.json"}], **kw)
load_dataset_generator(data_pars)
log(*s, n = 0, m = 1, **kw)
logger_handler_console(formatter = None)
logger_handler_file(isrotate = False, rotate_time = "midnight", formatter = None, log_file_used = None)
logger_setup(logger_name = None, log_file = None, formatter = 'FORMATTER_0', isrotate = False, isconsole_output = True, logging_level = 'info', )
logger_setup2(name = __name__, level = None)
pd_to_keyvalue_dict(dfa, colkey =  [ "shop_id", "l2_genre_id" ], col_list = 'item_id', to_file = "")
pd_to_scipy_sparse_matrix(df)
test_log()
tf_dataset(dataset_pars)

-------------------------methods----------------------
Downloader.__init__(self, url)
Downloader._transform_dropbox_url(self)
Downloader._transform_gdrive_url(self)
Downloader._transform_github_url(self)
Downloader.adjust_url(self)
Downloader.clean_netloc(self)
Downloader.download(self, filepath = '')
Downloader.get_filename(self, headers)
dict2.__init__(self, d)
dictLazy.__getitem__(self, key)
dictLazy.__init__(self, *args, **kw)
dictLazy.__iter__(self)
dictLazy.__len__(self)
logger_class.__init__(self, config_file = None, verbose = True)
logger_class.debug(self, *s, level = 1)
logger_class.load_config(self, config_file_path = None)
logger_class.log(self, *s, level = 1)


utilmy/zml/source/util_feature.py
-------------------------functions----------------------
col_extractname(col_onehot)
col_remove(cols, colsremove, mode = "exact")
estimator_boostrap_bayes(err, alpha = 0.05, )
estimator_bootstrap(err, custom_stat = None, alpha = 0.05, n_iter = 10000)
estimator_std_normal(err, alpha = 0.05, )
feature_correlation_cat(df, colused)
feature_importance_perm(clf, Xtrain, ytrain, cols, n_repeats = 8, scoring = 'neg_root_mean_squared_error', show_graph = 1)
feature_selection_multicolinear(df, threshold = 1.0)
fetch_dataset(url_dataset, path_target = None, file_target = None)
fetch_spark_koalas(path_data_x, path_data_y = '', colid = "jobId", n_sample = -1)
load(name, path)
load_dataset(path_data_x, path_data_y = '', colid = "jobId", n_sample = -1)
load_features(name, path)
load_function_uri(uri_name="myfolder/myfile.py = "myfolder/myfile.py::myFunction")
log(*s, n = 0, m = 1, **kw)
log2(*s, **kw)
log3(*s, **kw)
metrics_eval(metric_list = ["mean_squared_error"], ytrue = None, ypred = None, ypred_proba = None, return_dict = False)
np_conv_to_one_col(np_array, sep_char = "_")
os_get_function_name()
os_getcwd()
pa_read_file(path =   'folder_parquet/', cols = None, n_rows = 1000, file_start = 0, file_end = 100000, verbose = 1, )
pa_write_file(df, path =   'folder_parquet/', cols = None, n_rows = 1000, partition_cols = None, overwrite = True, verbose = 1, filesystem  =  'hdfs')
params_check(pars, check_list, name = "")
pd_col_fillna(dfref, colname = None, method = "frequent", value = None, colgroupby = None, return_val = "dataframe,param", )
pd_col_filter(df, filter_val = None, iscol = 1)
pd_col_merge_onehot(df, colname)
pd_col_to_num(df, colname = None, default = np.nan)
pd_col_to_onehot(dfref, colname = None, colonehot = None, return_val = "dataframe,column")
pd_colcat_mapping(df, colname)
pd_colcat_mergecol(df, col_list, x0, colid = "easy_id")
pd_colcat_toint(dfref, colname, colcat_map = None, suffix = None)
pd_colcat_tonum(df, colcat = "all", drop_single_label = False, drop_fact_dict = True)
pd_colnum_normalize(df0, colname, pars, suffix = "_norm", return_val = 'dataframe,param')
pd_colnum_tocat(df, colname = None, colexclude = None, colbinmap = None, bins = 5, suffix = "_bin", method = "uniform", na_value = -1, return_val = "dataframe,param", params={"KMeans_n_clusters" = {"KMeans_n_clusters": 8, "KMeans_init": 'k-means++', "KMeans_n_init": 10,"KMeans_max_iter": 300, "KMeans_tol": 0.0001, "KMeans_precompute_distances": 'auto',"KMeans_verbose": 0, "KMeans_random_state": None,"KMeans_copy_x": True, "KMeans_n_jobs": None, "KMeans_algorithm": 'auto'})
pd_colnum_tocat_stat(df, feature, target_col, bins, cuts = 0)
pd_feature_generate_cross(df, cols, cols_cross_input = None, pct_threshold = 0.2, m_combination = 2)
pd_pipeline_apply(df, pipeline)
pd_read_file(path_glob = "*.pkl", ignore_index = True, cols = None, verbose = False, nrows = -1, concat_sort = True, n_pool = 1, drop_duplicates = None, col_filter = None, col_filter_val = None, **kw)
pd_stat_correl_pair(df, coltarget = None, colname = None)
pd_stat_dataset_shift(dftrain, dftest, colused, nsample = 10000, buckets = 5, axis = 0)
pd_stat_datashift_psi(expected, actual, buckettype = 'bins', buckets = 10, axis = 0)
pd_stat_distribution_colnum(df, nrows = 2000, verbose = False)
pd_stat_histogram(df, bins = 50, coltarget = "diff")
pd_stat_pandas_profile(df, savefile = "report.html", title = "Pandas Profile")
pd_stat_shift_changes(df, target_col, features_list = 0, bins = 10, df_test = 0)
pd_stat_shift_trend_changes(df, feature, target_col, threshold = 0.03)
pd_stat_shift_trend_correlation(df, df_test, colname, target_col)
save(df, name, path = None)
save_features(df, name, path = None)
save_list(path, name_list, glob)
test_get_classification_data(name = None)
test_heteroscedacity(y, y_pred, pred_value_only = 1)
test_mutualinfo(error, Xtest, colname = None, bins = 5)
test_normality(error, distribution = "norm", test_size_limit = 5000)

-------------------------methods----------------------
dict2.__init__(self, d)


utilmy/zml/source/utils/__init__.py


utilmy/zml/source/utils/metrics.py


utilmy/zml/source/utils/util.py
-------------------------functions----------------------
create_appid(filename)
create_logfilename(filename)
create_uniqueid()
load(filename = "/folder1/keyname", isabsolutpath = 0, encoding1 = "utf-8")
load_arguments(config_file = None, arg_list = None)
logger_handler_console(formatter = None)
logger_handler_file(isrotate = False, rotate_time = "midnight", formatter = None, log_file_used = None)
logger_setup(logger_name = None, log_file = None, formatter = FORMATTER_1, isrotate = False, isconsole_output = True, logging_level = logging.DEBUG, )
logger_setup2(name = __name__, level = None)
os_make_dirs(filename)
printlog(s = "", s1 = "", s2 = "", s3 = "", s4 = "", s5 = "", s6 = "", s7 = "", s8 = "", s9 = "", s10 = "", app_id = "", logfile = None, iswritelog = True, )
save(variable_list, folder, globals_main = None)
save_all(variable_list, folder, globals_main = None)
sk_tree_get_ifthen(tree, feature_names, target_names, spacer_base = " ")
writelog(m = "", f = None)



utilmy/zml/source/utils/util_autofeature.py
-------------------------functions----------------------
create_model_name(save_folder, model_name)
data_loader(file_name = 'dataset/GOOG-year.csv')
load_arguments(config_file =  None)
optim_(modelname = "model_dl.1_lstm.py", pars =  {}, df  =  None, optim_engine = "optuna", optim_method = "normal/prune", save_folder = "model_save/", log_folder = "logs/", ntrials = 2)
optim_optuna(modelname = "model_dl.1_lstm.py", pars =  {}, df  =  None, optim_method = "normal/prune", save_folder = "/mymodel/", log_folder = "", ntrials = 2)
test_all()
test_fast()



utilmy/zml/source/utils/util_automl.py
-------------------------functions----------------------
import_(abs_module_path, class_name = None)
model_auto_automlgs(filepath= [ "train.csv", "test.csv" ],colX=None, coly=None,do="predict",outfolder="aaserialize/",model_type="regressor/classifier",params={ "csv_seprator"  =  [ "train.csv", "test.csv" ],colX=None, coly=None,do="predict",outfolder="aaserialize/",model_type="regressor/classifier",params={ "csv_seprator" : ",", "train_size" : 0.5, "score_metric" : "accuracy","n_folds": 3, "n_step": 10},param_space =  {'est__strategy':{"search":"choice",                         "space":["LightGBM"]},'est__n_estimators':{"search":"choice",                     "space":[150]},'est__colsample_bytree':{"search":"uniform",                "space":[0.8,0.95]},'est__subsample':{"search":"uniform",                       "space":[0.8,0.95]},'est__max_depth':{"search":"choice",                        "space":[5,6,7,8,9]},'est__learning_rate':{"search":"choice",                    "space":[0.07]}},generation=1,population_size=5,verbosity=2,)
model_auto_mlbox(filepath= [ "train.csv", "test.csv" ],colX=None, coly=None,do="predict",outfolder="aaserialize/",model_type="regressor/classifier",params={ "csv_seprator"  =  [ "train.csv", "test.csv" ],colX=None, coly=None,do="predict",outfolder="aaserialize/",model_type="regressor/classifier",params={ "csv_seprator" : ",", "train_size" : 0.5, "score_metric" : "accuracy","n_folds": 3, "n_step": 10},param_space =  {'est__strategy':{"search":"choice",                         "space":["LightGBM"]},'est__n_estimators':{"search":"choice",                     "space":[150]},'est__colsample_bytree':{"search":"uniform",                "space":[0.8,0.95]},'est__subsample':{"search":"uniform",                       "space":[0.8,0.95]},'est__max_depth':{"search":"choice",                        "space":[5,6,7,8,9]},'est__learning_rate':{"search":"choice",                    "space":[0.07]}},generation=1,population_size=5,verbosity=2,)
model_auto_tpot(df, colX, coly, outfolder = "aaserialize/", model_type = "regressor/classifier", train_size = 0.5, generation = 1, population_size = 5, verbosity = 2, )

-------------------------methods----------------------
dict2.__init__(self, d)


utilmy/zml/source/utils/util_credit.py
-------------------------functions----------------------
fun_get_segmentlimit(x, l1)
model_logistic_score(clf, df1, cols, coltarget, outype = "score")
np_drop_duplicates(l1)
pd_num_segment_limit(df, col_score = "scoress", coldefault = "y", ntotal_default = 491, def_list = None, nblock = 20.0)
split_train(X, y, split_ratio = 0.8)
split_train2(df1, ntrain = 10000, ntest = 100000, colused = None, coltarget = None, nratio = 0.4)
split_train_test(X, y, split_ratio = 0.8)
ztest()



utilmy/zml/source/utils/util_csv.py
-------------------------functions----------------------
csv_analysis()
csv_bigcompute()
csv_col_get_dict_categoryfreq(dircsv, filepattern = "*.csv", category_cols = [], maxline = -1, fileencoding = "utf-8")
csv_col_schema_toexcel(dircsv = "", filepattern = "*.csv", outfile = ".xlsx", returntable = 1, maxrow = 5000000, maxcol_pertable = 90, maxstrlen = "U80", )
csv_dtypes_getdict(df = None, csvfile = None)
csv_fast_processing()
csv_pivotable(dircsv = "", filepattern = "*.csv", fileh5 = ".h5", leftX = "col0", topY = "col2", centerZ = "coli", mapreduce = "sum", chunksize = 500000, tablename = "df", )
csv_row_mapreduce(dircsv = "", outfile = "", type_mapreduce = "sum", nrow = 1000000, chunk = 5000000)
csv_row_reduce_line(fromfile, tofile, condfilter, catval_tokeep, header = True, maxline = -1)
csv_row_reduce_line_manual(file_category, file_transact, file_reduced)
db_getdata()
db_meta_add("", []), schema = None, df_table_uri = None, df_table_columns = None)
db_meta_find(ALLDB, query = "", filter_db = [], filter_table = [], filter_column = [])
db_sql()
isnull(x)
str_to_unicode(x, encoding = "utf-8")
xl_get_rowcol(ws, i0, j0, imax, jmax)
xl_setstyle(file1)
xl_val(ws, colj, rowi)



utilmy/zml/source/utils/util_date.py
-------------------------functions----------------------
dateime_daytime(datetimex)
datenumpy_todatetime(tt, islocaltime = True)
datestring_todatetime(datelist, fmt="%Y-%m-%d %H = "%Y-%m-%d %H:%M:%S")
datetime_quarter(datetimex)
datetime_to_milisec(datelist)
datetime_toint(datelist)
datetime_tointhour(datelist)
datetime_tonumpydate(t, islocaltime = True)
datetime_tostring(datelist, fmt="%Y-%m-%d %H = "%Y-%m-%d %H:%M:%S")
datetime_weekday(datelist)
datetime_weekday_fast(dateval)
np_dict_tolist(dd)
np_dict_tostr_key(dd)
np_dict_tostr_val(dd)
pd_datestring_split(dfref, coldate, fmt="%Y-%m-%d %H = "%Y-%m-%d %H:%M:%S", return_val = "split")



utilmy/zml/source/utils/util_deep.py
-------------------------functions----------------------
tf_to_dot(graph)



utilmy/zml/source/utils/util_import.py
-------------------------methods----------------------
dict2.__init__(self, d)


utilmy/zml/source/utils/util_metric.py
-------------------------functions----------------------
average_precision(r)
dcg_at_k(r, k, method = 0)
mean_average_precision(rs)
mean_reciprocal_rank(rs)
ndcg_at_k(r, k, method = 0)
precision_at_k(r, k)
r_precision(r)



utilmy/zml/source/utils/util_model.py
-------------------------functions----------------------
import_(abs_module_path, class_name = None)
model_catboost_classifier(Xtrain, Ytrain, Xcolname = None, pars={"learning_rate" = {"learning_rate": 0.1, "iterations": 1000, "random_seed": 0, "loss_function": "MultiClass", }, isprint = 0, )
model_lightgbm_kfold(df, colname = None, num_folds = 2, stratified = False, colexclude = None, debug = False)
pd_dim_reduction(df, colname, colprefix = "colsvd", method = "svd", dimpca = 2, model_pretrain = None, return_val = "dataframe,param", )
sk_cluster(Xmat, method = "kmode", ), kwds={"metric" = {"metric": "euclidean", "min_cluster_size": 150, "min_samples": 3}, isprint = 1, preprocess={"norm" = {"norm": False}, )
sk_error(ypred, ytrue, method = "r2", sample_weight = None, multioutput = None)
sk_feature_concept_shift(df)
sk_feature_covariate_shift(dftrain, dftest, colname, nsample = 10000)
sk_feature_evaluation(clf, df, kbest = 30, colname_best = None, dfy = None)
sk_feature_impt(clf, colname, model_type = "logistic")
sk_feature_prior_shift()
sk_feature_selection(clf, method = "f_classif", colname = None, kbest = 50, Xtrain = None, ytrain = None)
sk_metric_roc_auc(y_test, ytest_pred, ytest_proba)
sk_metric_roc_auc_multiclass(n_classes = 3, y_test = None, y_test_pred = None, y_predict_proba = None)
sk_metric_roc_optimal_cutoff(ytest, ytest_proba)
sk_metrics_eval(clf, Xtest, ytest, cv = 1, metrics = ["f1_macro", "accuracy", "precision_macro", "recall_macro"])
sk_model_ensemble_weight(model_list, acclevel, maxlevel = 0.88)
sk_model_eval(clf, istrain = 1, Xtrain = None, ytrain = None, Xval = None, yval = None)
sk_model_eval_classification(clf, istrain = 1, Xtrain = None, ytrain = None, Xtest = None, ytest = None)
sk_model_eval_classification_cv(clf, X, y, test_size = 0.5, ncv = 1, method = "random")
sk_model_eval_regression(clf, istrain = 1, Xtrain = None, ytrain = None, Xval = None, yval = None)
sk_model_votingpredict(estimators, voting, ww, X_test)
sk_params_search_best(clf, X, y, 0, 1, 5)}, method = "gridsearch", param_search={"scorename" = {"scorename": "r2", "cv": 5, "population_size": 5, "generations_number": 3}, )
sk_score_get(name = "r2")
sk_showconfusion(Y, Ypred, isprint = True)
sk_showmetrics(y_test, ytest_pred, ytest_proba, target_names = ["0", "1"], return_stat = 0)

-------------------------methods----------------------
dict2.__init__(self, d)
model_template1.__init__(self, alpha = 0.5, low_y_cut = -0.09, high_y_cut = 0.09, ww0 = 0.95)
model_template1.fit(self, X, Y = None)
model_template1.predict(self, X, y = None, ymedian = None)
model_template1.score(self, X, Ytrue = None, ymedian = None)


utilmy/zml/source/utils/util_optim.py
-------------------------functions----------------------
create_model_name(save_folder, model_name)
data_loader(file_name = 'dataset/GOOG-year.csv')
load_arguments(config_file =  None)
optim(modelname = "model_dl.1_lstm.py", pars =  {}, df  =  None, optim_engine = "optuna", optim_method = "normal/prune", save_folder = "model_save/", log_folder = "logs/", ntrials = 2)
optim_optuna(modelname = "model_dl.1_lstm.py", pars =  {}, df  =  None, optim_method = "normal/prune", save_folder = "/mymodel/", log_folder = "", ntrials = 2)
test_all()
test_fast()



utilmy/zml/source/utils/util_pipeline.py
-------------------------functions----------------------
pd_grid_search(full_pipeline, X, y)
pd_pipeline(bin_cols, text_col, X, y)



utilmy/zml/source/utils/util_plot.py
-------------------------functions----------------------
pd_colnum_tocat_stat(input_data, feature, target_col, bins, cuts = 0)
pd_stat_distribution_trend_correlation(grouped, grouped_test, feature, target_col)
plot_XY(xx, yy, zcolor = None, tsize = None, labels = None, title = "", xlabel = "", ylabel = "", zcolor_label = "", 8, 6), dpi = 75, savefile = "", color_dot = "Blues", doreturn = 0, )
plot_XY_plotly(xx, yy, towhere = "url")
plot_XY_seaborn(X, Y, Zcolor = None)
plot_Y(Yval, typeplot = ".b", tsize = None, labels = None, title = "", xlabel = "", ylabel = "", zcolor_label = "", 8, 6), dpi = 75, savefile = "", color_dot = "Blues", doreturn = 0, )
plot_cluster_2D(X_2dim, target_class, target_names)
plot_cluster_hiearchy(Xmat_dist, p = 30, truncate_mode = None, color_threshold = None, get_leaves = True, orientation = "top", labels = None, count_sort = False, distance_sort = False, show_leaf_counts = True, do_plot = 1, no_labels = False, leaf_font_size = None, leaf_rotation = None, leaf_label_func = None, show_contracted = False, link_color_func = None, ax = None, above_threshold_color = "b", annotate_above = 0, )
plot_cluster_pca(Xmat, Xcluster_label = None, metric = "euclidean", dimpca = 2, whiten = True, isprecompute = False, savefile = "", doreturn = 1, )
plot_cluster_tsne(Xmat, Xcluster_label = None, metric = "euclidean", perplexity = 50, ncomponent = 2, savefile = "", isprecompute = False, returnval = True, )
plot_col_correl_matrix(df, cols, annot = True, size = 30)
plot_col_correl_target(df, cols, coltarget, nb_to_show = 10, ascending = False)
plot_col_distribution(df, col_include = None, col_exclude = None, pars={"binsize" = {"binsize": 20})
plot_col_univariate(input_data, feature, target_col, trend_correlation = None)
plot_cols_with_NaNs(df, nb_to_show)
plot_distance_heatmap(Xmat_dist, Xcolname)
plot_distribution_density(Xsample, kernel = "gaussian", N = 10, bandwith = 1 / 10.0)
plot_pair(df, Xcolname = None, Ycoltarget = None)
plot_plotly()
plot_univariate_histogram(feature, data, target_col, bins = 10, data_test = 0)
plot_univariate_plots(data, target_col, features_list = 0, bins = 10, data_test = 0)
plotbar(df, colname, figsize = (20, 10)
plotxy(12, 10), title = "feature importance", savefile = "myfile.png")



utilmy/zml/source/utils/util_sql.py
-------------------------functions----------------------
sql_create_dbengine(type1 = "", dbname = "", login = "", password = "", url = "localhost", port = 5432)
sql_delete_table(name, dbengine)
sql_get_dbschema(dburl="sqlite = "sqlite:///aapackage/store/yahoo.db", dbengine = None, isprint = 0)
sql_insert_csv(csvfile, dbtable, dbengine, col_drop = [])
sql_insert_csv2(csvfile = "", dbtable = "", columns = [], dbengine = None, nrows = 10000)
sql_insert_df(df, dbtable, dbengine, col_drop = ["id"], verbose = 1)
sql_insert_excel(file1 = ".xls", dbengine = None, dbtype = "")
sql_mysql_insert_excel()
sql_pivotable(dbcon, ss = "select  ")
sql_postgres_create_table(mytable = "", database = "", username = "", password = "")
sql_postgres_pivot()
sql_postgres_query_to_csv(sqlr = "SELECT ticker,shortratio,sector1_id, FROM stockfundamental", csv_out = "")
sql_query(sqlr = "SELECT ticker,shortratio,sector1_id, FROM stockfundamental", dbengine = None, output = "df", dburl="sqlite = "sqlite:///aaserialize/store/finviz.db", )



utilmy/zml/source/utils/util_stat.py
-------------------------functions----------------------
np_conditional_entropy(x, y)
np_correl_cat_cat_cramers_v(x, y)
np_correl_cat_cat_theils_u(x, y)
np_correl_cat_num_ratio(cat_array, num_array)
np_transform_pca(X, dimpca = 2, whiten = True)
pd_num_correl_associations(df, colcat = None, mark_columns = False, theil_u = False, plot = True, return_results = False, **kwargs)
sk_distribution_kernel_bestbandwidth(X, kde)
sk_distribution_kernel_sample(kde = None, n = 1)
stat_hypothesis_test_permutation(df, variable, classes, repetitions)

-------------------------methods----------------------
dict2.__init__(self, d)


utilmy/zml/source/utils/util_text.py
-------------------------functions----------------------
coltext_lemmatizer(text)
coltext_stemmer(text, sep = " ")
coltext_stemporter(text)
coltext_stopwords(text, stopwords = None, sep = " ")
get_stopwords(lang)
pd_coltext_clean(dfref, colname, stopwords)
pd_coltext_clean_advanced(dfref, colname, fromword, toword)
pd_coltext_countvect(df, coltext, word_tokeep = None, word_minfreq = 1, return_val = "dataframe,param")
pd_coltext_encoder(df)
pd_coltext_fillna(df, colname, val = "")
pd_coltext_hashing(df, coltext, n_features = 20)
pd_coltext_minhash(dfref, colname, n_component = 2, model_pretrain_dict = None, return_val = "dataframe,param")
pd_coltext_tdidf(df, coltext, word_tokeep = None, word_minfreq = 1, return_val = "dataframe,param")
pd_coltext_tdidf_multi(df, coltext, coltext_freq, ntoken = 100, word_tokeep_dict = None, stopwords = None, return_val = "dataframe,param", )
pd_coltext_wordfreq(df, coltext, sep = " ")
pd_fromdict(ddict, colname)



utilmy/zml/source/utils/util_text_embedding.py
-------------------------functions----------------------
test_MDVEncoder()

-------------------------methods----------------------
AdHocIndependentPDF.__init__(self, fisher_kernel = True, dtype = np.float64, ngram_range = (2, 4)
AdHocIndependentPDF.fit(self, X, y = None)
AdHocIndependentPDF.transform(self, X)
AdHocNgramsMultinomialMixture.__init__(self, n_iters = 10, fisher_kernel = True, ngram_range = (2, 4)
AdHocNgramsMultinomialMixture._e_step(self, D, unqD, X, unqX, theta, beta)
AdHocNgramsMultinomialMixture._m_step(self, D, _doc_topic_posterior)
AdHocNgramsMultinomialMixture.fit(self, X, y = None)
AdHocNgramsMultinomialMixture.transform(self, X)
ColumnEncoder.__init__(self, encoder_name, reduction_method = None, 2, 4), categories = "auto", dtype = np.float64, handle_unknown = "ignore", clf_type = None, n_components = None, )
ColumnEncoder._get_most_frequent(self, X)
ColumnEncoder.fit(self, X, y = None)
ColumnEncoder.get_feature_names(self)
ColumnEncoder.transform(self, X)
DimensionalityReduction.__init__(self, method_name = None, n_components = None, column_names = None)
DimensionalityReduction.fit(self, X, y = None)
DimensionalityReduction.transform(self, X)
MDVEncoder.__init__(self, clf_type)
MDVEncoder.fit(self, X, y = None)
MDVEncoder.transform(self, X)
NgramNaiveFisherKernel.__init__(self, 2, 4), categories = "auto", dtype = np.float64, handle_unknown = "ignore", hashing_dim = None, n_prototypes = None, random_state = None, n_jobs = None, )
NgramNaiveFisherKernel._ngram_presence_fisher_kernel(self, strings, cats)
NgramNaiveFisherKernel._ngram_presence_fisher_kernel2(self, strings, cats)
NgramNaiveFisherKernel.fit(self, X, y = None)
NgramNaiveFisherKernel.transform(self, X)
NgramsMultinomialMixture.__init__(self, n_topics = 10, max_iters = 100, fisher_kernel = True, beta_init_type = None, max_mean_change_tol = 1e-5, 2, 4), )
NgramsMultinomialMixture._e_step(self, D, unqD, X, unqX, theta, beta)
NgramsMultinomialMixture._get_most_frequent(self, X)
NgramsMultinomialMixture._m_step(self, D, _doc_topic_posterior)
NgramsMultinomialMixture._max_mean_change(self, last_beta, beta)
NgramsMultinomialMixture.fit(self, X, y = None)
NgramsMultinomialMixture.transform(self, X)
PasstroughEncoder.__init__(self, passthrough = True)
PasstroughEncoder.fit(self, X, y = None)
PasstroughEncoder.transform(self, X)
PretrainedBert.fit(self, X, y = None)
PretrainedBert.transform(self, X: list)
PretrainedFastText.__init__(self, n_components, language = "english")
PretrainedFastText.fit(self, X, y = None)
PretrainedFastText.transform(self, X)
PretrainedGensim.__get_word_embedding(self, word, model)
PretrainedGensim.__word_forms(self, word)
PretrainedGensim.fit(self, X, y = None)
PretrainedGensim.transform(self, X: dict)
PretrainedWord2Vec.__init__(self, n_components = None, language = "english", model_path = None, bert_args={'bert_model' = {'bert_model': None, 'bert_dataset_name': None, 'oov': 'sum', 'ctx': None})
PretrainedWord2Vec.fit(self, X, y = None)
PretrainedWord2Vec.transform(self, X)


utilmy/zml/source/utils/ztest.py
-------------------------methods----------------------
dict2.__init__(self, d)


utilmy/zml/titanic_classifier.py
-------------------------functions----------------------
check()
config1()
global_pars_update(model_dict, data_name, config_name)
pd_col_myfun(df = None, col = None, pars = {})



utilmy/zml/toutlier.py
-------------------------functions----------------------
global_pars_update(model_dict, data_name, config_name, dir_data = None, dir_input_tr = None, dir_input_te = None)
post_process_fun(y)
pre_process_fun(y)



utilmy/zml/tsampler.py
-------------------------functions----------------------
config_sampler()
global_pars_update(model_dict, data_name, config_name)
log(*s)
test_batch(nsample = 1000)



utilmy/zml/tseries.py
-------------------------functions----------------------
config1()
global_pars_update(model_dict, data_name, config_name)
pd_dsa2_custom(df: pd.DataFrame, col: list = None, pars: dict = None)



utilmy/zml/zgitutil.py
-------------------------functions----------------------
_filter_on_size(size = 0, f = files)
_run(*args)
add(size = 10000000)
commit(mylist)
main()
path_leaf(path)



utilmy/zml/ztemplate.py
-------------------------functions----------------------
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
get_dataset(Xtrain, cols_type_received, cols_ref)
get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
init(*kw, **kwargs)
load_model(path = "")
load_model(path = "")
predict(Xpred = None, data_pars = None, compute_pars = {}, out_pars = {}, **kw)
predict(Xpred = None, data_pars = None, compute_pars = {}, out_pars = {}, **kw)
reset()
reset()
save(path = None, info = None)
save(path = None, info = None)
test(config = '')
test_helper(model_pars, data_pars, compute_pars)

-------------------------methods----------------------
MY_MODEL_CLASS.__init__(cpars)
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy/zrecs/contrib/azureml_designer_modules/entries/map_entry.py


utilmy/zrecs/contrib/azureml_designer_modules/entries/ndcg_entry.py


utilmy/zrecs/contrib/azureml_designer_modules/entries/precision_at_k_entry.py


utilmy/zrecs/contrib/azureml_designer_modules/entries/recall_at_k_entry.py


utilmy/zrecs/contrib/azureml_designer_modules/entries/score_sar_entry.py
-------------------------functions----------------------
joblib_loader(load_from_dir, model_spec)

-------------------------methods----------------------
ScoreSARModule.__init__(self, model, input_data)
ScoreSARModule.input_data(self)
ScoreSARModule.model(self)
ScoreSARModule.predict_ratings(self, items_to_predict, normalize)
ScoreSARModule.recommend_items(self, ranking_metric, top_k, sort_top_k, remove_seen, normalize)


utilmy/zrecs/contrib/azureml_designer_modules/entries/stratified_splitter_entry.py


utilmy/zrecs/contrib/azureml_designer_modules/entries/train_sar_entry.py
-------------------------functions----------------------
joblib_dumper(data, file_name = None)



utilmy/zrecs/contrib/sarplus/python/setup.py
-------------------------methods----------------------
get_pybind_include.__init__(self, user = False)
get_pybind_include.__str__(self)


utilmy/zrecs/docs/source/conf.py


utilmy/zrecs/examples/04_model_select_and_optimize/train_scripts/svd_training.py
-------------------------functions----------------------
main()
svd_training(args)



utilmy/zrecs/examples/04_model_select_and_optimize/train_scripts/wide_deep_training.py
-------------------------functions----------------------
_log(metric, value)



utilmy/zrecs/examples/06_benchmarks/benchmark_utils.py
-------------------------functions----------------------
predict_als(model, test)
predict_fastai(model, test)
predict_svd(model, test)
prepare_metrics_als(train, test)
prepare_metrics_fastai(train, test)
prepare_training_als(train, test)
prepare_training_cornac(train, test)
prepare_training_fastai(train, test)
prepare_training_lightgcn(train, test)
prepare_training_ncf(train, test)
prepare_training_sar(train, test)
prepare_training_svd(train, test)
ranking_metrics_pyspark(test, predictions, k = DEFAULT_K)
ranking_metrics_python(test, predictions, k = DEFAULT_K)
rating_metrics_pyspark(test, predictions)
rating_metrics_python(test, predictions)
recommend_k_als(model, test, train, top_k = DEFAULT_K, remove_seen = True)
recommend_k_cornac(model, test, train, top_k = DEFAULT_K, remove_seen = True)
recommend_k_fastai(model, test, train, top_k = DEFAULT_K, remove_seen = True)
recommend_k_lightgcn(model, test, train, top_k = DEFAULT_K, remove_seen = True)
recommend_k_ncf(model, test, train, top_k = DEFAULT_K, remove_seen = True)
recommend_k_sar(model, test, train, top_k = DEFAULT_K, remove_seen = True)
recommend_k_svd(model, test, train, top_k = DEFAULT_K, remove_seen = True)
train_als(params, data)
train_bivae(params, data)
train_bpr(params, data)
train_fastai(params, data)
train_lightgcn(params, data)
train_ncf(params, data)
train_sar(params, data)
train_svd(params, data)



utilmy/zrecs/recommenders/__init__.py


utilmy/zrecs/recommenders/datasets/__init__.py


utilmy/zrecs/recommenders/datasets/amazon_reviews.py
-------------------------functions----------------------
_create_instance(reviews_file, meta_file)
_create_item2cate(instance_file)
_create_vocab(train_file, user_vocab, item_vocab, cate_vocab)
_data_generating(input_file, train_file, valid_file, test_file, min_sequence = 1)
_data_generating_no_history_expanding(input_file, train_file, valid_file, test_file, min_sequence = 1)
_data_processing(input_file)
_download_reviews(name, dest_path)
_extract_reviews(file_path, zip_path)
_get_sampled_data(instance_file, sample_rate)
_meta_preprocessing(meta_readfile)
_negative_sampling_offline(instance_input_file, valid_file, test_file, valid_neg_nums = 4, test_neg_nums = 49)
_reviews_preprocessing(reviews_readfile)
data_preprocessing(reviews_file, meta_file, train_file, valid_file, test_file, user_vocab, item_vocab, cate_vocab, sample_rate = 0.01, valid_num_ngs = 4, test_num_ngs = 9, is_history_expanding = True, )
download_and_extract(name, dest_path)



utilmy/zrecs/recommenders/datasets/cosmos_cli.py
-------------------------functions----------------------
find_collection(client, dbid, id)
find_database(client, id)
read_collection(client, dbid, id)
read_database(client, id)



utilmy/zrecs/recommenders/datasets/covid_utils.py
-------------------------functions----------------------
clean_dataframe(df)
get_public_domain_text(df, container_name, azure_storage_account_name = "azureopendatastorage", azure_storage_sas_token = "", )
load_pandas_df(azure_storage_account_name = "azureopendatastorage", azure_storage_sas_token = "", container_name = "covid19temp", metadata_filename = "metadata.csv", )
remove_duplicates(df, cols)
remove_nan(df, cols)
retrieve_text(entry, container_name, azure_storage_account_name = "azureopendatastorage", azure_storage_sas_token = "", )



utilmy/zrecs/recommenders/datasets/criteo.py
-------------------------functions----------------------
download_criteo(size = "sample", work_directory = ".")
extract_criteo(size, compressed_file, path = None)
get_spark_schema(header = DEFAULT_HEADER)
load_pandas_df(size = "sample", local_cache_path = None, header = DEFAULT_HEADER)
load_spark_df(spark, size = "sample", header = DEFAULT_HEADER, local_cache_path = None, dbfs_datapath="dbfs = "dbfs:/FileStore/dac", dbutils = None, )



utilmy/zrecs/recommenders/datasets/download_utils.py
-------------------------functions----------------------
download_path(path = None)
maybe_download(url, filename = None, work_directory = ".", expected_bytes = None)
unzip_file(zip_src, dst_dir, clean_zip_file = False)



utilmy/zrecs/recommenders/datasets/mind.py
-------------------------functions----------------------
_newsample(nnn, ratio)
_read_news(filepath, news_words, news_entities, tokenizer)
download_and_extract_glove(dest_path)
download_mind(size = "small", dest_path = None)
extract_mind(train_zip, valid_zip, train_folder = "train", valid_folder = "valid", clean_zip_file = True, )
generate_embeddings(data_path, news_words, news_entities, train_entities, valid_entities, max_sentence = 10, word_embedding_dim = 100, )
get_train_input(session, train_file_path, npratio = 4)
get_user_history(train_history, valid_history, user_history_path)
get_valid_input(session, valid_file_path)
get_words_and_entities(train_news, valid_news)
load_glove_matrix(path_emb, word_dict, word_embedding_dim)
read_clickhistory(path, filename)
word_tokenize(sent)



utilmy/zrecs/recommenders/datasets/movielens.py
-------------------------functions----------------------
_get_schema(header, schema)
_load_item_df(size, item_datapath, movie_col, title_col, genres_col, year_col)
_maybe_download_and_extract(size, dest_path)
download_movielens(size, dest_path)
extract_movielens(size, rating_path, item_path, zip_path)
load_item_df(size = "100k", local_cache_path = None, movie_col = DEFAULT_ITEM_COL, title_col = None, genres_col = None, year_col = None, )
load_pandas_df(size = "100k", header = None, local_cache_path = None, title_col = None, genres_col = None, year_col = None, )
load_spark_df(spark, size = "100k", header = None, schema = None, local_cache_path = None, dbutils = None, title_col = None, genres_col = None, year_col = None, )

-------------------------methods----------------------
_DataFormat.__init__(self, sep, path, has_header = False, item_sep = None, item_path = None, item_has_header = False, )
_DataFormat.has_header(self)
_DataFormat.item_has_header(self)
_DataFormat.item_path(self)
_DataFormat.item_separator(self)
_DataFormat.path(self)
_DataFormat.separator(self)


utilmy/zrecs/recommenders/datasets/pandas_df_utils.py
-------------------------functions----------------------
filter_by(df, filter_by_df, filter_by_cols)
has_columns(df, columns)
has_same_base_dtype(df_1, df_2, columns = None)
lru_cache_df(maxsize, typed = False)
negative_feedback_sampler(df, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_label = DEFAULT_LABEL_COL, col_feedback = "feedback", ratio_neg_per_user = 1, pos_value = 1, neg_value = 0, seed = 42, )
user_item_pairs(user_df, item_df, user_col = DEFAULT_USER_COL, item_col = DEFAULT_ITEM_COL, user_item_filter_df = None, shuffle = True, seed = None, )

-------------------------methods----------------------
LibffmConverter.__init__(self, filepath = None)
LibffmConverter.fit(self, df, col_rating = DEFAULT_RATING_COL)
LibffmConverter.fit_transform(self, df, col_rating = DEFAULT_RATING_COL)
LibffmConverter.get_params(self)
LibffmConverter.transform(self, df)
PandasHash.__eq__(self, other)
PandasHash.__hash__(self)
PandasHash.__init__(self, pandas_object)


utilmy/zrecs/recommenders/datasets/python_splitters.py
-------------------------functions----------------------
_do_stratification(data, ratio = 0.75, min_rating = 1, filter_by = "user", is_random = True, seed = 42, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_timestamp = DEFAULT_TIMESTAMP_COL, )
numpy_stratified_split(X, ratio = 0.75, seed = 42)
python_chrono_split(data, ratio = 0.75, min_rating = 1, filter_by = "user", col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_timestamp = DEFAULT_TIMESTAMP_COL, )
python_random_split(data, ratio = 0.75, seed = 42)
python_stratified_split(data, ratio = 0.75, min_rating = 1, filter_by = "user", col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, seed = 42, )



utilmy/zrecs/recommenders/datasets/spark_splitters.py
-------------------------functions----------------------
_do_stratification_spark(data, ratio = 0.75, min_rating = 1, filter_by = "user", is_partitioned = True, is_random = True, seed = 42, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_timestamp = DEFAULT_TIMESTAMP_COL, )
spark_chrono_split(data, ratio = 0.75, min_rating = 1, filter_by = "user", col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_timestamp = DEFAULT_TIMESTAMP_COL, no_partition = False, )
spark_random_split(data, ratio = 0.75, seed = 42)
spark_stratified_split(data, ratio = 0.75, min_rating = 1, filter_by = "user", col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, seed = 42, )
spark_timestamp_split(data, ratio = 0.75, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_timestamp = DEFAULT_TIMESTAMP_COL, )



utilmy/zrecs/recommenders/datasets/sparse.py
-------------------------methods----------------------
AffinityMatrix.__init__(self, df, items_list = None, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_pred = DEFAULT_PREDICTION_COL, save_path = None, )
AffinityMatrix._gen_index(self)
AffinityMatrix.gen_affinity_matrix(self)
AffinityMatrix.map_back_sparse(self, X, kind)


utilmy/zrecs/recommenders/datasets/split_utils.py
-------------------------functions----------------------
_get_column_name(name, col_user, col_item)
min_rating_filter_pandas(data, min_rating = 1, filter_by = "user", col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, )
min_rating_filter_spark(data, min_rating = 1, filter_by = "user", col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, )
process_split_ratio(ratio)
split_pandas_data_with_ratios(data, ratios, seed = 42, shuffle = False)



utilmy/zrecs/recommenders/datasets/wikidata.py
-------------------------functions----------------------
find_wikidata_id(name, limit = 1, session = None)
get_session(session = None)
query_entity_description(entity_id, session = None)
query_entity_links(entity_id, session = None)
read_linked_entities(data)
search_wikidata(names, extras = None, describe = True, verbose = False)



utilmy/zrecs/recommenders/evaluation/__init__.py


utilmy/zrecs/recommenders/evaluation/python_evaluation.py
-------------------------functions----------------------
_check_column_dtypes(func)
_check_column_dtypes_diversity_serendipity(func)
_check_column_dtypes_novelty_coverage(func)
_get_cooccurrence_similarity(train_df, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_sim = DEFAULT_SIMILARITY_COL, )
_get_cosine_similarity(train_df, item_feature_df = None, item_sim_measure = DEFAULT_ITEM_SIM_MEASURE, col_item_features = DEFAULT_ITEM_FEATURES_COL, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_sim = DEFAULT_SIMILARITY_COL, )
_get_intralist_similarity(train_df, reco_df, item_feature_df = None, item_sim_measure = DEFAULT_ITEM_SIM_MEASURE, col_item_features = DEFAULT_ITEM_FEATURES_COL, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_sim = DEFAULT_SIMILARITY_COL, )
_get_item_feature_similarity(item_feature_df, col_item_features = DEFAULT_ITEM_FEATURES_COL, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_sim = DEFAULT_SIMILARITY_COL, )
_get_pairwise_items(df, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, )
auc(rating_true, rating_pred, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_prediction = DEFAULT_PREDICTION_COL, )
catalog_coverage(train_df, reco_df, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL)
distributional_coverage(train_df, reco_df, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL)
diversity(train_df, reco_df, item_feature_df = None, item_sim_measure = DEFAULT_ITEM_SIM_MEASURE, col_item_features = DEFAULT_ITEM_FEATURES_COL, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_sim = DEFAULT_SIMILARITY_COL, col_relevance = None, )
exp_var(rating_true, rating_pred, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_prediction = DEFAULT_PREDICTION_COL, )
get_top_k_items(dataframe, col_user = DEFAULT_USER_COL, col_rating = DEFAULT_RATING_COL, k = DEFAULT_K)
historical_item_novelty(train_df, reco_df, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, )
logloss(rating_true, rating_pred, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_prediction = DEFAULT_PREDICTION_COL, )
mae(rating_true, rating_pred, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_prediction = DEFAULT_PREDICTION_COL, )
map_at_k(rating_true, rating_pred, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_prediction = DEFAULT_PREDICTION_COL, relevancy_method = "top_k", k = DEFAULT_K, threshold = DEFAULT_THRESHOLD, )
merge_ranking_true_pred(rating_true, rating_pred, col_user, col_item, col_rating, col_prediction, relevancy_method, k = DEFAULT_K, threshold = DEFAULT_THRESHOLD, )
merge_rating_true_pred(rating_true, rating_pred, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_prediction = DEFAULT_PREDICTION_COL, )
ndcg_at_k(rating_true, rating_pred, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_prediction = DEFAULT_PREDICTION_COL, relevancy_method = "top_k", k = DEFAULT_K, threshold = DEFAULT_THRESHOLD, )
novelty(train_df, reco_df, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL)
precision_at_k(rating_true, rating_pred, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_prediction = DEFAULT_PREDICTION_COL, relevancy_method = "top_k", k = DEFAULT_K, threshold = DEFAULT_THRESHOLD, )
recall_at_k(rating_true, rating_pred, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_prediction = DEFAULT_PREDICTION_COL, relevancy_method = "top_k", k = DEFAULT_K, threshold = DEFAULT_THRESHOLD, )
rmse(rating_true, rating_pred, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_prediction = DEFAULT_PREDICTION_COL, )
rsquared(rating_true, rating_pred, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_prediction = DEFAULT_PREDICTION_COL, )
serendipity(train_df, reco_df, item_feature_df = None, item_sim_measure = DEFAULT_ITEM_SIM_MEASURE, col_item_features = DEFAULT_ITEM_FEATURES_COL, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_sim = DEFAULT_SIMILARITY_COL, col_relevance = None, )
user_diversity(train_df, reco_df, item_feature_df = None, item_sim_measure = DEFAULT_ITEM_SIM_MEASURE, col_item_features = DEFAULT_ITEM_FEATURES_COL, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_sim = DEFAULT_SIMILARITY_COL, col_relevance = None, )
user_item_serendipity(train_df, reco_df, item_feature_df = None, item_sim_measure = DEFAULT_ITEM_SIM_MEASURE, col_item_features = DEFAULT_ITEM_FEATURES_COL, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_sim = DEFAULT_SIMILARITY_COL, col_relevance = None, )
user_serendipity(train_df, reco_df, item_feature_df = None, item_sim_measure = DEFAULT_ITEM_SIM_MEASURE, col_item_features = DEFAULT_ITEM_FEATURES_COL, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_sim = DEFAULT_SIMILARITY_COL, col_relevance = None, )



utilmy/zrecs/recommenders/evaluation/spark_evaluation.py
-------------------------functions----------------------
_get_relevant_items_by_threshold(dataframe, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_prediction = DEFAULT_PREDICTION_COL, threshold = DEFAULT_THRESHOLD, )
_get_relevant_items_by_timestamp(dataframe, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_timestamp = DEFAULT_TIMESTAMP_COL, col_prediction = DEFAULT_PREDICTION_COL, k = DEFAULT_K, )
_get_top_k_items(dataframe, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_prediction = DEFAULT_PREDICTION_COL, k = DEFAULT_K, )

-------------------------methods----------------------
SparkDiversityEvaluation.__init__(self, train_df, reco_df, item_feature_df = None, item_sim_measure = DEFAULT_ITEM_SIM_MEASURE, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_relevance = None, )
SparkDiversityEvaluation._get_cooccurrence_similarity(self, n_partitions)
SparkDiversityEvaluation._get_cosine_similarity(self, n_partitions = 200)
SparkDiversityEvaluation._get_intralist_similarity(self, df)
SparkDiversityEvaluation._get_item_feature_similarity(self, n_partitions)
SparkDiversityEvaluation._get_pairwise_items(self, df)
SparkDiversityEvaluation.catalog_coverage(self)
SparkDiversityEvaluation.distributional_coverage(self)
SparkDiversityEvaluation.diversity(self)
SparkDiversityEvaluation.historical_item_novelty(self)
SparkDiversityEvaluation.novelty(self)
SparkDiversityEvaluation.serendipity(self)
SparkDiversityEvaluation.sim_cos(v1, v2)
SparkDiversityEvaluation.user_diversity(self)
SparkDiversityEvaluation.user_item_serendipity(self)
SparkDiversityEvaluation.user_serendipity(self)
SparkRankingEvaluation.__init__(self, rating_true, rating_pred, k = DEFAULT_K, relevancy_method = "top_k", col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_prediction = DEFAULT_PREDICTION_COL, threshold = DEFAULT_THRESHOLD, )
SparkRankingEvaluation._calculate_metrics(self)
SparkRankingEvaluation.map_at_k(self)
SparkRankingEvaluation.ndcg_at_k(self)
SparkRankingEvaluation.precision_at_k(self)
SparkRankingEvaluation.recall_at_k(self)
SparkRatingEvaluation.__init__(self, rating_true, rating_pred, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_prediction = DEFAULT_PREDICTION_COL, )
SparkRatingEvaluation.exp_var(self)
SparkRatingEvaluation.mae(self)
SparkRatingEvaluation.rmse(self)
SparkRatingEvaluation.rsquared(self)


utilmy/zrecs/recommenders/models/__init__.py


utilmy/zrecs/recommenders/models/cornac/__init__.py


utilmy/zrecs/recommenders/models/cornac/cornac_utils.py
-------------------------functions----------------------
predict(model, data, usercol = DEFAULT_USER_COL, itemcol = DEFAULT_ITEM_COL, predcol = DEFAULT_PREDICTION_COL, )
predict_ranking(model, data, usercol = DEFAULT_USER_COL, itemcol = DEFAULT_ITEM_COL, predcol = DEFAULT_PREDICTION_COL, remove_seen = False, )



utilmy/zrecs/recommenders/models/deeprec/__init__.py


utilmy/zrecs/recommenders/models/deeprec/deeprec_utils.py
-------------------------functions----------------------
cal_metric(labels, preds, metrics)
check_nn_config(f_config)
check_type(config)
create_hparams(flags)
dcg_score(y_true, y_score, k = 10)
download_deeprec_resources(azure_container_url, data_path, remote_resource_name)
flat_config(config)
hit_score(y_true, y_score, k = 10)
load_dict(filename)
load_yaml(filename)
mrr_score(y_true, y_score)
ndcg_score(y_true, y_score, k = 10)
prepare_hparams(yaml_file = None, **kwargs)



utilmy/zrecs/recommenders/models/fastai/__init__.py


utilmy/zrecs/recommenders/models/fastai/fastai_utils.py
-------------------------functions----------------------
cartesian_product(*arrays)
hide_fastai_progress_bar()
score(learner, test_df, user_col = cc.DEFAULT_USER_COL, item_col = cc.DEFAULT_ITEM_COL, prediction_col = cc.DEFAULT_PREDICTION_COL, top_k = None, )



utilmy/zrecs/recommenders/models/geoimc/__init__.py


utilmy/zrecs/recommenders/models/geoimc/geoimc_algorithm.py
-------------------------methods----------------------
IMCProblem.__init__(self, dataPtr, lambda1 = 1e-2, rank = 10)
IMCProblem._computeLoss_csrmatrix(a, b, cd, indices, indptr, residual_global)
IMCProblem._cost(self, params, residual_global)
IMCProblem._egrad(self, params, residual_global)
IMCProblem._loadTarget(self, )
IMCProblem._optimize(self, max_opt_time, max_opt_iter, verbosity)
IMCProblem.reset(self)
IMCProblem.solve(self, *args)


utilmy/zrecs/recommenders/models/geoimc/geoimc_data.py
-------------------------methods----------------------
DataPtr.__init__(self, data, entities)
DataPtr.get_data(self)
DataPtr.get_entity(self, of = "row")
Dataset.__init__(self, name, features_dim = 0, normalize = False, target_transform = "")
Dataset.generate_train_test_data(self, data, test_ratio = 0.3)
Dataset.normalize(self)
Dataset.reduce_dims(self)
ML_100K.__init__(self, **kwargs)
ML_100K._load_item_features(self, path)
ML_100K._load_user_features(self, path)
ML_100K._read_from_file(self, path)
ML_100K.df2coo(self, df)
ML_100K.load_data(self, path)


utilmy/zrecs/recommenders/models/geoimc/geoimc_predict.py
-------------------------methods----------------------
Inferer.__init__(self, method = "dot", k = 10, transformation = "")
Inferer._get_method(self, k)
Inferer.infer(self, dataPtr, W, **kwargs)
PlainScalarProduct.__init__(self, X, Y, **kwargs)
PlainScalarProduct.sim(self, **kwargs)


utilmy/zrecs/recommenders/models/geoimc/geoimc_utils.py
-------------------------functions----------------------
length_normalize(matrix)
mean_center(matrix)
reduce_dims(matrix, target_dim)



utilmy/zrecs/recommenders/models/lightfm/__init__.py


utilmy/zrecs/recommenders/models/lightfm/lightfm_utils.py
-------------------------functions----------------------
compare_metric(df_list, metric = "prec", stage = "test")
model_perf_plots(df)
prepare_all_predictions(data, uid_map, iid_map, interactions, model, num_threads, user_features = None, item_features = None, )
prepare_test_df(test_idx, uids, iids, uid_map, iid_map, weights)
similar_items(item_id, item_features, model, N = 10)
similar_users(user_id, user_features, model, N = 10)
track_model_metrics(model, train_interactions, test_interactions, k = 10, no_epochs = 100, no_threads = 8, show_plot = True, **kwargs)



utilmy/zrecs/recommenders/models/lightgbm/__init__.py


utilmy/zrecs/recommenders/models/lightgbm/lightgbm_utils.py
-------------------------functions----------------------
unpackbits(x, num_bits)

-------------------------methods----------------------
NumEncoder.__init__(self, cate_cols, nume_cols, label_col, threshold = 10, thresrate = 0.99)
NumEncoder.fit_transform(self, df)
NumEncoder.transform(self, df)


utilmy/zrecs/recommenders/models/ncf/__init__.py


utilmy/zrecs/recommenders/models/ncf/dataset.py
-------------------------methods----------------------
Dataset.__init__(self, train, test = None, n_neg = 4, n_neg_test = 100, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_timestamp = DEFAULT_TIMESTAMP_COL, binary = True, seed = None, )
Dataset._data_processing(self, train, test, binary)
Dataset._init_test_data(self)
Dataset._init_train_data(self)
Dataset._reindex(self, df, binary)
Dataset.negative_sampling(self)
Dataset.test_loader(self)
Dataset.train_loader(self, batch_size, shuffle = True)


utilmy/zrecs/recommenders/models/ncf/ncf_singlenode.py
-------------------------methods----------------------
NCF.__init__(self, n_users, n_items, model_type = "NeuMF", n_factors = 8, layer_sizes = [16, 8, 4], n_epochs = 50, batch_size = 64, learning_rate = 5e-3, verbose = 1, seed = None, )
NCF._create_model(self, )
NCF._load_neumf(self, gmf_dir, mlp_dir, alpha)
NCF._predict(self, user_input, item_input)
NCF.fit(self, data)
NCF.load(self, gmf_dir = None, mlp_dir = None, neumf_dir = None, alpha = 0.5)
NCF.predict(self, user_input, item_input, is_list = False)
NCF.save(self, dir_name)


utilmy/zrecs/recommenders/models/newsrec/__init__.py


utilmy/zrecs/recommenders/models/newsrec/newsrec_utils.py
-------------------------functions----------------------
check_nn_config(f_config)
check_type(config)
create_hparams(flags)
get_mind_data_set(type)
newsample(news, ratio)
prepare_hparams(yaml_file = None, **kwargs)
word_tokenize(sent)



utilmy/zrecs/recommenders/models/rbm/__init__.py


utilmy/zrecs/recommenders/models/rbm/rbm.py
-------------------------methods----------------------
RBM.__init__(self, hidden_units = 500, keep_prob = 0.7, init_stdv = 0.1, learning_rate = 0.004, minibatch_size = 100, training_epoch = 20, display_epoch = 10, sampling_protocol = [50, 70, 80, 90, 100], debug = False, with_metrics = False, seed = 42, )
RBM.accuracy(self, vp)
RBM.batch_training(self, num_minibatches)
RBM.binomial_sampling(self, pr)
RBM.data_pipeline(self)
RBM.display_metrics(self, Rmse_train, precision_train, precision_test)
RBM.eval_out(self)
RBM.fit(self, xtr, xtst)
RBM.free_energy(self, x)
RBM.generate_graph(self)
RBM.gibbs_protocol(self, i)
RBM.gibbs_sampling(self)
RBM.init_gpu(self)
RBM.init_metrics(self)
RBM.init_parameters(self)
RBM.init_training_session(self, xtr)
RBM.losses(self, vv)
RBM.multinomial_distribution(self, phi)
RBM.multinomial_sampling(self, pr)
RBM.placeholder(self)
RBM.predict(self, x, maps)
RBM.recommend_k_items(self, x, top_k = 10, remove_seen = True)
RBM.rmse(self, vp)
RBM.sample_hidden_units(self, vv)
RBM.sample_visible_units(self, h)
RBM.time(self)
RBM.train_test_precision(self, xtst)


utilmy/zrecs/recommenders/models/rlrmc/RLRMCalgorithm.py
-------------------------methods----------------------
RLRMCalgorithm.__init__(self, rank, C, model_param, initialize_flag = "random", max_time = 1000, maxiter = 100, seed = 42, )
RLRMCalgorithm._computeLoss_csrmatrix(a, b, cd, indices, indptr, residual_global)
RLRMCalgorithm._cost(self, weights, entries_train_csr_data, entries_train_csr_indices, entries_train_csr_indptr, residual_global, )
RLRMCalgorithm._egrad(self, weights, entries_train_csr_indices, entries_train_csr_indptr, residual_global, )
RLRMCalgorithm._init_train(self, entries_train_csr)
RLRMCalgorithm._my_stats(self, weights, given_stats, stats, residual_global, entries_validation_csr_data = None, entries_validation_csr_indices = None, entries_validation_csr_indptr = None, residual_validation_global = None, )
RLRMCalgorithm.fit(self, RLRMCdata, verbosity = 0)
RLRMCalgorithm.fit_and_evaluate(self, RLRMCdata, verbosity = 0)
RLRMCalgorithm.predict(self, user_input, item_input, low_memory = False)


utilmy/zrecs/recommenders/models/rlrmc/RLRMCdataset.py
-------------------------methods----------------------
RLRMCdataset.__init__(self, train, validation = None, test = None, mean_center = True, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_timestamp = DEFAULT_TIMESTAMP_COL, )
RLRMCdataset._data_processing(self, train, validation = None, test = None, mean_center = True)
RLRMCdataset._reindex(self, df)


utilmy/zrecs/recommenders/models/rlrmc/__init__.py


utilmy/zrecs/recommenders/models/rlrmc/conjugate_gradient_ms.py
-------------------------methods----------------------
ConjugateGradientMS.__init__(self, beta_type = BetaTypes.HestenesStiefel, orth_value = np.inf, linesearch = None, *args, **kwargs)
ConjugateGradientMS.solve(self, problem, x = None, reuselinesearch = False, compute_stats = None)


utilmy/zrecs/recommenders/models/sar/__init__.py


utilmy/zrecs/recommenders/models/sar/sar_singlenode.py
-------------------------methods----------------------
SARSingleNode.__init__(self, col_user = constants.DEFAULT_USER_COL, col_item = constants.DEFAULT_ITEM_COL, col_rating = constants.DEFAULT_RATING_COL, col_timestamp = constants.DEFAULT_TIMESTAMP_COL, col_prediction = constants.DEFAULT_PREDICTION_COL, similarity_type = JACCARD, time_decay_coefficient = 30, time_now = None, timedecay_formula = False, threshold = 1, normalize = False, )
SARSingleNode.compute_affinity_matrix(self, df, rating_col)
SARSingleNode.compute_coocurrence_matrix(self, df)
SARSingleNode.compute_time_decay(self, df, decay_column)
SARSingleNode.fit(self, df)
SARSingleNode.get_item_based_topk(self, items, top_k = 10, sort_top_k = True)
SARSingleNode.get_popularity_based_topk(self, top_k = 10, sort_top_k = True)
SARSingleNode.predict(self, test)
SARSingleNode.recommend_k_items(self, test, top_k = 10, sort_top_k = True, remove_seen = False)
SARSingleNode.score(self, test, remove_seen = False)
SARSingleNode.set_index(self, df)


utilmy/zrecs/recommenders/models/surprise/__init__.py


utilmy/zrecs/recommenders/models/surprise/surprise_utils.py
-------------------------functions----------------------
compute_ranking_predictions(algo, data, usercol = DEFAULT_USER_COL, itemcol = DEFAULT_ITEM_COL, predcol = DEFAULT_PREDICTION_COL, remove_seen = False, )
predict(algo, data, usercol = DEFAULT_USER_COL, itemcol = DEFAULT_ITEM_COL, predcol = DEFAULT_PREDICTION_COL, )
surprise_trainset_to_df(trainset, col_user = "uid", col_item = "iid", col_rating = "rating")



utilmy/zrecs/recommenders/models/tfidf/__init__.py


utilmy/zrecs/recommenders/models/tfidf/tfidf_utils.py
-------------------------methods----------------------
TfidfRecommender.__clean_text(self, text, for_BERT = False, verbose = False)
TfidfRecommender.__create_full_recommendation_dictionary(self, df_clean)
TfidfRecommender.__get_single_item_info(self, metadata, rec_id)
TfidfRecommender.__init__(self, id_col, tokenization_method = "scibert")
TfidfRecommender.__make_clickable(self, address)
TfidfRecommender.__organize_results_as_tabular(self, df_clean, k)
TfidfRecommender.clean_dataframe(self, df, cols_to_clean, new_col_name = "cleaned_text")
TfidfRecommender.fit(self, tf, vectors_tokenized)
TfidfRecommender.get_stop_words(self)
TfidfRecommender.get_tokens(self)
TfidfRecommender.get_top_k_recommendations(self, metadata, query_id, cols_to_keep = [], verbose = True)
TfidfRecommender.recommend_top_k_items(self, df_clean, k = 5)
TfidfRecommender.tokenize_text(1, 3), min_df = 0)


utilmy/zrecs/recommenders/models/vae/__init__.py


utilmy/zrecs/recommenders/models/vae/multinomial_vae.py
-------------------------methods----------------------
AnnealingCallback.__init__(self, beta, anneal_cap, total_anneal_steps)
AnnealingCallback.get_data(self)
AnnealingCallback.on_batch_end(self, epoch, logs = {})
AnnealingCallback.on_epoch_end(self, epoch, logs = {})
AnnealingCallback.on_train_begin(self, logs = {})
LossHistory.on_epoch_end(self, epoch, logs = {})
LossHistory.on_train_begin(self, logs = {})
Metrics.__init__(self, model, val_tr, val_te, mapper, k, save_path = None)
Metrics.get_data(self)
Metrics.on_epoch_end(self, batch, logs = {})
Metrics.on_train_begin(self, logs = {})
Metrics.recommend_k_items(self, x, k, remove_seen = True)
Mult_VAE.__init__(self, n_users, original_dim, intermediate_dim = 200, latent_dim = 70, n_epochs = 400, batch_size = 100, k = 100, verbose = 1, drop_encoder = 0.5, drop_decoder = 0.5, beta = 1.0, annealing = False, anneal_cap = 1.0, seed = None, save_path = None, )
Mult_VAE._create_model(self)
Mult_VAE._get_vae_loss(self, x, x_bar)
Mult_VAE._take_sample(self, args)
Mult_VAE.display_metrics(self)
Mult_VAE.fit(self, x_train, x_valid, x_val_tr, x_val_te, mapper)
Mult_VAE.get_optimal_beta(self)
Mult_VAE.ndcg_per_epoch(self)
Mult_VAE.nn_batch_generator(self, x_train)
Mult_VAE.recommend_k_items(self, x, k, remove_seen = True)


utilmy/zrecs/recommenders/models/vae/standard_vae.py
-------------------------methods----------------------
AnnealingCallback.__init__(self, beta, anneal_cap, total_anneal_steps)
AnnealingCallback.get_data(self)
AnnealingCallback.on_batch_end(self, epoch, logs = {})
AnnealingCallback.on_epoch_end(self, epoch, logs = {})
AnnealingCallback.on_train_begin(self, logs = {})
LossHistory.on_epoch_end(self, epoch, logs = {})
LossHistory.on_train_begin(self, logs = {})
Metrics.__init__(self, model, val_tr, val_te, mapper, k, save_path = None)
Metrics.get_data(self)
Metrics.on_epoch_end(self, batch, logs = {})
Metrics.on_train_begin(self, logs = {})
Metrics.recommend_k_items(self, x, k, remove_seen = True)
StandardVAE.__init__(self, n_users, original_dim, intermediate_dim = 200, latent_dim = 70, n_epochs = 400, batch_size = 100, k = 100, verbose = 1, drop_encoder = 0.5, drop_decoder = 0.5, beta = 1.0, annealing = False, anneal_cap = 1.0, seed = None, save_path = None, )
StandardVAE._create_model(self)
StandardVAE._get_vae_loss(self, x, x_bar)
StandardVAE._take_sample(self, args)
StandardVAE.display_metrics(self)
StandardVAE.fit(self, x_train, x_valid, x_val_tr, x_val_te, mapper)
StandardVAE.get_optimal_beta(self)
StandardVAE.ndcg_per_epoch(self)
StandardVAE.nn_batch_generator(self, x_train)
StandardVAE.recommend_k_items(self, x, k, remove_seen = True)


utilmy/zrecs/recommenders/models/vowpal_wabbit/__init__.py


utilmy/zrecs/recommenders/models/vowpal_wabbit/vw.py
-------------------------methods----------------------
VW.__del__(self)
VW.__init__(self, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_timestamp = DEFAULT_TIMESTAMP_COL, col_prediction = DEFAULT_PREDICTION_COL, **kwargs, )
VW.fit(self, df)
VW.parse_test_params(self, params)
VW.parse_train_params(self, params)
VW.predict(self, df)
VW.to_vw_cmd(params)
VW.to_vw_file(self, df, train = True)


utilmy/zrecs/recommenders/models/wide_deep/__init__.py


utilmy/zrecs/recommenders/models/wide_deep/wide_deep_utils.py
-------------------------functions----------------------
_build_deep_columns(user_ids, item_ids, user_dim, item_dim, item_feat_col = None, item_feat_shape = 1)
_build_wide_columns(user_ids, item_ids, hash_bucket_size = 1000)
build_feature_columns(users, items, user_col = DEFAULT_USER_COL, item_col = DEFAULT_ITEM_COL, item_feat_col = None, crossed_feat_dim = 1000, user_dim = 8, item_dim = 8, item_feat_shape = None, model_type = "wide_deep", )
build_model(model_dir = MODEL_DIR, ), ), linear_optimizer = "Ftrl", dnn_optimizer = "Adagrad", 128, 128), dnn_dropout = 0.0, dnn_batch_norm = True, log_every_n_iter = 1000, save_checkpoints_steps = 10000, seed = None, )



utilmy/zrecs/recommenders/tuning/__init__.py


utilmy/zrecs/recommenders/tuning/nni/__init__.py


utilmy/zrecs/recommenders/tuning/nni/ncf_training.py
-------------------------functions----------------------
_update_metrics(metrics_dict, metric, params, result)
get_params()
main(params)
ncf_training(params)



utilmy/zrecs/recommenders/tuning/nni/ncf_utils.py
-------------------------functions----------------------
combine_metrics_dicts(*metrics)
compute_test_results(model, train, test, rating_metrics, ranking_metrics)



utilmy/zrecs/recommenders/tuning/nni/nni_utils.py
-------------------------functions----------------------
check_experiment_status(wait = WAITING_TIME, max_retries = MAX_RETRIES)
check_metrics_written(wait = WAITING_TIME, max_retries = MAX_RETRIES)
check_stopped(wait = WAITING_TIME, max_retries = MAX_RETRIES)
get_experiment_status(status_url = NNI_STATUS_URL)
get_trials(optimize_mode)
start_nni(config_path, wait = WAITING_TIME, max_retries = MAX_RETRIES)
stop_nni()



utilmy/zrecs/recommenders/tuning/nni/svd_training.py
-------------------------functions----------------------
get_params()
main(params)
svd_training(params)



utilmy/zrecs/recommenders/tuning/parameter_sweep.py
-------------------------functions----------------------
generate_param_grid(params)



utilmy/zrecs/recommenders/utils/__init__.py


utilmy/zrecs/recommenders/utils/constants.py


utilmy/zrecs/recommenders/utils/general_utils.py
-------------------------functions----------------------
get_number_processors()
get_physical_memory()
invert_dictionary(dictionary)



utilmy/zrecs/recommenders/utils/gpu_utils.py
-------------------------functions----------------------
clear_memory_all_gpus()
get_cuda_version(unix_path = DEFAULT_CUDA_PATH_LINUX)
get_cudnn_version()
get_gpu_info()
get_number_gpus()



utilmy/zrecs/recommenders/utils/k8s_utils.py
-------------------------functions----------------------
nodes_to_replicas(n_cores_per_node, n_nodes = 3, cpu_cores_per_replica = 0.1)
qps_to_replicas(target_qps, processing_time, max_qp_replica = 1, target_utilization = 0.7)
replicas_to_qps(num_replicas, processing_time, max_qp_replica = 1, target_utilization = 0.7)



utilmy/zrecs/recommenders/utils/notebook_memory_management.py
-------------------------functions----------------------
pre_run_cell()
start_watching_memory()
stop_watching_memory()
watch_memory()



utilmy/zrecs/recommenders/utils/notebook_utils.py
-------------------------functions----------------------
is_databricks()
is_jupyter()



utilmy/zrecs/recommenders/utils/plot.py
-------------------------functions----------------------
line_graph(values, labels, x_guides = None, x_name = None, y_name = None, x_min_max = None, y_min_max = None, legend_loc = None, subplot = None, 5, 5), )



utilmy/zrecs/recommenders/utils/python_utils.py
-------------------------functions----------------------
binarize(a, threshold)
exponential_decay(value, max_val, half_life)
get_top_k_scored_items(scores, top_k, sort_top_k = False)
jaccard(cooccurrence)
lift(cooccurrence)
rescale(data, new_min = 0, new_max = 1, data_min = None, data_max = None)



utilmy/zrecs/recommenders/utils/spark_utils.py
-------------------------functions----------------------
start_or_get_spark(app_name = "Sample", url = "local[*]", memory = "10g", config = None, packages = None, jars = None, repository = None, )



utilmy/zrecs/recommenders/utils/tf_utils.py
-------------------------functions----------------------
_dataset(x, y = None, batch_size = 128, num_epochs = 1, shuffle = False, seed = None)
build_optimizer(name, lr = 0.001, **kwargs)
evaluation_log_hook(estimator, logger, true_df, y_col, eval_df, every_n_iter = 10000, model_dir = None, batch_size = 256, eval_fns = None, **eval_kwargs)
export_model(model, train_input_fn, eval_input_fn, tf_feat_cols, base_dir)
pandas_input_fn(df, feat_name_type)
pandas_input_fn_for_saved_model(df, feat_name_type)

-------------------------methods----------------------
MetricsLogger.__init__(self)
MetricsLogger.get_log(self)
MetricsLogger.log(self, metric, value)
_TrainLogHook.__init__(self, estimator, logger, true_df, y_col, eval_df, every_n_iter = 10000, model_dir = None, batch_size = 256, eval_fns = None, **eval_kwargs)
_TrainLogHook._log(self, tag, value)
_TrainLogHook.after_run(self, run_context, run_values)
_TrainLogHook.before_run(self, run_context)
_TrainLogHook.begin(self)
_TrainLogHook.end(self, session)


utilmy/zrecs/recommenders/utils/timer.py
-------------------------methods----------------------
Timer.__enter__(self)
Timer.__exit__(self, *args)
Timer.__init__(self)
Timer.__str__(self)
Timer.interval(self)
Timer.start(self)
Timer.stop(self)


utilmy/zrecs/setup.py


utilmy/zrecs/tests/__init__.py


utilmy/zrecs/tests/ci/run_pytest.py
-------------------------functions----------------------
create_arg_parser()



utilmy/zrecs/tests/ci/submit_azureml_pytest.py
-------------------------functions----------------------
create_arg_parser()
create_experiment(workspace, experiment_name)
create_run_config(cpu_cluster, docker_proc_type, conda_env_file)
setup_persistent_compute_target(workspace, cluster_name, vm_size, max_nodes)
setup_workspace(workspace_name, subscription_id, resource_group, cli_auth, location)
submit_experiment_to_azureml(test, test_folder, test_markers, junitxml, run_config, experiment)



utilmy/zrecs/tests/conftest.py
-------------------------functions----------------------
affinity_matrix(test_specs)
criteo_first_row()
deeprec_config_path()
deeprec_resource_path()
demo_usage_data(header, sar_settings)
demo_usage_data_spark(spark, demo_usage_data, header)
header()
kernel_name()
mind_resource_path(deeprec_resource_path)
notebooks()
output_notebook()
pandas_dummy(header)
pandas_dummy_timestamp(pandas_dummy, header)
path_notebooks()
python_dataset_ncf(test_specs_ncf)
sar_settings()
spark(tmp_path_factory, app_name = "Sample", url = "local[*]")
test_specs()
test_specs_ncf()
tmp(tmp_path_factory)
train_test_dummy_timestamp(pandas_dummy_timestamp)



utilmy/zrecs/tests/integration/__init__.py


utilmy/zrecs/tests/integration/examples/__init__.py


utilmy/zrecs/tests/integration/examples/test_notebooks_gpu.py
-------------------------functions----------------------
test_cornac_bivae_integration(notebooks, output_notebook, kernel_name, size, expected_values)
test_dkn_quickstart_integration(notebooks, output_notebook, kernel_name)
test_fastai_integration(notebooks, output_notebook, kernel_name, size, epochs, expected_values)
test_gpu_vm()
test_lightgcn_deep_dive_integration(notebooks, output_notebook, kernel_name, yaml_file, data_path, size, epochs, batch_size, expected_values, seed, )
test_lstur_quickstart_integration(notebooks, output_notebook, kernel_name, epochs, seed, MIND_type, expected_values)
test_naml_quickstart_integration(notebooks, output_notebook, kernel_name, epochs, seed, MIND_type, expected_values)
test_ncf_deep_dive_integration(notebooks, output_notebook, kernel_name, size, epochs, batch_size, expected_values, seed, )
test_ncf_integration(notebooks, output_notebook, kernel_name, size, epochs, expected_values, seed)
test_npa_quickstart_integration(notebooks, output_notebook, kernel_name, epochs, seed, MIND_type, expected_values)
test_nrms_quickstart_integration(notebooks, output_notebook, kernel_name, epochs, seed, MIND_type, expected_values)
test_slirec_quickstart_integration(notebooks, output_notebook, kernel_name, yaml_file, data_path, epochs, batch_size, expected_values, seed, )
test_wide_deep_integration(notebooks, output_notebook, kernel_name, size, steps, expected_values, seed, tmp)
test_xdeepfm_integration(notebooks, output_notebook, kernel_name, syn_epochs, criteo_epochs, expected_values, seed, )



utilmy/zrecs/tests/integration/examples/test_notebooks_pyspark.py
-------------------------functions----------------------
test_als_pyspark_integration(notebooks, output_notebook, kernel_name)
test_mmlspark_lightgbm_criteo_integration(notebooks, output_notebook, kernel_name)



utilmy/zrecs/tests/integration/examples/test_notebooks_python.py
-------------------------functions----------------------
test_baseline_deep_dive_integration(notebooks, output_notebook, kernel_name, size, expected_values)
test_cornac_bpr_integration(notebooks, output_notebook, kernel_name, size, expected_values)
test_geoimc_integration(notebooks, output_notebook, kernel_name, expected_values)
test_nni_tuning_svd(notebooks, output_notebook, kernel_name, tmp)
test_sar_single_node_integration(notebooks, output_notebook, kernel_name, size, expected_values)
test_surprise_svd_integration(notebooks, output_notebook, kernel_name, size, expected_values)
test_vw_deep_dive_integration(notebooks, output_notebook, kernel_name, size, expected_values)
test_wikidata_integration(notebooks, output_notebook, kernel_name, tmp)
test_xlearn_fm_integration(notebooks, output_notebook, kernel_name)



utilmy/zrecs/tests/integration/recommenders/__init__.py


utilmy/zrecs/tests/smoke/__init__.py


utilmy/zrecs/tests/smoke/examples/__init__.py


utilmy/zrecs/tests/smoke/examples/test_notebooks_gpu.py
-------------------------functions----------------------
test_cornac_bivae_smoke(notebooks, output_notebook, kernel_name)
test_fastai_smoke(notebooks, output_notebook, kernel_name)
test_gpu_vm()
test_lstur_smoke(notebooks, output_notebook, kernel_name)
test_naml_smoke(notebooks, output_notebook, kernel_name)
test_ncf_deep_dive_smoke(notebooks, output_notebook, kernel_name)
test_ncf_smoke(notebooks, output_notebook, kernel_name)
test_npa_smoke(notebooks, output_notebook, kernel_name)
test_nrms_smoke(notebooks, output_notebook, kernel_name)
test_wide_deep_smoke(notebooks, output_notebook, kernel_name, tmp)
test_xdeepfm_smoke(notebooks, output_notebook, kernel_name)



utilmy/zrecs/tests/smoke/examples/test_notebooks_pyspark.py
-------------------------functions----------------------
test_als_pyspark_smoke(notebooks, output_notebook, kernel_name)
test_mmlspark_lightgbm_criteo_smoke(notebooks, output_notebook, kernel_name)



utilmy/zrecs/tests/smoke/examples/test_notebooks_python.py
-------------------------functions----------------------
test_baseline_deep_dive_smoke(notebooks, output_notebook, kernel_name)
test_cornac_bpr_smoke(notebooks, output_notebook, kernel_name)
test_lightgbm_quickstart_smoke(notebooks, output_notebook, kernel_name)
test_mind_utils(notebooks, output_notebook, kernel_name, tmp)
test_sar_single_node_smoke(notebooks, output_notebook, kernel_name)
test_surprise_svd_smoke(notebooks, output_notebook, kernel_name)
test_vw_deep_dive_smoke(notebooks, output_notebook, kernel_name)



utilmy/zrecs/tests/smoke/recommenders/__init__.py


utilmy/zrecs/tests/unit/__init__.py


utilmy/zrecs/tests/unit/examples/__init__.py


utilmy/zrecs/tests/unit/examples/test_notebooks_gpu.py
-------------------------functions----------------------
test_dkn_quickstart(notebooks, output_notebook, kernel_name)
test_fastai(notebooks, output_notebook, kernel_name)
test_gpu_vm()
test_ncf(notebooks, output_notebook, kernel_name)
test_ncf_deep_dive(notebooks, output_notebook, kernel_name)
test_wide_deep(notebooks, output_notebook, kernel_name, tmp)
test_xdeepfm(notebooks, output_notebook, kernel_name)



utilmy/zrecs/tests/unit/examples/test_notebooks_pyspark.py
-------------------------functions----------------------
test_als_deep_dive_runs(notebooks, output_notebook, kernel_name)
test_als_pyspark_runs(notebooks, output_notebook, kernel_name)
test_data_split_runs(notebooks, output_notebook, kernel_name)
test_evaluation_diversity_runs(notebooks, output_notebook, kernel_name)
test_evaluation_runs(notebooks, output_notebook, kernel_name)
test_mmlspark_lightgbm_criteo_runs(notebooks, output_notebook, kernel_name)
test_spark_tuning(notebooks, output_notebook, kernel_name)



utilmy/zrecs/tests/unit/examples/test_notebooks_python.py
-------------------------functions----------------------
test_baseline_deep_dive_runs(notebooks, output_notebook, kernel_name)
test_cornac_deep_dive_runs(notebooks, output_notebook, kernel_name)
test_lightgbm(notebooks, output_notebook, kernel_name)
test_rlrmc_quickstart_runs(notebooks, output_notebook, kernel_name)
test_sar_deep_dive_runs(notebooks, output_notebook, kernel_name)
test_sar_single_node_runs(notebooks, output_notebook, kernel_name)
test_surprise_deep_dive_runs(notebooks, output_notebook, kernel_name)
test_template_runs(notebooks, output_notebook, kernel_name)
test_vw_deep_dive_runs(notebooks, output_notebook, kernel_name)
test_wikidata_runs(notebooks, output_notebook, kernel_name, tmp)



utilmy/zrecs/tests/unit/recommenders/__init__.py


utilmy/zrecs/tools/__init__.py


utilmy/zrecs/tools/databricks_install.py
-------------------------functions----------------------
create_egg(), local_eggname = "Recommenders.egg", overwrite = False, )
dbfs_file_exists(api_client, dbfs_path)
prepare_for_operationalization(cluster_id, api_client, dbfs_path, overwrite, spark_version)



utilmy/zrecs/tools/generate_conda_file.py


utilmy/zrecs/tools/generate_requirements_txt.py


utilmy/zzarchive/_HELP.py
-------------------------functions----------------------
fun_cython(a)
fun_python(a)
os_VS_build(self, lib_to_build)
os_VS_start(self, version)
os_compileVSsolution(dir1, flags1 = "", type1 = "devenv", compilerdir = "")
set_rc_version(rcfile, target_version)



utilmy/zzarchive/__init__.py


utilmy/zzarchive/alldata.py


utilmy/zzarchive/allmodule.py
-------------------------functions----------------------
pprint(table1, tablefmt = "simple")
pprint2(x)
str_convert_beforeprint(x)
str_to_unicode(x, encoding = 'utf-8')
str_to_utf8(x)



utilmy/zzarchive/allmodule_fin.py


utilmy/zzarchive/coke_functions.py
-------------------------functions----------------------
date_diffend(t)
date_diffsecond(str_t1, str_t0, fmt='YYYY-MM-DD HH = 'YYYY-MM-DD HH:mm:SS')
date_diffstart(t)
day(s)
daytime(d)
hour(s)
month(s)
np_dict_tolist(dd)
np_dict_tostr_key(dd)
np_dict_tostr_val(dd)
pd_date_splitall(df, coldate = 'purchased_at')
season(d)
weekday(s, fmt = 'YYYY-MM-DD', i0 = 0, i1 = 10)
year(s)



utilmy/zzarchive/datanalysis.py
-------------------------functions----------------------
col_feature_importance(Xcol, Ytarget)
col_pair_correl(Xcol, Ytarget)
col_pair_interaction(Xcol, Ytarget)
col_study_getcategorydict_freq(catedict)
col_study_summary(Xmat = [0.0, 0.0], Xcolname = ['col1', 'col2'], Xcolselect = [9, 9], isprint = 0)
csv_analysis()
csv_bigcompute()
csv_col_get_dict_categoryfreq(dircsv, filepattern = "*.csv", category_cols = [], maxline = -1, fileencoding = "utf-8")
csv_col_schema_toexcel(dircsv = "", filepattern = '*.csv', outfile = '.xlsx', returntable = 1, maxrow = 5000000, maxcol_pertable = 90, maxstrlen = 'U80')
csv_dtypes_getdict(df = None, csvfile = None)
csv_fast_processing()
csv_pivotable(dircsv = "", filepattern = '*.csv', fileh5 = '.h5', leftX = 'col0', topY = 'col2', centerZ = 'coli', mapreduce = 'sum', chunksize =  500000, tablename = 'df')
csv_row_mapreduce(dircsv = "", outfile = "", type_mapreduce = 'sum', nrow = 1000000, chunk =  5000000)
csv_row_reduce_line(fromfile, tofile, condfilter, catval_tokeep, header = True, maxline = -1)
csv_row_reduce_line_manual(file_category, file_transact, file_reduced)
db_getdata()
db_meta_add(metadb, dbname, new_table = ('', [])
db_meta_find(ALLDB, query = '', filter_db = [], filter_table = [], filter_column = [])
db_sql()
isnull(x)
optim_is_pareto_efficient(Xmat_cost, epsilon =  0.01, ret_boolean = 1)
pd_checkpoint()
pd_col_pair_plot(dfX, Xcolname_selectlist = None, dfY = None, Ycolname = None)
pd_col_study_distribution_show(df, col_include = None, col_exclude = None, pars={'binsize' = {'binsize':20})
pd_describe(df)
pd_filter_column(df_client_product, filter_val = [], iscol = 1)
pd_missing_show()
pd_stack_dflist(df_list)
pd_validation_struct()
plot_XY(xx, yy, zcolor = None, tsize = None, labels = None, title = '', xlabel = '', ylabel = '', zcolor_label = '', figsize = (8, 6)
plot_XY_plotly(xx, yy, towhere = 'url')
plot_XY_seaborn(X, Y, Zcolor = None)
plot_Y(Yval, typeplot = '.b', tsize = None, labels = None, title = '', xlabel = '', ylabel = '', zcolor_label = '', figsize = (8, 6)
plot_cluster_2D(X_2dim, target_class, target_names)
plot_cluster_hiearchy(Xmat_dist, p = 30, truncate_mode = None, color_threshold = None, get_leaves = True, orientation = 'top', labels = None, count_sort = False, distance_sort = False, show_leaf_counts = True, do_plot = 1, no_labels = False, leaf_font_size = None, leaf_rotation = None, leaf_label_func = None, show_contracted = False, link_color_func = None, ax = None, above_threshold_color = 'b', annotate_above = 0)
plot_cluster_pca(Xmat, Xcluster_label = None, metric = 'euclidean', dimpca = 2, whiten = True, isprecompute = False, savefile = '', doreturn = 1)
plot_cluster_tsne(Xmat, Xcluster_label = None, metric = 'euclidean', perplexity = 50, ncomponent = 2, savefile = '', isprecompute = False, returnval = True)
plot_col_pair(dfX, Xcolname_selectlist = None, dfY = None, Ycolname = None)
plot_distance_heatmap(Xmat_dist, Xcolname)
plot_distribution_density(Xsample, kernel = 'gaussian', N = 10, bandwith = 1 / 10.0)
sk_catboost_classifier(Xtrain, Ytrain, Xcolname = None, pars= {"learning_rate" =  {"learning_rate":0.1, "iterations":1000, "random_seed":0, "loss_function": "MultiClass" }, isprint = 0)
sk_catboost_regressor()
sk_cluster(Xmat, metric = 'jaccard')
sk_cluster_algo_custom(Xmat, algorithm, args, kwds, returnval = 1)
sk_cluster_distance_pair(Xmat, metric = 'jaccard')
sk_correl_rank(correl = [[1, 0], [0, 1]])
sk_distribution_kernel_bestbandwidth(kde)
sk_distribution_kernel_sample(kde = None, n = 1)
sk_error_r2(Ypred, y_true, sample_weight = None, multioutput = None)
sk_error_rmse(Ypred, Ytrue)
sk_feature_importance(clfrf, feature_name)
sk_gen_ensemble_weight(vv, acclevel, maxlevel = 0.88)
sk_model_auto_tpot(Xmat, y, outfolder = 'aaserialize/', model_type = 'regressor/classifier', train_size = 0.5, generation = 1, population_size = 5, verbosity = 2)
sk_optim_de(obj_fun, bounds, maxiter = 1, name1 = '', solver1 = None, isreset = 1, popsize = 15)
sk_params_search_best(Xmat, Ytarget, model1, param_grid={'alpha' = {'alpha':  np.linspace(0, 1, 5) }, method = 'gridsearch', param_search= {'scoretype' =  {'scoretype':'r2', 'cv':5, 'population_size':5, 'generations_number':3 })
sk_showconfusion(clfrf, X_train, Y_train, isprint = True)
sk_tree(Xtrain, Ytrain, nbtree, maxdepth, isprint1 = 1, njobs = 1)
sk_tree_get_ifthen(tree, feature_names, target_names, spacer_base = " ")
sk_votingpredict(estimators, voting, ww, X_test)
str_to_unicode(x, encoding = 'utf-8')
tf_transform_catlabel_toint(Xmat)
tf_transform_pca(Xmat, dimpca = 2, whiten = True)
xl_get_rowcol(ws, i0, j0, imax, jmax)
xl_getschema(dirxl = "", filepattern = '*.xlsx', dirlevel = 1, outfile = '.xlsx')
xl_setstyle(file1)
xl_val(ws, colj, rowi)

-------------------------methods----------------------
sk_model_template1.__init__(self, alpha = 0.5, low_y_cut = -0.09, high_y_cut = 0.09, ww0 = 0.95)
sk_model_template1.fit(self, X, Y = None)
sk_model_template1.predict(self, X, y = None, ymedian = None)
sk_model_template1.score(self, X, Ytrue = None, ymedian = None)
sk_stateRule.__init__(self, state, trigger, colname = [])
sk_stateRule.addrule(self, rulefun, name = '', desc = '')
sk_stateRule.eval(self, idrule, t, ktrig = 0)
sk_stateRule.help()


utilmy/zzarchive/excel.py
-------------------------functions----------------------
add_one(data)
double_sum(x, y)
get_workbook_name()
matrix_mult(x, y)
npdot()



utilmy/zzarchive/fast.py
-------------------------functions----------------------
_compute_overlaps(u, v)
cosine(u, v)
cross(vec1, vec2)
day(s)
daytime(d)
distance_jaccard(u, v)
distance_jaccard2(u, v)
distance_jaccard_X(X)
drawdown_calc_fast(price)
fastStrptime(val, format)
hour(s)
log_exp_sum2(a, b)
mean(x)
month(s)
norm(vec)
rmse(y, yhat)
season(d)
std(x)
weekday(s)
year(s)



utilmy/zzarchive/fast_parallel.py
-------------------------functions----------------------
task_find_best(tasks, n_top = 5)
task_parallel_job_01(name, param, datadict)
task_progress(tasks)
task_summary(tasks)



utilmy/zzarchive/filelock.py
-------------------------methods----------------------
FileLock.__del__(self)
FileLock.__enter__(self)
FileLock.__exit__(self, type, value, traceback)
FileLock.__init__(self, protected_file_path, timeout = None, delay = 1, lock_file_contents = None)
FileLock.acquire(self, blocking = True)
FileLock.available(self)
FileLock.locked(self)
FileLock.purge(self)
FileLock.release(self)


utilmy/zzarchive/function_custom.py
-------------------------functions----------------------
fun_obj(vv, ext)
getweight(ww, size = (9, 3)
mapping_calc_risk_elvis_v03(ss, tr, t, riskout)
mapping_calc_risk_v00(self, ss, tr, t, risk0)
mapping_calc_risk_v01(ss, tr, t, risk0)
mapping_calc_risk_v02(ss, tr, t, risk0)
mapping_risk_ww_v01(risk, wwmat, ww2)



utilmy/zzarchive/geospatial.py


utilmy/zzarchive/global01.py


utilmy/zzarchive/kagglegym.py
-------------------------functions----------------------
make()
r_score(y_true, y_pred, sample_weight = None, multioutput = None)

-------------------------methods----------------------
Environment.__init__(self)
Environment.__str__(self)
Environment.reset(self)
Environment.step(self, target)
Observation.__init__(self, train, target, features)


utilmy/zzarchive/linux.py
-------------------------functions----------------------
VS_build(self, lib_to_build)
VS_start(self, version)
aa_cleanmemory()
aa_getmodule_doc(module1, fileout = '')
aa_isanaconda()
acf(data)
and1(x, y, x3 = None, x4 = None, x5 = None, x6 = None, x7 = None, x8 = None)
comoment(xx, yy, nsample, kx, ky)
compileVSsolution(dir1, flags1 = "", type1 = "devenv", compilerdir = "")
date_add_bdays(from_date, add_days)
date_as_float(dt)
date_diffindays(intdate1, intdate2)
date_finddateid(date1, dateref)
date_generatedatetime(start = "20100101", nbday = 10, end = "")
date_now(i = 0)
date_remove_bdays(from_date, add_days)
datediff_inyear(startdate, enddate)
dateint_todatetime(datelist1)
datestring_todatetime(datelist1, format1 =  "%Y%m%d")
datestring_todatetime(datelist1, format1 =  "%Y%m%d")
datestring_toint(datelist1)
datestring_toint(datelist1)
datetime_toint(datelist1)
datetime_tostring(datelist1)
datetime_tostring(datelist1)
find(item, vec)
findhigher(x, vec)
findlower(x, vec)
finds(itemlist, vec)
findx(item, vec)
isfloat(value)
isint(x)
load_session(name = 'test_20160815')
np_cleanmatrix(m)
np_find(item, vec)
np_find_maxpos(values)
np_find_maxpos_2nd(numbers)
np_find_minpos(values)
np_findfirst(item, vec)
np_findlocalmax(v, trig)
np_findlocalmax2(v, trig)
np_findlocalmin(v, trig)
np_findlocalmin2(v, trig)
np_interpolate_nan(y)
np_ma(vv, n)
np_memory_array_adress(x)
np_remove_zeros(vv, axis1 = 1)
np_sort(arr, colid, asc = True)
np_sortbycol(arr, colid, asc = True)
np_sortbycolumn(arr, colid, asc = True)
np_stack(v1, v2 = None, v3 = None, v4 = None, v5 = None)
np_uniquerows(a)
numexpr_topanda(filename, expr, i0 = 0, imax = 1000, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
numexpr_vect_calc(filename, expr, i0 = 0, imax = 1000, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
pd_addcolumn(df1, name1 = 'new')
pd_array_todataframe(price, symbols = None, date1 = None, dotranspose = False)
pd_changeencoding(data, cols)
pd_create_colmap_nametoid(df)
pd_createdf(val1, col1 = None, idx1 = None)
pd_csv_topanda(filein1, filename, tablen = 'data')
pd_dataframe_toarray(df)
pd_date_intersection(qlist)
pd_extract_col_idx_val(df)
pd_getpanda_tonumpy(filename, nsize, tablen = 'data')
pd_getrandom_tonumpy(filename, nbdim, nbsample, tablen = 'data')
pd_insertcolumn(df, colname, vec)
pd_insertrows(df, rowval, index1 = None)
pd_load_panda2vec(filenameh5, store_id = 'data')
pd_remove_row(df, row_list_index = [23, 45])
pd_removecolumn(df1, name1)
pd_replacevalues(df, matrix)
pd_resetindex(df)
pd_save_vectopanda(vv, filenameh5)
pd_split_col_idx_val(df)
pd_storeadddf(df, dfname, dbfile='F = 'F:\temp_pandas.h5')
pd_storedumpinfo(dbfile='E = 'E:\_data\stock\intraday_google.h5')
plotsave(xx, yy, title1 = "")
plotshow(xx, yy, title1 = "")
save_session(name = '')
set_rc_version(rcfile, target_version)
sk_cluster_kmeans(x, nbcluster = 5, isplot = True)
sk_featureimportance(clfrf, feature_name)
sk_gen_ensemble_weight(vv, acclevel, maxlevel = 0.88)
sk_showconfusion(clfrf, X_train, Y_train, isprint = True)
sk_tree(Xtrain, Ytrain, nbtree, maxdepth, print1)
sk_tree_get_ifthen(tree, feature_names, target_names, spacer_base = " ")
sk_votingpredict(estimators, voting, ww, X_test)
sort(arr, colid, asc = 1)
sortcol(arr, colid, asc = 1)
textvect_topanda(vv, fileout = "")



utilmy/zzarchive/multiprocessfunc.py
-------------------------functions----------------------
bm_generator(bm, dt, n, type1)
func(val, lock)
init2(d)
init_global1(l, r)
integratene(its)
integratenp(its, nchunk)
integratenp2(its, nchunk)
list_append(count, id, out_list)
merge(d2)
multigbm_paralell_func(nbsimul, ww, voldt, drift, upper_cholesky, nbasset, n, price, type1 = 0, strike = 0, cp = 1)
multigbm_processfast7(nbsimul, s0, voldt, drift, upper_cholesky, nbasset, n, price)
ne_sin(x)
np_sin(value)
parzen_estimation(x_samples, point_x, h)
res_shared2()



utilmy/zzarchive/multithread.py
-------------------------functions----------------------
multithread_run(fun_async, input_list:list, n_pool = 5, start_delay = 0.1, verbose = True, **kw)
multithread_run_list(**kwargs)



utilmy/zzarchive/portfolio.py
-------------------------functions----------------------
_date_align(dateref, datei, tmax, closei)
_notnone(x)
_reshape(x)
array_todataframe(price, symbols = None, date1 = None)
calc_ranktable(close2, symjp1, nlag, refindex, funeval, funargs)
calcbasket_objext(RETURN, TMAX, riskind_i, wwmat, wwasset0, ww0, nbrange, criteria)
calcbasket_table(wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps =  0.000)
causality_y1_y2(price2, price1, maxlag)
cointegration(x, y)
correl_fast(xn, y, nx)
correl_reducebytrigger(correl2, trigger)
correlation_mat(Xmat, type1 = "robust", type2 = "correl")
data_jpsector()
date_align(array1, dateref)
date_alignfromdateref(array1, dateref)
date_earningquater(t1)
date_extract_dailyopenclosetime(spdateref1, market = 'us')
date_find_intradateid(datetimelist, stringdate = ['20160420223000'])
date_find_kday_fromintradaydate(kintraday, intradaydate, dailydate)
date_find_kintraday_fromdate(d1, intradaydate1, h1 = 9, m1 = 30)
date_finddateid(date1, dateref)
date_is_3rdfriday(s)
date_option_expiry(date)
datetime_convertzone1_tozone2(tt, fromzone = 'Japan', tozone = 'US/Eastern')
folio_concenfactor2(ww, masset = 12)
folio_cost_turnover(wwall, bsk, dateref, costbp)
folio_createvolta_asset(close, vol = 0.12, volrange = 120, lev = 1.0)
folio_fixedunitprice(price, fixedww, costpa = 0.0)
folio_fixedweightprice(price, fixedww, costpa = 0.0)
folio_fixedweightret(ret, fixedww)
folio_histogram(close)
folio_inverseetf(price, costpa = 0.0)
folio_leverageetf(price, lev = 1.0, costpa = 0.0)
folio_longshort_pct(long1, short1, ww = [1, -1], costpa = 0.0)
folio_longshort_unit(long1, short1, ww = [1, -1], costpa = 0.0, tlag = 1, istable = 1, wwschedule = [])
folio_longshort_unitfixed(long1, short1, nn = [1, -1], costpa = 0.0, tlag = 1, istable = 1)
folio_lowcorrelation(sym01, nstock, periodlist, dateref, close1, kbenchmark, badlist, costbppa = 0.02, showgraph = True)
folio_perfreport_schedule(sym, dateref, close, wwind, t0, scheduleperiod = "1monthend")
folio_riskpa(ret, targetvol = 0.1, volrange = 90, cap = 1.0)
folio_volta(bsk, targetvol = 0.11, volrange =  90, expocap = 1.5)
folio_volta2(bsk, riskind, par, targetvol = 0.11, volrange =  90, cap = 1.5, floor = 0.0, costbp = 0.0005)
folio_voltarget(bsk, targetvol = 0.11, volrange =  90, expocap = 1.5)
generate_sepvertical(asset1, tt, tmax, start = None, datebar = None)
get(close, timelag)
getdiff_fromquotes(close, timelag)
getlogret_fromquotes(close, timelag = 1)
getprice_fromret(ret, normprice = 100)
getret_fromquotes(close, timelag = 1)
isfloat(value)
isint(x)
load_asset_fromfile(file1)
max_withposition(values)
min_withposition(values)
norm_fast(y, ny)
np_distance_l1(x, y, wwerr)
np_distance_l1(x, y, wwerr)
np_similarity(x, y, wwerr = [], type1 = 0)
np_similarity(x, y, wwerr = [], type1 = 0)
pd_dataframe_toarray(df)
pd_transform_asset(q0, q1, type1 = "spread")
plot_check(close, tt0i = 20140102, tt1i = 20160815, dateref = [], sym = [], tickperday = 120)
plot_price(asset, y2 = None, y3 = None, y4 = None, y5 = None, sym = None, savename1 = '', tickperday = 20, date1 = None, graphsize = (10, 5)
plot_pricedate(date1, sym1, asset1, sym2 = None, bsk1 = None, verticaldate = None, savename1 = '', graphsize = (10, 5)
plot_priceintraday(data)
price_normalize100(ret, normprice = 100)
price_normalize_1d(ret, normprice = 100, dtype1 =  np.float32)
reg_slope(close, dateref, tlag, type1 = 'elasticv')
regression(yreturn, xreturn, type1 = "elasticv")
regression_allstocks_vs_riskfactors(symstock, pricestock, symriskfac, priceriskfac, nlaglist)
regression_fixedsymbolstock(sym, ret_close2, tsstart, tsample, ret_spy, spyclose, regonly = True)
regression_getpricefromww(spyclose, ww01, regasset01, ret_close2, tstart, tlag = 1)
rolling_cointegration(x, y)
rsk_calc_all_TA(df = 'panda_dataframe')
save_asset_tofile(file1, asset1, asset2 = None, asset3 = None, date1 = None, title1 = None)
similarity_correl(ret_close2, funargs)
sk_cov_fromcorrel(correl, ret_close1)
ta_highbandtrend1(close2, type1 = 0)
volhisto_fromprice(price, t, volrange, axis = 0)
volhisto_fromret(retbsk, t, volrange, axis = 0)
volhistorolling_fromprice(price, volrange)

-------------------------methods----------------------
folioCalc.__init__(self, sym, close, dateref)
folioCalc._regimecalc(self, t, wwextra)
folioCalc._weightcalc_constant(self, ww2, t)
folioCalc._weightcalc_generic(self, wwvec, t)
folioCalc._weightcalc_regime(self, wwvec, wwextra, t)
folioCalc.calc_baskettable(self, wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps = 0.000, showdetail = 0)
folioCalc.getweight(self)
folioCalc.help(self)
folioCalc.multiperiod_ww(self, t)
folioCalc.plot(self, wwvec = None, show1 = 1, tickperday = 60)
folioCalc.set_symclose(self, sym, close, dateref)
folioCalc.setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid = "spprice", lfun = None)
folioOptimizationF.__init__(self, sym, close, dateref)
folioOptimizationF._loss_obj(self, ww2, wwpenalty)
folioOptimizationF._mapping_calc_risk(self, ss, tr, t, risk0)
folioOptimizationF._objective_criteria(self, bsk)
folioOptimizationF._regimecalc(self, t, wwextra)
folioOptimizationF._weightcalc_constant(self, ww2, t)
folioOptimizationF._weightcalc_generic(self, wwvec, t)
folioOptimizationF._weightcalc_regime(self, wwvec, wwextra, t)
folioOptimizationF.calc_baskettable(self, wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps =  0.000, showdetail = 0)
folioOptimizationF.calc_optimal_weight(self, maxiter = 1, name1 = '', isreset = 1, popsize = 15)
folioOptimizationF.calcbasket_obj(self, wwvec)
folioOptimizationF.calcbasket_obj2(self, wwvec)
folioOptimizationF.getweight(self)
folioOptimizationF.help(self)
folioOptimizationF.mapping_risk_ww(self, risk, wwmat, ww2 = self.wwasset0)
folioOptimizationF.multiperiod_ww(self, t)
folioOptimizationF.plot(self, wwvec = None, show1 = 1, tickperday = 60)
folioOptimizationF.set_symclose(self, sym, close, dateref)
folioOptimizationF.setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid = "spprice", lfun = None)
folioRiskIndicator.__init__(self, sym, close, dateref)
folioRiskIndicator._regimecalc(self, t, wwextra)
folioRiskIndicator._weightcalc_generic(self, wwvec, t)
folioRiskIndicator._weightcalc_regime(self, wwvec, wwextra, t)
folioRiskIndicator.calc_optimal_weight(self, maxiter = 1, name1 = '', isreset = 1, popsize = 15)
folioRiskIndicator.calcrisk(self, wwvec = [], initval = 1)
folioRiskIndicator.set_symclose(self, sym, close, dateref)
folioRiskIndicator.setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid = "spprice", lfun = None)
index.__init__(self, id1, sym, ww, tstart)
index.__init__(self, id1, sym, ww, tstart)
index._udpate_wwindpct(self, t, bskt, hedgecost, wwpct_actual, wwpct_th)
index._wwpct_rebal(self, wwpct_actual, t, trebal)
index.calc_baskettable_pct(self, type1 = "table", showdetail = 0)
index.calc_baskettable_unit()
index.close(self)
index.help(self)
index.updatehisto(self)
searchSimilarity.__generate_return__(self, nlag)
searchSimilarity.__init__(self, filejpstock=r'E = r'E:/_data/stock/daily/20160616/jp', sym01 = ['7203'], symname = ['Toyota'], startdate =  20150101, enddate = 20160601, pricetype = "close")
searchSimilarity.__overweight__(self, px)
searchSimilarity.export_results()
searchSimilarity.get_rankresult(self, filetosave = '')
searchSimilarity.launch_search(self)
searchSimilarity.load_quotes_fromdb(self, picklefile = '')
searchSimilarity.set_searchcriteria(self, name1 = '7203', date1 = 20160301, date2 = 20160601, nlag = 1, searchperiodstart = 20120101, typesearch = "pattern2", )
searchSimilarity.show_comparison_graph(self, maxresult = 20, show_only_different_time = True, fromid = 0, fromend =  0, filenameout = '')
searchSimilarity.staticmethod(self, x)


utilmy/zzarchive/portfolio_withdate.py
-------------------------functions----------------------
_date_align(dateref, datei, tmax, closei)
_notnone(x)
_reshape(x)
array_todataframe(price, symbols = None, date1 = None)
calc_ranktable(close2, symjp1, nlag, refindex, funeval, funargs)
calcbasket_objext(RETURN, TMAX, riskind_i, wwmat, wwasset0, ww0, nbrange, criteria)
calcbasket_table(wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps =  0.000)
causality_y1_y2(price2, price1, maxlag)
cointegration(x, y)
correl_fast(xn, y, nx)
correl_reducebytrigger(correl2, trigger)
correlation_mat(Xmat, type1 = "robust", type2 = "correl")
data_jpsector()
date_add_bdays(from_date, add_days)
date_align(array1, dateref)
date_alignfromdateref(array1, dateref)
date_as_float(dt)
date_diffindays(intdate1, intdate2)
date_earningquater(t1)
date_extract_dailyopenclosetime(spdateref1, market = 'us')
date_find_intradateid(datetimelist, stringdate = ['20160420223000'])
date_find_kday_fromintradaydate(kintraday, intradaydate, dailydate)
date_find_kintraday_fromdate(d1, intradaydate1, h1 = 9, m1 = 30)
date_finddateid(date1, dateref)
date_generatedatetime(start = "20100101", nbday = 10, end = "")
date_getspecificdate(datelist, datetype1 = "yearend", outputype1 = "intdate", includelastdate = True, includefirstdate = False, )
date_is_3rdfriday(s)
date_option_expiry(date)
date_removetimezone(datelist)
date_todatetime(tlist)
datediff_inyear(startdate, enddate)
dateint_todatetime(datelist1)
dateint_tostring(datelist1, format1 = '%b-%y')
datenumpy_todatetime(tt, islocaltime = True)
datestring_todatetime(datelist1, format1 =  "%Y%m%d")
datestring_toint(datelist1)
datetime_convertzone1_tozone2(tt, fromzone = 'Japan', tozone = 'US/Eastern')
datetime_todate(tt)
datetime_toint(datelist1)
datetime_tointhour(datelist1)
datetime_tonumpypdate(t, islocaltime = True)
datetime_tostring(tt)
folio_concenfactor2(ww, masset = 12)
folio_cost_turnover(wwall, bsk, dateref, costbp)
folio_createvolta_asset(close, vol = 0.12, volrange = 120, lev = 1.0)
folio_fixedunitprice(price, fixedww, costpa = 0.0)
folio_fixedweightprice(price, fixedww, costpa = 0.0)
folio_fixedweightret(ret, fixedww)
folio_histogram(close)
folio_inverseetf(price, costpa = 0.0)
folio_leverageetf(price, lev = 1.0, costpa = 0.0)
folio_longshort_pct(long1, short1, ww = [1, -1], costpa = 0.0)
folio_longshort_unit(long1, short1, ww = [1, -1], costpa = 0.0, tlag = 1, istable = 1, wwschedule = [])
folio_longshort_unitfixed(long1, short1, nn = [1, -1], costpa = 0.0, tlag = 1, istable = 1)
folio_lowcorrelation(sym01, nstock, periodlist, dateref, close1, kbenchmark, badlist, costbppa = 0.02, showgraph = True)
folio_perfreport_schedule(sym, dateref, close, wwind, t0, scheduleperiod = "1monthend")
folio_riskpa(ret, targetvol = 0.1, volrange = 90, cap = 1.0)
folio_volta(bsk, targetvol = 0.11, volrange =  90, expocap = 1.5)
folio_volta2(bsk, riskind, par, targetvol = 0.11, volrange =  90, cap = 1.5, floor = 0.0, costbp = 0.0005)
folio_voltarget(bsk, targetvol = 0.11, volrange =  90, expocap = 1.5)
generate_sepvertical(asset1, tt, tmax, start = None, datebar = None)
get(close, timelag)
getdiff_fromquotes(close, timelag)
getlogret_fromquotes(close, timelag = 1)
getprice_fromret(ret, normprice = 100)
getret_fromquotes(close, timelag = 1)
isfloat(value)
isint(x)
load_asset_fromfile(file1)
max_withposition(values)
min_withposition(values)
norm_fast(y, ny)
np_distance_l1(x, y, wwerr)
np_distance_l1(x, y, wwerr)
np_similarity(x, y, wwerr = [], type1 = 0)
np_similarity(x, y, wwerr = [], type1 = 0)
pd_dataframe_toarray(df)
pd_transform_asset(q0, q1, type1 = "spread")
plot_check(close, tt0i = 20140102, tt1i = 20160815, dateref = [], sym = [], tickperday = 120)
plot_price(asset, y2 = None, y3 = None, y4 = None, y5 = None, sym = None, savename1 = '', tickperday = 20, date1 = None, graphsize = (10, 5)
plot_pricedate(date1, sym1, asset1, sym2 = None, bsk1 = None, verticaldate = None, savename1 = '', graphsize = (10, 5)
plot_priceintraday(data)
price_normalize100(ret, normprice = 100)
price_normalize_1d(ret, normprice = 100, dtype1 =  np.float32)
reg_slope(close, dateref, tlag, type1 = 'elasticv')
regression(yreturn, xreturn, type1 = "elasticv")
regression_allstocks_vs_riskfactors(symstock, pricestock, symriskfac, priceriskfac, nlaglist)
regression_fixedsymbolstock(sym, ret_close2, tsstart, tsample, ret_spy, spyclose, regonly = True)
regression_getpricefromww(spyclose, ww01, regasset01, ret_close2, tstart, tlag = 1)
rolling_cointegration(x, y)
rsk_calc_all_TA(df = 'panda_dataframe')
save_asset_tofile(file1, asset1, asset2 = None, asset3 = None, date1 = None, title1 = None)
similarity_correl(ret_close2, funargs)
sk_cov_fromcorrel(correl, ret_close1)
ta_highbandtrend1(close2, type1 = 0)
volhisto_fromprice(price, t, volrange, axis = 0)
volhisto_fromret(retbsk, t, volrange, axis = 0)
volhistorolling_fromprice(price, volrange)

-------------------------methods----------------------
folioCalc.__init__(self, sym, close, dateref)
folioCalc._regimecalc(self, t, wwextra)
folioCalc._weightcalc_constant(self, ww2, t)
folioCalc._weightcalc_generic(self, wwvec, t)
folioCalc._weightcalc_regime(self, wwvec, wwextra, t)
folioCalc.calc_baskettable(self, wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps = 0.000, showdetail = 0)
folioCalc.getweight(self)
folioCalc.help(self)
folioCalc.multiperiod_ww(self, t)
folioCalc.plot(self, wwvec = None, show1 = 1, tickperday = 60)
folioCalc.set_symclose(self, sym, close, dateref)
folioCalc.setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid = "spprice", lfun = None)
folioOptimizationF.__init__(self, sym, close, dateref)
folioOptimizationF._loss_obj(self, ww2, wwpenalty)
folioOptimizationF._mapping_calc_risk(self, ss, tr, t, risk0)
folioOptimizationF._objective_criteria(self, bsk)
folioOptimizationF._regimecalc(self, t, wwextra)
folioOptimizationF._weightcalc_constant(self, ww2, t)
folioOptimizationF._weightcalc_generic(self, wwvec, t)
folioOptimizationF._weightcalc_regime(self, wwvec, wwextra, t)
folioOptimizationF.calc_baskettable(self, wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps =  0.000, showdetail = 0)
folioOptimizationF.calc_optimal_weight(self, maxiter = 1, name1 = '', isreset = 1, popsize = 15)
folioOptimizationF.calcbasket_obj(self, wwvec)
folioOptimizationF.calcbasket_obj2(self, wwvec)
folioOptimizationF.getweight(self)
folioOptimizationF.help(self)
folioOptimizationF.mapping_risk_ww(self, risk, wwmat, ww2 = self.wwasset0)
folioOptimizationF.multiperiod_ww(self, t)
folioOptimizationF.plot(self, wwvec = None, show1 = 1, tickperday = 60)
folioOptimizationF.set_symclose(self, sym, close, dateref)
folioOptimizationF.setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid = "spprice", lfun = None)
folioRiskIndicator.__init__(self, sym, close, dateref)
folioRiskIndicator._regimecalc(self, t, wwextra)
folioRiskIndicator._weightcalc_generic(self, wwvec, t)
folioRiskIndicator._weightcalc_regime(self, wwvec, wwextra, t)
folioRiskIndicator.calc_optimal_weight(self, maxiter = 1, name1 = '', isreset = 1, popsize = 15)
folioRiskIndicator.calcrisk(self, wwvec = [], initval = 1)
folioRiskIndicator.set_symclose(self, sym, close, dateref)
folioRiskIndicator.setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid = "spprice", lfun = None)
index.__init__(self, id1, sym, ww, tstart)
index.__init__(self, id1, sym, ww, tstart)
index._udpate_wwindpct(self, t, bskt, hedgecost, wwpct_actual, wwpct_th)
index._wwpct_rebal(self, wwpct_actual, t, trebal)
index.calc_baskettable_pct(self, type1 = "table", showdetail = 0)
index.calc_baskettable_unit()
index.close(self)
index.help(self)
index.updatehisto(self)
searchSimilarity.__generate_return__(self, nlag)
searchSimilarity.__init__(self, filejpstock=r'E = r'E:/_data/stock/daily/20160616/jp', sym01 = ['7203'], symname = ['Toyota'], startdate =  20150101, enddate = 20160601, pricetype = "close")
searchSimilarity.__overweight__(self, px)
searchSimilarity.export_results()
searchSimilarity.get_rankresult(self, filetosave = '')
searchSimilarity.launch_search(self)
searchSimilarity.load_quotes_fromdb(self, picklefile = '')
searchSimilarity.set_searchcriteria(self, name1 = '7203', date1 = 20160301, date2 = 20160601, nlag = 1, searchperiodstart = 20120101, typesearch = "pattern2", )
searchSimilarity.show_comparison_graph(self, maxresult = 20, show_only_different_time = True, fromid = 0, fromend =  0, filenameout = '')
searchSimilarity.staticmethod(self, x)


utilmy/zzarchive/py2to3/_HELP.py
-------------------------functions----------------------
fun_cython(a)
fun_python(a)
os_VS_build(self, lib_to_build)
os_VS_start(self, version)
os_compileVSsolution(dir1, flags1 = "", type1 = "devenv", compilerdir = "")
set_rc_version(rcfile, target_version)



utilmy/zzarchive/py2to3/__init__.py


utilmy/zzarchive/py2to3/alldata.py


utilmy/zzarchive/py2to3/allmodule.py
-------------------------functions----------------------
pprint(table1, tablefmt = "simple")
pprint2(x)
str_convert_beforeprint(x)
str_to_unicode(x, encoding = 'utf-8')
str_to_utf8(x)



utilmy/zzarchive/py2to3/allmodule_fin.py


utilmy/zzarchive/py2to3/coke_functions.py
-------------------------functions----------------------
date_diffend(t)
date_diffsecond(str_t1, str_t0, fmt='YYYY-MM-DD HH = 'YYYY-MM-DD HH:mm:SS')
date_diffstart(t)
day(s)
daytime(d)
hour(s)
month(s)
np_dict_tolist(dd)
np_dict_tostr_key(dd)
np_dict_tostr_val(dd)
pd_date_splitall(df, coldate = 'purchased_at')
season(d)
weekday(s, fmt = 'YYYY-MM-DD', i0 = 0, i1 = 10)
year(s)



utilmy/zzarchive/py2to3/datanalysis.py
-------------------------functions----------------------
col_feature_importance(Xcol, Ytarget)
col_pair_correl(Xcol, Ytarget)
col_pair_interaction(Xcol, Ytarget)
col_pair_plot(dfX, Xcolname_selectlist = None, dfY = None, Ycolname = None)
col_study_distribution_show(df, col_include = None, col_exclude = None, pars={'binsize' = {'binsize':20})
col_study_getcategorydict_freq(catedict)
col_study_summary(Xmat = [0.0, 0.0], Xcolname = ['col1', 'col2'], Xcolselect = [9, 9], isprint = 0)
csv_analysis()
csv_bigcompute()
csv_col_get_dict_categoryfreq(dircsv, filepattern = "*.csv", category_cols = [], maxline = -1, fileencoding = "utf-8")
csv_col_schema_toexcel(dircsv = "", filepattern = '*.csv', outfile = '.xlsx', returntable = 1, maxrow = 5000000, maxcol_pertable = 90, maxstrlen = 'U80')
csv_dtypes_getdict(df = None, csvfile = None)
csv_fast_processing()
csv_pivotable(dircsv = "", filepattern = '*.csv', fileh5 = '.h5', leftX = 'col0', topY = 'col2', centerZ = 'coli', mapreduce = 'sum', chunksize =  500000, tablename = 'df')
csv_row_mapreduce(dircsv = "", outfile = "", type_mapreduce = 'sum', chunk =  5000000)
csv_row_reduce_line(fromfile, tofile, condfilter, catval_tokeep, maxline = -1)
csv_row_reduce_line_manual(file_category, file_transact, file_reduced)
db_getdata()
db_meta_add(metadb, dbname, new_table = ('', [])
db_meta_find(ALLDB, query = '', filter_db = [], filter_table = [], filter_column = [])
db_sql()
isnull(x)
optim_is_pareto_efficient(Xmat_cost, epsilon =  0.01, ret_boolean = 1)
pd_checkpoint()
pd_filter_column(df_client_product, filter_val = [], iscol = 1)
pd_missing_show()
pd_stack_dflist(df_list)
pd_validation_struct()
plot_XY(xx, yy, zcolor = None, tsize = None, labels = None, title = '', xlabel = '', ylabel = '', zcolor_label = '', figsize = (8, 6)
plot_XY_plotly(xx, yy, towhere = 'url')
plot_XY_seaborn(X, Y, Zcolor = None)
plot_Y(Yval, typeplot = '.b', tsize = None, labels = None, title = '', xlabel = '', ylabel = '', zcolor_label = '', figsize = (8, 6)
plot_cluster_2D(X_2dim, target_class, target_names)
plot_cluster_hiearchy(Xmat_dist, p = 30, truncate_mode = None, color_threshold = None, get_leaves = True, orientation = 'top', labels = None, count_sort = False, distance_sort = False, show_leaf_counts = True, no_plot = False, no_labels = False, leaf_font_size = None, leaf_rotation = None, leaf_label_func = None, show_contracted = False, link_color_func = None, ax = None, above_threshold_color = 'b')
plot_cluster_pca(Xmat, Xcluster_label = None, metric = 'euclidean', dimpca = 2, whiten = True, isprecompute = False, savefile = '', doreturn = 1)
plot_cluster_tsne(Xmat, Xcluster_label = None, metric = 'euclidean', perplexity = 50, ncomponent = 2, savefile = '', isprecompute = False, returnval = True)
plot_col_pair(dfX, Xcolname_selectlist = None, dfY = None, Ycolname = None)
plot_distance_heatmap(Xmat_dist, Xcolname)
plot_distribution_density(Xsample, kernel = 'gaussian', N = 10, bandwith = 1 / 10.0)
sk_cluster(Xmat, metric = 'jaccard')
sk_cluster_algo_custom(Xmat, algorithm, args, kwds, returnval = 1)
sk_cluster_distance_pair(Xmat, metric = 'jaccard')
sk_correl_rank(correl = [[1, 0], [0, 1]])
sk_distribution_kernel_bestbandwidth(kde)
sk_distribution_kernel_sample(kde = None, n = 1)
sk_error_r2(Ypred, y_true, sample_weight = None, multioutput = None)
sk_error_rmse(Ypred, Ytrue)
sk_feature_importance(clfrf, feature_name)
sk_gen_ensemble_weight(vv, acclevel, maxlevel = 0.88)
sk_model_auto_tpot(Xmat, y, outfolder = 'aaserialize/', model_type = 'regressor/classifier', train_size = 0.5, generation = 1, population_size = 5, verbosity = 2)
sk_optim_de(obj_fun, bounds, maxiter = 1, name1 = '', solver1 = None, isreset = 1, popsize = 15)
sk_params_search_best(Xmat, Ytarget, model1, param_grid={'alpha' = {'alpha':  np.linspace(0, 1, 5) }, method = 'gridsearch', param_search= {'scoretype' =  {'scoretype':'r2', 'cv':5, 'population_size':5, 'generations_number':3 })
sk_showconfusion(clfrf, X_train, Y_train, isprint = True)
sk_tree(Xtrain, Ytrain, nbtree, maxdepth, isprint1 = 1, njobs = 1)
sk_tree_get_ifthen(tree, feature_names, target_names, spacer_base = " ")
sk_votingpredict(estimators, voting, ww, X_test)
str_to_unicode(x, encoding = 'utf-8')
tf_transform_catlabel_toint(Xmat)
tf_transform_pca(Xmat, dimpca = 2, whiten = True)
xl_get_rowcol(ws, i0, j0, imax, jmax)
xl_getschema(dirxl = "", filepattern = '*.xlsx', dirlevel = 1, outfile = '.xlsx')
xl_setstyle(file1)
xl_val(ws, colj, rowi)

-------------------------methods----------------------
sk_model_template1.__init__(self, alpha = 0.5, low_y_cut = -0.09, high_y_cut = 0.09, ww0 = 0.95)
sk_model_template1.fit(self, X, Y = None)
sk_model_template1.predict(self, X, y = None, ymedian = None)
sk_model_template1.score(self, X, Ytrue = None, ymedian = None)
sk_stateRule.__init__(self, state, trigger, colname = [])
sk_stateRule.addrule(self, rulefun, name = '', desc = '')
sk_stateRule.eval(self, idrule, t, ktrig = 0)
sk_stateRule.help()


utilmy/zzarchive/py2to3/excel.py
-------------------------functions----------------------
add_one(data)
double_sum(x, y)
get_workbook_name()
matrix_mult(x, y)
npdot()



utilmy/zzarchive/py2to3/fast.py
-------------------------functions----------------------
_compute_overlaps(u, v)
cosine(u, v)
cross(vec1, vec2)
day(s)
daytime(d)
distance_jaccard(u, v)
distance_jaccard2(u, v)
distance_jaccard_X(X)
drawdown_calc_fast(price)
fastStrptime(val, format)
hour(s)
log_exp_sum2(a, b)
mean(x)
month(s)
norm(vec)
rmse(y, yhat)
season(d)
std(x)
weekday(s)
year(s)



utilmy/zzarchive/py2to3/fast_parallel.py
-------------------------functions----------------------
task_find_best(tasks, n_top = 5)
task_parallel_job_01(name, param, datadict)
task_progress(tasks)
task_summary(tasks)



utilmy/zzarchive/py2to3/filelock.py
-------------------------methods----------------------
FileLock.__del__(self)
FileLock.__enter__(self)
FileLock.__exit__(self, type, value, traceback)
FileLock.__init__(self, protected_file_path, timeout = None, delay = 1, lock_file_contents = None)
FileLock.acquire(self, blocking = True)
FileLock.available(self)
FileLock.locked(self)
FileLock.purge(self)
FileLock.release(self)


utilmy/zzarchive/py2to3/function_custom.py
-------------------------functions----------------------
fun_obj(vv, ext)
getweight(ww, size = (9, 3)
mapping_calc_risk_elvis_v03(ss, tr, t, riskout)
mapping_calc_risk_v00(self, ss, tr, t, risk0)
mapping_calc_risk_v01(ss, tr, t, risk0)
mapping_calc_risk_v02(ss, tr, t, risk0)
mapping_risk_ww_v01(risk, wwmat, ww2)



utilmy/zzarchive/py2to3/geospatial.py


utilmy/zzarchive/py2to3/global01.py


utilmy/zzarchive/py2to3/kagglegym.py
-------------------------functions----------------------
make()
r_score(y_true, y_pred, sample_weight = None, multioutput = None)

-------------------------methods----------------------
Environment.__init__(self)
Environment.__str__(self)
Environment.reset(self)
Environment.step(self, target)
Observation.__init__(self, train, target, features)


utilmy/zzarchive/py2to3/linux.py
-------------------------functions----------------------
VS_build(self, lib_to_build)
VS_start(self, version)
aa_cleanmemory()
aa_getmodule_doc(module1, fileout = '')
aa_isanaconda()
acf(data)
and1(x, y, x3 = None, x4 = None, x5 = None, x6 = None, x7 = None, x8 = None)
comoment(xx, yy, nsample, kx, ky)
compileVSsolution(dir1, flags1 = "", type1 = "devenv", compilerdir = "")
date_add_bdays(from_date, add_days)
date_as_float(dt)
date_diffindays(intdate1, intdate2)
date_finddateid(date1, dateref)
date_generatedatetime(start = "20100101", nbday = 10, end = "")
date_now(i = 0)
date_remove_bdays(from_date, add_days)
datediff_inyear(startdate, enddate)
dateint_todatetime(datelist1)
datestring_todatetime(datelist1, format1 =  "%Y%m%d")
datestring_todatetime(datelist1, format1 =  "%Y%m%d")
datestring_toint(datelist1)
datestring_toint(datelist1)
datetime_toint(datelist1)
datetime_tostring(datelist1)
datetime_tostring(datelist1)
find(item, vec)
findhigher(x, vec)
findlower(x, vec)
finds(itemlist, vec)
findx(item, vec)
isfloat(value)
isint(x)
load_session(name = 'test_20160815')
np_cleanmatrix(m)
np_find(item, vec)
np_find_maxpos(values)
np_find_maxpos_2nd(numbers)
np_find_minpos(values)
np_findfirst(item, vec)
np_findlocalmax(v, trig)
np_findlocalmax2(v, trig)
np_findlocalmin(v, trig)
np_findlocalmin2(v, trig)
np_interpolate_nan(y)
np_ma(vv, n)
np_memory_array_adress(x)
np_remove_zeros(vv, axis1 = 1)
np_sort(arr, colid, asc = True)
np_sortbycol(arr, colid, asc = True)
np_sortbycolumn(arr, colid, asc = True)
np_stack(v1, v2 = None, v3 = None, v4 = None, v5 = None)
np_uniquerows(a)
numexpr_topanda(filename, expr, i0 = 0, imax = 1000, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
numexpr_vect_calc(filename, expr, i0 = 0, imax = 1000, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
pd_addcolumn(df1, name1 = 'new')
pd_array_todataframe(price, symbols = None, date1 = None, dotranspose = False)
pd_changeencoding(data, cols)
pd_create_colmap_nametoid(df)
pd_createdf(val1, col1 = None, idx1 = None)
pd_csv_topanda(filein1, filename, tablen = 'data')
pd_dataframe_toarray(df)
pd_date_intersection(qlist)
pd_extract_col_idx_val(df)
pd_getpanda_tonumpy(filename, nsize, tablen = 'data')
pd_getrandom_tonumpy(filename, nbdim, nbsample, tablen = 'data')
pd_insertcolumn(df, colname, vec)
pd_insertrows(df, rowval, index1 = None)
pd_load_panda2vec(filenameh5, store_id = 'data')
pd_remove_row(df, row_list_index = [23, 45])
pd_removecolumn(df1, name1)
pd_replacevalues(df, matrix)
pd_resetindex(df)
pd_save_vectopanda(vv, filenameh5)
pd_split_col_idx_val(df)
pd_storeadddf(df, dfname, dbfile='F = 'F:\temp_pandas.h5')
pd_storedumpinfo(dbfile='E = 'E:\_data\stock\intraday_google.h5')
plotsave(xx, yy, title1 = "")
plotshow(xx, yy, title1 = "")
save_session(name = '')
set_rc_version(rcfile, target_version)
sk_cluster_kmeans(x, nbcluster = 5, isplot = True)
sk_featureimportance(clfrf, feature_name)
sk_gen_ensemble_weight(vv, acclevel, maxlevel = 0.88)
sk_showconfusion(clfrf, X_train, Y_train, isprint = True)
sk_tree(Xtrain, Ytrain, nbtree, maxdepth, print1)
sk_tree_get_ifthen(tree, feature_names, target_names, spacer_base = " ")
sk_votingpredict(estimators, voting, ww, X_test)
sort(arr, colid, asc = 1)
sortcol(arr, colid, asc = 1)
textvect_topanda(vv, fileout = "")



utilmy/zzarchive/py2to3/multiprocessfunc.py
-------------------------functions----------------------
bm_generator(bm, dt, n, type1)
func(val, lock)
init2(d)
init_global1(l, r)
integratene(its)
integratenp(its, nchunk)
integratenp2(its, nchunk)
list_append(count, id, out_list)
merge(d2)
multigbm_paralell_func(nbsimul, ww, voldt, drift, upper_cholesky, nbasset, n, price, type1 = 0, strike = 0, cp = 1)
multigbm_processfast7(nbsimul, s0, voldt, drift, upper_cholesky, nbasset, n, price)
ne_sin(x)
np_sin(value)
parzen_estimation(x_samples, point_x, h)
res_shared2()



utilmy/zzarchive/py2to3/portfolio.py
-------------------------functions----------------------
_date_align(dateref, datei, tmax, closei)
_notnone(x)
_reshape(x)
array_todataframe(price, symbols = None, date1 = None)
calc_ranktable(close2, symjp1, nlag, refindex, funeval, funargs)
calcbasket_objext(RETURN, TMAX, riskind_i, wwmat, wwasset0, ww0, nbrange, criteria)
calcbasket_table(wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps =  0.000)
causality_y1_y2(price2, price1, maxlag)
cointegration(x, y)
correl_fast(xn, y, nx)
correl_reducebytrigger(correl2, trigger)
correlation_mat(Xmat, type1 = "robust", type2 = "correl")
data_jpsector()
date_align(array1, dateref)
date_alignfromdateref(array1, dateref)
date_earningquater(t1)
date_extract_dailyopenclosetime(spdateref1, market = 'us')
date_find_intradateid(datetimelist, stringdate = ['20160420223000'])
date_find_kday_fromintradaydate(kintraday, intradaydate, dailydate)
date_find_kintraday_fromdate(d1, intradaydate1, h1 = 9, m1 = 30)
date_finddateid(date1, dateref)
date_is_3rdfriday(s)
date_option_expiry(date)
datetime_convertzone1_tozone2(tt, fromzone = 'Japan', tozone = 'US/Eastern')
folio_concenfactor2(ww, masset = 12)
folio_cost_turnover(wwall, bsk, dateref, costbp)
folio_createvolta_asset(close, vol = 0.12, volrange = 120, lev = 1.0)
folio_fixedunitprice(price, fixedww, costpa = 0.0)
folio_fixedweightprice(price, fixedww, costpa = 0.0)
folio_fixedweightret(ret, fixedww)
folio_histogram(close)
folio_inverseetf(price, costpa = 0.0)
folio_leverageetf(price, lev = 1.0, costpa = 0.0)
folio_longshort_pct(long1, short1, ww = [1, -1], costpa = 0.0)
folio_longshort_unit(long1, short1, ww = [1, -1], costpa = 0.0, tlag = 1, istable = 1, wwschedule = [])
folio_longshort_unitfixed(long1, short1, nn = [1, -1], costpa = 0.0, tlag = 1, istable = 1)
folio_lowcorrelation(sym01, nstock, periodlist, dateref, close1, kbenchmark, badlist, costbppa = 0.02, showgraph = True)
folio_perfreport_schedule(sym, dateref, close, wwind, t0, scheduleperiod = "1monthend")
folio_riskpa(ret, targetvol = 0.1, volrange = 90, cap = 1.0)
folio_volta(bsk, targetvol = 0.11, volrange =  90, expocap = 1.5)
folio_volta2(bsk, riskind, par, targetvol = 0.11, volrange =  90, cap = 1.5, floor = 0.0, costbp = 0.0005)
folio_voltarget(bsk, targetvol = 0.11, volrange =  90, expocap = 1.5)
generate_sepvertical(asset1, tt, tmax, start = None, datebar = None)
get(close, timelag)
getdiff_fromquotes(close, timelag)
getlogret_fromquotes(close, timelag = 1)
getprice_fromret(ret, normprice = 100)
getret_fromquotes(close, timelag = 1)
isfloat(value)
isint(x)
load_asset_fromfile(file1)
max_withposition(values)
min_withposition(values)
norm_fast(y, ny)
np_distance_l1(x, y, wwerr)
np_distance_l1(x, y, wwerr)
np_similarity(x, y, wwerr = [], type1 = 0)
np_similarity(x, y, wwerr = [], type1 = 0)
pd_dataframe_toarray(df)
pd_transform_asset(q0, q1, type1 = "spread")
plot_check(close, tt0i = 20140102, tt1i = 20160815, dateref = [], sym = [], tickperday = 120)
plot_price(asset, y2 = None, y3 = None, y4 = None, y5 = None, sym = None, savename1 = '', tickperday = 20, date1 = None, graphsize = (10, 5)
plot_pricedate(date1, sym1, asset1, sym2 = None, bsk1 = None, verticaldate = None, savename1 = '', graphsize = (10, 5)
plot_priceintraday(data)
price_normalize100(ret, normprice = 100)
price_normalize_1d(ret, normprice = 100, dtype1 =  np.float32)
reg_slope(close, dateref, tlag, type1 = 'elasticv')
regression(yreturn, xreturn, type1 = "elasticv")
regression_allstocks_vs_riskfactors(symstock, pricestock, symriskfac, priceriskfac, nlaglist)
regression_fixedsymbolstock(sym, ret_close2, tsstart, tsample, ret_spy, spyclose, regonly = True)
regression_getpricefromww(spyclose, ww01, regasset01, ret_close2, tstart, tlag = 1)
rolling_cointegration(x, y)
rsk_calc_all_TA(df = 'panda_dataframe')
save_asset_tofile(file1, asset1, asset2 = None, asset3 = None, date1 = None, title1 = None)
similarity_correl(ret_close2, funargs)
sk_cov_fromcorrel(correl, ret_close1)
ta_highbandtrend1(close2, type1 = 0)
volhisto_fromprice(price, t, volrange, axis = 0)
volhisto_fromret(retbsk, t, volrange, axis = 0)
volhistorolling_fromprice(price, volrange)

-------------------------methods----------------------
folioCalc.__init__(self, sym, close, dateref)
folioCalc._regimecalc(self, t, wwextra)
folioCalc._weightcalc_constant(self, ww2, t)
folioCalc._weightcalc_generic(self, wwvec, t)
folioCalc._weightcalc_regime(self, wwvec, wwextra, t)
folioCalc.calc_baskettable(self, wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps = 0.000, showdetail = 0)
folioCalc.getweight(self)
folioCalc.help(self)
folioCalc.multiperiod_ww(self, t)
folioCalc.plot(self, wwvec = None, show1 = 1, tickperday = 60)
folioCalc.set_symclose(self, sym, close, dateref)
folioCalc.setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid = "spprice", lfun = None)
folioOptimizationF.__init__(self, sym, close, dateref)
folioOptimizationF._loss_obj(self, ww2, wwpenalty)
folioOptimizationF._mapping_calc_risk(self, ss, tr, t, risk0)
folioOptimizationF._objective_criteria(self, bsk)
folioOptimizationF._regimecalc(self, t, wwextra)
folioOptimizationF._weightcalc_constant(self, ww2, t)
folioOptimizationF._weightcalc_generic(self, wwvec, t)
folioOptimizationF._weightcalc_regime(self, wwvec, wwextra, t)
folioOptimizationF.calc_baskettable(self, wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps =  0.000, showdetail = 0)
folioOptimizationF.calc_optimal_weight(self, maxiter = 1, name1 = '', isreset = 1, popsize = 15)
folioOptimizationF.calcbasket_obj(self, wwvec)
folioOptimizationF.calcbasket_obj2(self, wwvec)
folioOptimizationF.getweight(self)
folioOptimizationF.help(self)
folioOptimizationF.mapping_risk_ww(self, risk, wwmat, ww2 = self.wwasset0)
folioOptimizationF.multiperiod_ww(self, t)
folioOptimizationF.plot(self, wwvec = None, show1 = 1, tickperday = 60)
folioOptimizationF.set_symclose(self, sym, close, dateref)
folioOptimizationF.setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid = "spprice", lfun = None)
folioRiskIndicator.__init__(self, sym, close, dateref)
folioRiskIndicator._regimecalc(self, t, wwextra)
folioRiskIndicator._weightcalc_generic(self, wwvec, t)
folioRiskIndicator._weightcalc_regime(self, wwvec, wwextra, t)
folioRiskIndicator.calc_optimal_weight(self, maxiter = 1, name1 = '', isreset = 1, popsize = 15)
folioRiskIndicator.calcrisk(self, wwvec = [], initval = 1)
folioRiskIndicator.set_symclose(self, sym, close, dateref)
folioRiskIndicator.setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid = "spprice", lfun = None)
index.__init__(self, id1, sym, ww, tstart)
index.__init__(self, id1, sym, ww, tstart)
index._udpate_wwindpct(self, t, bskt, hedgecost, wwpct_actual, wwpct_th)
index._wwpct_rebal(self, wwpct_actual, t, trebal)
index.calc_baskettable_pct(self, type1 = "table", showdetail = 0)
index.calc_baskettable_unit()
index.close(self)
index.help(self)
index.updatehisto(self)
searchSimilarity.__generate_return__(self, nlag)
searchSimilarity.__init__(self, filejpstock=r'E = r'E:/_data/stock/daily/20160616/jp', sym01 = ['7203'], symname = ['Toyota'], startdate =  20150101, enddate = 20160601, pricetype = "close")
searchSimilarity.__overweight__(self, px)
searchSimilarity.export_results()
searchSimilarity.get_rankresult(self, filetosave = '')
searchSimilarity.launch_search(self)
searchSimilarity.load_quotes_fromdb(self, picklefile = '')
searchSimilarity.set_searchcriteria(self, name1 = '7203', date1 = 20160301, date2 = 20160601, nlag = 1, searchperiodstart = 20120101, typesearch = "pattern2", )
searchSimilarity.show_comparison_graph(self, maxresult = 20, show_only_different_time = True, fromid = 0, fromend =  0, filenameout = '')
searchSimilarity.staticmethod(self, x)


utilmy/zzarchive/py2to3/portfolio_withdate.py
-------------------------functions----------------------
_date_align(dateref, datei, tmax, closei)
_notnone(x)
_reshape(x)
array_todataframe(price, symbols = None, date1 = None)
calc_ranktable(close2, symjp1, nlag, refindex, funeval, funargs)
calcbasket_objext(RETURN, TMAX, riskind_i, wwmat, wwasset0, ww0, nbrange, criteria)
calcbasket_table(wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps =  0.000)
causality_y1_y2(price2, price1, maxlag)
cointegration(x, y)
correl_fast(xn, y, nx)
correl_reducebytrigger(correl2, trigger)
correlation_mat(Xmat, type1 = "robust", type2 = "correl")
data_jpsector()
date_add_bdays(from_date, add_days)
date_align(array1, dateref)
date_alignfromdateref(array1, dateref)
date_as_float(dt)
date_diffindays(intdate1, intdate2)
date_earningquater(t1)
date_extract_dailyopenclosetime(spdateref1, market = 'us')
date_find_intradateid(datetimelist, stringdate = ['20160420223000'])
date_find_kday_fromintradaydate(kintraday, intradaydate, dailydate)
date_find_kintraday_fromdate(d1, intradaydate1, h1 = 9, m1 = 30)
date_finddateid(date1, dateref)
date_generatedatetime(start = "20100101", nbday = 10, end = "")
date_getspecificdate(datelist, datetype1 = "yearend", outputype1 = "intdate", includelastdate = True, includefirstdate = False, )
date_is_3rdfriday(s)
date_option_expiry(date)
date_removetimezone(datelist)
date_todatetime(tlist)
datediff_inyear(startdate, enddate)
dateint_todatetime(datelist1)
dateint_tostring(datelist1, format1 = '%b-%y')
datenumpy_todatetime(tt, islocaltime = True)
datestring_todatetime(datelist1, format1 =  "%Y%m%d")
datestring_toint(datelist1)
datetime_convertzone1_tozone2(tt, fromzone = 'Japan', tozone = 'US/Eastern')
datetime_todate(tt)
datetime_toint(datelist1)
datetime_tointhour(datelist1)
datetime_tonumpypdate(t, islocaltime = True)
datetime_tostring(tt)
folio_concenfactor2(ww, masset = 12)
folio_cost_turnover(wwall, bsk, dateref, costbp)
folio_createvolta_asset(close, vol = 0.12, volrange = 120, lev = 1.0)
folio_fixedunitprice(price, fixedww, costpa = 0.0)
folio_fixedweightprice(price, fixedww, costpa = 0.0)
folio_fixedweightret(ret, fixedww)
folio_histogram(close)
folio_inverseetf(price, costpa = 0.0)
folio_leverageetf(price, lev = 1.0, costpa = 0.0)
folio_longshort_pct(long1, short1, ww = [1, -1], costpa = 0.0)
folio_longshort_unit(long1, short1, ww = [1, -1], costpa = 0.0, tlag = 1, istable = 1, wwschedule = [])
folio_longshort_unitfixed(long1, short1, nn = [1, -1], costpa = 0.0, tlag = 1, istable = 1)
folio_lowcorrelation(sym01, nstock, periodlist, dateref, close1, kbenchmark, badlist, costbppa = 0.02, showgraph = True)
folio_perfreport_schedule(sym, dateref, close, wwind, t0, scheduleperiod = "1monthend")
folio_riskpa(ret, targetvol = 0.1, volrange = 90, cap = 1.0)
folio_volta(bsk, targetvol = 0.11, volrange =  90, expocap = 1.5)
folio_volta2(bsk, riskind, par, targetvol = 0.11, volrange =  90, cap = 1.5, floor = 0.0, costbp = 0.0005)
folio_voltarget(bsk, targetvol = 0.11, volrange =  90, expocap = 1.5)
generate_sepvertical(asset1, tt, tmax, start = None, datebar = None)
get(close, timelag)
getdiff_fromquotes(close, timelag)
getlogret_fromquotes(close, timelag = 1)
getprice_fromret(ret, normprice = 100)
getret_fromquotes(close, timelag = 1)
isfloat(value)
isint(x)
load_asset_fromfile(file1)
max_withposition(values)
min_withposition(values)
norm_fast(y, ny)
np_distance_l1(x, y, wwerr)
np_distance_l1(x, y, wwerr)
np_similarity(x, y, wwerr = [], type1 = 0)
np_similarity(x, y, wwerr = [], type1 = 0)
pd_dataframe_toarray(df)
pd_transform_asset(q0, q1, type1 = "spread")
plot_check(close, tt0i = 20140102, tt1i = 20160815, dateref = [], sym = [], tickperday = 120)
plot_price(asset, y2 = None, y3 = None, y4 = None, y5 = None, sym = None, savename1 = '', tickperday = 20, date1 = None, graphsize = (10, 5)
plot_pricedate(date1, sym1, asset1, sym2 = None, bsk1 = None, verticaldate = None, savename1 = '', graphsize = (10, 5)
plot_priceintraday(data)
price_normalize100(ret, normprice = 100)
price_normalize_1d(ret, normprice = 100, dtype1 =  np.float32)
reg_slope(close, dateref, tlag, type1 = 'elasticv')
regression(yreturn, xreturn, type1 = "elasticv")
regression_allstocks_vs_riskfactors(symstock, pricestock, symriskfac, priceriskfac, nlaglist)
regression_fixedsymbolstock(sym, ret_close2, tsstart, tsample, ret_spy, spyclose, regonly = True)
regression_getpricefromww(spyclose, ww01, regasset01, ret_close2, tstart, tlag = 1)
rolling_cointegration(x, y)
rsk_calc_all_TA(df = 'panda_dataframe')
save_asset_tofile(file1, asset1, asset2 = None, asset3 = None, date1 = None, title1 = None)
similarity_correl(ret_close2, funargs)
sk_cov_fromcorrel(correl, ret_close1)
ta_highbandtrend1(close2, type1 = 0)
volhisto_fromprice(price, t, volrange, axis = 0)
volhisto_fromret(retbsk, t, volrange, axis = 0)
volhistorolling_fromprice(price, volrange)

-------------------------methods----------------------
folioCalc.__init__(self, sym, close, dateref)
folioCalc._regimecalc(self, t, wwextra)
folioCalc._weightcalc_constant(self, ww2, t)
folioCalc._weightcalc_generic(self, wwvec, t)
folioCalc._weightcalc_regime(self, wwvec, wwextra, t)
folioCalc.calc_baskettable(self, wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps = 0.000, showdetail = 0)
folioCalc.getweight(self)
folioCalc.help(self)
folioCalc.multiperiod_ww(self, t)
folioCalc.plot(self, wwvec = None, show1 = 1, tickperday = 60)
folioCalc.set_symclose(self, sym, close, dateref)
folioCalc.setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid = "spprice", lfun = None)
folioOptimizationF.__init__(self, sym, close, dateref)
folioOptimizationF._loss_obj(self, ww2, wwpenalty)
folioOptimizationF._mapping_calc_risk(self, ss, tr, t, risk0)
folioOptimizationF._objective_criteria(self, bsk)
folioOptimizationF._regimecalc(self, t, wwextra)
folioOptimizationF._weightcalc_constant(self, ww2, t)
folioOptimizationF._weightcalc_generic(self, wwvec, t)
folioOptimizationF._weightcalc_regime(self, wwvec, wwextra, t)
folioOptimizationF.calc_baskettable(self, wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps =  0.000, showdetail = 0)
folioOptimizationF.calc_optimal_weight(self, maxiter = 1, name1 = '', isreset = 1, popsize = 15)
folioOptimizationF.calcbasket_obj(self, wwvec)
folioOptimizationF.calcbasket_obj2(self, wwvec)
folioOptimizationF.getweight(self)
folioOptimizationF.help(self)
folioOptimizationF.mapping_risk_ww(self, risk, wwmat, ww2 = self.wwasset0)
folioOptimizationF.multiperiod_ww(self, t)
folioOptimizationF.plot(self, wwvec = None, show1 = 1, tickperday = 60)
folioOptimizationF.set_symclose(self, sym, close, dateref)
folioOptimizationF.setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid = "spprice", lfun = None)
folioRiskIndicator.__init__(self, sym, close, dateref)
folioRiskIndicator._regimecalc(self, t, wwextra)
folioRiskIndicator._weightcalc_generic(self, wwvec, t)
folioRiskIndicator._weightcalc_regime(self, wwvec, wwextra, t)
folioRiskIndicator.calc_optimal_weight(self, maxiter = 1, name1 = '', isreset = 1, popsize = 15)
folioRiskIndicator.calcrisk(self, wwvec = [], initval = 1)
folioRiskIndicator.set_symclose(self, sym, close, dateref)
folioRiskIndicator.setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid = "spprice", lfun = None)
index.__init__(self, id1, sym, ww, tstart)
index.__init__(self, id1, sym, ww, tstart)
index._udpate_wwindpct(self, t, bskt, hedgecost, wwpct_actual, wwpct_th)
index._wwpct_rebal(self, wwpct_actual, t, trebal)
index.calc_baskettable_pct(self, type1 = "table", showdetail = 0)
index.calc_baskettable_unit()
index.close(self)
index.help(self)
index.updatehisto(self)
searchSimilarity.__generate_return__(self, nlag)
searchSimilarity.__init__(self, filejpstock=r'E = r'E:/_data/stock/daily/20160616/jp', sym01 = ['7203'], symname = ['Toyota'], startdate =  20150101, enddate = 20160601, pricetype = "close")
searchSimilarity.__overweight__(self, px)
searchSimilarity.export_results()
searchSimilarity.get_rankresult(self, filetosave = '')
searchSimilarity.launch_search(self)
searchSimilarity.load_quotes_fromdb(self, picklefile = '')
searchSimilarity.set_searchcriteria(self, name1 = '7203', date1 = 20160301, date2 = 20160601, nlag = 1, searchperiodstart = 20120101, typesearch = "pattern2", )
searchSimilarity.show_comparison_graph(self, maxresult = 20, show_only_different_time = True, fromid = 0, fromend =  0, filenameout = '')
searchSimilarity.staticmethod(self, x)


utilmy/zzarchive/py2to3/report.py
-------------------------functions----------------------
map_show()
xl_create_pdf()
xl_create_pivot(infile, index_list = ["Manager", "Rep", "Product"], value_list = ["Price", "Quantity"])
xl_save_report(report, outfile)



utilmy/zzarchive/py2to3/rstatpy.py
-------------------------functions----------------------
stl(data, ns, np = None, nt = None, nl = None, isdeg = 0, itdeg = 1, ildeg = 1, nsjump = None, ntjump = None, nljump = None, ni = 2, no = 0, fulloutput = False)



utilmy/zzarchive/py2to3/util_min.py
-------------------------functions----------------------
a_get_pythonversion()
a_isanaconda()
isexist(a)
isfloat(x)
isint(x)
load(folder = '/folder1/keyname', isabsolutpath = 0)
os_file_exist(file1)
os_file_getname(path)
os_file_getpath(path)
os_file_gettext(file1)
os_file_listall(dir1, pattern = "*.*", dirlevel = 1, onlyfolder = 0)
os_file_mergeall(nfile, dir1, pattern1, deepness = 2)
os_file_read(file1)
os_file_rename(some_dir, pattern = "*.*", pattern2 = "", dirlevel = 1)
os_file_replacestring1(findStr, repStr, filePath)
os_file_replacestring2(findstr, replacestr, some_dir, pattern = "*.*", dirlevel = 1)
os_file_size(file1)
os_folder_copy(src, dst, symlinks = False, pattern1 = "*.py", fun_file_toignore = None)
os_folder_create(directory)
os_folder_robocopy(from_folder = '', to_folder = '', my_log='H = 'H:/robocopy_log.txt')
os_path_append(p1, p2 = None, p3 = None, p4 = None)
os_path_change(path1)
os_path_current()
os_path_norm(pth)
os_print_tofile(vv, file1, mode1 = 'a')
os_split_dir_file(dirfile)
os_wait_cpu(priority = 300, cpu_min = 50)
os_zip_checkintegrity(filezip1)
os_zipextractall(filezip_or_dir = "folder1/*.zip", tofolderextract = 'zdisk/test', isprint = 1)
os_zipfile(folderin, folderzipname, iscompress = True)
os_zipfolder(dir_tozip = '/zdisks3/output', zipname = '/zdisk3/output.zip', dir_prefix = None, iscompress = True)
py_importfromfile(modulename, dir1)
py_load_obj(folder = '/folder1/keyname', isabsolutpath = 0, encoding1 = 'utf-8')
py_memorysize(o, ids, hint = " deep_getsizeof(df_pd, set()
py_save_obj(obj, folder = '/folder1/keyname', isabsolutpath = 0)
save(obj, folder = '/folder1/keyname', isabsolutpath = 0)
save_test(folder = '/folder1/keyname', isabsolutpath = 0)
z_key_splitinto_dir_name(keyname)



utilmy/zzarchive/py2to3/util_ml.py
-------------------------functions----------------------
create_adam_optimizer(learning_rate, momentum)
create_bias_variable(name, shape)
create_weight_variable(name, shape)
parse_args(ppa = None, args =  {})
parse_args2(ppa = None)
tf_check()
tf_global_variables_initializer(sess = None)
visualize_result()

-------------------------methods----------------------
TextLoader.__init__(self, data_dir, batch_size, seq_length)
TextLoader.create_batches(self)
TextLoader.load_preprocessed(self, vocab_file, tensor_file)
TextLoader.next_batch(self)
TextLoader.preprocess(self, input_file, vocab_file, tensor_file)
TextLoader.reset_batch_pointer(self)


utilmy/zzarchive/py2to3/utilgeo.py
-------------------------functions----------------------
df_to_geojson(df, col_properties, lat = 'latitude', lon = 'longitude')



utilmy/zzarchive/py3/util.py
-------------------------functions----------------------
a_autoreload()
a_cleanmemory()
a_get_platform()
a_info_conda_jupyter()
a_isanaconda()
a_module_codesample(module_str = 'pandas')
a_module_doc(module_str = 'pandas')
a_module_generatedoc(module_str = "pandas", fileout = '')
a_run_ipython(cmd1)
a_start_log(id1 = '', folder = 'aaserialize/log/')
aa_unicode_ascii_utf8_issue()
aws_accesskey_get(access = '', key = '')
aws_conn_create(region = "ap-northeast-2", access = '', key = '')
aws_conn_do(action = '', region = "ap-northeast-2")
aws_conn_getallregions(conn = None)
aws_conn_getinfo(conn)
aws_ec2_allocate_elastic_ip(instance_id, region = "ap-northeast-2")
aws_ec2_create_con(contype = 'sftp/ssh', host = 'ip', port = 22, username = 'ubuntu', keyfilepath = '', password = '', keyfiletype = 'RSA', isprint = 1)
aws_ec2_python_script(script_path, args1, host)
aws_s3_file_read(filepath, isbinary = 1)
aws_s3_folder_printtall(bucket_name = 'zdisk')
aws_s3_getbucketconn(s3dir)
aws_s3_getfrom_s3(froms3dir = 'task01/', todir = '', bucket_name = 'zdisk')
aws_s3_puto_s3(fromdir_file = 'dir/file.zip', todir = 'bucket/folder1/folder2')
aws_s3_url_split(url)
date_add_bday(from_date, add_days)
date_add_bdays(from_date, add_days)
date_allinfo()
date_convert(t1, fromtype, totype)
date_diffinbday(intd2, intd1)
date_diffinday(intdate1, intdate2)
date_diffinyear(startdate, enddate)
date_finddateid(date1, dateref)
date_gencalendar(start = '2010-01-01', end = '2010-01-15', country = 'us')
date_generatedatetime(start = "20100101", nbday = 10, end = "")
date_getspecificdate(datelist, datetype1 = "yearend", outputype1 = "intdate", includelastdate = True, includefirstdate = False, )
date_holiday()
date_now(i = 0)
date_nowtime(type1 = 'str', format1= "%Y-%m-%d %H =  "%Y-%m-%d %H:%M:%S:%f")
date_remove_bdays(from_date, add_days)
date_tofloat(dt)
dateint_todatetime(datelist1)
datenumpy_todatetime(tt, islocaltime = True)
datestring_todatetime(datelist1, format1 =  "%Y%m%d")
datestring_toint(datelist1)
datetime_toint(datelist1)
datetime_tonumpydate(t, islocaltime = True)
datetime_tostring(datelist1)
find(xstring, list_string)
find_fuzzy(xstring, list_string)
findhigher(x, vec)
findlower(x, vec)
findnone(vec)
finds(itemlist, vec)
findx(item, vec)
gc_map_dict_to_bq_schema(source_dict, schema, dest_dict)
googledrive_get()
googledrive_list()
googledrive_put()
isexist(a)
isfloat(x)
isint(x)
load(folder = '/folder1/keyname', isabsolutpath = 0)
max_kpos(arr, kth)
min_kpos(arr, kth)
np_acf(data)
np_addcolumn(arr, nbcol)
np_addrow(arr, nbrow)
np_and1(x, y, x3 = None, x4 = None, x5 = None, x6 = None, x7 = None, x8 = None)
np_cleanmatrix(m)
np_comoment(xx, yy, nsample, kx, ky)
np_dict_tolist(dd, withkey = 0)
np_dict_tostr_key(dd)
np_dict_tostr_val(dd)
np_dictordered_create()
np_enumerate2(vec_1d)
np_find(item, vec)
np_find_maxpos(values)
np_find_maxpos_2nd(numbers)
np_find_minpos(values)
np_findfirst(item, vec)
np_findlocalmax(v, trig)
np_findlocalmax2(v, trig)
np_findlocalmin(v, trig)
np_findlocalmin2(v, trig)
np_int_tostr(i)
np_interpolate_nan(y)
np_list_flatten(seq)
np_list_tofreqdict(l1, wweight = [])
np_list_unique(seq)
np_ma(vv, n)
np_memory_array_adress(x)
np_mergelist(x0, x1)
np_minimize(fun_obj, x0 = [0.0], argext = (0, 0)
np_minimizeDE(fun_obj, bounds, name1, solver = None)
np_nan_helper(y)
np_numexpr_tohdfs(filename, expr, i0 = 0, imax = 1000, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
np_numexpr_vec_calc(filename, expr, i0 = 0, imax = 1000, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
np_pivotable_create(table, left, top, value)
np_pivottable_count(mylist)
np_remove_NA_INF_2d(X)
np_remove_zeros(vv, axis1 = 1)
np_removelist(x0, xremove = [])
np_sort(arr, colid, asc = 1)
np_sort(arr, colid, asc = 1)
np_sortbycol(arr, colid, asc = True)
np_sortbycolumn(arr, colid, asc = True)
np_sortcol(arr, colid, asc = 1)
np_stack(v1, v2 = None, v3 = None, v4 = None, v5 = None)
np_torecarray(arr, colname)
np_transform2d_int_1d(m2d, onlyhalf = False)
np_uniquerows(a)
obj_getclass_of_method(meth)
obj_getclass_property(pfi)
os_config_getfile(file1)
os_config_setfile(dict_params, outfile, mode1 = 'w+')
os_csv_process(file1)
os_extracttext_allfile(nfile, dir1, pattern1 = "*.html", htmltag = 'p', deepness = 2)
os_file_exist(file1)
os_file_getname(path)
os_file_getpath(path)
os_file_gettext(file1)
os_file_listall(dir1, pattern = "*.*", dirlevel = 1, onlyfolder = 0)
os_file_mergeall(nfile, dir1, pattern1, deepness = 2)
os_file_read(file1)
os_file_rename(some_dir, pattern = "*.*", pattern2 = "", dirlevel = 1)
os_file_replacestring1(findStr, repStr, filePath)
os_file_replacestring2(findstr, replacestr, some_dir, pattern = "*.*", dirlevel = 1)
os_file_size(file1)
os_folder_copy(src, dst, symlinks = False, pattern1 = "*.py", fun_file_toignore = None)
os_folder_create(directory)
os_folder_robocopy(from_folder = '', to_folder = '', my_log='H = 'H:/robocopy_log.txt')
os_gui_popup_show(txt)
os_path_append(p1, p2 = None, p3 = None, p4 = None)
os_path_change(path1)
os_path_current()
os_path_norm(pth)
os_print_tofile(vv, file1, mode1 = 'a')
os_process_2()
os_process_run(cmd_list = ['program', 'arg1', 'arg2'], capture_output = False)
os_processify_fun(func)
os_split_dir_file(dirfile)
os_zipextractall(filezip_or_dir = "folder1/*.zip", tofolderextract = 'zdisk/test', isprint = 1)
pd_addcol(df1, name1 = 'new')
pd_applyfun_col(df, newcol, ff, use_colname = "all/[colname]")
pd_array_todataframe(array, colname = None, index1 = None, dotranspose = False)
pd_cleanquote(q)
pd_create_colmapdict_nametoint(df)
pd_createdf(array1, col1 = None, idx1 = None)
pd_csv_randomread(filename, nsample = 10000, filemaxline = -1, dtype = None)
pd_dataframe_toarray(df)
pd_date_intersection(qlist)
pd_df_todict(df, colkey = 'table', excludekey = [''], onlyfirstelt =  True)
pd_dtypes(df, columns = [], targetype = 'category')
pd_dtypes_totype2(df, columns = [], targetype = 'category')
pd_dtypes_type1_totype2(df, fromtype = str, targetype = str)
pd_extract_col_idx_val(df)
pd_extract_col_uniquevalue_tocsv(df, colname = '', csvfile = '')
pd_find(df, regex_pattern = '*', col_restrict = [], isnumeric = False, doreturnposition = False)
pd_h5_addtable(df, tablename, dbfile='F = 'F:\temp_pandas.h5')
pd_h5_cleanbeforesave(df)
pd_h5_dumpinfo(dbfile='E = 'E:\_data\stock\intraday_google.h5')
pd_h5_fromcsv_tohdfs(dircsv = 'dir1/dir2/', filepattern = '*.csv', tofilehdfs = 'file1.h5', tablename = 'df', col_category = [], dtype0 = None, encoding = 'utf-8', chunksize =  2000000, mode = 'a', format = 'table', complib = None)
pd_h5_load(filenameh5='E = 'E:/_data/_data_outlier.h5', table_id = 'data', exportype = "pandas", rowstart = -1, rowend = -1, cols = [])
pd_h5_save(df, filenameh5='E = 'E:/_data/_data_outlier.h5', key = 'data')
pd_h5_tableinfo(filenameh5, table)
pd_info(df, doreturn = 1)
pd_info_memsize(df, memusage = 0)
pd_insertcol(df, colname, vec)
pd_insertdatecol(df_insider, format1="%Y-%m-%d %H = "%Y-%m-%d %H:%M:%S:%f")
pd_insertrow(df, rowval, index1 = None, isreset = 1)
pd_is_categorical(z)
pd_np_toh5file(numpyarr, fileout = "file.h5", table1 = 'data')
pd_removecol(df1, name1)
pd_removerow(df, row_list_index = [23, 45])
pd_replacevalues(df, matrix)
pd_resetindex(df)
pd_selectrow(df, **conditions)
pd_split_col_idx_val(df)
pd_splitdf_inlist(df, colid, type1 = "dict")
pd_str_encoding_change(df, cols, fromenc = 'iso-8859-1', toenc = 'utf-8')
pd_str_isascii(x)
pd_str_unicode_tostr(df, targetype = str)
plot_XY(xx, yy, zcolor = None, tsize = None, title1 = '', xlabel = '', ylabel = '', figsize = (8, 6)
plot_heatmap(frame, ax = None, cmap = None, vmin = None, vmax = None, interpolation = 'nearest')
print_topdf()
py_importfromfile(modulename, dir1)
py_load_obj(folder = '/folder1/keyname', isabsolutpath = 0, encoding1 = 'utf-8')
py_memorysize(o, ids, hint = " deep_getsizeof(df_pd, set()
py_save_obj(obj1, keyname)
py_save_obj_dill(obj1, keyname)
read_funding_data(path)
read_funding_data(path)
read_funding_data(path)
save(obj, folder = '/folder1/keyname', isabsolutpath = 0)
save_test(folder = '/folder1/keyname', isabsolutpath = 0)
session_guispyder_load(filename)
session_guispyder_save(filename)
session_load(name = 'test_20160815')
session_load_function(name = 'test_20160815')
session_save(name = 'test')
session_save_function(name = 'test')
session_spyder_showall()
sql_create_dbengine(type1 = '', dbname = '', login = '', password = '', url = 'localhost', port = 5432)
sql_delete_table(name, dbengine)
sql_get_dbschema(dburl='sqlite = 'sqlite:///aapackage/store/yahoo.db', dbengine = None, isprint = 0)
sql_getdate()
sql_insert_csv(csvfile, dbtable, dbengine, col_drop = [])
sql_insert_csv2(csvfile = '', dbtable = '', columns = [], dbengine = None, nrows =  10000)
sql_insert_df(df, dbtable, dbengine, col_drop = ['id'], verbose = 1)
sql_insert_excel(file1 = '.xls', dbengine = None, dbtype = '')
sql_mysql_insert_excel()
sql_pivotable(dbcon, ss = 'select  ')
sql_postgres_create_table(mytable = '', database = '', username = '', password = '')
sql_postgres_pivot()
sql_postgres_query_to_csv(sqlr = 'SELECT ticker,shortratio,sector1_id, FROM stockfundamental', csv_out = '')
sql_query(sqlr = 'SELECT ticker,shortratio,sector1_id, FROM stockfundamental', dbengine = None, output = 'df', dburl='sqlite = 'sqlite:///aaserialize/store/finviz.db')
str_empty_string_array(x, y = 1)
str_empty_string_array_numpy(nx, ny = 1)
str_is_az09char(x)
str_is_azchar(x)
str_isfloat(value)
str_make_unicode(input, errors = 'replace')
str_match_fuzzy(xstring, list_string)
str_parse_stringcalendar(cal)
str_reindent(s, numSpaces)
str_split2(delimiters, string, maxsplit = 0)
str_split_pattern(sep2, ll, maxsplit = 0)
str_to_unicode(x, encoding = 'utf-8')
str_to_utf8(x)
web_getjson_fromurl(url)
web_getlink_fromurl(url)
web_getrawhtml(url1)
web_gettext_fromhtml(file1, htmltag = 'p')
web_gettext_fromurl(url, htmltag = 'p')
web_importio_todataframe(apiurl1, isurl = 1)
web_restapi_toresp(apiurl1)
web_send_email(FROM, recipient, subject, body, login1 = "mizenjapan@gmail.com", pss1 = "sophieelise237", server1 = "smtp.gmail.com", port1 = 465)
web_send_email_tls(FROM, recipient, subject, body, login1 = "mizenjapan@gmail.com", pss1 = "sophieelise237", server1 = "smtp.gmail.com", port1 = 465)
web_sendurl(url1)
z_key_splitinto_dir_name(keyname)
ztest_processify()

-------------------------methods----------------------
FundingRecord.__str__(self)
FundingRecord.parse(klass, row)
aws_ec2_ssh.__init__(self, hostname, username = 'ubuntu', key_file = None, password = None)
aws_ec2_ssh._help_ssh(self)
aws_ec2_ssh.cmd2(self, cmd1)
aws_ec2_ssh.command(self, cmd)
aws_ec2_ssh.command_list(self, cmdlist)
aws_ec2_ssh.get(self, remotefile, localfile)
aws_ec2_ssh.get_all(self, remotepath, localpath)
aws_ec2_ssh.jupyter_kill(self)
aws_ec2_ssh.jupyter_start(self)
aws_ec2_ssh.listdir(self, remotedir)
aws_ec2_ssh.put(self, localfile, remotefile)
aws_ec2_ssh.put_all(self, localpath, remotepath)
aws_ec2_ssh.python_script(self, script_path, args1)
aws_ec2_ssh.sftp_walk(self, remotepath)
aws_ec2_ssh.write_command(self, text, remotefile)
testclass.__init__(self, x)
testclass.z_autotest(self)


utilmy/zzarchive/report.py
-------------------------functions----------------------
map_show()
xl_create_pdf()
xl_create_pivot(infile, index_list = ["Manager", "Rep", "Product"], value_list = ["Price", "Quantity"])
xl_save_report(report, outfile)



utilmy/zzarchive/rstatpy.py
-------------------------functions----------------------
stl(data, ns, np = None, nt = None, nl = None, isdeg = 0, itdeg = 1, ildeg = 1, nsjump = None, ntjump = None, nljump = None, ni = 2, no = 0, fulloutput = False)



utilmy/zzarchive/storage/aapackage_gen/34/Working Copy of util34.py
-------------------------functions----------------------
getmodule_doc(module1, fileout = '')



utilmy/zzarchive/storage/aapackage_gen/34/global01.py


utilmy/zzarchive/storage/aapackage_gen/34/util34.py
-------------------------functions----------------------
getmodule_doc(module1, fileout = '')



utilmy/zzarchive/storage/aapackage_gen/codeanalysis.py
-------------------------functions----------------------
dedent()
describe(obj)
describe2(module)
describe_builtin(obj)
describe_builtin2(obj, name1)
describe_func(obj, method = False)
describe_func2(obj, method = False, name1 = '')
describe_klass(obj)
describe_klass2(obj, name1 = '')
getmodule_doc(module1, file1 = 'moduledoc.txt')
indent()
printinfile(vv, file1)
wi(*args)
wi2(*args)



utilmy/zzarchive/storage/aapackage_gen/global01.py


utilmy/zzarchive/storage/aapackage_gen/old/Working Copy of util34.py
-------------------------functions----------------------
getmodule_doc(module1, fileout = '')



utilmy/zzarchive/storage/aapackage_gen/old/util27.py
-------------------------functions----------------------
getmodule_doc(module1, fileout = '')



utilmy/zzarchive/storage/aapackage_gen/old/util34.py
-------------------------functions----------------------
getmodule_doc(module1, fileout = '')



utilmy/zzarchive/storage/aapackage_gen/old/utils27.py
-------------------------functions----------------------
acf(data)
comoment(xx, yy, nsample, kx, ky)
convertcsv_topanda(filein1, filename, tablen = 'data')
getpanda_tonumpy(filename, nsize, tablen = 'data')
getrandom_tonumpy(filename, nbdim, nbsample, tablen = 'data')
load_frompanda(filenameh5)
numexpr_topanda(filename, i0 = 0, imax = 1000, expr, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
numexpr_vect_calc(filename, i0 = 0, imax = 1000, expr, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
parsePDF(url)
plotsave(xx, yy, title1 = "")
plotshow(xx, yy, title1 = "")
remove_zeros(vv, axis1 = 1)
save_topanda(vv, filenameh5)
sort_array(vv)
unique_rows(a)



utilmy/zzarchive/storage/aapackage_gen/old/utils34.py
-------------------------functions----------------------
acf(data)
comoment(xx, yy, nsample, kx, ky)
convertcsv_topanda(filein1, filename, tablen = 'data')
getpanda_tonumpy(filename, nsize, tablen = 'data')
getrandom_tonumpy(filename, nbdim, nbsample, tablen = 'data')
load_frompanda(filenameh5)
numexpr_topanda(filename, i0 = 0, imax = 1000, expr, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
numexpr_vect_calc(filename, i0 = 0, imax = 1000, expr, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
parsePDF(url)
plotsave(xx, yy, title1 = "")
plotshow(xx, yy, title1 = "")
remove_zeros(vv, axis1 = 1)
save_topanda(vv, filenameh5)
sort_array(vv)
unique_rows(a)



utilmy/zzarchive/storage/aapackage_gen/util.py
-------------------------functions----------------------
getmodule_doc(module1, fileout = '')



utilmy/zzarchive/storage/aapackagedev/random.py
-------------------------functions----------------------
Plot2D_random_save(dir1, title1, dimxmax, dimymax, dimstep, samplejump, nsamplegraph, )
Plot2D_random_show(dir1, title1, dimxmax, dimymax, dimstep, samplejump, nsamplegraph)
acf(data)
binary_process(a, z, k)
call_process(a, z, k)
comoment(xx, yy, nsample, kx, ky)
convert_csv2hd5f(filein1, filename)
doublecheck_outlier(fileoutlier, ijump, nsample = 4000, trigger1 = 0.1, )fileoutlier=   'E =    'E:\_data\_QUASI_SOBOL_gaussian_outlier.h5'fileoutlier, 'data')    #from filevv5 =  pdf.values   #to numpy vectordel pdfistartx= 0; istarty= 0nsample= 4000trigger1=  0.1crrmax = 250000kk=0(crrmax, 4), dtype = 'int')  #empty listvv5)[0]0, kkmax1, 1) :  #Decrasing: dimy0 to dimmindimx =  vv5[kk, 0];   dimy =  vv5[kk, 1]y0= dimy * ijump + istartyym= dimy* ijump + nsample + istartyyyu1= yy1[y0 =  dimy * ijump + istartyym= dimy* ijump + nsample + istartyyyu1= yy1[y0:ym];   yyu2= yy2[y0:ym];   yyu3= yy3[y0:ym]x0= dimx * ijump + istartxxm= dimx* ijump + nsample + istartxxxu1= yy1[x0:xm];   xxu2= yy2[x0:xm];   xxu3= yy3[x0:xm]"sum( xxu3 * yyu1)") / (nsample) # X3.Y moments"sum( xxu1 * yyu3)") / (nsample)"sum( xxu2 * yyu2)") / (nsample)abs(c22) > trigger1)  :)
getdvector(dimmax, istart, idimstart)
getoutlier_fromrandom(filename, jmax1, imax1, isamplejum, nsample, fileoutlier=   'E =    'E:\_data\_QUASI_SOBOL_gaussian_outlier.h5')
getoutlier_fromrandom_fast(filename, jmax1, imax1, isamplejum, nsample, trigger1 = 0.28, fileoutlier=   'E =    'E:\_data\_QUASI_SOBOL_gaussian_outlier.h5')
getrandom_tonumpy(filename, nbdim, nbsample)
lognormal_process2d(a1, z1, a2, z2, k)
numexpr_vect_calc(filename, i0 = 0, imax = 1000, expr, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
outlier_clean(vv2)
overwrite_data(fileoutlier, vv2)
pathScheme_(T, n, zz)
pathScheme_bb(T, n, zz)
pathScheme_std(T, n, zz)
permute(yy, kmax)
permute2(xx, yy, kmax)
plot_outlier(fileoutlier, kk)fileoutlier, 'data')    #from filevv =  df.values   #to numpy vectordel dfxx= vv[kk, 0]yy =  vv[kk, 1]xx, yy, s = 1 )[00, 1000, 00, 1000])nsample)+'sampl D_'+str(dimx)+' X D_'+str(dimy)tit1)'_img/'+tit1+'_outlier.jpg', dpi = 100))yy, kmax))
plotdensity(nsample, totdim, bin01, tit0, Ti = -1)
plotdensity2(nsample, totdim, bin01, tit0, process01, vol = 0.25, tt = 5, Ti = -1)
pricing01(totdim, nsample, a, strike, process01, aa = 0.25, itmax = -1, tt = 10)
testdensity(nsample, totdim, bin01, Ti = -1)
testdensity2d(nsample, totdim, bin01, nbasset)
testdensity2d2(nsample, totdim, bin01, nbasset, process01 = lognormal_process2d, a1 = 0.25, a2 = 0.25, kk = 1)



utilmy/zzarchive/storage/alldata.py


utilmy/zzarchive/storage/allmodule.py
-------------------------functions----------------------
aa_isanaconda()



utilmy/zzarchive/storage/benchmarktest.py
-------------------------functions----------------------
payoff1(pricepath)
payoff2(pricepath)
payoffeuro1(st)
payoffeuro1(st)



utilmy/zzarchive/storage/codeanalysis.py
-------------------------functions----------------------
dedent()
describe(obj)
describe2(module, type1 = 0)
describe_builtin(obj)
describe_builtin2(obj, name1)
describe_func(obj, method = False)
describe_func2(obj, method = False, name1 = '')
describe_func3(obj, method = False, name1 = '')
describe_klass(obj)
describe_klass2(obj, name1 = '')
getmodule_doc(module1, file2 = '')
indent()
printinfile(vv, file2)
wi(*args)
wi2(*args)



utilmy/zzarchive/storage/dbcheck.py


utilmy/zzarchive/storage/derivatives.py
-------------------------functions----------------------
CRR_option_value(S0, K, T, r, vol, otype, M = 4)
N(d)
brownian_logret(mu, vol, timegrid)
brownian_process(s0, vol, timegrid)
bs(S0, K, t, T, r, d, vol)
bsbinarycall(S0, K, t, T, r, d, vol)
bscall(S0, K, t, T, r, d, vol)
bsdelta(St, K, t, T, r, d, vol, cp1)
bsdvd(St, K, t, T, r, d, vol, cp)
bsgamma(St, K, t, T, r, d, vol, cp)
bsgammaspot(St, K, t, T, r, d, vol, cp)
bsput(S0, K, t, T, r, d, vol)
bsrho(St, K, t, T, r, d, vol, cp)
bsstrikedelta(s0, K, t, T, r, d, vol, cp1)
bsstrikegamma(s0, K, t, T, r, d, vol)
bstheta(St, K, t, T, r, d, vol, cp)
bsvanna(St, K, t, T, r, d, vol, cp)
bsvega(St, K, t, T, r, d, vol, cp)
bsvolga(St, K, t, T, r, d, vol, cp)
d1f(St, K, t, T, r, d, vol)
d2f(St, K, t, T, r, d, vol)
dN(d)
dN2d(x, y)
gbm_logret(mu, vol, timegrid)
gbm_process(s0, mu, vol, timegrid)
gbm_process2(s0, mu, vol, timegrid)
gbm_process_euro(s0, mu, vol, timegrid)
gbmjump_logret(s0, mu, vol, lamda, jump_mu, jump_vol, timegrid)
gbmjump_process(s0, mu, vol, lamda, jump_mu, jump_vol, timegrid)
gdelta(St, K, t, T, r, d, vol, pv)
generateall_multigbm1(process, ww, s0, mu, vol, corrmatrix, timegrid, nbsimul, nproc = -1, type1 = -1, strike = 0.0, cp = 1)
generateallmultigbmfast(process, s0, mu, vol, corrmatrix, timegrid, nbsimul, type1)
generateallmultigbmfast2(process, s0, mu, vol, corrmatrix, timegrid, nbsimul, type1)
generateallmultiprocess(process, s0, mu, vol, corrmatrix, timegrid, nbsimul)
generateallprocess(process, params01, timegrid1, nbsimul)
generateallprocess_gbmeuro(process, params01, timegrid1, nbsimul)
genmatrix(ni, nj, gg)
gensymmatrix(ni, nj, pp)
getbrowniandata(nbasset, step, simulk)
getpv(discount, payoff, allpriceprocess)
ggamma(St, K, t, T, r, d, vol, pv)
gtheta(St, K, t, T, r, d, vol, pv)
gvega(St, K, t, T, r, d, vol, pv)
jump_process(lamda, jumps_mu, jumps_vol, timegrid)
lgnormalmoment1(ww, fft, vol, corr, tt)
lgnormalmoment2(ww, fft, vol, corr, tt)
lgnormalmoment3(ww, fft, vol, corr, tt)
lgnormalmoment4(ww, fft, vol, corr, tt)
loadbrownian(nbasset, step, nbsimul)
logret_to_price(s0, log_ret)
logret_to_ret(log_returns)
multibrownian_logret(mu, vol, corrmatrix, timegrid)
multigbm_logret(mu, vol, corrmatrix, timegrid)
multigbm_process(s0, voldt, drift, upper_cholesky, nbasset, n, kk)
multigbm_processfast(s0, voldt, drift, upper_cholesky, nbasset, n, kk)
multigbm_processfast2(s0, voldt, drift, upper_cholesky, nbasset, n, kk)
multigbm_processfast3(s0, voldt, drift, upper_cholesky, nbasset, n, kk)
multilogret_to_price(s0, log_ret)
plot_greeks(function, greek)
plot_greeks(function, greek)
plot_values(function)
savebrownian(nbasset, step, nbsimul)
solve_momentmatch3(ww, b0, fft, vol, corr, tt)
timegrid(timestep, maturityyears)



utilmy/zzarchive/storage/dl_utils.py
-------------------------functions----------------------
feats_len(fname)
file_len(fname)
get_all_data(file)
get_batch_data(file, index, size)
get_xy(line)
init_weight(hidden1, hidden2, acti_type)
log(msg, file = "")
log_p(msg, file = "")
logfile(msg, file)
save_prediction(file, prediction)
save_weights(file, tuple_weights)



utilmy/zzarchive/storage/excel.py
-------------------------functions----------------------
add_one(data)
double_sum(x, y)
get_workbook_name()
matrix_mult(x, y)
npdot()



utilmy/zzarchive/storage/global01.py


utilmy/zzarchive/storage/installNewPackage.py


utilmy/zzarchive/storage/java.py
-------------------------functions----------------------
compileJAVA(javafile)
compileJAVAtext(classname, javatxt, path1 = "")
directorygetalltext(dir1, filetype1 = "*.*", withMeta = 0, fileout = "")
directorygetalltext2(dir1, filetype1 = "*.*", type1 = 0, fileout = "")
execute_javamain(java_file)
getfpdffulltext(pdfile1)
getfulltext(file1, withMeta = 0)
importFolderJAR(dir1 = "", dirlevel = 1)
importFromMaven()
importJAR(path1 = "", path2 = "", path3 = "", path4 = "")
inspectJAR(dir1)
java_print(x)
javaerror(jpJavaException)
launchPDFbox()
launchTIKA()
listallfile(some_dir, pattern = "*.*", dirlevel = 1)
loadSingleton(class1)
showLoadedClass()
writeText(text, filename)



utilmy/zzarchive/storage/multiprocessfunc.py
-------------------------functions----------------------
bm_generator(bm, dt, n, type1)
func(val, lock)
init2(d)
init_global1(l, r)
integratene(its)
integratenp(its, nchunk)
integratenp2(its, nchunk)
list_append(count, id, out_list)
merge(d2)
multigbm_paralell_func(nbsimul, ww, voldt, drift, upper_cholesky, nbasset, n, price, type1 = 0, strike = 0, cp = 1)
multigbm_processfast7(nbsimul, s0, voldt, drift, upper_cholesky, nbasset, n, price)
ne_sin(x)
np_sin(value)
parzen_estimation(x_samples, point_x, h)
res_shared2()



utilmy/zzarchive/storage/panda_util.py
-------------------------functions----------------------
array_toexcel(vv, wk, r1)subset = 'rownum', take_last=True)level=0))a) = True)level=0))a):)
csv_topanda(filein1, filename, tablen = 'data', lineterminator=",")
database_topanda()
df_topanda(vv, filenameh5, colname = 'data')
excel_topanda(filein, fileout)
excel_topandas(filein, fileout)
folder_topanda()
getrandom_tonumpy(filename, nbdim, nbsample, tablen = 'data')
load_frompanda(filenameh5, colname = "data")
numexpr_topanda(filename, i0 = 0, imax = 1000, expr, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
numpy_topanda(vv, fileout = "", colname = "data")
panda_todabatase()
panda_toexcel()
panda_tofolder()
panda_tonumpy(filename, nsize, tablen = 'data')
remove_zeros()
sort_array()
sqlquery_topanda()
unique_rows(a)



utilmy/zzarchive/storage/portfolio.py
-------------------------functions----------------------
_date_align(dateref, datei, tmax, closei)
_notnone(x)
_reshape(x)
array_todataframe(price, symbols = None, date1 = None)
calc_optimal_weight(args, bounds, maxiter = 1)
calc_print_correlrank(close2, symjp1, nlag, refindexname, toprank2 = 5, customnameid = [], customnameid2 = [])
calc_ranktable(close2, symjp1, nlag, refindex, funeval, funargs)
calc_statestock(close2, dateref, symfull)
calcbasket_obj(wwvec, *data)
calcbasket_table(wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps =  0.000)
causality_y1_y2(price2, price1, maxlag)
cointegration(x, y)
correl_fast(xn, y, nx)
correl_rankbystock(stkid = [2, 5, 6], correl = [[1, 0], [0, 1]])
correl_reducebytrigger(correl2, trigger)
correlation_mat(matx, type1 = "robust", type2 = "correl")
data_jpsector()
dataframe_toarray(df)
date_add_bdays(from_date, add_days)
date_align(array1, dateref)
date_alignfromdateref(array1, dateref)
date_as_float(dt)
date_diffindays(intdate1, intdate2)
date_earningquater(t1)
date_extract_dailyopenclosetime(spdateref1, market = 'us')
date_find_intradateid(datetimelist, stringdate = ['20160420223000'])
date_find_kday_fromintradaydate(kintraday, intradaydate, dailydate)
date_find_kintraday_fromdate(d1, intradaydate1, h1 = 9, m1 = 30)
date_finddateid(date1, dateref)
date_generatedatetime(start = "20100101", nbday = 10, end = "")
date_getspecificdate(datelist, datetype1 = "yearend", outputype1 = "intdate", includelastdate = True, includefirstdate = False, )
date_is_3rdfriday(s)
date_option_expiry(date)
date_removetimezone(datelist)
date_todatetime(tlist)
datediff_inyear(startdate, enddate)
dateint_todatetime(datelist1)
dateint_tostring(datelist1, format1 = '%b-%y')
datenumpy_todatetime(tt, islocaltime = True)
datestring_todatetime(datelist1, format1 =  "%Y%m%d")
datestring_toint(datelist1)
datetime_convertzone1_tozone2(tt, fromzone = 'Japan', tozone = 'US/Eastern')
datetime_todate(tt)
datetime_toint(datelist1)
datetime_tointhour(datelist1)
datetime_tonumpypdate(t, islocaltime = True)
datetime_tostring(datelist1)
fitness(p)
folio_cost_turnover(wwall, bsk, dateref)
folio_fixedweightprice(price, fixedww, costpa = 0.0)
folio_fixedweightret(ret, fixedww)
folio_inverseetf(price, costpa = 0.0)
folio_lowcorrelation(sym01, nstock, periodlist, dateref, close1, kbenchmark, badlist, costbppa = 0.02, showgraph = True)
folio_riskpa(ret, targetvol = 0.1, volrange = 90)
folio_volta(bsk, targetvol = 0.11, volrange =  90, expocap = 1.5)
folio_voltarget(bsk, targetvol = 0.11, volrange =  90, expocap = 1.5)
generate_sepvertical(asset1, tt, tmax, start = None, datebar = None)
get_price2book(symbol)
getdiff_fromquotes(close, timelag)
getlogret_fromquotes(close, timelag = 1)
getprice_fromret(ret, normprice = 100)
getret_fromquotes(close, timelag = 1)
imp_close_dateref(sym01, sdate = 20100101, edate = 20160628, datasource = '', typeprice = "close")
imp_csvquote_topanda(file1, filenameh5, dfname = 'sym1', fromzone = 'Japan', tozone = 'UTC')
imp_errorticker(symbols, start = "20150101", end = "20160101")
imp_findticker(tickerlist, sym01, symname)
imp_finviz()
imp_finviz_financials()
imp_finviz_news()
imp_getcsvname(name1, date1, inter, tframe)
imp_googleIntradayQuoteSave(name1, date1, inter, tframe, dircsv)
imp_googleQuoteList(symbols, date1, date2, inter = 23400, tframe = 2000, dircsv = '', intraday1 = True)
imp_googleQuoteSave(name1, date1, date2, dircsv)
imp_numpyclose_frompandas(dbfile, symlist = [], t0 = 20010101, t1 = 20010101, priceid = "close", maxasset = 2500, tmax2 = 2000)
imp_panda_checkquote(quotes)
imp_panda_cleanquotes(df, datefilter)
imp_panda_db_dumpinfo(dbfile='E = 'E:\_data\stock\intraday_google.h5')
imp_panda_getListquote(symbols, close1 = 'close', start='12/18/2015 00 = '12/18/2015 00:00:00+00:00', end='3/1/2016 00 = '3/1/2016 00:00:00+00:00', freq = '0d0h10min', filepd= 'E =  'E:\_data\stock\intraday_google.h5', tozone = 'Japan', fillna = True, interpo = True)
imp_panda_getquote(filenameh5, dfname = "data")
imp_panda_insertfoldercsv(dircsv, filepd= r'E =  r'E:\_data\stock\intraday_google.h5', fromtimezone = 'Japan', tozone = 'UTC')
imp_panda_removeDuplicate(filepd=  'E =   'E:\_data\stock\intraday_google.h5')
imp_panda_storecopy()
imp_pd_merge_database(filepdfrom, filepdto)
imp_quote_tohdfs(sym, qqlist, filenameh5, fromzone = 'Japan', tozone = 'UTC')
imp_quotes_errordate(quotes, dateref)
imp_quotes_fromtxt(stocklist01, filedir='E = 'E:/_data/stock/daily/20160610/jp', startdate = 20150101, endate = 20160616)
imp_screening_addrecommend(string1, dbname = 'stock_recommend')
imp_yahoo_financials_url(ticker_symbol, statement = "is", quarterly = False)
imp_yahoo_periodic_figure(soup, yahoo_figure)
imp_yahooticker(symbols, start = "20150101", end = "20160101", type1 = 1)
isfloat(value)
isint(x)
load_asset_fromfile(file1)
max_withposition(values)
min_withposition(values)
norm_fast(y, ny)
np_countretsign(x)
np_distance_l1(x, y, wwerr)
np_similarity(x, y, wwerr = [], type1 = 0)
np_trendtest(x, alpha  =  0.05)
objective_criteria(bsk, criteria, date1 = None)
pd_filterbydate(df, dtref = None, start='2016-06-06 00 = '2016-06-06 00:00:00', end='2016-06-14 00 = '2016-06-14 00:00:00', freq = '0d0h05min', timezone = 'Japan')
pd_transform_asset(q0, q1, type1 = "spread")
plot_price(asset, y2 = None, y3 = None, y4 = None, y5 = None, sym = None, savename1 = '', tickperday = 20, date1 = None, graphsize = (10, 5)
plot_pricedate(date1, sym1, asset1, sym2 = None, bsk1 = None, verticaldate = None, savename1 = '', graphsize = (10, 5)
plot_priceintraday(data)
price_normalize100(ret, normprice = 100)
price_normalize_1d(ret, normprice = 100, dtype1 =  np.float16)
regression(yreturn, xreturn, type1 = "elasticv")
regression_allstocks_vs_riskfactors(symstock, pricestock, symriskfac, priceriskfac, nlaglist)
regression_fixedsymbolstock(sym, ret_close2, tsstart, tsample, ret_spy, spyclose, regonly = True)
regression_getpricefromww(spyclose, ww01, regasset01, ret_close2, tstart, tlag = 1)
rolling_cointegration(x, y)
rsk_calc_all_TA(df = 'panda_dataframe')
save_asset_tofile(file1, asset1, asset2 = None, asset3 = None, date1 = None, title1 = None)
similarity_correl(ret_close2, funargs)
sk_cov_fromcorrel(correl, ret_close1)
ta_highbandtrend1(close2, type1 = 0)
ta_lowbandtrend1(close2, type1 = 0)
volhisto_fromprice(price, t, volrange, axis = 0)
volhisto_fromret(retbsk, t, volrange, axis = 0)

-------------------------methods----------------------
Quote.__init__(self)
Quote.__repr__(self)
Quote.append(self, dt, open_, high, low, close, volume)
Quote.read_csv(self, filename)
Quote.to_csv(self)
Quote.write_csv(self, filename)
googleIntradayQuote.__init__(self, symbol, interval_seconds = 300, num_days = 5)
googleQuote.__init__(self, symbol, start_date, end_date = datetime.date.today()
index.__init__(self, id1, sym, ww, tstart)
index._objective_criteria(self, bsk)
index._statecalc(self)
index._weightcalc_constant(self, ww2, t)
index._weightcalc_generic(self, wwvec, t)
index._weightcalc_regime2(self, wwvec, t)
index.calc_optimal_weight(self, maxiter = 1)
index.calcbasket_obj(self, wwvec)
index.close()
index.help()
index.updatehisto()
searchSimilarity.__generate_return__(self, nlag)
searchSimilarity.__init__(self, filejpstock=r'E = r'E:/_data/stock/daily/20160616/jp', sym01 = ['7203'], symname = ['Toyota'], startdate =  20150101, enddate = 20160601, pricetype = "close")
searchSimilarity.__overweight__(self, px)
searchSimilarity.export_results(self, filename)
searchSimilarity.get_rankresult(self, filetosave = '')
searchSimilarity.launch_search(self)
searchSimilarity.load_quotes_fromdb(self, picklefile = '')
searchSimilarity.set_searchcriteria(self, name1 = '7203', date1 = 20160301, date2 = 20160601, nlag = 1, searchperiodstart = 20120101, typesearch = "pattern2", )
searchSimilarity.show_comparison_graph(self, maxresult = 20, show_only_different_time = True, fromid = 0, fromend =  0, filenameout = '')
searchSimilarity.staticmethod(self, x)


utilmy/zzarchive/storage/rec_data.py
-------------------------functions----------------------
_build_interaction_matrix(rows, cols, data)
_download_movielens(dest_path)
_get_movie_raw_metadata()
_get_movielens_path()
_get_raw_movielens_data()
_parse(data)
get_dense_triplets(uids, pids, nids, num_users, num_items)
get_movielens_data()
get_movielens_item_metadata(use_item_ids)
get_triplets(mat)



utilmy/zzarchive/storage/rec_metrics.py
-------------------------functions----------------------
full_auc(model, ground_truth)
precision_at_k(model, ground_truth, k, user_features = None, item_features = None)
predict(model, uid, pids)



utilmy/zzarchive/storage/sobol.py
-------------------------functions----------------------
Plot2D_random_save(dir1, title1, dimxmax, dimymax, dimstep, samplejump, nsamplegraph, )
Plot2D_random_show(dir1, title1, dimxmax, dimymax, dimstep, samplejump, nsamplegraph)
acf(data)
binary_process(a, z, k)
call_process(a, z, k)
comoment(xx, yy, nsample, kx, ky)
convert_csv2hd5f(filein1, filename)
doublecheck_outlier(fileoutlier, ijump, nsample = 4000, trigger1 = 0.1, )
getdvector(dimmax, istart, idimstart)
getoutlier_fromrandom(filename, jmax1, imax1, isamplejum, nsample, fileoutlier=   'E =    'E:\_data\_QUASI_SOBOL_gaussian_outlier.h5')
getoutlier_fromrandom_fast(filename, jmax1, imax1, isamplejum, nsample, trigger1 = 0.28, fileoutlier=   'E =    'E:\_data\_QUASI_SOBOL_gaussian_outlier.h5')
getrandom_tonumpy(filename, nbdim, nbsample)
lognormal_process2d(a1, z1, a2, z2, k)
numexpr_vect_calc(filename, i0, imax, expr, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
outlier_clean(vv2)
overwrite_data(fileoutlier, vv2)
pathScheme_(T, n, zz)
pathScheme_bb(T, n, zz)
pathScheme_std(T, n, zz)
permute(yy, kmax)
permute2(xx, yy, kmax)
plot_outlier(fileoutlier, kk)
plotdensity(nsample, totdim, bin01, tit0, Ti = -1)
plotdensity2(nsample, totdim, bin01, tit0, process01, vol = 0.25, tt = 5, Ti = -1)
pricing01(totdim, nsample, a, strike, process01, aa = 0.25, itmax = -1, tt = 10)
testdensity(nsample, totdim, bin01, Ti = -1)
testdensity2d(nsample, totdim, bin01, nbasset)
testdensity2d2(nsample, totdim, bin01, nbasset, process01 = lognormal_process2d, a1 = 0.25, a2 = 0.25, kk = 1)



utilmy/zzarchive/storage/stateprocessor.py
-------------------------functions----------------------
and2(tuple1)
ff(x, symfull = symfull)
gap(close, t0, t1, lag)
get_stocklist(clf, s11, initial, show1 = 1)
get_treeselect(stk, s1 = s1, xnewdata = None, newsample = 5, show1 = 1, nbtree = 5, depthtree = 10)
load_patternstate(name1)
perf(close, t0, t1)
printn(ss, symfull = symfull, s1 = s1)
process_stock(stkstr, show1 = 1)
show(ll, s1 = s1)
sort(x, col, asc)
store_patternstate(tree, sym1, theme, symfull = symfull)



utilmy/zzarchive/storage/symbolicmath.py
-------------------------functions----------------------
EEvarbrownian(ff1d)
EEvarbrownian2d(ff)
N(x)
bs(s0, K, t, T, r, d, vol)
bsbinarycall(s0, K, t, T, r, d, vol)
bscall(s0, K, t, T, r, d, vol)
bsdelta(St, K, t, T, r, d, vol, cp1)
bsdvd(St, K, t, T, r, d, vol, cp)
bsgamma(St, K, t, T, r, d, vol, cp)
bsgammaspot(St, K, t, T, r, d, vol, cp)
bsput(s0, K, t, T, r, d, vol)
bsrho(St, K, t, T, r, d, vol, cp)
bsstrikedelta(s0, K, t, T, r, d, vol, cp1)
bsstrikegamma(s0, K, t, T, r, d, vol)
bstheta(St, K, t, T, r, d, vol, cp)
bsvanna(St, K, t, T, r, d, vol, cp)
bsvega(St, K, t, T, r, d, vol, cp)
bsvolga(St, K, t, T, r, d, vol, cp)
d1f(St, K, t, T, r, d, vol)
d1xf(St, K, t, T, r, d, vol)
d2f(St, K, t, T, r, d, vol)
d2xf(St, K, t, T, r, d, vol)
dN(x)
decomposecorrel(m1)
diffn(ff, x0, kk)
dnn(x, y, p)
dnn2(x, y, p)
factorpoly(pp)
lagrangian2d(ll)
nn(x)
nn2(x, y, p)
print2(a0, a1 = '', a2 = '', a3 = '', a4 = '', a5 = '', a6 = '', a7 = '', a8 = '')
spp()
taylor2(ff, x0, n)



utilmy/zzarchive/storage/technical_indicator.py
-------------------------functions----------------------
ACCDIST(df, n)
ADX(df, n, n_ADX)
ATR(df, n)
BBANDS(df, n)
CCI(df, n)
COPP(df, n)
Chaikin(df)
DONCH(df, n)
EMA(df, n)
EOM(df, n)
FORCE(df, n)
KELCH(df, n)
KST(df, r1, r2, r3, r4, n1, n2, n3, n4)
MA(df, n)
MACD(df, n_fast, n_slow)
MFI(df, n)
MOM(df, n)
MassI(df)
OBV(df, n)
PPSR(df)
RET(df, n)
RMI(df, n = 14, m = 10)
ROC(df, n)
RSI(df, n = 14)
RWI(df, nn, nATR)
STDDEV(df, n)
STO(df)
STOK(df)
TRIX(df, n)
TSI(df, r, s)
ULTOSC(df)
Vortex(df, n)
date_earningquater(t1)
date_option_expiry(date)
distance(df, tk, tkname)
distance_day(df, tk, tkname)
findhigher(item, vec)
findlower(item, vec)
linearreg(a, *args)
nbday_high(df, n)
nbday_high(df, n)
nbday_low(df, n)
nbtime_reachtop(df, n, trigger = 0.005)
np_find(item, vec)
np_find_maxpos(values)
np_find_minpos(values)
np_findlocalmax(v)
np_findlocalmin(v)
np_sortbycolumn(arr, colid, asc = True)
optionexpiry_dist(df)
qearning_dist(df)
supportmaxmin1(df1)



utilmy/zzarchive/storage/testmulti.py
-------------------------functions----------------------
mc01()
mc02()
multiprocess(processes, samples, x, widths)
random_tree(Data)
random_tree(Data)
serial(samples, x, widths)
test01()
test01()



utilmy/zzarchive/storage/theano_imdb.py
-------------------------functions----------------------
get_dataset_file(dataset, default_dataset, origin)
load_data(path = "imdb.pkl", n_words = 100000, valid_portion = 0.1, maxlen = None, sort_by_len = True)
prepare_data(seqs, labels, maxlen = None)



utilmy/zzarchive/storage/theano_lstm.py
-------------------------functions----------------------
_p(pp, name)
adadelta(lr, tparams, grads, x, mask, y, cost)
build_model(tparams, options)
dropout_layer(state_before, use_noise, trng)
get_dataset(name)
get_layer(name)
get_minibatches_idx(n, minibatch_size, shuffle = False)
init_params(options)
init_tparams(params)
load_params(path, params)
lstm_layer(tparams, state_below, options, prefix = 'lstm', mask = None)
numpy_floatX(data)
ortho_weight(ndim)
param_init_lstm(options, params, prefix = 'lstm')
pred_error(f_pred, prepare_data, data, iterator, verbose = False)
pred_probs(f_pred_prob, prepare_data, data, iterator, verbose = False)
rmsprop(lr, tparams, grads, x, mask, y, cost)
sgd(lr, tparams, grads, x, mask, y, cost)
train_lstm(dim_proj = 128, # word embeding dimension and LSTM number of hidden units.patience = 10, # Number of epoch to wait before early stop if no progressmax_epochs = 5000, # The maximum number of epoch to rundispFreq = 10, # Display to stdout the training progress every N updatesdecay_c = 0., # Weight decay for the classifier applied to the U weights.not used for adadelta and rmsprop)n_words = 10000, # Vocabulary sizeprobably need momentum and decaying learning rate).encoder = 'lstm', # TODO: can be removed must be lstm.saveto = 'lstm_model.npz', # The best model will be saved therevalidFreq = 370, # Compute the validation error after this number of update.saveFreq = 1110, # Save the parameters after every saveFreq updatesmaxlen = 100, # Sequence longer then this get ignoredbatch_size = 16, # The batch size during training.valid_batch_size = 64, # The batch size used for validation/test set.dataset = 'imdb', noise_std = 0., use_dropout = True, # if False slightly faster, but worst test errorreload_model = None, # Path to a saved model we want to start from.test_size = -1, # If >0, we keep only this number of test example.)
unzip(zipped)
zipp(params, tparams)



utilmy/zzarchive/util.py
-------------------------functions----------------------
a_autoreload()
a_cleanmemory()
a_info_conda_jupyter()
a_isanaconda()
a_module_codesample(module_str = 'pandas')
a_module_doc(module_str = 'pandas')
a_module_generatedoc(module_str = "pandas", fileout = '')
a_run_ipython(cmd1)
a_start_log(id1 = '', folder = 'aaserialize/log/')
aa_unicode_ascii_utf8_issue()
date_add_bday(from_date, add_days)
date_add_bdays(from_date, add_days)
date_allinfo()
date_convert(t1, fromtype, totype)
date_diffinbday(intd2, intd1)
date_diffinday(intdate1, intdate2)
date_diffinyear(startdate, enddate)
date_finddateid(date1, dateref)
date_gencalendar(start = '2010-01-01', end = '2010-01-15', country = 'us')
date_generatedatetime(start = "20100101", nbday = 10, end = "")
date_getspecificdate(datelist, datetype1 = "yearend", outputype1 = "intdate", includelastdate = True, includefirstdate = False, )
date_holiday()
date_now(i = 0)
date_nowtime(type1 = 'str', format1= "%Y-%m-%d %H =  "%Y-%m-%d %H:%M:%S:%f")
date_remove_bdays(from_date, add_days)
date_tofloat(dt)
dateint_todatetime(datelist1)
datenumpy_todatetime(tt, islocaltime = True)
datestring_todatetime(datelist1, format1 =  "%Y%m%d")
datestring_toint(datelist1)
datetime_toint(datelist1)
datetime_tonumpydate(t, islocaltime = True)
datetime_tostring(datelist1)
find(item, vec)
findhigher(x, vec)
findlower(x, vec)
findnone(vec)
finds(itemlist, vec)
findx(item, vec)
googledrive_get()
googledrive_list()
googledrive_put()
isexist(a)
isfloat(x)
isint(x)
load(folder = '/folder1/keyname', isabsolutpath = 0)
max_kpos(arr, kth)
min_kpos(arr, kth)
np_acf(data)
np_addcolumn(arr, nbcol)
np_addrow(arr, nbrow)
np_and1(x, y, x3 = None, x4 = None, x5 = None, x6 = None, x7 = None, x8 = None)
np_cleanmatrix(m)
np_comoment(xx, yy, nsample, kx, ky)
np_dict_tolist(dd, withkey = 0)
np_dict_tostr_key(dd)
np_dict_tostr_val(dd)
np_dictordered_create()
np_enumerate2(vec_1d)
np_find(item, vec)
np_find_maxpos(values)
np_find_maxpos_2nd(numbers)
np_find_minpos(values)
np_findfirst(item, vec)
np_findlocalmax(v, trig)
np_findlocalmax2(v, trig)
np_findlocalmin(v, trig)
np_findlocalmin2(v, trig)
np_int_tostr(i)
np_interpolate_nan(y)
np_list_flatten(seq)
np_list_tofreqdict(l1, wweight = [])
np_list_unique(seq)
np_ma(vv, n)
np_map_dict_to_bq_schema(source_dict, schema, dest_dict)
np_memory_array_adress(x)
np_mergelist(x0, x1)
np_minimize(fun_obj, x0 = [0.0], argext = (0, 0)
np_minimizeDE(fun_obj, bounds, name1, maxiter = 10, popsize = 5, solver = None)
np_nan_helper(y)
np_numexpr_tohdfs(filename, expr, i0 = 0, imax = 1000, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
np_numexpr_vec_calc(filename, expr, i0 = 0, imax = 1000, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
np_pivotable_create(table, left, top, value)
np_pivottable_count(mylist)
np_remove_NA_INF_2d(X)
np_remove_zeros(vv, axis1 = 1)
np_removelist(x0, xremove = [])
np_sort(arr, colid, asc = 1)
np_sort(arr, colid, asc = 1)
np_sortbycol(arr, colid, asc = True)
np_sortbycolumn(arr, colid, asc = True)
np_sortcol(arr, colid, asc = 1)
np_stack(v1, v2 = None, v3 = None, v4 = None, v5 = None)
np_torecarray(arr, colname)
np_transform2d_int_1d(m2d, onlyhalf = False)
np_uniquerows(a)
obj_getclass_of_method(meth)
obj_getclass_property(pfi)
os_config_getfile(file1)
os_config_setfile(dict_params, outfile, mode1 = 'w+')
os_csv_process(file1)
os_file_are_same_file_types(paths)
os_file_exist(file1)
os_file_extracttext(output_file, dir1, pattern1 = "*.html", htmltag = 'p', deepness = 2)
os_file_get_file_extension(file_path)
os_file_get_path_from_stream(maybe_stream)
os_file_getname(path)
os_file_getpath(path)
os_file_gettext(file1)
os_file_isame(file1, file2)
os_file_listall(dir1, pattern = "*.*", dirlevel = 1, onlyfolder = 0)
os_file_mergeall(nfile, dir1, pattern1, deepness = 2)
os_file_norm_paths(paths, marker = '*')
os_file_normpath(path)
os_file_read(file1)
os_file_rename(some_dir, pattern = "*.*", pattern2 = "", dirlevel = 1)
os_file_replace(source_file_path, pattern, substring)
os_file_replacestring1(findStr, repStr, filePath)
os_file_replacestring2(findstr, replacestr, some_dir, pattern = "*.*", dirlevel = 1)
os_file_size(file1)
os_file_try_to_get_extension(path_or_strm)
os_folder_copy(src, dst, symlinks = False, pattern1 = "*.py", fun_file_toignore = None)
os_folder_create(directory)
os_folder_is_path(path_or_stream)
os_folder_robocopy(from_folder = '', to_folder = '', my_log='H = 'H:/robocopy_log.txt')
os_gui_popup_show(txt)
os_path_append(p1, p2 = None, p3 = None, p4 = None)
os_path_change(path1)
os_path_current()
os_path_norm(pth)
os_print_tofile(vv, file1, mode1 = 'a')
os_process_2()
os_process_run(cmd_list = ['program', 'arg1', 'arg2'], capture_output = False)
os_processify_fun(func)
os_split_dir_file(dirfile)
os_wait_cpu(priority = 300, cpu_min = 50)
os_zipextractall(filezip_or_dir = "folder1/*.zip", tofolderextract = 'zdisk/test', isprint = 1)
pd_applyfun_col(df, newcol, ff, use_colname = "all/[colname]")
pd_array_todataframe(array, colname = None, index1 = None, dotranspose = False)
pd_col_addfrom_dfmap(df, dfmap, colkey, colval, df_colused, df_colnew, exceptval = -1, inplace =  True)
pd_create_colmapdict_nametoint(df)
pd_createdf(array1, col1 = None, idx1 = None)
pd_csv_randomread(filename, nsample = 10000, filemaxline = -1, dtype = None)
pd_dataframe_toarray(df)
pd_date_intersection(qlist)
pd_df_todict(df1, colkey = 'table', excludekey = [''], onlyfirstelt =  True)
pd_df_todict2(df1, colkey = 'table', excludekey = [''], onlyfirstelt =  True)
pd_dtypes(df, columns = [], targetype = 'category')
pd_dtypes_totype2(df, columns = [], targetype = 'category')
pd_dtypes_type1_totype2(df, fromtype = str, targetype = str)
pd_extract_col_idx_val(df)
pd_extract_col_uniquevalue_tocsv(df, colname = '', csvfile = '')
pd_find(df, regex_pattern = '*', col_restrict = [], isnumeric = False, doreturnposition = False)
pd_h5_addtable(df, tablename, dbfile='F = 'F:\temp_pandas.h5')
pd_h5_cleanbeforesave(df)
pd_h5_dumpinfo(dbfile='E = 'E:\_data\stock\intraday_google.h5')
pd_h5_fromcsv_tohdfs(dircsv = 'dir1/dir2/', filepattern = '*.csv', tofilehdfs = 'file1.h5', tablename = 'df', col_category = [], dtype0 = None, encoding = 'utf-8', chunksize =  2000000, mode = 'a', format = 'table', complib = None)
pd_h5_load(filenameh5='E = 'E:/_data/_data_outlier.h5', table_id = 'data', exportype = "pandas", rowstart = -1, rowend = -1, cols = [])
pd_h5_save(df, filenameh5='E = 'E:/_data/_data_outlier.h5', key = 'data')
pd_h5_tableinfo(filenameh5, table)
pd_info(df, doreturn = 1)
pd_info_memsize(df, memusage = 0)
pd_insertdatecol(df_insider, format1="%Y-%m-%d %H = "%Y-%m-%d %H:%M:%S:%f")
pd_insertrow(df, rowval, index1 = None, isreset = 1)
pd_is_categorical(z)
pd_np_toh5file(numpyarr, fileout = "file.h5", table1 = 'data')
pd_removecol(df1, name1)
pd_removerow(df, row_list_index = [23, 45])
pd_replacevalues(df, matrix)
pd_resetindex(df)
pd_row_findlast(df, colid = 0, emptyrowid = None)
pd_row_select(df, **conditions)
pd_split_col_idx_val(df)
pd_splitdf_inlist(df, colid, type1 = "dict")
pd_str_encoding_change(df, cols, fromenc = 'iso-8859-1', toenc = 'utf-8')
pd_str_isascii(x)
pd_str_unicode_tostr(df, targetype = str)
plot_XY(xx, yy, zcolor = None, tsize = None, title1 = '', xlabel = '', ylabel = '', figsize = (8, 6)
plot_heatmap(frame, ax = None, cmap = None, vmin = None, vmax = None, interpolation = 'nearest')
print_topdf()
py_exception_print()
py_importfromfile(modulename, dir1)
py_load_obj(folder = '/folder1/keyname', isabsolutpath = 0, encoding1 = 'utf-8')
py_log_write(LOGFILE, prefix)
py_memorysize(o, ids, hint = " deep_getsizeof(df_pd, set()
py_save_obj(obj1, keyname = '', otherfolder = 0)
py_save_obj_dill(obj1, keyname = '', otherfolder = 0)
read_funding_data(path)
read_funding_data(path)
read_funding_data(path)
save(obj, folder = '/folder1/keyname', isabsolutpath = 0)
save_test(folder = '/folder1/keyname', isabsolutpath = 0)
session_guispyder_load(filename)
session_guispyder_save(filename)
session_load(name = 'test_20160815')
session_load_function(name = 'test_20160815')
session_save(name = 'test')
session_save_function(name = 'test')
session_spyder_showall()
sql_getdate()
str_empty_string_array(x, y = 1)
str_empty_string_array_numpy(nx, ny = 1)
str_is_az09char(x)
str_is_azchar(x)
str_isfloat(value)
str_make_unicode(input, errors = 'replace')
str_match_fuzzy(xstring, list_string)
str_parse_stringcalendar(cal)
str_reindent(s, numSpaces)
str_split2(delimiters, string, maxsplit = 0)
str_split_pattern(sep2, ll, maxsplit = 0)
str_to_unicode(x, encoding = 'utf-8')
str_to_utf8(x)
z_key_splitinto_dir_name(keyname)
ztest_processify()

-------------------------methods----------------------
FundingRecord.__str__(self)
FundingRecord.parse(klass, row)
testclass.__init__(self, x)
testclass.z_autotest(self)


utilmy/zzarchive/util_aws.py
-------------------------functions----------------------
aws_accesskey_get(access = '', key = '')
aws_conn_create(region = "ap-northeast-2", access = '', key = '')
aws_conn_do(action = '', region = "ap-northeast-2")
aws_conn_getallregions(conn = None)
aws_conn_getinfo(conn)
aws_credentials(account = None)
aws_ec2_allocate_elastic_ip(con, instance_id = "", elastic_ip = '', region = "ap-northeast-2")
aws_ec2_cmd_ssh(cmdlist =   ["ls " ], host = 'ip', doreturn = 0, ssh = None, username = 'ubuntu', keyfilepath = '')
aws_ec2_create_con(contype = 'sftp/ssh', host = 'ip', port = 22, username = 'ubuntu', keyfilepath = '', password = '', keyfiletype = 'RSA', isprint = 1)
aws_ec2_get_id(ipadress = '', instance_id = '')
aws_ec2_get_instanceid(con, ip_address)
aws_ec2_printinfo(instance = None, ipadress = "", instance_id = "")
aws_ec2_python_script(script_path, args1, host)
aws_ec2_res_start(con, region, key_name, ami_id, inst_type = "cx2.2", min_count  = 1, max_count  = 1, pars= {"security_group" =  {"security_group": [""], "disk_size": 25, "disk_type": "ssd", "volume_type": "gp2"})
aws_ec2_res_stop(con, ipadress = "", instance_id = "")
aws_ec2_spot_start(con, region, key_name = "ecsInstanceRole", inst_type = "cx2.2", ami_id = "", pricemax = 0.15, elastic_ip = '', pars= {"security_group" =  {"security_group": [""], "disk_size": 25, "disk_type": "ssd", "volume_type": "gp2"})
aws_ec2_spot_stop(con, ipadress = "", instance_id = "")
aws_s3_file_read(bucket1, filepath, isbinary = 1)
aws_s3_folder_printtall(bucket_name = 'zdisk')
aws_s3_getbucketconn(s3dir)
aws_s3_getfrom_s3(froms3dir = 'task01/', todir = '', bucket_name = 'zdisk')
aws_s3_puto_s3(fromdir_file = 'dir/file.zip', todir = 'bucket/folder1/folder2')
aws_s3_url_split(url)
ztest_01()

-------------------------methods----------------------
aws_ec2_ssh.__init__(self, hostname, username = 'ubuntu', key_file = None, password = None)
aws_ec2_ssh._help_ssh(self)
aws_ec2_ssh.cmd2(self, cmd1)
aws_ec2_ssh.command(self, cmd)
aws_ec2_ssh.command_list(self, cmdlist)
aws_ec2_ssh.get(self, remotefile, localfile)
aws_ec2_ssh.get_all(self, remotepath, localpath)
aws_ec2_ssh.jupyter_kill(self)
aws_ec2_ssh.jupyter_start(self)
aws_ec2_ssh.listdir(self, remotedir)
aws_ec2_ssh.put(self, localfile, remotefile)
aws_ec2_ssh.put_all(self, localpath, remotepath)
aws_ec2_ssh.python_script(self, script_path, args1)
aws_ec2_ssh.sftp_walk(self, remotepath)
aws_ec2_ssh.write_command(self, text, remotefile)


utilmy/zzarchive/util_min.py
-------------------------functions----------------------
a_get_pythonversion()
a_isanaconda()
isexist(a)
isfloat(x)
isint(x)
load(folder = '/folder1/keyname', isabsolutpath = 0)
os_file_exist(file1)
os_file_getname(path)
os_file_getpath(path)
os_file_gettext(file1)
os_file_listall(dir1, pattern = "*.*", dirlevel = 1, onlyfolder = 0)
os_file_mergeall(nfile, dir1, pattern1, deepness = 2)
os_file_read(file1)
os_file_rename(some_dir, pattern = "*.*", pattern2 = "", dirlevel = 1)
os_file_replacestring1(findStr, repStr, filePath)
os_file_replacestring2(findstr, replacestr, some_dir, pattern = "*.*", dirlevel = 1)
os_file_size(file1)
os_folder_copy(src, dst, symlinks = False, pattern1 = "*.py", fun_file_toignore = None)
os_folder_create(directory)
os_folder_robocopy(from_folder = '', to_folder = '', my_log='H = 'H:/robocopy_log.txt')
os_path_append(p1, p2 = None, p3 = None, p4 = None)
os_path_change(path1)
os_path_current()
os_path_norm(pth)
os_print_tofile(vv, file1, mode1 = 'a')
os_split_dir_file(dirfile)
os_wait_cpu(priority = 300, cpu_min = 50)
os_zip_checkintegrity(filezip1)
os_zipextractall(filezip_or_dir = "folder1/*.zip", tofolderextract = 'zdisk/test', isprint = 1)
os_zipfile(folderin, folderzipname, iscompress = True)
os_zipfolder(dir_tozip = '/zdisks3/output', zipname = '/zdisk3/output.zip', dir_prefix = None, iscompress = True)
py_importfromfile(modulename, dir1)
py_load_obj(folder = '/folder1/keyname', isabsolutpath = 0, encoding1 = 'utf-8')
py_memorysize(o, ids, hint = " deep_getsizeof(df_pd, set()
py_save_obj(obj, folder = '/folder1/keyname', isabsolutpath = 0)
save(obj, folder = '/folder1/keyname', isabsolutpath = 0)
save_test(folder = '/folder1/keyname', isabsolutpath = 0)
z_key_splitinto_dir_name(keyname)



utilmy/zzarchive/util_ml.py
-------------------------functions----------------------
create_adam_optimizer(learning_rate, momentum)
create_bias_variable(name, shape)
create_weight_variable(name, shape)
parse_args(ppa = None, args =  {})
parse_args2(ppa = None)
tf_check()
tf_global_variables_initializer(sess = None)
visualize_result()

-------------------------methods----------------------
TextLoader.__init__(self, data_dir, batch_size, seq_length)
TextLoader.create_batches(self)
TextLoader.load_preprocessed(self, vocab_file, tensor_file)
TextLoader.next_batch(self)
TextLoader.preprocess(self, input_file, vocab_file, tensor_file)
TextLoader.reset_batch_pointer(self)


utilmy/zzarchive/util_sql.py
-------------------------functions----------------------
sql_create_dbengine(type1 = '', dbname = '', login = '', password = '', url = 'localhost', port = 5432)
sql_delete_table(name, dbengine)
sql_get_dbschema(dburl='sqlite = 'sqlite:///aapackage/store/yahoo.db', dbengine = None, isprint = 0)
sql_insert_csv(csvfile, dbtable, dbengine, col_drop = [])
sql_insert_csv2(csvfile = '', dbtable = '', columns = [], dbengine = None, nrows =  10000)
sql_insert_df(df, dbtable, dbengine, col_drop = ['id'], verbose = 1)
sql_insert_excel(file1 = '.xls', dbengine = None, dbtype = '')
sql_mysql_insert_excel()
sql_pivotable(dbcon, ss = 'select  ')
sql_postgres_create_table(mytable = '', database = '', username = '', password = '')
sql_postgres_pivot()
sql_postgres_query_to_csv(sqlr = 'SELECT ticker,shortratio,sector1_id, FROM stockfundamental', csv_out = '')
sql_query(sqlr = 'SELECT ticker,shortratio,sector1_id, FROM stockfundamental', dbengine = None, output = 'df', dburl='sqlite = 'sqlite:///aaserialize/store/finviz.db')



utilmy/zzarchive/util_web.py
-------------------------functions----------------------
web_getjson_fromurl(url)
web_getlink_fromurl(url)
web_getrawhtml(url1)
web_gettext_fromhtml(file1, htmltag = 'p')
web_gettext_fromurl(url, htmltag = 'p')
web_importio_todataframe(apiurl1, isurl = 1)
web_restapi_toresp(apiurl1)
web_send_email(FROM, recipient, subject, body, login1 = "mizenjapan@gmail.com", pss1 = "sophieelise237", server1 = "smtp.gmail.com", port1 = 465)
web_send_email_tls(FROM, recipient, subject, body, login1 = "mizenjapan@gmail.com", pss1 = "sophieelise237", server1 = "smtp.gmail.com", port1 = 465)
web_sendurl(url1)



utilmy/zzarchive/utilgeo.py
-------------------------functions----------------------
df_to_geojson(df, col_properties, lat = 'latitude', lon = 'longitude')



utilmy/zzarchive/zzarchive/zutil.py
-------------------------functions----------------------
_os_file_search_fast(fname, texts = None, mode = "regex/str")
a_cleanmemory()
a_help()
a_info_conda_jupyter()
a_run_cmd(cmd1)
a_run_ipython(cmd1)
a_start_log(id1 = "", folder = "aaserialize/log/")
aa_unicode_ascii_utf8_issue()
date_add_bday(from_date, add_days)
date_add_bdays(from_date, add_days)
date_allinfo()
date_diffinbday(intd2, intd1)
date_diffinday(intdate1, intdate2)
date_finddateid(date1, dateref)
date_gencalendar(start = "2010-01-01", end = "2010-01-15", country = "us")
date_generatedatetime(start = "20100101", nbday = 10, end = "")
date_holiday()
date_now(i = 0)
date_nowtime(type1 = "str", format1="%Y-%m-%d %H = "%Y-%m-%d %H:%M:%S:%f")
date_remove_bdays(from_date, add_days)
dateint_todatetime(datelist1)
datenumpy_todatetime(tt, islocaltime = True)
datestring_todatetime(datelist1, format1 = "%Y%m%d")
datestring_toint(datelist1)
datetime_toint(datelist1)
datetime_tonumpydate(t, islocaltime = True)
datetime_tostring(datelist1)
find(xstring, list_string)
find_fuzzy(xstring, list_string)
findhigher(x, vec)
findlower(x, vec)
findnone(vec)
finds(itemlist, vec)
findx(item, vec)
isanaconda()
isfloat(x)
isint(x)
load(folder = "/folder1/keyname", isabsolutpath = 0)
np_addcolumn(arr, nbcol)
np_addrow(arr, nbrow)
np_and1(x, y, x3 = None, x4 = None, x5 = None, x6 = None, x7 = None, x8 = None)
np_cleanmatrix(m)
np_dict_tolist(dd, withkey = 0)
np_dict_tostr_key(dd)
np_dict_tostr_val(dd)
np_dictordered_create()
np_enumerate2(vec_1d)
np_find(item, vec)
np_find_maxpos(values)
np_find_maxpos_2nd(numbers)
np_find_minpos(values)
np_findfirst(item, vec)
np_findlocalmax(v, trig)
np_findlocalmax2(v, trig)
np_findlocalmin(v, trig)
np_findlocalmin2(v, trig)
np_int_tostr(i)
np_interpolate_nan(y)
np_list_flatten(seq)
np_list_tofreqdict(l1, wweight = None)
np_list_unique(seq)
np_ma(vv, n)
np_max_kpos(arr, kth)
np_memory_array_adress(x)
np_mergelist(x0, x1)
np_min_kpos(arr, kth)
np_minimize(fun_obj, x0 = None, argext = (0, 0)
np_minimize_de(fun_obj, bounds, name1, maxiter = 10, popsize = 5, solver = None)
np_nan_helper(y)
np_numexpr_vec_calc()
np_pivotable_create(table, left, top, value)
np_pivottable_count(mylist)
np_remove_na_inf_2d(x)
np_remove_zeros(vv, axis1 = 1)
np_removelist(x0, xremove = None)
np_sort(arr, colid, asc = 1)
np_sortbycol(arr, colid, asc = True)
np_sortbycolumn(arr, colid, asc = True)
np_sortcol(arr, colid, asc = 1)
np_stack(v1, v2 = None, v3 = None, v4 = None, v5 = None)
np_torecarray(arr, colname)
np_transform2d_int_1d(m2d, onlyhalf = False)
np_uniquerows(a)
os_config_getfile(file1)
os_config_setfile(dict_params, outfile, mode1 = "w+")
os_csv_process(file1)
os_file_are_same_file_types(paths)
os_file_exist(file1)
os_file_extracttext(output_file, dir1, pattern1 = "*.html", htmltag = "p", deepness = 2)
os_file_get_extension(file_path)
os_file_get_path_from_stream(maybe_stream)
os_file_getname(path)
os_file_getpath(path)
os_file_gettext(file1)
os_file_isame(file1, file2)
os_file_listall(dir1, pattern = "*.*", dirlevel = 1, onlyfolder = 0)
os_file_mergeall(nfile, dir1, pattern1, deepness = 2)
os_file_norm_paths(paths, marker = "*")
os_file_normpath(path)
os_file_read(file1)
os_file_rename(some_dir, pattern = "*.*", pattern2 = "", dirlevel = 1)
os_file_replace(source_file_path, pattern, substring)
os_file_replacestring1(find_str, rep_str, file_path)
os_file_replacestring2(findstr, replacestr, some_dir, pattern = "*.*", dirlevel = 1)
os_file_search_content(srch_pattern = None, mode = "str", dir1 = "", file_pattern = "*.*", dirlevel = 1)
os_file_size(file1)
os_file_try_to_get_extension(path_or_strm)
os_folder_copy(src, dst, symlinks = False, pattern1 = "*.py", fun_file_toignore = None)
os_folder_create(directory)
os_folder_is_path(path_or_stream)
os_folder_robocopy(from_folder = "", to_folder = "", my_log="H = "H:/robocopy_log.txt")
os_gui_popup_show(txt)
os_path_append(p1, p2 = None, p3 = None, p4 = None)
os_path_change(path1)
os_path_current()
os_path_norm(pth)
os_platform()
os_print_tofile(vv, file1, mode1 = "a")
os_process_2()
os_process_run(cmd_list, capture_output = False)
os_split_dir_file(dirfile)
os_wait_cpu(priority = 300, cpu_min = 50)
os_zip_checkintegrity(filezip1)
os_zipextractall(filezip_or_dir = "folder1/*.zip", tofolderextract = "zdisk/test", isprint = 1)
os_zipfile(folderin, folderzipname, iscompress = True)
os_zipfolder(dir_tozip = "/zdisks3/output", zipname = "/zdisk3/output.zip", dir_prefix = True, iscompress=Trueimport shutil_ = iscompressdir_tozip = dir_tozip if dir_tozip[-1] != "/" else dir_tozip[ = Trueimport shutil_ = iscompressdir_tozip = dir_tozip if dir_tozip[-1] != "/" else dir_tozip[:-1]if dir_prefix:)
pd_applyfun_col(df, newcol, ff, use_colname = "all/[colname]")
pd_array_todataframe(array, colname = None, index1 = None, dotranspose = False)
pd_col_addfrom_dfmap(df, dfmap, colkey, colval, df_colused, df_colnew, exceptval = -1, inplace = Truedfmap, colkey = colkey, colval=colval)rowi) = colval)rowi):)
pd_create_colmapdict_nametoint(df)
pd_createdf(array1, col1 = None, idx1 = None)
pd_csv_randomread(filename, nsample = 10000, filemaxline = -1, dtype = None)
pd_dataframe_toarray(df)
pd_date_intersection(qlist)
pd_df_todict(df, colkey = "table", excludekey = ("", )
pd_df_todict2(df, colkey = "table", excludekey = ("", )
pd_dtypes(df, columns = ()
pd_dtypes_totype2(df, columns = ()
pd_dtypes_type1_totype2(df, fromtype = str, targetype = str)
pd_extract_col_idx_val(df)
pd_extract_col_uniquevalue_tocsv(df, colname = "", csvfile = "")
pd_find(df, regex_pattern = "*", col_restrict = None, isnumeric = False, doreturnposition = False)
pd_h5_addtable(df, tablename, dbfile="F = "F:\temp_pandas.h5")
pd_h5_cleanbeforesave(df)
pd_h5_dumpinfo(dbfile=r"E = r"E:\_data\stock\intraday_google.h5")
pd_h5_fromcsv_tohdfs(dircsv = "dir1/dir2/", filepattern = "*.csv", tofilehdfs = "file1.h5", tablename = "df", ), dtype0 = None, encoding = "utf-8", chunksize = 2000000, mode = "a", form = "table", complib = None, )
pd_h5_load(filenameh5="E = "E:/_data/_data_outlier.h5", table_id = "data", exportype = "pandas", rowstart = -1, rowend = -1, ), )
pd_h5_save(df, filenameh5="E = "E:/_data/_data_outlier.h5", key = "data")
pd_h5_tableinfo(filenameh5, table)
pd_info(df, doreturn = 1)
pd_info_memsize(df, memusage = 0)
pd_insertdatecol(df, col, format1="%Y-%m-%d %H = "%Y-%m-%d %H:%M:%S:%f")
pd_insertrow(df, rowval, index1 = None, isreset = 1)
pd_is_categorical(z)
pd_np_toh5file(numpyarr, fileout = "file.h5", table1 = "data")
pd_removecol(df1, name1)
pd_removerow(df, row_list_index = (23, 45)
pd_replacevalues(df, matrix)
pd_resetindex(df)
pd_row_findlast(df, colid = 0, emptyrowid = None)
pd_row_select(df, **conditions)
pd_split_col_idx_val(df)
pd_splitdf_inlist(df, colid, type1 = "dict")
pd_str_encoding_change(df, cols, fromenc = "iso-8859-1", toenc = "utf-8")
pd_str_isascii(x)
pd_str_unicode_tostr(df, targetype = str)
pd_toexcel(df, outfile = "file.xlsx", sheet_name = "sheet1", append = 1, returnfile = 1)
pd_toexcel_many(outfile = "file1.xlsx", df1 = None, df2 = None, df3 = None, df4 = None, df5 = None, df6 = Nonedf1, outfile, sheet_name="df1")if df2 is not None = "df1")if df2 is not None:)
print_object_tofile(vv, txt, file1="d = "d:/regression_output.py")
print_progressbar(iteration, total, prefix = "", suffix = "", decimals = 1, bar_length = 100)
py_autoreload()
py_importfromfile(modulename, dir1)
py_load_obj(folder = "/folder1/keyname", isabsolutpath = 0, encoding1 = "utf-8")
py_memorysize(o, ids, hint = " deep_getsizeof(df_pd, set()
py_save_obj(obj1, keyname = "", otherfolder = 0)
py_save_obj_dill(obj1, keyname = "", otherfolder = 0)
save(obj, folder = "/folder1/keyname", isabsolutpath = 0)
save_test(folder = "/folder1/keyname", isabsolutpath = 0)
session_load_function(name = "test_20160815")
session_save_function(name = "test")
str_empty_string_array(x, y = 1)
str_empty_string_array_numpy(nx, ny = 1)
str_is_az09char(x)
str_is_azchar(x)
str_isfloat(value)
str_make_unicode(input_str, errors = "replace")
str_match_fuzzy(xstring, list_string)
str_parse_stringcalendar(cal)
str_reindent(s, num_spaces)
str_split2(delimiters, string, maxsplit = 0)
str_split_pattern(sep2, ll, maxsplit = 0)
str_to_unicode(x, encoding = "utf-8")
str_to_utf8(x)
z_key_splitinto_dir_name(keyname)



utilmy/zzarchive/zzarchive/zutil_features.py
-------------------------functions----------------------
col_extractname(col_onehot)
col_remove(cols, colsremove, mode = "exact")
feature_correlation_cat(df, colused)
feature_importance_perm(clf, Xtrain, ytrain, cols, n_repeats = 8, scoring = 'neg_root_mean_squared_error', show_graph = 1)
feature_selection_multicolinear(df, threshold = 1.0)
fetch_dataset(url_dataset, path_target = None, file_target = None)
fetch_spark_koalas(path_data_x, path_data_y = '', colid = "jobId", n_sample = -1)
load(name, path)
load_dataset(path_data_x, path_data_y = '', colid = "jobId", n_sample = -1)
load_features(name, path)
load_function_uri(uri_name="myfolder/myfile.py = "myfolder/myfile.py::myFunction")
log(*s, n = 0, m = 1, **kw)
log2(*s, **kw)
log3(*s, **kw)
metrics_eval(metric_list = ["mean_squared_error"], ytrue = None, ypred = None, ypred_proba = None, return_dict = False)
np_conv_to_one_col(np_array, sep_char = "_")
os_get_function_name()
os_getcwd()
pa_read_file(path =   'folder_parquet/', cols = None, n_rows = 1000, file_start = 0, file_end = 100000, verbose = 1, )
pa_write_file(df, path =   'folder_parquet/', cols = None, n_rows = 1000, partition_cols = None, overwrite = True, verbose = 1, filesystem  =  'hdfs')
params_check(pars, check_list, name = "")
pd_col_fillna(dfref, colname = None, method = "frequent", value = None, colgroupby = None, return_val = "dataframe,param", )
pd_col_filter(df, filter_val = None, iscol = 1)
pd_col_merge_onehot(df, colname)
pd_col_to_num(df, colname = None, default = np.nan)
pd_col_to_onehot(dfref, colname = None, colonehot = None, return_val = "dataframe,column")
pd_colcat_mapping(df, colname)
pd_colcat_mergecol(df, col_list, x0, colid = "easy_id")
pd_colcat_toint(dfref, colname, colcat_map = None, suffix = None)
pd_colcat_tonum(df, colcat = "all", drop_single_label = False, drop_fact_dict = True)
pd_colnum_normalize(df0, colname, pars, suffix = "_norm", return_val = 'dataframe,param')
pd_colnum_tocat(df, colname = None, colexclude = None, colbinmap = None, bins = 5, suffix = "_bin", method = "uniform", na_value = -1, return_val = "dataframe,param", params={"KMeans_n_clusters" = {"KMeans_n_clusters": 8, "KMeans_init": 'k-means++', "KMeans_n_init": 10,"KMeans_max_iter": 300, "KMeans_tol": 0.0001, "KMeans_precompute_distances": 'auto',"KMeans_verbose": 0, "KMeans_random_state": None,"KMeans_copy_x": True, "KMeans_n_jobs": None, "KMeans_algorithm": 'auto'})
pd_colnum_tocat_stat(df, feature, target_col, bins, cuts = 0)
pd_feature_generate_cross(df, cols, cols_cross_input = None, pct_threshold = 0.2, m_combination = 2)
pd_pipeline_apply(df, pipeline)
pd_read_file(path_glob = "*.pkl", ignore_index = True, cols = None, verbose = False, nrows = -1, concat_sort = True, n_pool = 1, drop_duplicates = None, col_filter = None, col_filter_val = None, **kw)
pd_stat_correl_pair(df, coltarget = None, colname = None)
pd_stat_dataset_shift(dftrain, dftest, colused, nsample = 10000, buckets = 5, axis = 0)
pd_stat_datashift_psi(expected, actual, buckettype = 'bins', buckets = 10, axis = 0)
pd_stat_distribution_colnum(df, nrows = 2000, verbose = False)
pd_stat_histogram(df, bins = 50, coltarget = "diff")
pd_stat_pandas_profile(df, savefile = "report.html", title = "Pandas Profile")
pd_stat_shift_changes(df, target_col, features_list = 0, bins = 10, df_test = 0)
pd_stat_shift_trend_changes(df, feature, target_col, threshold = 0.03)
pd_stat_shift_trend_correlation(df, df_test, colname, target_col)
save(df, name, path = None)
save_features(df, name, path = None)
save_list(path, name_list, glob)
test_get_classification_data(name = None)

-------------------------methods----------------------
dict2.__init__(self, d)
