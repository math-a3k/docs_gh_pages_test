

.\utilmy\__init__.py


.\utilmy\adatasets.py
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
test_dataset_classification_fake(nrows = 500)
test_dataset_classification_petfinder(nrows = 1000)
test_dataset_classifier_covtype(nrows = 500)
test_dataset_regression_fake(nrows = 500, n_features = 17)



.\utilmy\configs\__init__.py


.\utilmy\configs\test.py
-------------------------functions----------------------
create_fixtures_data(tmp_path)
test_validate_yaml_failed_silent(tmp_path)
test_validate_yaml_types(tmp_path)
test_validate_yaml_types_failed(tmp_path)



.\utilmy\configs\util_config.py
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



.\utilmy\data.py
-------------------------functions----------------------
help()
log(*s)



.\utilmy\dates.py
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
pd_date_split(df, coldate  =   'time_key', prefix_col  = "", verbose = False)
random_dates(start, end, size)
random_genders(size, p = None)
test()



.\utilmy\debug.py
-------------------------functions----------------------
help()
log(*s)
log10(*s, nmax = 60)
log5(*s)
log_trace(msg = "", dump_path = "", globs = None)
print_everywhere()
profiler_start()
profiler_stop()



.\utilmy\decorators.py
-------------------------functions----------------------
dummy_func()
profiled_sum()
profiler_context()
profiler_decorator(func)
profiler_decorator_base(fnc)
profiler_decorator_base_test()
test0()
thread_decorator(func)
thread_decorator_test()
timeout_decorator(seconds = 10, error_message = os.strerror(errno.ETIME)
timeout_decorator_test()
timer_decorator(func)



.\utilmy\deeplearning\__init__.py


.\utilmy\deeplearning\keras\nsl_graph_loss.py
-------------------------functions----------------------
create_fake_neighbor(x, max_neighbors)
map_func(x_batch, y_batch, neighbors, neighbor_weights)
test_step(x, y, model, loss_fn)
test_step(x, y, model, loss_fn)
train_step(x, y, model, loss_fn, optimizer)
train_step(x, y, model, loss_fn, optimizer)



.\utilmy\deeplearning\keras\template_train.py
-------------------------functions----------------------
clean1(ll)
config_save(cc, path)
image_check(name, img, renorm = False)
label_get_data()
log(*s)
log2(*s)
log3(*s)
metric_accuracy(y_val, y_pred_head, class_dict)
model_reload(model_reload_name, cc, )
np_remove_duplicates(seq)
os_path_copy(in_dir, path, ext = "*.py")
param_set()
params_set2()
pd_category_filter(df, category_map)
pd_get_dummies(df, cols_cat, cat_dict:dict, only_onehot = True)
save_best(model, model_dir2, curr_loss, best_loss, counter, epoch, dd)
save_model_state(model, model_dir2)
train_step(x, model, y_label_list = None)
train_stop(counter, patience)
valid_image_check(img_list, path = "", tag = "", y_labels = "", n_sample = 3, renorm = True)
valid_image_original(img_list, path, tag, y_labels, n_sample = None)
validation_step(x, model, y_label_list = None)

-------------------------methods----------------------
LearningRateDecay.plot(self, epochs, title = "Learning Rate Schedule", path = None)


.\utilmy\deeplearning\keras\util_dataloader.py
-------------------------functions----------------------
data_add_onehot(dfref, img_dir, labels_col)
data_get_sample(batch_size, x_train, labels_val)
data_to_y_onehot_list(df, dfref, labels_col)
log(*s)

-------------------------methods----------------------
CustomDataGenerator.__getitem__(self, idx)
CustomDataGenerator.__init__(self, x, y, batch_size = 32, augmentations = None)
CustomDataGenerator.__len__(self)
CustomDataGenerator_img.__getitem__(self, idx)
CustomDataGenerator_img.__init__(self, img_dir, label_path, class_list, split = 'train', batch_size = 8, transforms = None)
CustomDataGenerator_img.__len__(self)
CustomDataGenerator_img.on_epoch_end(self)
SprinklesTransform.__init__(self, num_holes = 30, side_length = 5, always_apply = False, p = 1.0)
SprinklesTransform.apply(self, image, **params)


.\utilmy\deeplearning\keras\util_layers.py
-------------------------functions----------------------
help()
log(*s)
log2(*s)
make_classifier(class_dict)
make_decoder()
make_encoder(n_outputs = 1)

-------------------------methods----------------------
DFC_VAE.__init__(self, latent_dim, class_dict)
DFC_VAE.call(self, x, training = True, mask = None, y_label_list =  None)
DFC_VAE.decode(self, z, apply_sigmoid = False)
DFC_VAE.encode(self, x)
DFC_VAE.reparameterize(self, z_mean, z_logsigma)


.\utilmy\deeplearning\keras\util_loss.py
-------------------------functions----------------------
clf_loss_macro_soft_f1(y, y_hat)
learning_rate_schedule(mode = "step", epoch = 1, cc = None)
log(*s)
loss_schedule(mode = "step", epoch = 1)
metric_accuracy(y_test, y_pred, dd)
perceptual_loss_function(x, x_recon, z_mean, z_logsigma, kl_weight = 0.00005, y_label_heads = None, y_pred_heads = None, clf_loss_fn = None)

-------------------------methods----------------------
LearningRateDecay.plot(self, epochs, title = "Learning Rate Schedule", path = None)
StepDecay.__call__(self, epoch)
StepDecay.__init__(self, init_lr = 0.01, factor = 0.25, drop_every = 5)


.\utilmy\deeplearning\keras\util_train.py
-------------------------functions----------------------
clean1(ll)
config_save(cc, path)
image_check(name, img, renorm = False)
label_get_data()
log(*s)
log2(*s)
log3(*s)
metric_accuracy(y_val, y_pred_head, class_dict)
model_reload(model_reload_name, cc, )
np_remove_duplicates(seq)
os_path_copy(in_dir, path, ext = "*.py")
pd_category_filter(df, category_map)
pd_get_dummies(df, cols_cat, cat_dict:dict, only_onehot = True)
save_best(model, model_dir2, curr_loss, best_loss, counter, epoch, dd)
save_model_state(model, model_dir2)
tf_compute_set(cc:dict)
train_step(x, model, y_label_list = None)
train_stop(counter, patience)
valid_image_check(img_list, path = "", tag = "", y_labels = "", n_sample = 3, renorm = True)
valid_image_original(img_list, path, tag, y_labels, n_sample = None)
validation_step(x, model, y_label_list = None)

-------------------------methods----------------------
LearningRateDecay.plot(self, epochs, title = "Learning Rate Schedule", path = None)


.\utilmy\deeplearning\keras\vq_vae2.py
-------------------------functions----------------------
encoder_Base(latent_dim)
get_vqvae_layer_hierarchical(latent_dim = 16, num_embeddings = 64)
plot_original_reconstructed(orig, rec)

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


.\utilmy\deeplearning\torch\util_train.py


.\utilmy\deeplearning\util_dl.py
-------------------------functions----------------------
down_page(query, out_dir = "query1", genre_en = '', id0 = "", cat = "", npage = 1)
gpu_free()
gpu_usage()
help()
log(*s)
log2(*s)
tensorboard_log(pars_dict:dict = None, writer = None, verbose = True)
test()
tf_check()



.\utilmy\deeplearning\util_image.py
-------------------------functions----------------------
help()
image_center_crop(img, dim)
image_center_crop(img, dim)
image_check_npz(path_npz, keys = ['train'], path = "", tag = "", n_sample = 3, renorm = True)
image_create_cache()
image_face_blank(in_dir = "", level  =  "/*", out_dir = f"", npool = 30)
image_face_blank(in_dir = "", level  =  "/*", out_dir = f"", npool = 30)
image_merge(image_list, n_dim, padding_size, max_height, total_width)
image_read(filepath_or_buffer: Union[str, io.BytesIO])
image_read(filepath_or_buffer: Union[str, io.BytesIO])
image_remove_bg(in_dir = "", out_dir = "", level = 1)
image_remove_extra_padding(img, inverse = False, removedot = True)
image_resize(img, size = (256, 256)
image_resize_pad(img, size = (256, 256)
image_resize_pad(img, size = (256, 256)
image_save(out_dir, name = "cache1")
image_save_tocache(out_dir, name = "cache1")
image_text_blank(in_dir, out_dir, level = "/*")
log(*s)
log2(*s)
os_path_check(path, n = 5)
padding_generate(paddings_number: int  =  1, min_padding: int  =  1, max_padding: int  =  1)
prepro_image(image_path:str, xdim = 1, ydim = 1)
prepro_images(image_paths, nmax = 10000000)
prepro_images_multi(image_paths, npool = 30, prepro_image = None)
test()



.\utilmy\deeplearning\util_topk.py
-------------------------functions----------------------
topk()
topk_export()
topk_nearest_vector(x0, vector_list, topk = 3)
topk_predict()



.\utilmy\deeplearning\zz_prepro.py
-------------------------functions----------------------
create_train_npz()
create_train_parquet()
data_add_onehot(dfref, img_dir, labels_col)
folder_size()
gzip()
image_check()
image_resize(out_dir = "")
log(*s)
model_deletes(dry = 0)
predict(name = None)
prepro_image0(image_path)
prepro_images(image_paths, nmax = 10000000)
prepro_images2(image_paths)
prepro_images_multi(image_paths, npool = 30, prepro_image = None)
run_multiprocess(myfun, list_args, npool = 10, **kwargs)
test()
unzip(in_dir, out_dir)



.\utilmy\deeplearning\zz_utils_dl2.py
-------------------------functions----------------------
check_tf()
clean1(ll)
clf_loss_macro_soft_f1(y, y_hat)
config_save(cc, path)
create_train_npz()
create_train_parquet()
data_add_onehot(dfref, img_dir, labels_col)
data_get_sample(batch_size, x_train, labels_val)
data_to_y_onehot_list(df, dfref, labels_col)
down_ichiba()
down_page(query, out_dir = "query1", genre_en = '', id0 = "", cat = "", npage = 1)
folder_size()
gpu_free()
gpu_usage()
gzip()
image_center_crop(img, dim)
image_check(path_npz, keys = ['train'], path = "", tag = "", n_sample = 3, renorm = True)
image_check_npz(path_npz, keys = ['train'], path = "", tag = "", n_sample = 3, renorm = True)
image_create_cache()
image_face_blank(in_dir = "", level  =  "/*", out_dir = f"", npool = 30)
image_load(pathi, mode = 'cache')
image_merge(image_list, n_dim, padding_size, max_height, total_width)
image_read(filepath_or_buffer: Union[str, io.BytesIO])
image_remove_bg(in_dir = "", out_dir = "", level = 1)
image_remove_extra_padding(img, inverse = False, removedot = True)
image_resize(img, size = (256, 256)
image_resize2(image, width = None, height = None, inter = cv2.INTER_AREA)
image_resize_pad(img, size = (256, 256)
image_save(out_dir)
image_text_blank(in_dir, out_dir, level = "/*")
label_get_data()
learning_rate_schedule(mode = "step", epoch = 1, cc = None)
log(*s)
log2(*s)
log3(*s)
loss_schedule(mode = "step", epoch = 1)
make_classifier(class_dict)
make_decoder()
make_encoder(n_outputs = 1)
metric_accuracy2(y_test, y_pred, dd)
metric_accuracy_test(y_test, y_pred, dd)
metric_accuracy_val(y_val, y_pred_head, class_dict)
model_deletes(dry = 0)
model_reload(model_reload_name, cc, )
np_remove_duplicates(seq)
os_path_check(path, n = 5)
os_path_copy(in_dir, path, ext = "*.py")
padding_generate(paddings_number: int  =  1, min_padding: int  =  1, max_padding: int  =  1)
pd_category_filter(df, category_map)
pd_get_dummies(df, cols_cat, cat_dict:dict, only_onehot = True)
perceptual_loss_function(x, x_recon, z_mean, z_logsigma, kl_weight = 0.00005, y_label_heads = None, y_pred_heads = None, clf_loss_fn = None)
predict(name = None)
prepro_image(image_path)
prepro_images(image_paths, nmax = 10000000)
prepro_images_multi(image_paths, npool = 30, prepro_image = None)
run_multiprocess(myfun, list_args, npool = 10, **kwargs)
save_best(model, model_dir2, curr_loss, best_loss, counter, epoch, dd)
save_model_state(model, model_dir2)
test()
topk()
topk_export()
topk_nearest_vector(x0, vector_list, topk = 3)
topk_predict()
train_step(x, model, y_label_list = None)
train_stop(counter, patience)
unzip(in_dir, out_dir)
valid_image_check(img_list, path = "", tag = "", y_labels = "", n_sample = 3, renorm = True)
valid_image_original(img_list, path, tag, y_labels, n_sample = None)
validation_step(x, model, y_label_list = None)

-------------------------methods----------------------
CustomDataGenerator_img.__getitem__(self, idx)
CustomDataGenerator_img.__init__(self, img_dir, label_path, class_list, split = 'train', batch_size = 8, transforms = None)
CustomDataGenerator_img.__len__(self)
CustomDataGenerator_img.on_epoch_end(self)
DFC_VAE.__init__(self, latent_dim, class_dict)
DFC_VAE.call(self, x, training = True, mask = None, y_label_list =  None)
DFC_VAE.decode(self, z, apply_sigmoid = False)
DFC_VAE.encode(self, x)
DFC_VAE.reparameterize(self, z_mean, z_logsigma)
LearningRateDecay.plot(self, epochs, title = "Learning Rate Schedule", path = None)
RealCustomDataGenerator.__getitem__(self, idx)
RealCustomDataGenerator.__init__(self, image_dir, label_path, class_dict, split = 'train', batch_size = 8, transforms = None, shuffle = False, img_suffix = ".png")
RealCustomDataGenerator.__len__(self)
RealCustomDataGenerator._load_data(self, label_path)
RealCustomDataGenerator.on_epoch_end(self)
SprinklesTransform.__init__(self, num_holes = 100, side_length = 10, always_apply = False, p = 1.0)
SprinklesTransform.apply(self, image, **params)
StepDecay.__call__(self, epoch)
StepDecay.__init__(self, init_lr = 0.01, factor = 0.25, drop_every = 5)


.\utilmy\distributed.py
-------------------------functions----------------------
date_now(fmt = "%Y-%m-%d %H =  "%Y-%m-%d %H:%M:%S %Z%z")
help()
load(to_file = "")
load_serialize(name)
log(*s)
log2(*s)
log_mem(*s)
os_lock_acquireLock(plock:str = "tmp/plock.lock")
os_lock_execute(fun_run, fun_args = None, ntry = 5, plock = "tmp/plock.lock")
os_lock_releaseLock(locked_file_descriptor)
save(dd, to_file = "", verbose = False)
save_serialize(name, value)
test1_functions()
test2_funtions_thread()
test3_index()
test_all()
time_sleep_random(nmax = 5)

-------------------------methods----------------------
IndexLock.__init__(self, findex, plock)
IndexLock.get(self)
IndexLock.put(self, val = "", ntry = 100, plock = "tmp/plock.lock")


.\utilmy\docs\__init__.py


.\utilmy\docs\cli.py
-------------------------functions----------------------
os_remove(filepath)
run_cli()



.\utilmy\docs\code_parser.py
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



.\utilmy\docs\generate_doc.py
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



.\utilmy\docs\test.py
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



.\utilmy\excel\xlvba.py
-------------------------functions----------------------
invokenumpy()
invokesklearn()
load_csv(csvfile)
loaddf()



.\utilmy\graph.py


.\utilmy\iio.py


.\utilmy\images\util_exceptions.py
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



.\utilmy\images\util_image.py
-------------------------functions----------------------
image_merge(image_list, n_dim, padding_size, max_height, total_width)
image_read(filepath_or_buffer: Union[str, io.BytesIO])
image_remove_extra_padding(img, inverse = False, removedot = True)
image_resize(image, width = None, height = None, inter = cv2.INTER_AREA)
padding_generate(paddings_number: int  =  1, min_padding: int  =  1, max_padding: int  =  1)



.\utilmy\images\util_image1.py
-------------------------functions----------------------
deps()
log(*s)
maintain_aspect_ratio_resize(image, width = None, height = None, inter = cv2.INTER_AREA)
read_image(filepath_or_buffer: typing.Union[str, io.BytesIO])
visualize_in_row(**images)



.\utilmy\images\utils.py
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



.\utilmy\keyvalue.py
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


.\utilmy\logs\__init__.py


.\utilmy\logs\test_log.py
-------------------------functions----------------------
test1()
test2()
test_launch_server()
test_server()

-------------------------methods----------------------
LoggingStreamHandler.handle(self)


.\utilmy\logs\util_log.py
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



.\utilmy\nnumpy.py


.\utilmy\oos.py
-------------------------functions----------------------
help()
is_float(x)
is_int(x)
log(*s)
log10(*s, nmax = 60)
log2(*s)
log5(*s)
log_trace(msg = "", dump_path = "", globs = None)
np_add_remove(set_, to_remove, to_add)
np_list_intersection(l1, l2)
os_clean_memory(varlist, globx)
os_copy(dirin = None, dirout = None, nlevel = 10, nfile = 100000, cmd_fallback = "")
os_copy_safe(dirin = None, dirout = None, nlevel = 10, nfile = 100000, cmd_fallback = "")
os_cpu()
os_file_check(fp)
os_file_replacestring(findstr, replacestr, some_dir, pattern = "*.*", dirlevel = 1)
os_get_function_name()
os_getcwd()
os_import(mod_name = "myfile.config.model", globs = None, verbose = True)
os_makedirs(dir_or_file)
os_memory()
os_path_size(path  =  '.')
os_path_split(fpath:str = "")
os_platform_ip()
os_platform_os()
os_removedirs(path)
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
to_datetime(x)
to_dict(**kw)
to_float(x)
to_int(x)
to_timeunix(datex = "2018-01-16")
z_os_search_fast(fname, texts = None, mode = "regex/str")

-------------------------methods----------------------
dict_to_namespace.__init__(self, d)


.\utilmy\parallel.py
-------------------------functions----------------------
help()
log(*s)
log2(*s)
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



.\utilmy\ppandas.py
-------------------------functions----------------------
is_float(x)
is_int(x)
np_add_remove(set_, to_remove, to_add)
np_list_intersection(l1, l2)
pd_add_noise(df, level = 0.05, cols_exclude:list = [])
pd_cols_unique_count(df, cols_exclude:list = [], nsample = -1)
pd_del(df, cols:list)
pd_dtype_count_unique(df, col_continuous = [])
pd_dtype_getcontinuous(df, cols_exclude:list = [], nsample = -1)
pd_dtype_to_category(df, col_exclude, treshold = 0.5)
pd_show(df, nrows = 100, reader = 'notepad.exe', **kw)
to_datetime(x)
to_dict(**kw)
to_float(x)
to_int(x)
to_timeunix(datex = "2018-01-16")

-------------------------methods----------------------
dict_to_namespace.__init__(self, d)


.\utilmy\prepro\prepro.py
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



.\utilmy\prepro\prepro_rec.py
-------------------------functions----------------------
_preprocess_criteo(df, **kw)
_preprocess_movielens(df, **kw)



.\utilmy\prepro\prepro_text.py
-------------------------functions----------------------
log(*s, n = 0, m = 1)
log_pd(df, *s, n = 0, m = 1)
logs(*s)
nlp_get_stopwords()
pd_coltext(df, col, stopwords =  None, pars = None)
pd_coltext_clean(df, col, stopwords =  None, pars = None)
pd_coltext_universal_google(df, col, pars = {})
pd_coltext_wordfreq(df, col, stopwords, ntoken = 100)



.\utilmy\prepro\prepro_tseries.py
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



.\utilmy\prepro\run_feature_profile.py
-------------------------functions----------------------
log(*s, n = 0, m = 0)
run_profile(path_data = None, path_output = "data/out/ztmp/", n_sample = 5000)



.\utilmy\spark\main.py
-------------------------functions----------------------
main()
spark_init(config:dict)



.\utilmy\spark\script\hadoopVersion.py


.\utilmy\spark\script\pysparkTest.py
-------------------------functions----------------------
inside(p)



.\utilmy\spark\setup.py


.\utilmy\spark\src\__init__.py


.\utilmy\spark\src\functions\GetFamiliesFromUserAgent.py
-------------------------functions----------------------
getall_families_from_useragent(ua_string)



.\utilmy\spark\src\tables\table_predict_session_length.py
-------------------------functions----------------------
preprocess(spark, conf, check = True)
run(spark:SparkSession, config_path: str = 'config.yaml', mode:str = 'train,pred')



.\utilmy\spark\src\tables\table_predict_url_unique.py
-------------------------functions----------------------
preprocess(spark, conf, check = True)
run(spark:SparkSession, config_path: str = 'config.yaml', mode:str = 'train,pred')



.\utilmy\spark\src\tables\table_predict_volume.py
-------------------------functions----------------------
model_predict(df:pd.DataFrame, conf_model:dict, verbose:bool = True)
model_train(df:object, conf_model:dict, verbose:bool = True)
preprocess(spark, conf, check = True)
run(spark:SparkSession, config_path: str = 'config.yaml')



.\utilmy\spark\src\tables\table_user_log.py
-------------------------functions----------------------
create_userid(userlogDF:pyspark.sql.DataFrame)
run(spark:SparkSession, config_name:str)



.\utilmy\spark\src\tables\table_user_session_log.py
-------------------------functions----------------------
run(spark:SparkSession, config_name = 'config.yaml')



.\utilmy\spark\src\tables\table_user_session_stats.py
-------------------------functions----------------------
run(spark:SparkSession, config_name: str = 'config.yaml')



.\utilmy\spark\src\util_models.py
-------------------------functions----------------------
Predict(spark, df_m:pyspark.sql.DataFrame, features:list, regressor:str, path:str = None, conf_model:dict = None)
TimeSeriesSplit(df_m:pyspark.sql.DataFrame, splitRatio:float, sparksession:object)
Train(spark, df_m:pyspark.sql.DataFrame, features:list, regressor:str, path:str = None, conf_model:dict = None)
os_makedirs(path:str)



.\utilmy\spark\src\utils.py
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


.\utilmy\spark\tests\__init__.py


.\utilmy\spark\tests\conftest.py
-------------------------functions----------------------
config()
spark_session(config: dict)



.\utilmy\spark\tests\test_common.py
-------------------------functions----------------------
assert_equal_spark_df(expected_sorted_df: DataFrame, actual_sorted_df: DataFrame, df_name: str)
assert_equal_spark_df_schema(expected_schema: [tuple], actual_schema: [tuple], df_name: str)
assert_equal_spark_df_sorted(expected_sorted_df: DataFrame, actual_sorted_df: DataFrame, df_name: str)



.\utilmy\spark\tests\test_functions.py
-------------------------functions----------------------
test_getall_families_from_useragent(spark_session: SparkSession)



.\utilmy\spark\tests\test_table_user_log.py
-------------------------functions----------------------
test_table_user_log_run(spark_session: SparkSession, config: dict)



.\utilmy\spark\tests\test_table_user_session_log.py
-------------------------functions----------------------
test_table_user_session_log(spark_session: SparkSession)
test_table_user_session_log_run(spark_session: SparkSession)
test_table_usersession_log_stats(spark_session: SparkSession, config: dict)



.\utilmy\spark\tests\test_table_user_session_stats.py
-------------------------functions----------------------
test_table_user_session_stats(spark_session: SparkSession)
test_table_user_session_stats_ip(spark_session: SparkSession, config: dict)
test_table_user_session_stats_run(spark_session: SparkSession)



.\utilmy\spark\tests\test_table_volume_predict.py
-------------------------functions----------------------
test_preprocess(spark_session: SparkSession, config: dict)



.\utilmy\spark\tests\test_utils.py
-------------------------functions----------------------
test_spark_check(spark_session: SparkSession, config: dict)



.\utilmy\tabular.py
-------------------------functions----------------------
estimator_boostrap_bayes(err, alpha = 0.05, )
estimator_bootstrap(err, custom_stat = None, alpha = 0.05, n_iter = 10000)
estimator_std_normal(err, alpha = 0.05, )
log(*s)
np_col_extractname(col_onehot)
np_conv_to_one_col(np_array, sep_char = "_")
np_list_remove(cols, colsremove, mode = "exact")
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
test_anova(df, col1, col2)
test_heteroscedacity(y, y_pred, pred_value_only = 1)
test_hypothesis(df_obs, df_ref, method = '', **kw)
test_mutualinfo(error, Xtest, colname = None, bins = 5)
test_normality(df, column, test_type)
test_normality2(df, column, test_type)
test_plot_qqplot(df, col_name)
y_adjuster_log(y_true, y_pred_log, error_func, **kwargs)



.\utilmy\templates\__init__.py


.\utilmy\templates\cli.py
-------------------------functions----------------------
run_cli()
template_copy(name, out_dir)
template_show()



.\utilmy\templates\templist\pypi_package\mygenerator\__init__.py


.\utilmy\templates\templist\pypi_package\mygenerator\dataset.py
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


.\utilmy\templates\templist\pypi_package\mygenerator\pipeline.py
-------------------------functions----------------------
run_generate_numbers_sequence(sequence: str, min_spacing: int  =  1, max_spacing: int  =  10, image_width: int  =  280, ### image_widthoutput_path: str  =  "./", config_file: str  =  "config/config.yaml", )
run_generate_phone_numbers(num_images: int  =  10, min_spacing: int  =  1, max_spacing: int  =  10, image_width: int  =  280, output_path: str  =  "./", config_file: str  =  "config/config.yaml", )



.\utilmy\templates\templist\pypi_package\mygenerator\transform.py
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


.\utilmy\templates\templist\pypi_package\mygenerator\util_exceptions.py
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



.\utilmy\templates\templist\pypi_package\mygenerator\util_image.py
-------------------------functions----------------------
image_merge(image_list, n_dim, padding_size, max_height, total_width)
image_read(filepath_or_buffer: Union[str, io.BytesIO])
image_remove_extra_padding(img, inverse = False, removedot = True)
image_resize(image, width = None, height = None, inter = cv2.INTER_AREA)
padding_generate(paddings_number: int  =  1, min_padding: int  =  1, max_padding: int  =  1)



.\utilmy\templates\templist\pypi_package\mygenerator\utils.py
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



.\utilmy\templates\templist\pypi_package\mygenerator\validate.py
-------------------------functions----------------------
image_padding_get(img, threshold = 0, inverse = True)
image_padding_load(img_path, threshold = 15)
image_padding_validate(final_image, min_padding, max_padding)
run_image_padding_validate(min_spacing: int  =  1, max_spacing: int  =  1, image_width: int  =  5, input_path: str  =  "", inverse_image: bool  =  True, config_file: str  =  "default", **kwargs, )



.\utilmy\templates\templist\pypi_package\run_pipy.py
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


.\utilmy\templates\templist\pypi_package\setup.py
-------------------------functions----------------------
get_current_githash()



.\utilmy\templates\templist\pypi_package\tests\__init__.py


.\utilmy\templates\templist\pypi_package\tests\conftest.py


.\utilmy\templates\templist\pypi_package\tests\test_common.py


.\utilmy\templates\templist\pypi_package\tests\test_dataset.py
-------------------------functions----------------------
test_image_dataset_get_image_only()
test_image_dataset_get_label_list()
test_image_dataset_get_sampe()
test_image_dataset_len()
test_nlp_dataset_len()



.\utilmy\templates\templist\pypi_package\tests\test_import.py
-------------------------functions----------------------
test_import()



.\utilmy\templates\templist\pypi_package\tests\test_pipeline.py
-------------------------functions----------------------
test_generate_phone_numbers(tmp_path)



.\utilmy\templates\templist\pypi_package\tests\test_transform.py
-------------------------functions----------------------
create_font_files(font_dir)
test_chars_to_images_transform()
test_combine_images_horizontally_transform()
test_scale_image_transform()
test_text_to_image_transform(tmp_path)



.\utilmy\templates\templist\pypi_package\tests\test_util_image.py
-------------------------functions----------------------
create_blank_image(width, height, rgb_color = (0, 0, 0)
test_image_merge()
test_image_read(tmp_path)
test_image_remove_extra_padding()
test_image_resize()



.\utilmy\templates\templist\pypi_package\tests\test_validate.py
-------------------------functions----------------------
test_image_padding_get()



.\utilmy\text.py
-------------------------functions----------------------
help()
help_get_codesource(func)
log(*s)
pd_text_getcluster(df:pd.DataFrame, col:str = 'col', threshold = 0.5, num_perm:int = 5, npool = 1, chunk  =  100000)
pd_text_hash_create_lsh(df, col, sep = " ", threshold = 0.7, num_perm = 10, npool = 1, chunk  =  20000)
pd_text_similarity(df: pd.DataFrame, cols = [], algo = '')
test()
test()
test_lsh()



.\utilmy\tseries\util_tseries.py


.\utilmy\utilmy.py
-------------------------functions----------------------
git_current_hash(mode = 'full')
git_repo_root()
help_create(modulename = 'utilmy.nnumpy', prefixs = None)
import_function(fun_name = None, module_name = None)
load(to_file = "")
pd_generate_data(ncols = 7, nrows = 100)
pd_getdata(verbose = True)
pd_random(ncols = 7, nrows = 100)
save(dd, to_file = "", verbose = False)

-------------------------methods----------------------
Session.__init__(self, dir_session = "ztmp/session/", )
Session.load(self, name, glob:dict = None, tag = "")
Session.load_session(self, folder, globs = None)
Session.save(self, name, glob = None, tag = "")
Session.save_session(self, folder, globs, tag = "")
Session.show(self)


.\utilmy\utils.py
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
to_file(s, filep)



.\utilmy\viz\__init__.py


.\utilmy\viz\embedding.py
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


.\utilmy\viz\util_map.py


.\utilmy\viz\vizhtml.py
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


.\utilmy\viz\zarchive\__init__.py


.\utilmy\viz\zarchive\toptoolbar.py
-------------------------methods----------------------
TopToolbar.__init__(self)


.\utilmy\zarchive\_HELP.py
-------------------------functions----------------------
fun_cython(a)
fun_python(a)
os_VS_build(self, lib_to_build)
os_VS_start(self, version)
os_compileVSsolution(dir1, flags1 = "", type1 = "devenv", compilerdir = "")
set_rc_version(rcfile, target_version)



.\utilmy\zarchive\__init__.py


.\utilmy\zarchive\alldata.py


.\utilmy\zarchive\allmodule.py
-------------------------functions----------------------
pprint(table1, tablefmt = "simple")
pprint2(x)
str_convert_beforeprint(x)
str_to_unicode(x, encoding = 'utf-8')
str_to_utf8(x)



.\utilmy\zarchive\allmodule_fin.py


.\utilmy\zarchive\coke_functions.py
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



.\utilmy\zarchive\datanalysis.py
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


.\utilmy\zarchive\excel.py
-------------------------functions----------------------
add_one(data)
double_sum(x, y)
get_workbook_name()
matrix_mult(x, y)
npdot()



.\utilmy\zarchive\fast.py
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



.\utilmy\zarchive\fast_parallel.py
-------------------------functions----------------------
task_find_best(tasks, n_top = 5)
task_parallel_job_01(name, param, datadict)
task_progress(tasks)
task_summary(tasks)



.\utilmy\zarchive\filelock.py
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


.\utilmy\zarchive\function_custom.py
-------------------------functions----------------------
fun_obj(vv, ext)
getweight(ww, size = (9, 3)
mapping_calc_risk_elvis_v03(ss, tr, t, riskout)
mapping_calc_risk_v00(self, ss, tr, t, risk0)
mapping_calc_risk_v01(ss, tr, t, risk0)
mapping_calc_risk_v02(ss, tr, t, risk0)
mapping_risk_ww_v01(risk, wwmat, ww2)



.\utilmy\zarchive\geospatial.py


.\utilmy\zarchive\global01.py


.\utilmy\zarchive\kagglegym.py
-------------------------functions----------------------
make()
r_score(y_true, y_pred, sample_weight = None, multioutput = None)

-------------------------methods----------------------
Environment.__init__(self)
Environment.__str__(self)
Environment.reset(self)
Environment.step(self, target)
Observation.__init__(self, train, target, features)


.\utilmy\zarchive\linux.py
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



.\utilmy\zarchive\multiprocessfunc.py
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



.\utilmy\zarchive\multithread.py
-------------------------functions----------------------
multithread_run(fun_async, input_list:list, n_pool = 5, start_delay = 0.1, verbose = True, **kw)
multithread_run_list(**kwargs)



.\utilmy\zarchive\portfolio.py
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


.\utilmy\zarchive\portfolio_withdate.py
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


.\utilmy\zarchive\py2to3\_HELP.py
-------------------------functions----------------------
fun_cython(a)
fun_python(a)
os_VS_build(self, lib_to_build)
os_VS_start(self, version)
os_compileVSsolution(dir1, flags1 = "", type1 = "devenv", compilerdir = "")
set_rc_version(rcfile, target_version)



.\utilmy\zarchive\py2to3\__init__.py


.\utilmy\zarchive\py2to3\alldata.py


.\utilmy\zarchive\py2to3\allmodule.py
-------------------------functions----------------------
pprint(table1, tablefmt = "simple")
pprint2(x)
str_convert_beforeprint(x)
str_to_unicode(x, encoding = 'utf-8')
str_to_utf8(x)



.\utilmy\zarchive\py2to3\allmodule_fin.py


.\utilmy\zarchive\py2to3\coke_functions.py
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



.\utilmy\zarchive\py2to3\datanalysis.py
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


.\utilmy\zarchive\py2to3\excel.py
-------------------------functions----------------------
add_one(data)
double_sum(x, y)
get_workbook_name()
matrix_mult(x, y)
npdot()



.\utilmy\zarchive\py2to3\fast.py
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



.\utilmy\zarchive\py2to3\fast_parallel.py
-------------------------functions----------------------
task_find_best(tasks, n_top = 5)
task_parallel_job_01(name, param, datadict)
task_progress(tasks)
task_summary(tasks)



.\utilmy\zarchive\py2to3\filelock.py
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


.\utilmy\zarchive\py2to3\function_custom.py
-------------------------functions----------------------
fun_obj(vv, ext)
getweight(ww, size = (9, 3)
mapping_calc_risk_elvis_v03(ss, tr, t, riskout)
mapping_calc_risk_v00(self, ss, tr, t, risk0)
mapping_calc_risk_v01(ss, tr, t, risk0)
mapping_calc_risk_v02(ss, tr, t, risk0)
mapping_risk_ww_v01(risk, wwmat, ww2)



.\utilmy\zarchive\py2to3\geospatial.py


.\utilmy\zarchive\py2to3\global01.py


.\utilmy\zarchive\py2to3\kagglegym.py
-------------------------functions----------------------
make()
r_score(y_true, y_pred, sample_weight = None, multioutput = None)

-------------------------methods----------------------
Environment.__init__(self)
Environment.__str__(self)
Environment.reset(self)
Environment.step(self, target)
Observation.__init__(self, train, target, features)


.\utilmy\zarchive\py2to3\linux.py
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



.\utilmy\zarchive\py2to3\multiprocessfunc.py
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



.\utilmy\zarchive\py2to3\portfolio.py
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


.\utilmy\zarchive\py2to3\portfolio_withdate.py
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


.\utilmy\zarchive\py2to3\report.py
-------------------------functions----------------------
map_show()
xl_create_pdf()
xl_create_pivot(infile, index_list = ["Manager", "Rep", "Product"], value_list = ["Price", "Quantity"])
xl_save_report(report, outfile)



.\utilmy\zarchive\py2to3\rstatpy.py
-------------------------functions----------------------
stl(data, ns, np = None, nt = None, nl = None, isdeg = 0, itdeg = 1, ildeg = 1, nsjump = None, ntjump = None, nljump = None, ni = 2, no = 0, fulloutput = False)



.\utilmy\zarchive\py2to3\util_min.py
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



.\utilmy\zarchive\py2to3\util_ml.py
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


.\utilmy\zarchive\py2to3\utilgeo.py
-------------------------functions----------------------
df_to_geojson(df, col_properties, lat = 'latitude', lon = 'longitude')



.\utilmy\zarchive\py3\util.py
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


.\utilmy\zarchive\report.py
-------------------------functions----------------------
map_show()
xl_create_pdf()
xl_create_pivot(infile, index_list = ["Manager", "Rep", "Product"], value_list = ["Price", "Quantity"])
xl_save_report(report, outfile)



.\utilmy\zarchive\rstatpy.py
-------------------------functions----------------------
stl(data, ns, np = None, nt = None, nl = None, isdeg = 0, itdeg = 1, ildeg = 1, nsjump = None, ntjump = None, nljump = None, ni = 2, no = 0, fulloutput = False)



.\utilmy\zarchive\storage\aapackage_gen\34\Working Copy of util34.py
-------------------------functions----------------------
getmodule_doc(module1, fileout = '')



.\utilmy\zarchive\storage\aapackage_gen\34\global01.py


.\utilmy\zarchive\storage\aapackage_gen\34\util34.py
-------------------------functions----------------------
getmodule_doc(module1, fileout = '')



.\utilmy\zarchive\storage\aapackage_gen\codeanalysis.py
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



.\utilmy\zarchive\storage\aapackage_gen\global01.py


.\utilmy\zarchive\storage\aapackage_gen\old\Working Copy of util34.py
-------------------------functions----------------------
getmodule_doc(module1, fileout = '')



.\utilmy\zarchive\storage\aapackage_gen\old\util27.py
-------------------------functions----------------------
getmodule_doc(module1, fileout = '')



.\utilmy\zarchive\storage\aapackage_gen\old\util34.py
-------------------------functions----------------------
getmodule_doc(module1, fileout = '')



.\utilmy\zarchive\storage\aapackage_gen\old\utils27.py
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



.\utilmy\zarchive\storage\aapackage_gen\old\utils34.py
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



.\utilmy\zarchive\storage\aapackage_gen\util.py
-------------------------functions----------------------
getmodule_doc(module1, fileout = '')



.\utilmy\zarchive\storage\aapackagedev\random.py
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



.\utilmy\zarchive\storage\alldata.py


.\utilmy\zarchive\storage\allmodule.py
-------------------------functions----------------------
aa_isanaconda()



.\utilmy\zarchive\storage\benchmarktest.py
-------------------------functions----------------------
payoff1(pricepath)
payoff2(pricepath)
payoffeuro1(st)
payoffeuro1(st)



.\utilmy\zarchive\storage\codeanalysis.py
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



.\utilmy\zarchive\storage\dbcheck.py


.\utilmy\zarchive\storage\derivatives.py
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



.\utilmy\zarchive\storage\dl_utils.py
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



.\utilmy\zarchive\storage\excel.py
-------------------------functions----------------------
add_one(data)
double_sum(x, y)
get_workbook_name()
matrix_mult(x, y)
npdot()



.\utilmy\zarchive\storage\global01.py


.\utilmy\zarchive\storage\installNewPackage.py


.\utilmy\zarchive\storage\java.py
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



.\utilmy\zarchive\storage\multiprocessfunc.py
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



.\utilmy\zarchive\storage\panda_util.py
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



.\utilmy\zarchive\storage\portfolio.py
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


.\utilmy\zarchive\storage\rec_data.py
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



.\utilmy\zarchive\storage\rec_metrics.py
-------------------------functions----------------------
full_auc(model, ground_truth)
precision_at_k(model, ground_truth, k, user_features = None, item_features = None)
predict(model, uid, pids)



.\utilmy\zarchive\storage\sobol.py
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



.\utilmy\zarchive\storage\stateprocessor.py
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



.\utilmy\zarchive\storage\symbolicmath.py
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



.\utilmy\zarchive\storage\technical_indicator.py
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



.\utilmy\zarchive\storage\testmulti.py
-------------------------functions----------------------
mc01()
mc02()
multiprocess(processes, samples, x, widths)
random_tree(Data)
random_tree(Data)
serial(samples, x, widths)
test01()
test01()



.\utilmy\zarchive\storage\theano_imdb.py
-------------------------functions----------------------
get_dataset_file(dataset, default_dataset, origin)
load_data(path = "imdb.pkl", n_words = 100000, valid_portion = 0.1, maxlen = None, sort_by_len = True)
prepare_data(seqs, labels, maxlen = None)



.\utilmy\zarchive\storage\theano_lstm.py
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



.\utilmy\zarchive\util.py
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


.\utilmy\zarchive\util_aws.py
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


.\utilmy\zarchive\util_min.py
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



.\utilmy\zarchive\util_ml.py
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


.\utilmy\zarchive\util_sql.py
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



.\utilmy\zarchive\util_web.py
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



.\utilmy\zarchive\utilgeo.py
-------------------------functions----------------------
df_to_geojson(df, col_properties, lat = 'latitude', lon = 'longitude')



.\utilmy\zarchive\zzarchive\zutil.py
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



.\utilmy\zarchive\zzarchive\zutil_features.py
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
