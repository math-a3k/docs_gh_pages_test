

utilmy\adatasets.py
-------------------------functions----------------------
test_all()
test0()
test1()
log(*s)
log2(*s)
dataset_classifier_XXXXX(nrows = 500, **kw)
pd_train_test_split(df, coly = None)
pd_train_test_split2(df, coly)
dataset_classifier_pmlb(name = '', return_X_y = False)
test_dataset_classifier_covtype(nrows = 500)
test_dataset_regression_fake(nrows = 500, n_features = 17)
test_dataset_classification_fake(nrows = 500)
test_dataset_classification_petfinder(nrows = 1000)
fetch_dataset(url_dataset, path_target = None, file_target = None)



utilmy\data.py
-------------------------functions----------------------
log(*s)
help()



utilmy\dates.py
-------------------------functions----------------------
test_all()
random_dates(start, end, size)
random_genders(size, p = None)
log(*s)
pd_date_split(df, coldate  =   'time_key', prefix_col  = "", sep = "/", verbose = False)
date_now(fmt="%Y-%m-%d %H = "%Y-%m-%d %H:%M:%S %Z%z", add_days = 0, timezone = 'Asia/Tokyo')
date_is_holiday(array)
date_weekmonth2(d)
date_weekmonth(d)
date_weekyear2(dt)
date_weekday_excel(x)
date_weekyear_excel(x)
date_generate(start = '2018-01-01', ndays = 100)



utilmy\debug.py
-------------------------functions----------------------
log(*s)
help()
print_everywhere()
log5(*s)
log_trace(msg = "", dump_path = "", globs = None)
profiler_start()
profiler_stop()



utilmy\decorators.py
-------------------------functions----------------------
test_all()
test_decorators()
test_decorators2()
thread_decorator(func)
timeout_decorator(seconds = 10, error_message = os.strerror(errno.ETIME)
timer_decorator(func)
profiler_context()
profiler_decorator(func)
profiler_decorator_base(fnc)
test0()
thread_decorator_test()
profiler_decorator_base_test()
timeout_decorator_test()
profiled_sum()
dummy_func()



utilmy\distributed.py
-------------------------functions----------------------
test_funtions_thread()
test_index()
test_tofilesafe()
test_all()
to_file_safe(msg:str, fpath:str)
os_lock_acquireLock(plock:str = "tmp/plock.lock")
os_lock_releaseLock(locked_file_descriptor)
os_lock_execute(fun_run, fun_args = None, ntry = 5, plock = "tmp/plock.lock", sleep = 5)
date_now(fmt = "%Y-%m-%d %H =  "%Y-%m-%d %H:%M:%S %Z%z")
time_sleep_random(nmax = 5)
save(dd, to_file = "", verbose = False)
load(to_file = "")
load_serialize(name)
save_serialize(name, value)

-------------------------methods----------------------
toFile.__init__(self, fpath)
toFile.write(self, msg)
IndexLock.__init__(self, findex, file_lock = None, min_size = 5, skip_comment = True, ntry = 20)
IndexLock.read(self, )
IndexLock.save_isok(self, flist:list)
IndexLock.save_filter(self, val:list = None)
IndexLock.get(self, **kw)
IndexLock.put(self, val:list = None)
Index0.__init__(self, findex:str = "ztmp_file.txt", ntry = 10)
Index0.read(self, )
Index0.save(self, flist:list)
Index0.save_filter(self, val:list = None)


utilmy\graph.py


utilmy\iio.py


utilmy\nnumpy.py
-------------------------functions----------------------
test0()
test1()
to_dict(**kw)
to_timeunix(datex = "2018-01-16")
to_datetime(x)
np_list_intersection(l1, l2)
np_add_remove(set_, to_remove, to_add)
to_float(x, valdef = -1)
to_int(x, valdef = -1)
is_int(x)
is_float(x)

-------------------------methods----------------------
LRUCache.__init__(self, max_size = 4)
LRUCache._move_latest(self, key)
LRUCache.__getitem__(self, key, default = None)
LRUCache.__setitem__(self, key, value)
fixedDict.__init__(self, *args, **kwds)
fixedDict.__setitem__(self, key, value)
fixedDict._check_size_limit(self)
dict_to_namespace.__init__(self, d)


utilmy\oos.py
-------------------------functions----------------------
os_wait_processes(nhours = 7)
os_path_size(path  =  '.')
os_path_split(fpath:str = "")
os_file_replacestring(findstr, replacestr, some_dir, pattern = "*.*", dirlevel = 1)
os_walk(path, pattern = "*", dirlevel = 50)
os_copy_safe(dirin = None, dirout = None, nlevel = 5, nfile = 5000, logdir = "./", pattern = "*", exclude = "", force = False, sleep = 0.5, cmd_fallback = "", verbose = Trueimport shutil, time, os, globflist = [] ; dirinj = dirinnlevel) =  [] ; dirinj = dirinnlevel):)
os_merge_safe(dirin_list = None, dirout = None, nlevel = 5, nfile = 5000, nrows = 10**8, cmd_fallback  =  "umount /mydrive/  && mount /mydrive/  ", sleep = 0.3)
z_os_search_fast(fname, texts = None, mode = "regex/str")
os_search_content(srch_pattern = None, mode = "str", dir1 = "", file_pattern = "*.*", dirlevel = 1)
os_get_function_name()
os_variable_init(ll, globs)
os_import(mod_name = "myfile.config.model", globs = None, verbose = True)
os_variable_exist(x, globs, msg = "")
os_variable_check(ll, globs = None, do_terminate = True)
os_clean_memory(varlist, globx)
os_system_list(ll, logfile = None, sleep_sec = 10)
os_file_check(fp)
os_to_file(txt = "", filename = "ztmp.txt", mode = 'a')
os_platform_os()
os_cpu()
os_platform_ip()
os_memory()
os_sleep_cpu(cpu_min = 30, sleep = 10, interval = 5, msg =  "", verbose = True)
os_sizeof(o, ids, hint = " deep_getsizeof(df_pd, set()
os_removedirs(path, verbose = False)
os_getcwd()
os_system(ll, logfile = None, sleep_sec = 10)
os_makedirs(dir_or_file)
print_everywhere()
log5(*s)
log_trace(msg = "", dump_path = "", globs = None)
profiler_start()
profiler_stop()

-------------------------methods----------------------
toFileSafe.__init__(self, fpath)
toFileSafe.write(self, msg)
toFileSafe.log(self, msg)
toFileSafe.w(self, msg)


utilmy\parallel.py
-------------------------functions----------------------
pd_read_file2(path_glob = "*.pkl", ignore_index = True, cols = None, verbose = False, nrows = -1, nfile = 1000000, concat_sort = True, n_pool = 1, npool = None, drop_duplicates = None, col_filter = None, col_filter_val = None, dtype_reduce = None, fun_apply = None, use_ext = None, **kw)
pd_groupby_parallel2(df, colsgroup = None, fun_apply = None, npool: int  =  1, **kw, )
pd_groupby_parallel(df, colsgroup = None, fun_apply = None, npool: int  =  1, **kw, )
pd_apply_parallel(df, fun_apply = None, npool = 5, verbose = True)
multiproc_run(fun_async, input_list: list, n_pool = 5, start_delay = 0.1, verbose = True, input_fixed:dict = None, npool = None, **kw)
multithread_run(fun_async, input_list: list, n_pool = 5, start_delay = 0.1, verbose = True, input_fixed:dict = None, npool = None, **kw)
multiproc_tochunk(flist, npool = 2)
multithread_run_list(**kwargs)
z_pd_read_file3(path_glob = "*.pkl", ignore_index = True, cols = None, verbose = False, nrows = -1, concat_sort = True, n_pool = 1, npool = None, drop_duplicates = None, col_filter = None, col_filter_val = None, dtype_reduce = None, **kw)
zz_pd_read_file3(path_glob = "*.pkl", ignore_index = True, cols = None, nrows = -1, concat_sort = True, n_pool = 1, npool = None, drop_duplicates = None, col_filter = None, col_filter_val = None, dtype_reduce = None, fun_apply = None, max_file = -1, #### apply function for each subverbose = False, **kw)
zz_pd_groupby_parallel5(df, colsgroup = None, fun_apply = None, npool = 5, verbose = False, **kw)
ztest1()
ztest2()



utilmy\ppandas.py
-------------------------functions----------------------
pd_to_mapdict(df, colkey = 'ranid', colval = 'item_tag', naval = '0', colkey_type = 'str', colval_type = 'str', npool = 5, nrows = 900900900, verbose = True)
pd_to_hiveparquet(dirin, dirout = "/ztmp_hive_parquet/df.parquet", verbose = False)
pd_random(nrows = 100)
pd_merge(df1, df2, on = None, colkeep = None)
pd_plot_multi(df, plot_type = None, cols_axe1:list = [], cols_axe2:list = [], figsize = (8, 4)
pd_plot_histogram(dfi, path_save = None, nbin = 20.0, q5 = 0.005, q95 = 0.995, nsample =  -1, show = False, clear = True)
pd_filter(df, filter_dict = "shop_id=11, l1_genre_id>600, l2_genre_id<80311,", verbose = False)
pd_to_file(df, filei, check = 0, verbose = True, show = 'shape', **kw)
pd_sample_strat(df, col, n)
pd_cartesian(df1, df2)
pd_col_bins(df, col, nbins = 5)
pd_dtype_reduce(dfm, int0  = 'int32', float0  =  'float32')
pd_dtype_count_unique(df, col_continuous = [])
pd_dtype_to_category(df, col_exclude, treshold = 0.5)
pd_dtype_getcontinuous(df, cols_exclude:list = [], nsample = -1)
pd_del(df, cols:list)
pd_add_noise(df, level = 0.05, cols_exclude:list = [])
pd_cols_unique_count(df, cols_exclude:list = [], nsample = -1)
pd_show(df, nrows = 100, reader = 'notepad.exe', **kw)
to_dict(**kw)
to_timeunix(datex = "2018-01-16")
to_datetime(x)
np_list_intersection(l1, l2)
np_add_remove(set_, to_remove, to_add)
to_float(x)
to_int(x)
is_int(x)
is_float(x)

-------------------------methods----------------------
dict_to_namespace.__init__(self, d)


utilmy\tabular.py
-------------------------functions----------------------
test_anova(df, col1, col2)
test_normality2(df, column, test_type)
test_plot_qqplot(df, col_name)
test_heteroscedacity(y, y_pred, pred_value_only = 1)
test_normality(df, column, test_type)
test_mutualinfo(error, Xtest, colname = None, bins = 5)
test_hypothesis(df_obs, df_ref, method = '', **kw)
estimator_std_normal(err, alpha = 0.05, )
estimator_boostrap_bayes(err, alpha = 0.05, )
estimator_bootstrap(err, custom_stat = None, alpha = 0.05, n_iter = 10000)
pd_train_test_split_time(df, test_period  =  40, cols = None, coltime  = "time_key", sort = True, minsize = 5, n_sample = 5, verbose = False)
pd_to_scipy_sparse_matrix(df)
pd_stat_correl_pair(df, coltarget = None, colname = None)
pd_stat_pandas_profile(df, savefile = "report.html", title = "Pandas Profile")
pd_stat_distribution_colnum(df, nrows = 2000, verbose = False)
pd_stat_histogram(df, bins = 50, coltarget = "diff")
np_col_extractname(col_onehot)
np_list_remove(cols, colsremove, mode = "exact")
pd_stat_shift_trend_changes(df, feature, target_col, threshold = 0.03)
pd_stat_shift_trend_correlation(df, df_test, colname, target_col)
pd_stat_shift_changes(df, target_col, features_list = 0, bins = 10, df_test = 0)
np_conv_to_one_col(np_array, sep_char = "_")
pd_data_drift_detect_alibi(df:pd.DataFrame, ### Reference datasetdf_new:pd.DataFrame, ### Test dataset to be checkedmethod:str = "'regressoruncertaintydrift','classifieruncertaintydrift','ksdrift','mmddrift','learnedkerneldrift','chisquaredrift','tabulardrift', 'classifierdrift','spotthediffdrift'", backend:str = 'tensorflow,pytorch', model = None, ### Pre-trained modelp_val = 0.05, **kwargs)



utilmy\utilmy.py
-------------------------functions----------------------
import_function(fun_name = None, module_name = None)
help_create(modulename = 'utilmy.nnumpy', prefixs = None)
pd_random(ncols = 7, nrows = 100)
pd_generate_data(ncols = 7, nrows = 100)
pd_getdata(verbose = True)
glob_glob(dirin, nfile = 1000)
sys_exit(msg = "exited", err_int = 0)
sys_install(cmd = "")
test_all()
git_repo_root()
git_current_hash(mode = 'full')
save(dd, to_file = "", verbose = False)
load(to_file = "")

-------------------------methods----------------------
Index0.__init__(self, findex:str = "ztmp_file.txt")
Index0.read(self, )
Index0.save(self, flist:list)
Session.__init__(self, dir_session = "ztmp/session/", )
Session.show(self)
Session.save(self, name, glob = None, tag = "")
Session.load(self, name, glob:dict = None, tag = "")
Session.save_session(self, folder, globs, tag = "")
Session.load_session(self, folder, globs = None)


utilmy\utils.py
-------------------------functions----------------------
test_all()
test0()
test1()
log(*s)
log2(*s)
logw(*s)
loge(*s)
logger_setup()
config_load(config_path: Optional[Union[str, pathlib.Path]]  =  None)
dataset_donwload(url, path_target)
os_extract_archive(file_path, path = ".", archive_format = "auto")
to_file(s, filep)



utilmy\util_batch.py
-------------------------functions----------------------
test_funtions_thread()
test_index()
test_os_process_find_name()
test_all()
now_weekday_isin(day_week = None, timezone = 'jp')
now_hour_between(hour1="12 = "12:45", hour2="13 = "13:45", timezone = "jp")
now_daymonth_isin(day_month, timezone = "jp")
date_now_jp(fmt = "%Y%m%d", add_days = 0, add_hours = 0, timezone = 'jp')
time_sleep_random(nmax = 5)
batchLog(object)
os_wait_filexist(flist, sleep = 300)
os_wait_fileexist2(dirin, ntry_max = 100, sleep_time = 300)
os_wait_cpu_ram_lower(cpu_min = 30, sleep = 10, interval = 5, msg =  "", name_proc = None, verbose = True)
os_process_find_name(name = r"((.*/)
os_wait_program_end(cpu_min = 30, sleep = 60, interval = 5, msg =  "", program_name = None, verbose = True)
to_file_safe(msg:str, fpath:str)
os_lock_acquireLock(plock:str = "tmp/plock.lock")
os_lock_releaseLock(locked_file_descriptor)
main()

-------------------------methods----------------------
toFile.__init__(self, fpath)
toFile.write(self, msg)
IndexLock.__init__(self, findex, file_lock = None, min_size = 5, skip_comment = True, ntry = 20)
IndexLock.read(self, )
IndexLock.save_isok(self, flist:list)
IndexLock.save_filter(self, val:list = None)
IndexLock.get(self, **kw)
IndexLock.put(self, val:list = None)
Index0.__init__(self, findex:str = "ztmp_file.txt", ntry = 10)
Index0.read(self, )
Index0.save(self, flist:list)
Index0.save_filter(self, val:list = None)


utilmy\util_cpu.py
-------------------------functions----------------------
log(*argv)
os_getparent(dir0)
ps_process_monitor_child(pid, logfile = None, duration = None, interval = None)
ps_wait_process_completion(subprocess_list, waitsec = 10)
ps_wait_ressourcefree(cpu_max = 90, mem_max = 90, waitsec = 15)
ps_get_cpu_percent(process)
ps_get_memory_percent(process)
ps_all_children(pr)
ps_get_process_status(pr)
ps_process_isdead(pid)
ps_get_computer_resources_usage()
ps_find_procs_by_name(name = r"((.*/)
os_launch(commands)
ps_terminate(processes)
os_extract_commands(csv_file, has_header = False)
ps_is_issue(p)
ps_net_send(tperiod = 5)
ps_is_issue_system()
monitor_maintain()
os_python_environment()
os_environment()
os_is_wndows()
np_avg(list)
np_pretty_nb(num, suffix = "")
monitor_nodes()
os_generate_cmdline()

-------------------------methods----------------------
NodeStats.__init__(self, num_connected_users = 0, num_pids = 0, cpu_count = 0, cpu_percent = None, mem_total = 0, mem_avail = 0, swap_total = 0, swap_avail = 0, disk_io = None, disk_usage = None, net = None, )
NodeStats.mem_used(self)
IOThroughputAggregator.__init__(self)
IOThroughputAggregator.aggregate(self, cur_read, cur_write)
NodeStatsCollector.__init__(self, pool_id, node_id, refresh_interval = _DEFAULT_STATS_UPDATE_INTERVAL, app_insights_key = None, )
NodeStatsCollector.init(self)
NodeStatsCollector._get_network_usage(self)
NodeStatsCollector._get_disk_io(self)
NodeStatsCollector._get_disk_usage(self)
NodeStatsCollector._sample_stats(self)
NodeStatsCollector._collect_stats(self)
NodeStatsCollector._send_stats(self, stats)
NodeStatsCollector._log_stats(self, stats)
NodeStatsCollector.run(self)


utilmy\util_download.py


utilmy\util_hadoop.py


utilmy\util_sampling.py
-------------------------functions----------------------
test()
reservoir_sampling(src, nsample, temp_fac = 1.5, rs = None)



utilmy\util_zip.py
-------------------------functions----------------------
unzip(in_dir, out_dir)
gzip(dirin = '/mydir', dirout = "./")
dir_size(dirin = "mypath", dirout = "./save.txt")
dataset_donwload(url, path_target)
dataset_get_path(cfg: dict)
os_extract_archive(file_path, path = ".", archive_format = "auto")
to_file(s, filep)



utilmy\zzz_text.py
-------------------------functions----------------------
test()
test_lsh()
pd_text_hash_create_lsh(df, col, sep = " ", threshold = 0.7, num_perm = 10, npool = 1, chunk  =  20000)
pd_text_getcluster(df:pd.DataFrame, col:str = 'col', threshold = 0.5, num_perm:int = 5, npool = 1, chunk  =  100000)
pd_text_similarity(df: pd.DataFrame, cols = [], algo = '')



utilmy\__init__.py


utilmy\codeparser\__init__.py


utilmy\configs\test.py
-------------------------functions----------------------
create_fixtures_data(tmp_path)
test_validate_yaml_types(tmp_path)
test_validate_yaml_types_failed(tmp_path)
test_validate_yaml_failed_silent(tmp_path)



utilmy\configs\util_config.py
-------------------------functions----------------------
log(*s)
loge(*s)
test_yamlschema()
test_pydanticgenrator()
test4()
test_example()
config_load(config_path:    str   =  None, path_default:   str   =  None, config_default: dict  =  None, save_default:   bool  =  False, to_dataclass:   bool  =  True, )
config_isvalid_yamlschema(config_dict: dict, schema_path: str  =  'config_val.yaml', silent: bool  =  False)
config_isvalid_pydantic(config_dict: dict, pydanctic_schema: str  =  'config_py.yaml', silent: bool  =  False)
convert_yaml_to_box(yaml_path: str)
convert_dict_to_pydantic(config_dict: dict, schema_name: str)
pydantic_model_generator(input_file: Union[Path, str], input_file_type, output_file: Path, **kwargs, )
global_verbosity(cur_path, path_relative = "/../../config.json", default = 5, key = 'verbosity', )
zzz_config_load_validate(config_path: str, schema_path: str, silent: bool  =  False)



utilmy\configs\__init__.py


utilmy\db\keyvalue.py
-------------------------functions----------------------
os_environ_set(name, value)
os_path_size(folder = None)
db_init(db_dir:str = "path", globs = None)
db_flush(db_dir)
db_size(db_dir =  None)
db_merge()
db_create_dict_pandas(df = None, cols = None, colsu = None)
db_load_dict(df, colkey, colval, verbose = True)
diskcache_load(db_path_or_object = "", size_limit = 100000000000, verbose = True)
diskcache_save(df, colkey, colvalue, db_path = "./dbcache.db", size_limit = 100000000000, timeout = 10, shards = 1, tbreak = 1, ## Break during insert to prevent big WAL file**kw)
diskcache_save2(df, colkey, colvalue, db_path = "./dbcache.db", size_limit = 100000000000, timeout = 10, shards = 1, npool = 10, sqlmode =  'fast', verbose = True)
diskcache_getkeys(cache)
diskcache_keycount(cache)
diskcache_getall(cache, limit = 1000000000)
diskcache_get(cache)
diskcache_config(db_path = None, task = 'commit')

-------------------------methods----------------------
DBlist.__init__(self, config_dict = None, config_path = None)
DBlist.add(self, db_path)
DBlist.remove(self, db_path)
DBlist.list(self, show = True)
DBlist.info(self, )
DBlist.clean(self, )
DBlist.check(self, db_path = None)
DBlist.show(self, db_path = None, n = 4)


utilmy\deeplearning\util_dl.py
-------------------------functions----------------------
tf_check()
gpu_usage()
gpu_available()
down_page(query, out_dir = "query1", genre_en = '', id0 = "", cat = "", npage = 1)
create_train_npz()
create_train_parquet()
model_deletes(dry = 0)



utilmy\deeplearning\util_embedding.py
-------------------------functions----------------------
np_str_to_array(vv, l2_norm = True, mdim  =  200)
viz_run(dirin = "in/model.vec", dirout = "ztmp/", nmax = 100)
topk(topk = 100, dname = None, pattern = "df_*1000*.parquet", filter1 = None)
topk_nearest_vector(x0, vector_list, topk = 3)
sim_score2(path = "")
simscore_cosinus_calc(embs, words)
faiss_create_index(df_or_path = None, col = 'emb', dir_out = "", db_type  =  "IVF4096,Flat", nfile = 1000, emb_dim = 200)
faiss_topk(df = None, root = None, colid = 'id', colemb = 'emb', faiss_index = None, topk = 200, npool = 1, nrows = 10**7, nfile = 1000)
np_matrix_to_str2(m, map_dict)
np_matrix_to_str(m, map_dict)
np_matrix_to_str_sim(m)
np_str_to_array(vv, l2_norm = True, mdim  =  200)
topk_predict()
topk(topk = 100, dname = None, pattern = "df_*1000*.parquet", filter1 = None)
topk_nearest_vector(x0, vector_list, topk = 3)
topk_export()
convert_txt_to_vector_parquet(dirin = None, dirout = None, skip = 0, nmax = 10**8)
data_add_onehot(dfref, img_dir, labels_col)
test()
unzip(in_dir, out_dir)

-------------------------methods----------------------
TopToolbar.__init__(self)
vizEmbedding.__init__(self, path = "myembed.parquet", num_clusters = 5, sep = ";", config:dict = None)
vizEmbedding.run_all(self, mode = "mds", col_embed = 'embed', ndim = 2, nmax =  5000, dir_out = "ztmp/", ntest = 10000)
vizEmbedding.dim_reduction(self, mode = "mds", col_embed = 'embed', ndim = 2, nmax =  5000, dir_out = None, ntest = 10000, npool = 2)
vizEmbedding.create_clusters(self, after_dim_reduction = True)
vizEmbedding.create_visualization(self, dir_out = "ztmp/", mode = 'd3', cols_label = None, show_server = False, **kw)
vizEmbedding.draw_hiearchy(self)


utilmy\deeplearning\util_yolo.py
-------------------------functions----------------------
test_convert_to_yolov5()
test_yolov5_from_xml()
test_all()
yolo_extract_info_from_xml(xml_file:str)
convert_to_yolov5(info_dict:Dict, names:Dict, output:str)
yolov5_from_xml(xml_file_path:str  =  "None", xml_folder:str =  "None", output:str = "None")



utilmy\deeplearning\zz_util_topk.py
-------------------------functions----------------------
np_matrix_to_str2(m, map_dict)
np_matrix_to_str(m, map_dict)
np_matrix_to_str_sim(m)
np_str_to_array(vv, l2_norm = True, mdim  =  200)
topk_predict()
topk()
topk_nearest_vector(x0, vector_list, topk = 3)
topk_export()
convert_txt_to_vector_parquet(dirin = None, dirout = None, skip = 0, nmax = 10**8)
data_add_onehot(dfref, img_dir, labels_col)
test()
unzip(in_dir, out_dir)
gzip()
predict(name = None)
folder_size()



utilmy\deeplearning\__init__.py


utilmy\docs\cli.py


utilmy\docs\code_parser.py
-------------------------functions----------------------
export_stats_pertype(in_path:str = None, type:str = None, out_path:str = None)
export_stats_perfile(in_path:str = None, out_path:str = None)
export_stats_perrepo_txt(in_path:str = None, out_path:str = None, repo_name:str = None)
export_stats_perrepo(in_path:str = None, out_path:str = None, repo_name:str = None)
export_stats_repolink_txt(repo_link: str, out_path:str = None)
export_stats_repolink(repo_link: str, out_path:str = None)
export_call_graph_url(repo_link: str, out_path:str = None)
export_call_graph(repo_link: str, out_path:str = None)
get_list_function_name(file_path)
get_list_class_name(file_path)
get_list_class_methods(file_path)
get_list_variable_global(file_path)
_get_docs(all_lines, index_1, func_lines)
get_list_function_info(file_path)
get_list_class_info(file_path)
get_list_method_info(file_path)
get_list_method_stats(file_path)
get_list_class_stats(file_path)
get_list_function_stats(file_path)
get_stats(df:pd.DataFrame, file_path:str)
get_file_stats(file_path)
get_list_imported_func(file_path: str)
get_list_import_class_as(file_path: str)
_get_words(row)
_get_functions(row)
_get_avg_char_per_word(row)
_validate_file(file_path)
_clean_data(array)
_remove_empty_line(line)
_remmove_commemt_line(line)
_get_and_clean_all_lines(file_path)
_get_all_line(file_path)
_get_all_lines_in_function(function_name, array, indentMethod = '')
_get_all_lines_in_class(class_name, array)
_get_all_lines_define_function(function_name, array, indentMethod = '')
_get_define_function_stats(array)
_get_function_stats(array, indent)
write_to_file(uri, type, list_functions, list_classes, list_imported, dict_functions, list_class_as, out_path)
test_example()



utilmy\docs\generate_doc.py
-------------------------functions----------------------
markdown_create_function(uri, name, type, args_name, args_type, args_value, start_line, list_docs, prefix = "")
markdown_create_file(list_info, prefix = '')
markdown_createall(dfi, prefix = "")
table_create_row(uri, name, type, start_line, list_funtions, prefix)
table_all_row(list_rows)
table_create(uri, name, type, start_line, list_funtions, prefix)
run_markdown(repo_stat_file, output = 'docs/doc_main.md', prefix="https = "https://github.com/user/repo/tree/a")
run_table(repo_stat_file, output = 'docs/doc_table.md', prefix="https = "https://github.com/user/repo/tree/a")
test()



utilmy\docs\test.py
-------------------------functions----------------------
log(data)
list_buy_price(start, bottom, delta)
calculateSellPrice(enter, profit)
list_sell_price(start, top, delta)
calculateBuyPrice(enter, profit)
get_list_price()
trading_up()
trading_down()
update_price()



utilmy\docs\__init__.py


utilmy\excel\xlvba.py
-------------------------functions----------------------
load_csv(csvfile)
invokenumpy()
invokesklearn()
loaddf()



utilmy\images\util_image.py
-------------------------functions----------------------
run_multiprocess(myfun, list_args, npool = 10, **kwargs)
image_cache_create()
image_cache_check(db_path:str = "db_images.cache", dirout:str = "tmp/", tag = "cache1")
image_cache_save(image_path_list:str = "db_images.cache", db_dir:str = "tmp/", tag = "cache1")
image_check_npz(path_npz, keys = ['train'], path = "", tag = "", n_sample = 3, renorm = True)
image_read(filepath_or_buffer: Union[str, io.BytesIO])
image_save()
image_show_in_row(image_list:dict = None)
image_resize_ratio(image, width = None, height = None, inter = cv2.INTER_AREA)
image_center_crop(img, dim)
image_resize_pad(img, size = (256, 256)
image_resize(image, width = None, height = None, inter = cv2.INTER_AREA)
image_resize2(image, width = None, height = None, inter = cv2.INTER_AREA)
image_padding_generate(paddings_number: int  =  1, min_padding: int  =  1, max_padding: int  =  1)
image_merge(image_list, n_dim, padding_size, max_height, total_width)
image_remove_extra_padding(img, inverse = False, removedot = True)
image_remove_bg(in_dir = "", out_dir = "", level = 1)
os_path_check(path, n = 5)
image_face_blank(in_dir = "", level  =  "/*", out_dir = f"", npool = 30)
image_text_blank(in_dir, out_dir, level = "/*")
image_check(path_npz, keys = ['train'], path = "", tag = "", n_sample = 3, renorm = True)



utilmy\logs\test_log.py
-------------------------functions----------------------
test1()
test2()
test_launch_server()
test_server()

-------------------------methods----------------------
LoggingStreamHandler.handle(self)


utilmy\logs\util_log.py
-------------------------functions----------------------
logger_setup(log_config_path: str  =  None, log_template: str  =  "default", **kwargs)
log(log_config_path: str  =  None, log_template: str  =  "default", **kwargs)
log2(*s)
log3(*s)
logw(*s)
logc(*s)
loge(*s)
logr(*s)
test()
z_logger_stdout_override()
z_logger_custom_1()



utilmy\logs\util_log_std.py
-------------------------functions----------------------
create_appid(filename)
create_logfilename(filename)
create_uniqueid()
logger_setup(logger_name = None, log_file = None, formatter = FORMATTER_1, isrotate = False, isconsole_output = True, logging_level = logging.DEBUG, )
logger_handler_console(formatter = None)
logger_handler_file(isrotate = False, rotate_time = "midnight", formatter = None, log_file_used = None)
logger_setup2(name = __name__, level = None)
printlog(s = "", s1 = "", s2 = "", s3 = "", s4 = "", s5 = "", s6 = "", s7 = "", s8 = "", s9 = "", s10 = "", app_id = "", logfile = None, iswritelog = True, )
writelog(m = "", f = None)
load_arguments(config_file = None, arg_list = None)



utilmy\logs\__init__.py


utilmy\nlp\rank_fusion.py
-------------------------functions----------------------
get_fusion_alg(text)
parse_svmlight_rank(filepath)
sort_by_score_and_id(elem1, elem2)
parse_svmlight_score(filepath)
parse_trec(filepath, idIsFilename = False)
norm_minmax(ranks, lowest, highest)
norm_zscore(ranks, lowest, highest)
comb(rank_list, fusion_function, params)
print_comb(ranks, max_k, outstream, rank_name)
folder_merge(base_path, norm, merge_function, params, max_k, rank_name, output)
file_merge(base_path, norm, merge_function, params, max_k, rank_name, output)

-------------------------methods----------------------
prettyfloat.__repr__(self)
prettyfloat.__str__(self)
prettyint.__repr__(self)
prettyint.__str__(self)


utilmy\nlp\rank_fusion_functions.py
-------------------------functions----------------------
isr(result_list, params)
log_isr(result_list, params)
logn_isr(result_list, params)
expn_isr(result_list, params)
logn_rrf(result_list, params)
expn_rrf(result_list, params)
rrf(result_list, params)
rr(result_list, params)
votes(result_list, params)
mnz(result_list, params)
sum(result_list, params)
max(result_list, params)
min(result_list, params)
condor(doc_id_scores)
compareCondor(item1, item2)



utilmy\nlp\util_cluster.py


utilmy\nlp\util_cocount.py
-------------------------functions----------------------
corpus_generate(outfile = "data.cor", unique_words_needed = 1000)
train_model(dirinput = "./data.cor", dirout = "./modelout/model.bin", **params)
load_model(dirin = "./modelout/model.bin")
create_1gram_stats(dirin, w_to_id)
cocount_calc_matrix(dirin = "gen_text_dist3.txt", dense = True)
cocount_get_topk(matrix, w_to_id)
cocount_matrix_to_dict(matrix, w_to_id)
cocount_norm(matrix)
get_top_k(w, ccount_name_dict, ccount_score_dict, top = 5)
calc_comparison_stats(model, ccount_name_dict, ccount_score_dict, corpus_file = "data.cor", top = 20, output_dir = "./no_ss_test")
corpus_generate_from_cocount(dirin = "./data.cor", dirout = "gen_text_dist3.txt", unique_words = 100, sentences_count = 1000)
corpus_add_prefix(dirin = "gen_text_dist3.txt", dirout = "gen_text_dist4.txt")
run_all()



utilmy\nlp\util_gensim.py
-------------------------functions----------------------
help()
test_all()
test_gensim1()
bigram_load_convert(path)
bigram_write_seq(rr = 0, dirin = None, dirout = None, tag = "")
bigram_get_seq3(ranid, itemtag, lname, pnorm)
bigram_get_list(ranid, mode = 'name, proba')
ccount_get_sample(lname, lproba = None, pnorm = None, k = 5)
np_intersec(va, vb)
generate_random_bigrams(n_words = 100, word_length = 4, bigrams_length = 5000)
write_random_sentences_from_bigrams_to_file(dirout, n_sentences = 14000)
gensim_model_load(dirin, modeltype = 'fastext', **kw)
gensim_model_train_save(model_or_path = None, dirinput = 'lee_background.cor', dirout = "./modelout/model", epochs = 1, pars: dict  =  None, **kw)
gensim_model_check(model_path)
text_preprocess(sentence, lemmatizer, stop_words)
text_generate_random_sentences(dirout = None, n_sentences = 5, )
embedding_model_to_parquet(model_vector_path = "model.vec", nmax = 500)
embedding_to_parquet(dirin = None, dirout = None, skip = 0, nmax = 10 ** 8, is_linevalid_fun=Nonedirout);dirout);4)if is_linevalid_fun is None = Nonedirout);dirout);4)if is_linevalid_fun is None:  #### Validate linew):)
embedding_load_parquet(dirin = "df.parquet", nmax = 500)
np_str_to_array(vv, l2_norm = True, mdim = 200)



utilmy\nlp\util_nlp.py
-------------------------functions----------------------
plot_distributions(dtf, x, max_cat = 20, top = None, y = None, bins = None, figsize = (10, 5)
add_detect_lang(data, column)
add_text_length(data, column)
add_sentiment(data, column, algo = "vader", sentiment_range = (-1, 1)
create_stopwords(lst_langs = ["english"], lst_add_words = [], lst_keep_words = [])
utils_preprocess_text(txt, lst_regex = None, punkt = True, lower = True, slang = True, lst_stopwords = None, stemm = False, lemm = True)
add_preprocessed_text(data, column, lst_regex = None, punkt = False, lower = False, slang = False, lst_stopwords = None, stemm = False, lemm = False, remove_na = True)
word_freq(corpus, ngrams = [1, 2, 3], top = 10, figsize = (10, 7)
plot_wordcloud(corpus, max_words = 150, max_font_size = 35, figsize = (10, 10)
add_word_freq(data, column, lst_words, freq = "count")
ner_displacy(txt, ner = None, lst_tag_filter = None, title = None, serve = False)
utils_ner_text(txt, ner = None, lst_tag_filter = None, grams_join = "_")
utils_lst_count(lst, top = None)
utils_ner_features(lst_dics_tuples, tag)
add_ner_spacy(data, column, ner = None, lst_tag_filter = None, grams_join = "_", create_features = True)
tags_freq(tags, top = 30, figsize = (10, 5)
retrain_ner_spacy(train_data, output_dir, model = "blank", n_iter = 100)
dtf_partitioning(dtf, y, test_size = 0.3, shuffle = False)
add_encode_variable(dtf, column)
evaluate_multi_classif(y_test, predicted, predicted_prob, figsize = (15, 5)
fit_bow(corpus, vectorizer = None, vocabulary = None)
features_selection(X, y, X_names, top = None, print_top = 10)
sparse2dtf(X, dic_vocabulary, X_names, prefix = "")
fit_ml_classif(X_train, y_train, X_test, vectorizer = None, classifier = None)
explainer_lime(model, y_train, txt_instance, top = 10)
utils_preprocess_ngrams(corpus, ngrams = 1, grams_join = " ", lst_ngrams_detectors = [])
create_ngrams_detectors(corpus, grams_join = " ", lst_common_terms = [], min_count = 5, top = 10, figsize = (10, 7)
fit_w2v(corpus, ngrams = 1, grams_join = " ", lst_ngrams_detectors = [], min_count = 1, size = 300, window = 20, sg = 1, epochs = 100)
embedding_w2v(x, nlp = None, value_na = 0)
plot_w2v(lst_words = None, nlp = None, plot_type = "2d", top = 20, annotate = True, figsize = (10, 5)
vocabulary_embeddings(dic_vocabulary, nlp = None)
text2seq(corpus, ngrams = 1, grams_join = " ", lst_ngrams_detectors = [], fitted_tokenizer = None, top = None, oov = None, maxlen = None)
utils_plot_keras_training(training)
fit_dl_classif(X_train, y_train, X_test, encode_y = False, dic_y_mapping = None, model = None, weights = None, epochs = 100, batch_size = 256)
explainer_attention(model, tokenizer, txt_instance, lst_ngrams_detectors = [], top = 5, figsize = (5, 3)
explainer_shap(model, X_train, X_instance, dic_vocabulary, class_names, top = 10)
get_similar_words(lst_words, top, nlp = None)
word_clustering(corpus, nlp = None, ngrams = 1, grams_join = " ", lst_ngrams_detectors = [], n_clusters = 3)
fit_lda(corpus, ngrams = 1, grams_join = " ", lst_ngrams_detectors = [], n_topics = 3, figsize = (10, 7)
plot_w2v_cluster(dic_words = None, nlp = None, plot_type = "2d", annotate = True, figsize = (10, 5)
utils_bert_embedding(txt, tokenizer, nlp, log = False)
embedding_bert(x, tokenizer = None, nlp = None, log = False)
tokenize_bert(corpus, tokenizer = None, maxlen = None)
fit_bert_classif(X_train, y_train, X_test, encode_y = False, dic_y_mapping = None, model = None, epochs = 100, batch_size = 64)
utils_cosine_sim(a, b, nlp = None)
predict_similarity_classif(X, dic_y)
explainer_similarity_classif(tokenizer, nlp, dic_clusters, txt_instance, token_level = False, top = 5, figsize = (20, 10)
utils_string_matching(a, lst_b, threshold = None, top = None)
vlookup(lst_left, lst_right, threshold = 0.7, top = 1)
display_string_matching(a, b, both = True, sentences = True, titles = [])
fit_seq2seq(X_train, y_train, X_embeddings, y_embeddings, model = None, build_encoder_decoder = True, epochs = 100, batch_size = 64)
predict_seq2seq(X_test, encoder_model, decoder_model, fitted_tokenizer, special_tokens = ("<START>", "<END>")
evaluate_summary(y_test, predicted)
textrank(corpus, ratio = 0.2)
bart(corpus, ratio = 0.2)



utilmy\nlp\util_rank.py
-------------------------functions----------------------
rank_biased_overlap(list1, list2, p = 0.9)
rbo_find_p()
rank_topk_kendall(a:list, b:list, topk = 5, p = 0)

-------------------------methods----------------------
RankingSimilarity.__init__(self, S: Union[List, np.ndarray], T: Union[List, np.ndarray], verbose = False)
RankingSimilarity.assert_p(self, p: float)
RankingSimilarity._bound_range(self, value: float)
RankingSimilarity.rbo(self, k: Optional[float]  =  None, p: float  =  1.0, ext: bool  =  False)
RankingSimilarity.rbo_ext(self, p = 0.98)
RankingSimilarity.top_weightness(self, p: Optional[float]  =  None, d: Optional[int]  =  None)


utilmy\nlp\util_rankmerge.py
-------------------------functions----------------------
log(*s)
test1()
test()
test_rankadjust2(df1, df2)
rank_adjust2(ll1, ll2, kk =  1)
rank_generatefake(ncorrect = 30, nsize = 100)
rank_generate_fake(dict_full, list_overlap, nsize = 100, ncorrect = 20)
rank_fillna(df)
rank_eval(rank_true, dfmerged, nrank = 100)
rank_score(rank1, rank2, adjust = 1.0, kk = 1.0)
rank_merge_v4(ll1, ll2)
rank_merge_v3(list1, list2, maxrank = 100)
rank_merge(ll1, ll2)
rank_merge_v2(list1, list2, nrank)



utilmy\nlp\util_sentence.py
-------------------------functions----------------------
help()
test_all()
test3()
test2()
test1()
model_load(model_path)
model_get_embed(model)
get_embed(model_emb, word)
model_finetune_classifier(model_path, df, n_labels = 3, lrate = 1e-5)
embed_compare_class_sim(model, embed_a, embed_b, embed_c, embed_d)

-------------------------methods----------------------
SentenceEncoder.__init__(self, num_labels = None)
SentenceEncoder.call(self, inputs, **kwargs)


utilmy\optim\util_hyper.py
-------------------------functions----------------------
log(*s)
run_hyper_optuna(obj_fun, pars_dict_init, pars_dict_range, engine_pars, ntrials = 3)
create_model_name(save_folder, model_name)
optim(modelname = "model_dl.1_lstm.py", pars =  {}, df  =  None, optim_engine = "optuna", optim_method = "normal/prune", save_folder = "model_save/", log_folder = "logs/", ntrials = 2)
optim_optuna(modelname = "model_dl.1_lstm.py", pars =  {}, df  =  None, optim_method = "normal/prune", save_folder = "/mymodel/", log_folder = "", ntrials = 2)
load_arguments(config_file =  None)
data_loader(file_name = 'dataset/GOOG-year.csv')
test_all()
test_fast()



utilmy\optim\util_optim.py
-------------------------functions----------------------
test_use_operon()



utilmy\prepro\prepro.py
-------------------------functions----------------------
log(*s)
log2(*s)
log3(*s)
log4(*s, n = 0, m = 1)
log4_pd(name, df, *s)
_pd_colnum(df, col, pars)
_pd_colnum_fill_na_median(df, col, pars)
prepro_load(prefix, pars)
prepro_save(prefix, pars, df_new, cols_new, prepro)
pd_col_atemplate(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_coly_clean(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_coly(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_colnum_normalize(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_colnum_quantile_norm(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_colnum_bin(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_colnum_binto_onehot(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_colcat_to_onehot(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_colcat_bin(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_colcross(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_coldate(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_colcat_encoder_generic(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_colcat_minhash(df: pd.DataFrame, col: list = None, pars: dict = None)
os_convert_topython_code(txt)
save_json(js, pfile, mode = 'a')
pd_col_genetic_transform(df: pd.DataFrame, col: list = None, pars: dict = None)
test()



utilmy\prepro\prepro_rec.py
-------------------------functions----------------------
_preprocess_criteo(df, **kw)
_preprocess_movielens(df, **kw)



utilmy\prepro\prepro_text.py
-------------------------functions----------------------
log(*s, n = 0, m = 1)
logs(*s)
log_pd(df, *s, n = 0, m = 1)
pd_coltext_clean(df, col, stopwords =  None, pars = None)
pd_coltext_wordfreq(df, col, stopwords, ntoken = 100)
nlp_get_stopwords()
pd_coltext(df, col, stopwords =  None, pars = None)
pd_coltext_universal_google(df, col, pars = {})



utilmy\prepro\prepro_tseries.py
-------------------------functions----------------------
log(*s)
log2(*s)
log3(*s)
logd(*s, n = 0, m = 0)
pd_prepro_custom(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_prepro_custom2(df: pd.DataFrame, cols: list = None, pars: dict = None)
pd_ts_date(df: pd.DataFrame, cols: list = None, pars: dict = None)
pd_ts_groupby(df: pd.DataFrame, cols: list = None, pars: dict = None)
pd_ts_onehot(df: pd.DataFrame, cols: list = None, pars: dict = None)
pd_ts_autoregressive(df: pd.DataFrame, cols: list = None, pars: dict = None)
pd_ts_rolling(df: pd.DataFrame, cols: list = None, pars: dict = None)
pd_ts_lag(df: pd.DataFrame, cols: list = None, pars: dict = None)
pd_ts_difference(df: pd.DataFrame, cols: list = None, pars: dict = None)
pd_ts_tsfresh_features(df: pd.DataFrame, cols: list = None, pars: dict = None)
pd_ts_deltapy_generic(df: pd.DataFrame, cols: list = None, pars: dict = None)
test_get_sampledata(url="https = "https://github.com/firmai/random-assets-two/raw/master/numpy/tsla.csv")
test_deltapy_all()
test_prepro_v1()
test_deltapy_get_method(df)
test_deltapy_all2()
m5_dataset()



utilmy\prepro\run_feature_profile.py
-------------------------functions----------------------
log(*s, n = 0, m = 0)
run_profile(path_data = None, path_output = "data/out/ztmp/", n_sample = 5000)



utilmy\prepro\util_feature.py
-------------------------functions----------------------
log(*s, n = 0, m = 1, **kw)
log2(*s, **kw)
log3(*s, **kw)
os_get_function_name()
os_getcwd()
pa_read_file(path =   'folder_parquet/', cols = None, n_rows = 1000, file_start = 0, file_end = 100000, verbose = 1, )
pa_write_file(df, path =   'folder_parquet/', cols = None, n_rows = 1000, partition_cols = None, overwrite = True, verbose = 1, filesystem  =  'hdfs')
test_get_classification_data(name = None)
params_check(pars, check_list, name = "")
save_features(df, name, path = None)
load_features(name, path)
save_list(path, name_list, glob)
save(df, name, path = None)
load(name, path)
pd_read_file(path_glob = "*.pkl", ignore_index = True, cols = None, verbose = False, nrows = -1, concat_sort = True, n_pool = 1, drop_duplicates = None, col_filter = None, col_filter_val = None, **kw)
load_dataset(path_data_x, path_data_y = '', colid = "jobId", n_sample = -1)
fetch_spark_koalas(path_data_x, path_data_y = '', colid = "jobId", n_sample = -1)
fetch_dataset(url_dataset, path_target = None, file_target = None)
load_function_uri(uri_name="myfolder/myfile.py = "myfolder/myfile.py::myFunction")
metrics_eval(metric_list = ["mean_squared_error"], ytrue = None, ypred = None, ypred_proba = None, return_dict = False)
pd_stat_dataset_shift(dftrain, dftest, colused, nsample = 10000, buckets = 5, axis = 0)
pd_stat_datashift_psi(expected, actual, buckettype = 'bins', buckets = 10, axis = 0)
estimator_std_normal(err, alpha = 0.05, )
estimator_boostrap_bayes(err, alpha = 0.05, )
estimator_bootstrap(err, custom_stat = None, alpha = 0.05, n_iter = 10000)
test_heteroscedacity(y, y_pred, pred_value_only = 1)
test_normality(error, distribution = "norm", test_size_limit = 5000)
test_mutualinfo(error, Xtest, colname = None, bins = 5)
feature_importance_perm(clf, Xtrain, ytrain, cols, n_repeats = 8, scoring = 'neg_root_mean_squared_error', show_graph = 1)
feature_selection_multicolinear(df, threshold = 1.0)
feature_correlation_cat(df, colused)
pd_feature_generate_cross(df, cols, cols_cross_input = None, pct_threshold = 0.2, m_combination = 2)
pd_col_to_onehot(dfref, colname = None, colonehot = None, return_val = "dataframe,column")
pd_colcat_mergecol(df, col_list, x0, colid = "easy_id")
pd_colcat_tonum(df, colcat = "all", drop_single_label = False, drop_fact_dict = True)
pd_colcat_mapping(df, colname)
pd_colcat_toint(dfref, colname, colcat_map = None, suffix = None)
pd_colnum_tocat(df, colname = None, colexclude = None, colbinmap = None, bins = 5, suffix = "_bin", method = "uniform", na_value = -1, return_val = "dataframe,param", params={"KMeans_n_clusters" = {"KMeans_n_clusters": 8, "KMeans_init": 'k-means++', "KMeans_n_init": 10,"KMeans_max_iter": 300, "KMeans_tol": 0.0001, "KMeans_precompute_distances": 'auto',"KMeans_verbose": 0, "KMeans_random_state": None,"KMeans_copy_x": True, "KMeans_n_jobs": None, "KMeans_algorithm": 'auto'})
pd_colnum_normalize(df0, colname, pars, suffix = "_norm", return_val = 'dataframe,param')
pd_col_merge_onehot(df, colname)
pd_col_to_num(df, colname = None, default = np.nan)
pd_col_filter(df, filter_val = None, iscol = 1)
pd_col_fillna(dfref, colname = None, method = "frequent", value = None, colgroupby = None, return_val = "dataframe,param", )
pd_pipeline_apply(df, pipeline)
pd_stat_correl_pair(df, coltarget = None, colname = None)
pd_stat_pandas_profile(df, savefile = "report.html", title = "Pandas Profile")
pd_stat_distribution_colnum(df, nrows = 2000, verbose = False)
pd_stat_histogram(df, bins = 50, coltarget = "diff")
col_extractname(col_onehot)
col_remove(cols, colsremove, mode = "exact")
pd_colnum_tocat_stat(df, feature, target_col, bins, cuts = 0)
pd_stat_shift_trend_changes(df, feature, target_col, threshold = 0.03)
pd_stat_shift_trend_correlation(df, df_test, colname, target_col)
pd_stat_shift_changes(df, target_col, features_list = 0, bins = 10, df_test = 0)
np_conv_to_one_col(np_array, sep_char = "_")

-------------------------methods----------------------
dict2.__init__(self, d)


utilmy\prepro\__init__.py


utilmy\recsys\ab.py
-------------------------functions----------------------
log(*s)
test_ab_getstat()
test_np_calculate_z_val()
test_np_calculate_confidence_interval()
test_np_calculate_ab_dist()
test_pd_generate_ctr_data()
test_np_calculate_min_sample_size()
get_ab_test_data(vars_also = False)
test_plot_binom_dist()
test_plot_ab()
test_zplot()
test_all()
ab_getstat(df, treatment_col = 'treatment', measure_col = 'metric', attribute_cols = 'attrib', control_label = 'A', variation_label = 'B', inference_method = 'means_delta', hypothesis = None, alpha = .05, experiment_name = 'exp', dirout = None, tag = None, **kwargs)
np_calculate_z_val(sig_level = 0.05, two_tailed = True)
np_calculate_confidence_interval(sample_mean = 0, sample_std = 1, sample_size = 1, sig_level = 0.05)
np_calculate_ab_dist(stderr, d_hat = 0, group_type = 'control')
pd_generate_ctr_data(N_A, N_B, p_A, p_B, days = None, control_label = 'A', test_label = 'B', seed = None)
np_calculate_min_sample_size(bcr, mde, power = 0.8, sig_level = 0.05)
plot_confidence_interval(ax, mu, s, sig_level = 0.05, color = 'grey')
plot_norm_dist(ax, mu, std, with_CI = False, sig_level = 0.05, label = None)
plot_binom_dist(ax, A_converted, A_cr, A_total, B_converted, B_cr, B_total)
plot_null_hypothesis_dist(ax, stderr)
plot_alternate_hypothesis_dist(ax, stderr, d_hat)
show_area(ax, d_hat, stderr, sig_level, area_type = 'power')
plot_ab(ax, N_A, N_B, bcr, d_hat, sig_level = 0.05, show_power = False, show_alpha = False, show_beta = False, show_p_value = False, show_legend = True)
zplot(ax, area = 0.95, two_tailed = True, align_right = False)
abplot_CI_bars(N, X, sig_level = 0.05, dmin = None)
funnel_CI_plot(A, B, sig_level = 0.05)
_pooled_prob(N_A, N_B, X_A, X_B)
_pooled_SE(N_A, N_B, X_A, X_B)
_p_val(N_A, N_B, p_A, p_B)



utilmy\recsys\__init__.py


utilmy\spark\main.py
-------------------------functions----------------------
main()



utilmy\spark\setup.py


utilmy\sparse\test_model1.py
-------------------------methods----------------------
EASE.__init__(self)
EASE._get_users_and_items(self, df)
EASE.fit(self, df, lambda_: float  =  0.5, implicit = True)
EASE.predict(self, train, users, items, k)


utilmy\sparse\test_model2.py


utilmy\stats\__init__.py


utilmy\tabular\util_drift.py


utilmy\templates\cli.py
-------------------------functions----------------------
run_cli()
template_show()
template_copy(name, out_dir)



utilmy\templates\__init__.py


utilmy\tseries\util_tseries.py


utilmy\viz\css.py
-------------------------functions----------------------
fontsize_css(size)



utilmy\viz\embedding.py
-------------------------functions----------------------
log(*s)
embedding_load_word2vec(model_vector_path = "model.vec", nmax  =  500)
embedding_load_parquet(path = "df.parquet", nmax  =  500)
tokenize_text(text)
run(dir_in = "in/model.vec", dir_out = "ztmp/", nmax = 100)

-------------------------methods----------------------
vizEmbedding.__init__(self, path = "myembed.parquet", num_clusters = 5, sep = ";", config:dict = None)
vizEmbedding.run_all(self, mode = "mds", col_embed = 'embed', ndim = 2, nmax =  5000, dir_out = "ztmp/")
vizEmbedding.dim_reduction(self, mode = "mds", col_embed = 'embed', ndim = 2, nmax =  5000, dir_out = None)
vizEmbedding.create_clusters(self, after_dim_reduction = True)
vizEmbedding.create_visualization(self, dir_out = "ztmp/", mode = 'd3', cols_label = None, show_server = False, **kw)
vizEmbedding.draw_hiearchy(self)


utilmy\viz\template1.py


utilmy\viz\test_vizhtml.py
-------------------------functions----------------------
test_getdata(verbose = True)
test1(verbose = False)
test2(verbose = False)



utilmy\viz\util_map.py


utilmy\viz\vizhtml.py
-------------------------functions----------------------
mlpd3_add_tooltip(fig, points, labels)
pd_plot_scatter_get_data(df0:pd.DataFrame, colx: str = None, coly: str = None, collabel: str = None, colclass1: str = None, colclass2: str = None, nmax: int = 20000, **kw)
pd_plot_scatter_matplot(df:pd.DataFrame, colx: str = None, coly: str = None, collabel: str = None, colclass1: str = None, colclass2: str = None, cfg: dict  =  {}, mode = 'd3', save_path: str = '', verbose = True, **kw)
pd_plot_histogram_matplot(df:pd.DataFrame, col: str = '', colormap:str = 'RdYlBu', title: str = '', nbin = 20.0, q5 = 0.005, q95 = 0.995, nsample = -1, save_img: str = "", xlabel: str = None, ylabel: str = None, verbose = True, **kw)
pd_plot_tseries_matplot(df:pd.DataFrame, plot_type: str = None, coly1: list  =  [], coly2: list  =  [], 8, 4), spacing = 0.1, verbose = True, **kw))
mpld3_server_start()
pd_plot_highcharts(df)
pd_plot_scatter_highcharts(df0:pd.DataFrame, colx:str = None, coly:str = None, collabel: str = None, colclass1: str = None, colclass2: str = None, colclass3: str = None, nsample = 10000, cfg:dict = {}, mode = 'd3', save_img = '', verbose = True, **kw)
pd_plot_tseries_highcharts(df0, coldate:str = None, date_format  =  None, coly1:list  = [], coly2:list  = [], figsize:tuple  =   None, title:str = None, xlabel:str = None, y1label:str = None, y2label:str = None, cfg:dict = {}, mode = 'd3', save_img = "", verbose = True, **kw)
pd_plot_histogram_highcharts(df:pd.DataFrame, colname:str = None, binsNumber = None, binWidth = None, color:str = '#7CB5EC', title:str = "", xaxis_label:str =  "x-axis", yaxis_label:str = "y-axis", cfg:dict = {}, mode = 'd3', save_img = "", show = False, verbose = True, **kw)
html_show_chart_highchart(html_code, verbose = True)
html_show(html_code, verbose = True)
images_to_html(dir_input = "*.png", title = "", verbose = False)
colormap_get_names()
pd_plot_network(df:pd.DataFrame, cola: str = 'col_node1', colb: str = 'col_node2', coledge: str = 'col_edge', colweight: str = "weight", html_code:bool  =  True)
help_get_codesource(func)
to_float(x)
zz_css_get_template(css_name:str =  "A4_size")
zz_test_get_random_data(n = 100)
zz_pd_plot_histogram_highcharts_old(df, col, figsize = None, title = None, cfg:dict = {}, mode = 'd3', save_img = '')

-------------------------methods----------------------
mpld3_TopToolbar.__init__(self)


utilmy\viz\__init__.py


utilmy\webscraper\cli_arxiv.py
-------------------------functions----------------------
main(url = "", path_pdf = "data/scraper/v1/pdf/", path_txt = "data/scraper/v1/txt/", npage_max = 1, tag = "v1")
parse_main_page(url)
process_and_paginate_soup(response_soup)
process_url(url_data, idx, list_len, path_pdf = "", path_txt = "")



utilmy\webscraper\cli_openreview.py
-------------------------methods----------------------
URLData.title_normalized(self)
URLData.pdf_title(self)
URLData.txt_title(self)
PageParser.__init__(self)
PageParser.parse(self, url, page_limit)
PageParser.construct_api_url(self, url)
PageParser.process_api_response(self, response)
PageParser.generate_next_url(self, current_url)
PageParser.construct_pdf_url(note)
PDFExtractor.__init__(self, pdf_path, txt_path)
PDFExtractor.extract(self, url_data: URLData)
OpenreviewScraper.__init__(self, url = "", npage_max = 1, path_pdf = "", path_txt = "")
OpenreviewScraper.run(self)


utilmy\webscraper\cli_reddit.py
-------------------------methods----------------------
URLData.fixed_post_id(self)
URLData.completed_url(self)
URLData.new_url(self)
URLData.sanitized_title(self)
RedditPageScraper.__init__(self, path_txt)
RedditPageScraper.parse(self, url, nposts)
RedditPageScraper.extract(self, url_data: URLData)
RedditPageScraper.request(self, url)
RedditPageScraper.replace_url(self, url, page_count)
RedditScraper.__init__(self, url = "", nposts = 20, path_txt = "")
RedditScraper.run(self)


utilmy\webscraper\ner_extractor.py


utilmy\webscraper\pdf_scraper.py
-------------------------functions----------------------
main(url = "", path_pdf = "data/scraper/v1/pdf/", path_txt = "data/scraper/v1/txt/", npage_max = 1, tag = "v1")
parse_main_page(url)
process_and_paginate_soup(response_soup)
process_url(url_data, idx, list_len, path_pdf = "", path_txt = "")



utilmy\webscraper\scrape_batch.py
-------------------------functions----------------------
test_extract_to_pandas()
extract_to_pandas(html, table_id = None)
download_page()



utilmy\webscraper\util_search.py
-------------------------functions----------------------
run()
googleSearch(query)

-------------------------methods----------------------
Search.__init__(self)
Search.repos_user()


utilmy\webscraper\util_web.py
-------------------------functions----------------------
web_get_url_loginpassword(url_list = None, browser = "phantomjs", login = "", password = "", phantomjs_path="D = "D:/_devs/webserver/phantomjs-1.9.8/phantomjs.exe", pars = None, if pars is None)
web_send_email_tls(FROM, recipient, subject, body, login1 = "mizenjapan@gmail.com", pss1 = "sophieelise237", server1 = "smtp.gmail.com", port1 = 465, )
web_sendurl(url1)



utilmy\webscraper\__init__.py


utilmy\zml\core_deploy.py
-------------------------functions----------------------
load_arguments()



utilmy\zml\core_run.py
-------------------------functions----------------------
log(*s)
get_global_pars(config_uri = "")
get_config_path(config = '')
data_profile2(config = '')
data_profile(config = '')
preprocess(config = '', nsample = None)
train(config = '', nsample = None)
check(config='outlier_predict.py = 'outlier_predict.py::titanic_lightgbm')
predict(config = '', nsample = None)
train_sampler(config = '', nsample = None)
transform(config = '', nsample = None)
hyperparam_wrapper(config_full = "", ntrials = 2, n_sample = 5000, debug = 1, path_output          =  "data/output/titanic1/", path_optuna_storage  =  'data/output/optuna_hyper/optunadb.db', metric_name = 'accuracy_score', mdict_range = None)
deploy()



utilmy\zml\core_test.py
-------------------------functions----------------------
os_bash(cmd)
log_separator(space = 140)
log_info_repo(arg = None)
to_logfile(prefix = "", dateformat='+%Y-%m-%d_%H = '+%Y-%m-%d_%H:%M:%S,%3N')
os_file_current_path()
os_system(cmd, dolog = 1, prefix = "", dateformat='+%Y-%m-%d_%H = '+%Y-%m-%d_%H:%M:%S,%3N')
json_load(path)
log_remote_start(arg = None)
log_remote_push(name = None)



utilmy\zml\datasketch_hashing.py
-------------------------functions----------------------
create_hash(df, column_name, threshold, num_perm)
find_clusters(df, column_name, threshold, num_perm)



utilmy\zml\titanic_classifier.py
-------------------------functions----------------------
global_pars_update(model_dict, data_name, config_name)
config1()
pd_col_myfun(df = None, col = None, pars = {})
check()



utilmy\zml\toutlier.py
-------------------------functions----------------------
global_pars_update(model_dict, data_name, config_name, dir_data = None, dir_input_tr = None, dir_input_te = None)
post_process_fun(y)
pre_process_fun(y)



utilmy\zml\tsampler.py
-------------------------functions----------------------
global_pars_update(model_dict, data_name, config_name)
config_sampler()
log(*s)
test_batch(nsample = 1000)



utilmy\zml\tseries.py
-------------------------functions----------------------
global_pars_update(model_dict, data_name, config_name)
config1()
pd_dsa2_custom(df: pd.DataFrame, col: list = None, pars: dict = None)



utilmy\zml\zgitutil.py
-------------------------functions----------------------
path_leaf(path)
_run(*args)
commit(mylist)
_filter_on_size(size = 0, f = files)
add(size = 10000000)
main()



utilmy\zml\zlocal.py
-------------------------functions----------------------
make_dc(d, name = 'd_dataclass')



utilmy\zml\ztemplate.py
-------------------------functions----------------------
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(Xpred = None, data_pars = None, compute_pars = {}, out_pars = {}, **kw)
reset()
save(path = None, info = None)
load_model(path = "")
init(*kw, **kwargs)
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(Xpred = None, data_pars = None, compute_pars = {}, out_pars = {}, **kw)
reset()
save(path = None, info = None)
load_model(path = "")
get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
get_dataset(Xtrain, cols_type_received, cols_ref)
test(config = '')
test_helper(model_pars, data_pars, compute_pars)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)
MY_MODEL_CLASS.__init__(cpars)
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zml\__init__.py


utilmy\zzarchive\alldata.py


utilmy\zzarchive\allmodule.py
-------------------------functions----------------------
pprint(table1, tablefmt = "simple")
pprint2(x)
str_convert_beforeprint(x)
str_to_utf8(x)
str_to_unicode(x, encoding = 'utf-8')



utilmy\zzarchive\allmodule_fin.py


utilmy\zzarchive\coke_functions.py
-------------------------functions----------------------
date_diffsecond(str_t1, str_t0, fmt='YYYY-MM-DD HH = 'YYYY-MM-DD HH:mm:SS')
date_diffstart(t)
date_diffend(t)
np_dict_tolist(dd)
np_dict_tostr_val(dd)
np_dict_tostr_key(dd)
day(s)
month(s)
year(s)
hour(s)
weekday(s, fmt = 'YYYY-MM-DD', i0 = 0, i1 = 10)
season(d)
daytime(d)
pd_date_splitall(df, coldate = 'purchased_at')



utilmy\zzarchive\datanalysis.py
-------------------------functions----------------------
pd_filter_column(df_client_product, filter_val = [], iscol = 1)
pd_missing_show()
pd_describe(df)
pd_stack_dflist(df_list)
pd_validation_struct()
pd_checkpoint()
xl_setstyle(file1)
xl_val(ws, colj, rowi)
isnull(x)
xl_get_rowcol(ws, i0, j0, imax, jmax)
xl_getschema(dirxl = "", filepattern = '*.xlsx', dirlevel = 1, outfile = '.xlsx')
str_to_unicode(x, encoding = 'utf-8')
csv_dtypes_getdict(df = None, csvfile = None)
csv_fast_processing()
csv_col_schema_toexcel(dircsv = "", filepattern = '*.csv', outfile = '.xlsx', returntable = 1, maxrow = 5000000, maxcol_pertable = 90, maxstrlen = 'U80')
csv_col_get_dict_categoryfreq(dircsv, filepattern = "*.csv", category_cols = [], maxline = -1, fileencoding = "utf-8")
csv_row_reduce_line(fromfile, tofile, condfilter, catval_tokeep, header = True, maxline = -1)
csv_analysis()
csv_row_reduce_line_manual(file_category, file_transact, file_reduced)
csv_row_mapreduce(dircsv = "", outfile = "", type_mapreduce = 'sum', nrow = 1000000, chunk =  5000000)
csv_pivotable(dircsv = "", filepattern = '*.csv', fileh5 = '.h5', leftX = 'col0', topY = 'col2', centerZ = 'coli', mapreduce = 'sum', chunksize =  500000, tablename = 'df')
csv_bigcompute()
db_getdata()
db_sql()
db_meta_add(metadb, dbname, new_table = ('', [])
db_meta_find(ALLDB, query = '', filter_db = [], filter_table = [], filter_column = [])
col_study_getcategorydict_freq(catedict)
col_feature_importance(Xcol, Ytarget)
pd_col_study_distribution_show(df, col_include = None, col_exclude = None, pars={'binsize' = {'binsize':20})
col_study_summary(Xmat = [0.0, 0.0], Xcolname = ['col1', 'col2'], Xcolselect = [9, 9], isprint = 0)
pd_col_pair_plot(dfX, Xcolname_selectlist = None, dfY = None, Ycolname = None)
col_pair_correl(Xcol, Ytarget)
col_pair_interaction(Xcol, Ytarget)
plot_col_pair(dfX, Xcolname_selectlist = None, dfY = None, Ycolname = None)
tf_transform_catlabel_toint(Xmat)
tf_transform_pca(Xmat, dimpca = 2, whiten = True)
plot_distance_heatmap(Xmat_dist, Xcolname)
plot_cluster_2D(X_2dim, target_class, target_names)
plot_cluster_tsne(Xmat, Xcluster_label = None, metric = 'euclidean', perplexity = 50, ncomponent = 2, savefile = '', isprecompute = False, returnval = True)
plot_cluster_pca(Xmat, Xcluster_label = None, metric = 'euclidean', dimpca = 2, whiten = True, isprecompute = False, savefile = '', doreturn = 1)
plot_cluster_hiearchy(Xmat_dist, p = 30, truncate_mode = None, color_threshold = None, get_leaves = True, orientation = 'top', labels = None, count_sort = False, distance_sort = False, show_leaf_counts = True, do_plot = 1, no_labels = False, leaf_font_size = None, leaf_rotation = None, leaf_label_func = None, show_contracted = False, link_color_func = None, ax = None, above_threshold_color = 'b', annotate_above = 0)
plot_distribution_density(Xsample, kernel = 'gaussian', N = 10, bandwith = 1 / 10.0)
plot_Y(Yval, typeplot = '.b', tsize = None, labels = None, title = '', xlabel = '', ylabel = '', zcolor_label = '', figsize = (8, 6)
plot_XY(xx, yy, zcolor = None, tsize = None, labels = None, title = '', xlabel = '', ylabel = '', zcolor_label = '', figsize = (8, 6)
plot_XY_plotly(xx, yy, towhere = 'url')
plot_XY_seaborn(X, Y, Zcolor = None)
optim_is_pareto_efficient(Xmat_cost, epsilon =  0.01, ret_boolean = 1)
sk_catboost_classifier(Xtrain, Ytrain, Xcolname = None, pars= {"learning_rate" =  {"learning_rate":0.1, "iterations":1000, "random_seed":0, "loss_function": "MultiClass" }, isprint = 0)
sk_catboost_regressor()
sk_model_auto_tpot(Xmat, y, outfolder = 'aaserialize/', model_type = 'regressor/classifier', train_size = 0.5, generation = 1, population_size = 5, verbosity = 2)
sk_params_search_best(Xmat, Ytarget, model1, param_grid={'alpha' = {'alpha':  np.linspace(0, 1, 5) }, method = 'gridsearch', param_search= {'scoretype' =  {'scoretype':'r2', 'cv':5, 'population_size':5, 'generations_number':3 })
sk_distribution_kernel_bestbandwidth(kde)
sk_distribution_kernel_sample(kde = None, n = 1)
sk_correl_rank(correl = [[1, 0], [0, 1]])
sk_error_r2(Ypred, y_true, sample_weight = None, multioutput = None)
sk_error_rmse(Ypred, Ytrue)
sk_cluster_distance_pair(Xmat, metric = 'jaccard')
sk_cluster(Xmat, metric = 'jaccard')
sk_cluster_algo_custom(Xmat, algorithm, args, kwds, returnval = 1)
sk_optim_de(obj_fun, bounds, maxiter = 1, name1 = '', solver1 = None, isreset = 1, popsize = 15)
sk_feature_importance(clfrf, feature_name)
sk_showconfusion(clfrf, X_train, Y_train, isprint = True)
sk_tree(Xtrain, Ytrain, nbtree, maxdepth, isprint1 = 1, njobs = 1)
sk_gen_ensemble_weight(vv, acclevel, maxlevel = 0.88)
sk_votingpredict(estimators, voting, ww, X_test)
sk_tree_get_ifthen(tree, feature_names, target_names, spacer_base = " ")

-------------------------methods----------------------
sk_model_template1.__init__(self, alpha = 0.5, low_y_cut = -0.09, high_y_cut = 0.09, ww0 = 0.95)
sk_model_template1.fit(self, X, Y = None)
sk_model_template1.predict(self, X, y = None, ymedian = None)
sk_model_template1.score(self, X, Ytrue = None, ymedian = None)
sk_stateRule.__init__(self, state, trigger, colname = [])
sk_stateRule.addrule(self, rulefun, name = '', desc = '')
sk_stateRule.eval(self, idrule, t, ktrig = 0)
sk_stateRule.help()


utilmy\zzarchive\excel.py
-------------------------functions----------------------
get_workbook_name()
double_sum(x, y)
add_one(data)
matrix_mult(x, y)
npdot()



utilmy\zzarchive\fast.py
-------------------------functions----------------------
day(s)
month(s)
year(s)
hour(s)
weekday(s)
season(d)
daytime(d)
fastStrptime(val, format)
drawdown_calc_fast(price)
std(x)
mean(x)
_compute_overlaps(u, v)
distance_jaccard2(u, v)
distance_jaccard(u, v)
distance_jaccard_X(X)
cosine(u, v)
rmse(y, yhat)
cross(vec1, vec2)
norm(vec)
log_exp_sum2(a, b)



utilmy\zzarchive\fast_parallel.py
-------------------------functions----------------------
task_summary(tasks)
task_progress(tasks)
task_find_best(tasks, n_top = 5)
task_parallel_job_01(name, param, datadict)



utilmy\zzarchive\filelock.py
-------------------------methods----------------------
FileLock.__init__(self, protected_file_path, timeout = None, delay = 1, lock_file_contents = None)
FileLock.locked(self)
FileLock.available(self)
FileLock.acquire(self, blocking = True)
FileLock.release(self)
FileLock.__enter__(self)
FileLock.__exit__(self, type, value, traceback)
FileLock.__del__(self)
FileLock.purge(self)


utilmy\zzarchive\function_custom.py
-------------------------functions----------------------
mapping_calc_risk_elvis_v03(ss, tr, t, riskout)
mapping_calc_risk_v02(ss, tr, t, risk0)
mapping_calc_risk_v01(ss, tr, t, risk0)
mapping_risk_ww_v01(risk, wwmat, ww2)
mapping_calc_risk_v00(self, ss, tr, t, risk0)
getweight(ww, size = (9, 3)
fun_obj(vv, ext)



utilmy\zzarchive\geospatial.py


utilmy\zzarchive\global01.py


utilmy\zzarchive\kagglegym.py
-------------------------functions----------------------
r_score(y_true, y_pred, sample_weight = None, multioutput = None)
make()

-------------------------methods----------------------
Observation.__init__(self, train, target, features)
Environment.__init__(self)
Environment.reset(self)
Environment.step(self, target)
Environment.__str__(self)


utilmy\zzarchive\linux.py
-------------------------functions----------------------
load_session(name = 'test_20160815')
save_session(name = '')
isfloat(value)
isint(x)
aa_isanaconda()
aa_cleanmemory()
aa_getmodule_doc(module1, fileout = '')
np_interpolate_nan(y)
and1(x, y, x3 = None, x4 = None, x5 = None, x6 = None, x7 = None, x8 = None)
sortcol(arr, colid, asc = 1)
sort(arr, colid, asc = 1)
np_ma(vv, n)
np_cleanmatrix(m)
np_sortbycolumn(arr, colid, asc = True)
np_sortbycol(arr, colid, asc = True)
np_findfirst(item, vec)
np_find(item, vec)
find(item, vec)
findx(item, vec)
finds(itemlist, vec)
findhigher(x, vec)
findlower(x, vec)
np_find_minpos(values)
np_find_maxpos(values)
np_find_maxpos_2nd(numbers)
np_findlocalmax2(v, trig)
np_findlocalmin2(v, trig)
np_findlocalmax(v, trig)
np_findlocalmin(v, trig)
np_stack(v1, v2 = None, v3 = None, v4 = None, v5 = None)
np_uniquerows(a)
np_remove_zeros(vv, axis1 = 1)
np_sort(arr, colid, asc = True)
np_memory_array_adress(x)
sk_featureimportance(clfrf, feature_name)
sk_showconfusion(clfrf, X_train, Y_train, isprint = True)
sk_tree(Xtrain, Ytrain, nbtree, maxdepth, print1)
sk_gen_ensemble_weight(vv, acclevel, maxlevel = 0.88)
sk_votingpredict(estimators, voting, ww, X_test)
sk_tree_get_ifthen(tree, feature_names, target_names, spacer_base = " ")
pd_array_todataframe(price, symbols = None, date1 = None, dotranspose = False)
pd_date_intersection(qlist)
pd_resetindex(df)
pd_create_colmap_nametoid(df)
pd_dataframe_toarray(df)
pd_changeencoding(data, cols)
pd_createdf(val1, col1 = None, idx1 = None)
pd_insertcolumn(df, colname, vec)
pd_insertrows(df, rowval, index1 = None)
pd_replacevalues(df, matrix)
pd_storeadddf(df, dfname, dbfile='F = 'F:\temp_pandas.h5')
pd_storedumpinfo(dbfile='E = 'E:\_data\stock\intraday_google.h5')
pd_remove_row(df, row_list_index = [23, 45])
pd_extract_col_idx_val(df)
pd_split_col_idx_val(df)
pd_addcolumn(df1, name1 = 'new')
pd_removecolumn(df1, name1)
pd_save_vectopanda(vv, filenameh5)
pd_load_panda2vec(filenameh5, store_id = 'data')
pd_csv_topanda(filein1, filename, tablen = 'data')
pd_getpanda_tonumpy(filename, nsize, tablen = 'data')
pd_getrandom_tonumpy(filename, nbdim, nbsample, tablen = 'data')
sk_cluster_kmeans(x, nbcluster = 5, isplot = True)
datetime_tostring(datelist1)
date_remove_bdays(from_date, add_days)
date_add_bdays(from_date, add_days)
datestring_todatetime(datelist1, format1 =  "%Y%m%d")
datetime_toint(datelist1)
dateint_todatetime(datelist1)
date_diffindays(intdate1, intdate2)
date_finddateid(date1, dateref)
datestring_toint(datelist1)
date_now(i = 0)
date_as_float(dt)
datediff_inyear(startdate, enddate)
date_generatedatetime(start = "20100101", nbday = 10, end = "")
datetime_tostring(datelist1)
datestring_todatetime(datelist1, format1 =  "%Y%m%d")
datestring_toint(datelist1)
numexpr_vect_calc(filename, expr, i0 = 0, imax = 1000, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
numexpr_topanda(filename, expr, i0 = 0, imax = 1000, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
textvect_topanda(vv, fileout = "")
comoment(xx, yy, nsample, kx, ky)
acf(data)
plotsave(xx, yy, title1 = "")
plotshow(xx, yy, title1 = "")
compileVSsolution(dir1, flags1 = "", type1 = "devenv", compilerdir = "")
VS_start(self, version)
VS_build(self, lib_to_build)
set_rc_version(rcfile, target_version)



utilmy\zzarchive\multiprocessfunc.py
-------------------------functions----------------------
multigbm_paralell_func(nbsimul, ww, voldt, drift, upper_cholesky, nbasset, n, price, type1 = 0, strike = 0, cp = 1)
func(val, lock)
multigbm_processfast7(nbsimul, s0, voldt, drift, upper_cholesky, nbasset, n, price)
bm_generator(bm, dt, n, type1)
merge(d2)
integratenp2(its, nchunk)
integratenp(its, nchunk)
integratene(its)
parzen_estimation(x_samples, point_x, h)
init2(d)
init_global1(l, r)
np_sin(value)
ne_sin(x)
res_shared2()
list_append(count, id, out_list)



utilmy\zzarchive\multithread.py
-------------------------functions----------------------
multithread_run(fun_async, input_list:list, n_pool = 5, start_delay = 0.1, verbose = True, **kw)
multithread_run_list(**kwargs)



utilmy\zzarchive\portfolio.py
-------------------------functions----------------------
data_jpsector()
date_earningquater(t1)
date_is_3rdfriday(s)
date_option_expiry(date)
date_find_kday_fromintradaydate(kintraday, intradaydate, dailydate)
date_find_kintraday_fromdate(d1, intradaydate1, h1 = 9, m1 = 30)
date_find_intradateid(datetimelist, stringdate = ['20160420223000'])
datetime_convertzone1_tozone2(tt, fromzone = 'Japan', tozone = 'US/Eastern')
date_extract_dailyopenclosetime(spdateref1, market = 'us')
date_finddateid(date1, dateref)
date_alignfromdateref(array1, dateref)
_date_align(dateref, datei, tmax, closei)
date_align(array1, dateref)
min_withposition(values)
max_withposition(values)
_reshape(x)
_notnone(x)
plot_price(asset, y2 = None, y3 = None, y4 = None, y5 = None, sym = None, savename1 = '', tickperday = 20, date1 = None, graphsize = (10, 5)
plot_priceintraday(data)
plot_check(close, tt0i = 20140102, tt1i = 20160815, dateref = [], sym = [], tickperday = 120)
plot_pricedate(date1, sym1, asset1, sym2 = None, bsk1 = None, verticaldate = None, savename1 = '', graphsize = (10, 5)
generate_sepvertical(asset1, tt, tmax, start = None, datebar = None)
save_asset_tofile(file1, asset1, asset2 = None, asset3 = None, date1 = None, title1 = None)
load_asset_fromfile(file1)
array_todataframe(price, symbols = None, date1 = None)
pd_dataframe_toarray(df)
isfloat(value)
isint(x)
correlation_mat(Xmat, type1 = "robust", type2 = "correl")
correl_reducebytrigger(correl2, trigger)
sk_cov_fromcorrel(correl, ret_close1)
cointegration(x, y)
causality_y1_y2(price2, price1, maxlag)
reg_slope(close, dateref, tlag, type1 = 'elasticv')
rolling_cointegration(x, y)
regression(yreturn, xreturn, type1 = "elasticv")
regression_fixedsymbolstock(sym, ret_close2, tsstart, tsample, ret_spy, spyclose, regonly = True)
regression_getpricefromww(spyclose, ww01, regasset01, ret_close2, tstart, tlag = 1)
regression_allstocks_vs_riskfactors(symstock, pricestock, symriskfac, priceriskfac, nlaglist)
getdiff_fromquotes(close, timelag)
getret_fromquotes(close, timelag = 1)
getlogret_fromquotes(close, timelag = 1)
getprice_fromret(ret, normprice = 100)
price_normalize100(ret, normprice = 100)
price_normalize_1d(ret, normprice = 100, dtype1 =  np.float32)
norm_fast(y, ny)
correl_fast(xn, y, nx)
volhisto_fromret(retbsk, t, volrange, axis = 0)
volhisto_fromprice(price, t, volrange, axis = 0)
volhistorolling_fromprice(price, volrange)
rsk_calc_all_TA(df = 'panda_dataframe')
ta_highbandtrend1(close2, type1 = 0)
pd_transform_asset(q0, q1, type1 = "spread")
calcbasket_table(wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps =  0.000)
folio_createvolta_asset(close, vol = 0.12, volrange = 120, lev = 1.0)
folio_lowcorrelation(sym01, nstock, periodlist, dateref, close1, kbenchmark, badlist, costbppa = 0.02, showgraph = True)
folio_leverageetf(price, lev = 1.0, costpa = 0.0)
folio_inverseetf(price, costpa = 0.0)
folio_longshort_unit(long1, short1, ww = [1, -1], costpa = 0.0, tlag = 1, istable = 1, wwschedule = [])
folio_longshort_unitfixed(long1, short1, nn = [1, -1], costpa = 0.0, tlag = 1, istable = 1)
folio_longshort_pct(long1, short1, ww = [1, -1], costpa = 0.0)
folio_histogram(close)
folio_voltarget(bsk, targetvol = 0.11, volrange =  90, expocap = 1.5)
folio_volta(bsk, targetvol = 0.11, volrange =  90, expocap = 1.5)
folio_volta2(bsk, riskind, par, targetvol = 0.11, volrange =  90, cap = 1.5, floor = 0.0, costbp = 0.0005)
folio_fixedunitprice(price, fixedww, costpa = 0.0)
folio_fixedweightprice(price, fixedww, costpa = 0.0)
folio_fixedweightret(ret, fixedww)
folio_cost_turnover(wwall, bsk, dateref, costbp)
folio_riskpa(ret, targetvol = 0.1, volrange = 90, cap = 1.0)
folio_perfreport_schedule(sym, dateref, close, wwind, t0, scheduleperiod = "1monthend")
folio_concenfactor2(ww, masset = 12)
calcbasket_objext(RETURN, TMAX, riskind_i, wwmat, wwasset0, ww0, nbrange, criteria)
calc_ranktable(close2, symjp1, nlag, refindex, funeval, funargs)
similarity_correl(ret_close2, funargs)
np_similarity(x, y, wwerr = [], type1 = 0)
np_distance_l1(x, y, wwerr)
np_similarity(x, y, wwerr = [], type1 = 0)
np_distance_l1(x, y, wwerr)
get(close, timelag)

-------------------------methods----------------------
index.__init__(self, id1, sym, ww, tstart)
index.close(self)
index.updatehisto(self)
index.help(self)
index.__init__(self, id1, sym, ww, tstart)
index.calc_baskettable_pct(self, type1 = "table", showdetail = 0)
index._wwpct_rebal(self, wwpct_actual, t, trebal)
index._udpate_wwindpct(self, t, bskt, hedgecost, wwpct_actual, wwpct_th)
index.calc_baskettable_unit()
folioCalc.__init__(self, sym, close, dateref)
folioCalc.set_symclose(self, sym, close, dateref)
folioCalc.setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid = "spprice", lfun = None)
folioCalc._weightcalc_generic(self, wwvec, t)
folioCalc._weightcalc_regime(self, wwvec, wwextra, t)
folioCalc._regimecalc(self, t, wwextra)
folioCalc._weightcalc_constant(self, ww2, t)
folioCalc.getweight(self)
folioCalc.calc_baskettable(self, wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps = 0.000, showdetail = 0)
folioCalc.plot(self, wwvec = None, show1 = 1, tickperday = 60)
folioCalc.multiperiod_ww(self, t)
folioCalc.help(self)
folioRiskIndicator.__init__(self, sym, close, dateref)
folioRiskIndicator.set_symclose(self, sym, close, dateref)
folioRiskIndicator.setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid = "spprice", lfun = None)
folioRiskIndicator.calcrisk(self, wwvec = [], initval = 1)
folioRiskIndicator._weightcalc_generic(self, wwvec, t)
folioRiskIndicator._weightcalc_regime(self, wwvec, wwextra, t)
folioRiskIndicator._regimecalc(self, t, wwextra)
folioRiskIndicator.calc_optimal_weight(self, maxiter = 1, name1 = '', isreset = 1, popsize = 15)
folioOptimizationF.__init__(self, sym, close, dateref)
folioOptimizationF.set_symclose(self, sym, close, dateref)
folioOptimizationF.setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid = "spprice", lfun = None)
folioOptimizationF.calcbasket_obj2(self, wwvec)
folioOptimizationF.calcbasket_obj(self, wwvec)
folioOptimizationF._loss_obj(self, ww2, wwpenalty)
folioOptimizationF._weightcalc_generic(self, wwvec, t)
folioOptimizationF._weightcalc_regime(self, wwvec, wwextra, t)
folioOptimizationF._regimecalc(self, t, wwextra)
folioOptimizationF._weightcalc_constant(self, ww2, t)
folioOptimizationF.calc_optimal_weight(self, maxiter = 1, name1 = '', isreset = 1, popsize = 15)
folioOptimizationF.getweight(self)
folioOptimizationF.calc_baskettable(self, wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps =  0.000, showdetail = 0)
folioOptimizationF.plot(self, wwvec = None, show1 = 1, tickperday = 60)
folioOptimizationF._objective_criteria(self, bsk)
folioOptimizationF.multiperiod_ww(self, t)
folioOptimizationF.help(self)
folioOptimizationF.mapping_risk_ww(self, risk, wwmat, ww2 = self.wwasset0)
folioOptimizationF._mapping_calc_risk(self, ss, tr, t, risk0)
searchSimilarity.__init__(self, filejpstock=r'E = r'E:/_data/stock/daily/20160616/jp', sym01 = ['7203'], symname = ['Toyota'], startdate =  20150101, enddate = 20160601, pricetype = "close")
searchSimilarity.load_quotes_fromdb(self, picklefile = '')
searchSimilarity.__generate_return__(self, nlag)
searchSimilarity.__overweight__(self, px)
searchSimilarity.set_searchcriteria(self, name1 = '7203', date1 = 20160301, date2 = 20160601, nlag = 1, searchperiodstart = 20120101, typesearch = "pattern2", )
searchSimilarity.launch_search(self)
searchSimilarity.show_comparison_graph(self, maxresult = 20, show_only_different_time = True, fromid = 0, fromend =  0, filenameout = '')
searchSimilarity.staticmethod(self, x)
searchSimilarity.get_rankresult(self, filetosave = '')
searchSimilarity.export_results()


utilmy\zzarchive\portfolio_withdate.py
-------------------------functions----------------------
data_jpsector()
date_earningquater(t1)
date_is_3rdfriday(s)
date_option_expiry(date)
date_find_kday_fromintradaydate(kintraday, intradaydate, dailydate)
date_find_kintraday_fromdate(d1, intradaydate1, h1 = 9, m1 = 30)
date_find_intradateid(datetimelist, stringdate = ['20160420223000'])
datetime_convertzone1_tozone2(tt, fromzone = 'Japan', tozone = 'US/Eastern')
date_extract_dailyopenclosetime(spdateref1, market = 'us')
datetime_tostring(tt)
datestring_todatetime(datelist1, format1 =  "%Y%m%d")
datetime_todate(tt)
datetime_toint(datelist1)
datetime_tointhour(datelist1)
dateint_tostring(datelist1, format1 = '%b-%y')
dateint_todatetime(datelist1)
datenumpy_todatetime(tt, islocaltime = True)
datetime_tonumpypdate(t, islocaltime = True)
date_diffindays(intdate1, intdate2)
date_finddateid(date1, dateref)
datestring_toint(datelist1)
date_as_float(dt)
date_todatetime(tlist)
date_removetimezone(datelist)
datediff_inyear(startdate, enddate)
date_generatedatetime(start = "20100101", nbday = 10, end = "")
date_add_bdays(from_date, add_days)
date_getspecificdate(datelist, datetype1 = "yearend", outputype1 = "intdate", includelastdate = True, includefirstdate = False, )
date_alignfromdateref(array1, dateref)
_date_align(dateref, datei, tmax, closei)
date_align(array1, dateref)
min_withposition(values)
max_withposition(values)
_reshape(x)
_notnone(x)
plot_price(asset, y2 = None, y3 = None, y4 = None, y5 = None, sym = None, savename1 = '', tickperday = 20, date1 = None, graphsize = (10, 5)
plot_priceintraday(data)
plot_check(close, tt0i = 20140102, tt1i = 20160815, dateref = [], sym = [], tickperday = 120)
plot_pricedate(date1, sym1, asset1, sym2 = None, bsk1 = None, verticaldate = None, savename1 = '', graphsize = (10, 5)
generate_sepvertical(asset1, tt, tmax, start = None, datebar = None)
save_asset_tofile(file1, asset1, asset2 = None, asset3 = None, date1 = None, title1 = None)
load_asset_fromfile(file1)
array_todataframe(price, symbols = None, date1 = None)
pd_dataframe_toarray(df)
isfloat(value)
isint(x)
correlation_mat(Xmat, type1 = "robust", type2 = "correl")
correl_reducebytrigger(correl2, trigger)
sk_cov_fromcorrel(correl, ret_close1)
cointegration(x, y)
causality_y1_y2(price2, price1, maxlag)
reg_slope(close, dateref, tlag, type1 = 'elasticv')
rolling_cointegration(x, y)
regression(yreturn, xreturn, type1 = "elasticv")
regression_fixedsymbolstock(sym, ret_close2, tsstart, tsample, ret_spy, spyclose, regonly = True)
regression_getpricefromww(spyclose, ww01, regasset01, ret_close2, tstart, tlag = 1)
regression_allstocks_vs_riskfactors(symstock, pricestock, symriskfac, priceriskfac, nlaglist)
getdiff_fromquotes(close, timelag)
getret_fromquotes(close, timelag = 1)
getlogret_fromquotes(close, timelag = 1)
getprice_fromret(ret, normprice = 100)
price_normalize100(ret, normprice = 100)
price_normalize_1d(ret, normprice = 100, dtype1 =  np.float32)
norm_fast(y, ny)
correl_fast(xn, y, nx)
volhisto_fromret(retbsk, t, volrange, axis = 0)
volhisto_fromprice(price, t, volrange, axis = 0)
volhistorolling_fromprice(price, volrange)
rsk_calc_all_TA(df = 'panda_dataframe')
ta_highbandtrend1(close2, type1 = 0)
pd_transform_asset(q0, q1, type1 = "spread")
calcbasket_table(wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps =  0.000)
folio_createvolta_asset(close, vol = 0.12, volrange = 120, lev = 1.0)
folio_lowcorrelation(sym01, nstock, periodlist, dateref, close1, kbenchmark, badlist, costbppa = 0.02, showgraph = True)
folio_leverageetf(price, lev = 1.0, costpa = 0.0)
folio_inverseetf(price, costpa = 0.0)
folio_longshort_unit(long1, short1, ww = [1, -1], costpa = 0.0, tlag = 1, istable = 1, wwschedule = [])
folio_longshort_unitfixed(long1, short1, nn = [1, -1], costpa = 0.0, tlag = 1, istable = 1)
folio_longshort_pct(long1, short1, ww = [1, -1], costpa = 0.0)
folio_histogram(close)
folio_voltarget(bsk, targetvol = 0.11, volrange =  90, expocap = 1.5)
folio_volta(bsk, targetvol = 0.11, volrange =  90, expocap = 1.5)
folio_volta2(bsk, riskind, par, targetvol = 0.11, volrange =  90, cap = 1.5, floor = 0.0, costbp = 0.0005)
folio_fixedunitprice(price, fixedww, costpa = 0.0)
folio_fixedweightprice(price, fixedww, costpa = 0.0)
folio_fixedweightret(ret, fixedww)
folio_cost_turnover(wwall, bsk, dateref, costbp)
folio_riskpa(ret, targetvol = 0.1, volrange = 90, cap = 1.0)
folio_perfreport_schedule(sym, dateref, close, wwind, t0, scheduleperiod = "1monthend")
folio_concenfactor2(ww, masset = 12)
calcbasket_objext(RETURN, TMAX, riskind_i, wwmat, wwasset0, ww0, nbrange, criteria)
calc_ranktable(close2, symjp1, nlag, refindex, funeval, funargs)
similarity_correl(ret_close2, funargs)
np_similarity(x, y, wwerr = [], type1 = 0)
np_distance_l1(x, y, wwerr)
np_similarity(x, y, wwerr = [], type1 = 0)
np_distance_l1(x, y, wwerr)
get(close, timelag)

-------------------------methods----------------------
index.__init__(self, id1, sym, ww, tstart)
index.close(self)
index.updatehisto(self)
index.help(self)
index.__init__(self, id1, sym, ww, tstart)
index.calc_baskettable_pct(self, type1 = "table", showdetail = 0)
index._wwpct_rebal(self, wwpct_actual, t, trebal)
index._udpate_wwindpct(self, t, bskt, hedgecost, wwpct_actual, wwpct_th)
index.calc_baskettable_unit()
folioCalc.__init__(self, sym, close, dateref)
folioCalc.set_symclose(self, sym, close, dateref)
folioCalc.setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid = "spprice", lfun = None)
folioCalc._weightcalc_generic(self, wwvec, t)
folioCalc._weightcalc_regime(self, wwvec, wwextra, t)
folioCalc._regimecalc(self, t, wwextra)
folioCalc._weightcalc_constant(self, ww2, t)
folioCalc.getweight(self)
folioCalc.calc_baskettable(self, wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps = 0.000, showdetail = 0)
folioCalc.plot(self, wwvec = None, show1 = 1, tickperday = 60)
folioCalc.multiperiod_ww(self, t)
folioCalc.help(self)
folioRiskIndicator.__init__(self, sym, close, dateref)
folioRiskIndicator.set_symclose(self, sym, close, dateref)
folioRiskIndicator.setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid = "spprice", lfun = None)
folioRiskIndicator.calcrisk(self, wwvec = [], initval = 1)
folioRiskIndicator._weightcalc_generic(self, wwvec, t)
folioRiskIndicator._weightcalc_regime(self, wwvec, wwextra, t)
folioRiskIndicator._regimecalc(self, t, wwextra)
folioRiskIndicator.calc_optimal_weight(self, maxiter = 1, name1 = '', isreset = 1, popsize = 15)
folioOptimizationF.__init__(self, sym, close, dateref)
folioOptimizationF.set_symclose(self, sym, close, dateref)
folioOptimizationF.setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid = "spprice", lfun = None)
folioOptimizationF.calcbasket_obj2(self, wwvec)
folioOptimizationF.calcbasket_obj(self, wwvec)
folioOptimizationF._loss_obj(self, ww2, wwpenalty)
folioOptimizationF._weightcalc_generic(self, wwvec, t)
folioOptimizationF._weightcalc_regime(self, wwvec, wwextra, t)
folioOptimizationF._regimecalc(self, t, wwextra)
folioOptimizationF._weightcalc_constant(self, ww2, t)
folioOptimizationF.calc_optimal_weight(self, maxiter = 1, name1 = '', isreset = 1, popsize = 15)
folioOptimizationF.getweight(self)
folioOptimizationF.calc_baskettable(self, wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps =  0.000, showdetail = 0)
folioOptimizationF.plot(self, wwvec = None, show1 = 1, tickperday = 60)
folioOptimizationF._objective_criteria(self, bsk)
folioOptimizationF.multiperiod_ww(self, t)
folioOptimizationF.help(self)
folioOptimizationF.mapping_risk_ww(self, risk, wwmat, ww2 = self.wwasset0)
folioOptimizationF._mapping_calc_risk(self, ss, tr, t, risk0)
searchSimilarity.__init__(self, filejpstock=r'E = r'E:/_data/stock/daily/20160616/jp', sym01 = ['7203'], symname = ['Toyota'], startdate =  20150101, enddate = 20160601, pricetype = "close")
searchSimilarity.load_quotes_fromdb(self, picklefile = '')
searchSimilarity.__generate_return__(self, nlag)
searchSimilarity.__overweight__(self, px)
searchSimilarity.set_searchcriteria(self, name1 = '7203', date1 = 20160301, date2 = 20160601, nlag = 1, searchperiodstart = 20120101, typesearch = "pattern2", )
searchSimilarity.launch_search(self)
searchSimilarity.show_comparison_graph(self, maxresult = 20, show_only_different_time = True, fromid = 0, fromend =  0, filenameout = '')
searchSimilarity.staticmethod(self, x)
searchSimilarity.get_rankresult(self, filetosave = '')
searchSimilarity.export_results()


utilmy\zzarchive\report.py
-------------------------functions----------------------
map_show()
xl_create_pivot(infile, index_list = ["Manager", "Rep", "Product"], value_list = ["Price", "Quantity"])
xl_save_report(report, outfile)
xl_create_pdf()



utilmy\zzarchive\rstatpy.py
-------------------------functions----------------------
stl(data, ns, np = None, nt = None, nl = None, isdeg = 0, itdeg = 1, ildeg = 1, nsjump = None, ntjump = None, nljump = None, ni = 2, no = 0, fulloutput = False)



utilmy\zzarchive\util.py
-------------------------functions----------------------
session_load_function(name = 'test_20160815')
session_save_function(name = 'test')
py_save_obj_dill(obj1, keyname = '', otherfolder = 0)
session_spyder_showall()
session_guispyder_save(filename)
session_guispyder_load(filename)
session_load(name = 'test_20160815')
session_save(name = 'test')
aa_unicode_ascii_utf8_issue()
isexist(a)
isfloat(x)
isint(x)
a_isanaconda()
a_run_ipython(cmd1)
a_autoreload()
a_start_log(id1 = '', folder = 'aaserialize/log/')
a_cleanmemory()
a_module_codesample(module_str = 'pandas')
a_module_doc(module_str = 'pandas')
a_module_generatedoc(module_str = "pandas", fileout = '')
a_info_conda_jupyter()
os_zipextractall(filezip_or_dir = "folder1/*.zip", tofolderextract = 'zdisk/test', isprint = 1)
os_folder_copy(src, dst, symlinks = False, pattern1 = "*.py", fun_file_toignore = None)
os_folder_create(directory)
os_folder_robocopy(from_folder = '', to_folder = '', my_log='H = 'H:/robocopy_log.txt')
os_file_replace(source_file_path, pattern, substring)
os_file_replacestring1(findStr, repStr, filePath)
os_file_replacestring2(findstr, replacestr, some_dir, pattern = "*.*", dirlevel = 1)
os_file_getname(path)
os_file_getpath(path)
os_file_gettext(file1)
os_file_listall(dir1, pattern = "*.*", dirlevel = 1, onlyfolder = 0)
os_file_rename(some_dir, pattern = "*.*", pattern2 = "", dirlevel = 1)
os_gui_popup_show(txt)
os_print_tofile(vv, file1, mode1 = 'a')
os_path_norm(pth)
os_path_change(path1)
os_path_current()
os_file_exist(file1)
os_file_size(file1)
os_file_read(file1)
os_file_isame(file1, file2)
os_file_get_file_extension(file_path)
os_file_normpath(path)
os_folder_is_path(path_or_stream)
os_file_get_path_from_stream(maybe_stream)
os_file_try_to_get_extension(path_or_strm)
os_file_are_same_file_types(paths)
os_file_norm_paths(paths, marker = '*')
os_file_mergeall(nfile, dir1, pattern1, deepness = 2)
os_file_extracttext(output_file, dir1, pattern1 = "*.html", htmltag = 'p', deepness = 2)
os_path_append(p1, p2 = None, p3 = None, p4 = None)
os_wait_cpu(priority = 300, cpu_min = 50)
os_split_dir_file(dirfile)
os_process_run(cmd_list = ['program', 'arg1', 'arg2'], capture_output = False)
os_process_2()
py_importfromfile(modulename, dir1)
py_memorysize(o, ids, hint = " deep_getsizeof(df_pd, set()
save(obj, folder = '/folder1/keyname', isabsolutpath = 0)
load(folder = '/folder1/keyname', isabsolutpath = 0)
save_test(folder = '/folder1/keyname', isabsolutpath = 0)
py_save_obj(obj1, keyname = '', otherfolder = 0)
py_load_obj(folder = '/folder1/keyname', isabsolutpath = 0, encoding1 = 'utf-8')
z_key_splitinto_dir_name(keyname)
sql_getdate()
obj_getclass_of_method(meth)
obj_getclass_property(pfi)
print_topdf()
os_config_setfile(dict_params, outfile, mode1 = 'w+')
os_config_getfile(file1)
os_csv_process(file1)
read_funding_data(path)
read_funding_data(path)
read_funding_data(path)
str_match_fuzzy(xstring, list_string)
str_parse_stringcalendar(cal)
str_make_unicode(input, errors = 'replace')
str_empty_string_array(x, y = 1)
str_empty_string_array_numpy(nx, ny = 1)
str_isfloat(value)
str_is_azchar(x)
str_is_az09char(x)
str_reindent(s, numSpaces)
str_split2(delimiters, string, maxsplit = 0)
str_split_pattern(sep2, ll, maxsplit = 0)
pd_str_isascii(x)
str_to_utf8(x)
str_to_unicode(x, encoding = 'utf-8')
np_minimize(fun_obj, x0 = [0.0], argext = (0, 0)
np_minimizeDE(fun_obj, bounds, name1, maxiter = 10, popsize = 5, solver = None)
np_remove_NA_INF_2d(X)
np_addcolumn(arr, nbcol)
np_addrow(arr, nbrow)
np_int_tostr(i)
np_dictordered_create()
np_list_unique(seq)
np_list_tofreqdict(l1, wweight = [])
np_list_flatten(seq)
np_dict_tolist(dd, withkey = 0)
np_dict_tostr_val(dd)
np_dict_tostr_key(dd)
np_removelist(x0, xremove = [])
np_transform2d_int_1d(m2d, onlyhalf = False)
np_mergelist(x0, x1)
np_enumerate2(vec_1d)
np_pivottable_count(mylist)
np_nan_helper(y)
np_interpolate_nan(y)
np_and1(x, y, x3 = None, x4 = None, x5 = None, x6 = None, x7 = None, x8 = None)
np_sortcol(arr, colid, asc = 1)
np_sort(arr, colid, asc = 1)
np_ma(vv, n)
np_cleanmatrix(m)
np_torecarray(arr, colname)
np_sortbycolumn(arr, colid, asc = True)
np_sortbycol(arr, colid, asc = True)
min_kpos(arr, kth)
max_kpos(arr, kth)
np_findfirst(item, vec)
np_find(item, vec)
find(item, vec)
findnone(vec)
findx(item, vec)
finds(itemlist, vec)
findhigher(x, vec)
findlower(x, vec)
np_find_minpos(values)
np_find_maxpos(values)
np_find_maxpos_2nd(numbers)
np_findlocalmax2(v, trig)
np_findlocalmin2(v, trig)
np_findlocalmax(v, trig)
np_findlocalmin(v, trig)
np_stack(v1, v2 = None, v3 = None, v4 = None, v5 = None)
np_uniquerows(a)
np_remove_zeros(vv, axis1 = 1)
np_sort(arr, colid, asc = 1)
np_memory_array_adress(x)
np_pivotable_create(table, left, top, value)
pd_info(df, doreturn = 1)
pd_info_memsize(df, memusage = 0)
pd_row_findlast(df, colid = 0, emptyrowid = None)
pd_row_select(df, **conditions)
pd_csv_randomread(filename, nsample = 10000, filemaxline = -1, dtype = None)
pd_array_todataframe(array, colname = None, index1 = None, dotranspose = False)
pd_dataframe_toarray(df)
pd_createdf(array1, col1 = None, idx1 = None)
pd_create_colmapdict_nametoint(df)
pd_extract_col_idx_val(df)
pd_extract_col_uniquevalue_tocsv(df, colname = '', csvfile = '')
pd_split_col_idx_val(df)
pd_splitdf_inlist(df, colid, type1 = "dict")
pd_find(df, regex_pattern = '*', col_restrict = [], isnumeric = False, doreturnposition = False)
pd_dtypes_totype2(df, columns = [], targetype = 'category')
pd_dtypes(df, columns = [], targetype = 'category')
pd_df_todict2(df1, colkey = 'table', excludekey = [''], onlyfirstelt =  True)
pd_df_todict(df1, colkey = 'table', excludekey = [''], onlyfirstelt =  True)
pd_col_addfrom_dfmap(df, dfmap, colkey, colval, df_colused, df_colnew, exceptval = -1, inplace =  True)
pd_applyfun_col(df, newcol, ff, use_colname = "all/[colname]")
pd_date_intersection(qlist)
pd_is_categorical(z)
pd_str_encoding_change(df, cols, fromenc = 'iso-8859-1', toenc = 'utf-8')
pd_str_unicode_tostr(df, targetype = str)
pd_dtypes_type1_totype2(df, fromtype = str, targetype = str)
pd_resetindex(df)
pd_insertdatecol(df_insider, format1="%Y-%m-%d %H = "%Y-%m-%d %H:%M:%S:%f")
pd_replacevalues(df, matrix)
pd_removerow(df, row_list_index = [23, 45])
pd_removecol(df1, name1)
pd_insertrow(df, rowval, index1 = None, isreset = 1)
pd_h5_cleanbeforesave(df)
pd_h5_addtable(df, tablename, dbfile='F = 'F:\temp_pandas.h5')
pd_h5_tableinfo(filenameh5, table)
pd_h5_dumpinfo(dbfile='E = 'E:\_data\stock\intraday_google.h5')
pd_h5_save(df, filenameh5='E = 'E:/_data/_data_outlier.h5', key = 'data')
pd_h5_load(filenameh5='E = 'E:/_data/_data_outlier.h5', table_id = 'data', exportype = "pandas", rowstart = -1, rowend = -1, cols = [])
pd_h5_fromcsv_tohdfs(dircsv = 'dir1/dir2/', filepattern = '*.csv', tofilehdfs = 'file1.h5', tablename = 'df', col_category = [], dtype0 = None, encoding = 'utf-8', chunksize =  2000000, mode = 'a', format = 'table', complib = None)
pd_np_toh5file(numpyarr, fileout = "file.h5", table1 = 'data')
date_allinfo()
date_convert(t1, fromtype, totype)
datetime_tostring(datelist1)
date_remove_bdays(from_date, add_days)
date_add_bdays(from_date, add_days)
datenumpy_todatetime(tt, islocaltime = True)
datetime_tonumpydate(t, islocaltime = True)
datestring_todatetime(datelist1, format1 =  "%Y%m%d")
datetime_toint(datelist1)
date_holiday()
date_add_bday(from_date, add_days)
dateint_todatetime(datelist1)
date_diffinday(intdate1, intdate2)
date_diffinyear(startdate, enddate)
date_diffinbday(intd2, intd1)
date_gencalendar(start = '2010-01-01', end = '2010-01-15', country = 'us')
date_finddateid(date1, dateref)
datestring_toint(datelist1)
date_now(i = 0)
date_nowtime(type1 = 'str', format1= "%Y-%m-%d %H =  "%Y-%m-%d %H:%M:%S:%f")
date_tofloat(dt)
date_generatedatetime(start = "20100101", nbday = 10, end = "")
np_numexpr_vec_calc(filename, expr, i0 = 0, imax = 1000, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
np_numexpr_tohdfs(filename, expr, i0 = 0, imax = 1000, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
np_comoment(xx, yy, nsample, kx, ky)
np_acf(data)
plot_XY(xx, yy, zcolor = None, tsize = None, title1 = '', xlabel = '', ylabel = '', figsize = (8, 6)
plot_heatmap(frame, ax = None, cmap = None, vmin = None, vmax = None, interpolation = 'nearest')
np_map_dict_to_bq_schema(source_dict, schema, dest_dict)
googledrive_get()
googledrive_put()
googledrive_list()
os_processify_fun(func)
ztest_processify()
date_getspecificdate(datelist, datetype1 = "yearend", outputype1 = "intdate", includelastdate = True, includefirstdate = False, )
py_exception_print()
py_log_write(LOGFILE, prefix)

-------------------------methods----------------------
testclass.__init__(self, x)
testclass.z_autotest(self)
FundingRecord.parse(klass, row)
FundingRecord.__str__(self)


utilmy\zzarchive\utilgeo.py
-------------------------functions----------------------
df_to_geojson(df, col_properties, lat = 'latitude', lon = 'longitude')



utilmy\zzarchive\util_aws.py
-------------------------functions----------------------
aws_credentials(account = None)
aws_ec2_get_instanceid(con, ip_address)
aws_ec2_allocate_elastic_ip(con, instance_id = "", elastic_ip = '', region = "ap-northeast-2")
aws_ec2_printinfo(instance = None, ipadress = "", instance_id = "")
aws_ec2_spot_start(con, region, key_name = "ecsInstanceRole", inst_type = "cx2.2", ami_id = "", pricemax = 0.15, elastic_ip = '', pars= {"security_group" =  {"security_group": [""], "disk_size": 25, "disk_type": "ssd", "volume_type": "gp2"})
aws_ec2_get_id(ipadress = '', instance_id = '')
aws_ec2_spot_stop(con, ipadress = "", instance_id = "")
aws_ec2_res_start(con, region, key_name, ami_id, inst_type = "cx2.2", min_count  = 1, max_count  = 1, pars= {"security_group" =  {"security_group": [""], "disk_size": 25, "disk_type": "ssd", "volume_type": "gp2"})
aws_ec2_res_stop(con, ipadress = "", instance_id = "")
aws_accesskey_get(access = '', key = '')
aws_conn_do(action = '', region = "ap-northeast-2")
aws_conn_getallregions(conn = None)
aws_conn_create(region = "ap-northeast-2", access = '', key = '')
aws_conn_getinfo(conn)
aws_s3_url_split(url)
aws_s3_getbucketconn(s3dir)
aws_s3_puto_s3(fromdir_file = 'dir/file.zip', todir = 'bucket/folder1/folder2')
aws_s3_getfrom_s3(froms3dir = 'task01/', todir = '', bucket_name = 'zdisk')
aws_s3_folder_printtall(bucket_name = 'zdisk')
aws_s3_file_read(bucket1, filepath, isbinary = 1)
aws_ec2_cmd_ssh(cmdlist =   ["ls " ], host = 'ip', doreturn = 0, ssh = None, username = 'ubuntu', keyfilepath = '')
aws_ec2_python_script(script_path, args1, host)
aws_ec2_create_con(contype = 'sftp/ssh', host = 'ip', port = 22, username = 'ubuntu', keyfilepath = '', password = '', keyfiletype = 'RSA', isprint = 1)
ztest_01()

-------------------------methods----------------------
aws_ec2_ssh.__init__(self, hostname, username = 'ubuntu', key_file = None, password = None)
aws_ec2_ssh.command(self, cmd)
aws_ec2_ssh.put(self, localfile, remotefile)
aws_ec2_ssh.put_all(self, localpath, remotepath)
aws_ec2_ssh.get(self, remotefile, localfile)
aws_ec2_ssh.sftp_walk(self, remotepath)
aws_ec2_ssh.get_all(self, remotepath, localpath)
aws_ec2_ssh.write_command(self, text, remotefile)
aws_ec2_ssh.python_script(self, script_path, args1)
aws_ec2_ssh.command_list(self, cmdlist)
aws_ec2_ssh.listdir(self, remotedir)
aws_ec2_ssh.jupyter_kill(self)
aws_ec2_ssh.jupyter_start(self)
aws_ec2_ssh.cmd2(self, cmd1)
aws_ec2_ssh._help_ssh(self)


utilmy\zzarchive\util_min.py
-------------------------functions----------------------
os_wait_cpu(priority = 300, cpu_min = 50)
isexist(a)
isfloat(x)
isint(x)
a_isanaconda()
os_zip_checkintegrity(filezip1)
os_zipfile(folderin, folderzipname, iscompress = True)
os_zipfolder(dir_tozip = '/zdisks3/output', zipname = '/zdisk3/output.zip', dir_prefix = None, iscompress = True)
os_zipextractall(filezip_or_dir = "folder1/*.zip", tofolderextract = 'zdisk/test', isprint = 1)
os_folder_copy(src, dst, symlinks = False, pattern1 = "*.py", fun_file_toignore = None)
os_folder_create(directory)
os_folder_robocopy(from_folder = '', to_folder = '', my_log='H = 'H:/robocopy_log.txt')
os_file_replacestring1(findStr, repStr, filePath)
os_file_replacestring2(findstr, replacestr, some_dir, pattern = "*.*", dirlevel = 1)
os_file_getname(path)
os_file_getpath(path)
os_file_gettext(file1)
os_file_listall(dir1, pattern = "*.*", dirlevel = 1, onlyfolder = 0)
os_file_rename(some_dir, pattern = "*.*", pattern2 = "", dirlevel = 1)
os_print_tofile(vv, file1, mode1 = 'a')
a_get_pythonversion()
os_path_norm(pth)
os_path_change(path1)
os_path_current()
os_file_exist(file1)
os_file_size(file1)
os_file_read(file1)
os_file_mergeall(nfile, dir1, pattern1, deepness = 2)
os_path_append(p1, p2 = None, p3 = None, p4 = None)
os_split_dir_file(dirfile)
py_importfromfile(modulename, dir1)
py_memorysize(o, ids, hint = " deep_getsizeof(df_pd, set()
save(obj, folder = '/folder1/keyname', isabsolutpath = 0)
load(folder = '/folder1/keyname', isabsolutpath = 0)
save_test(folder = '/folder1/keyname', isabsolutpath = 0)
py_save_obj(obj, folder = '/folder1/keyname', isabsolutpath = 0)
py_load_obj(folder = '/folder1/keyname', isabsolutpath = 0, encoding1 = 'utf-8')
z_key_splitinto_dir_name(keyname)



utilmy\zzarchive\util_ml.py
-------------------------functions----------------------
create_weight_variable(name, shape)
create_bias_variable(name, shape)
create_adam_optimizer(learning_rate, momentum)
tf_check()
parse_args(ppa = None, args =  {})
parse_args2(ppa = None)
tf_global_variables_initializer(sess = None)
visualize_result()

-------------------------methods----------------------
TextLoader.__init__(self, data_dir, batch_size, seq_length)
TextLoader.preprocess(self, input_file, vocab_file, tensor_file)
TextLoader.load_preprocessed(self, vocab_file, tensor_file)
TextLoader.create_batches(self)
TextLoader.next_batch(self)
TextLoader.reset_batch_pointer(self)


utilmy\zzarchive\util_sql.py
-------------------------functions----------------------
sql_create_dbengine(type1 = '', dbname = '', login = '', password = '', url = 'localhost', port = 5432)
sql_query(sqlr = 'SELECT ticker,shortratio,sector1_id, FROM stockfundamental', dbengine = None, output = 'df', dburl='sqlite = 'sqlite:///aaserialize/store/finviz.db')
sql_get_dbschema(dburl='sqlite = 'sqlite:///aapackage/store/yahoo.db', dbengine = None, isprint = 0)
sql_delete_table(name, dbengine)
sql_insert_excel(file1 = '.xls', dbengine = None, dbtype = '')
sql_insert_df(df, dbtable, dbengine, col_drop = ['id'], verbose = 1)
sql_insert_csv(csvfile, dbtable, dbengine, col_drop = [])
sql_insert_csv2(csvfile = '', dbtable = '', columns = [], dbengine = None, nrows =  10000)
sql_postgres_create_table(mytable = '', database = '', username = '', password = '')
sql_postgres_query_to_csv(sqlr = 'SELECT ticker,shortratio,sector1_id, FROM stockfundamental', csv_out = '')
sql_postgres_pivot()
sql_mysql_insert_excel()
sql_pivotable(dbcon, ss = 'select  ')



utilmy\zzarchive\util_web.py
-------------------------functions----------------------
web_restapi_toresp(apiurl1)
web_getrawhtml(url1)
web_importio_todataframe(apiurl1, isurl = 1)
web_getjson_fromurl(url)
web_gettext_fromurl(url, htmltag = 'p')
web_gettext_fromhtml(file1, htmltag = 'p')
web_getlink_fromurl(url)
web_send_email(FROM, recipient, subject, body, login1 = "mizenjapan@gmail.com", pss1 = "sophieelise237", server1 = "smtp.gmail.com", port1 = 465)
web_send_email_tls(FROM, recipient, subject, body, login1 = "mizenjapan@gmail.com", pss1 = "sophieelise237", server1 = "smtp.gmail.com", port1 = 465)
web_sendurl(url1)



utilmy\zzarchive\_HELP.py
-------------------------functions----------------------
os_compileVSsolution(dir1, flags1 = "", type1 = "devenv", compilerdir = "")
os_VS_build(self, lib_to_build)
set_rc_version(rcfile, target_version)
os_VS_start(self, version)
fun_cython(a)
fun_python(a)



utilmy\zzarchive\__init__.py


utilmy\zzml\setup.py


utilmy\zzml\versioneer.py
-------------------------functions----------------------
get_root()
get_config_from_root(root)
register_vcs_handler(vcs, method)
run_command(commands, args, cwd = None, verbose = False, hide_stderr = False, env = None)
get_keywords()
get_config(root)
register_vcs_handler(vcs, method)
run_command(commands, args, cwd = None, verbose = False, hide_stderr = False, env = None)
versions_from_parentdir(parentdir_prefix, root, verbose)
git_get_keywords(versionfile_abs)
git_versions_from_keywords(keywords, tag_prefix, verbose)
git_pieces_from_vcs(tag_prefix, root, verbose, run_command = run_command)
plus_or_dot(pieces)
render_pep440(pieces)
render_pep440_pre(pieces)
render_pep440_post(pieces)
render_pep440_old(pieces)
render_git_describe(pieces)
render_git_describe_long(pieces)
render(pieces)
get_versions()
git_get_keywords(versionfile_abs)
git_versions_from_keywords(keywords, tag_prefix, verbose)
git_pieces_from_vcs(tag_prefix, root, verbose, run_command = run_command)
do_vcs_install(manifest_in, versionfile_source, ipy)
versions_from_parentdir(parentdir_prefix, root, verbose)
get_versions()
versions_from_file(filename)
write_to_version_file(filename, versions)
plus_or_dot(pieces)
render_pep440(pieces)
render_pep440_pre(pieces)
render_pep440_post(pieces)
render_pep440_old(pieces)
render_git_describe(pieces)
render_git_describe_long(pieces)
render(pieces)
get_versions()
get_version()
get_cmdclass()
do_setup()
scan_setup_py()



utilmy\codeparser\project_graph\setup.py


utilmy\codeparser\project_graph\test_script.py
-------------------------functions----------------------
sleep_one_seconds()
sleep_two_seconds()



utilmy\codeparser\project_graph\test_script_argparse.py
-------------------------functions----------------------
sleep_one_seconds()
sleep_two_seconds()
foo()



utilmy\db\qdrant\dbvector.py
-------------------------methods----------------------
Client.__init__(self, host  =  'localhost', port  =  6333, table = 'default')
Client.connect(self, table)
Client.table_create(self, table, vector_size = 768)
Client.table_view(self, table)
Client.get_multi(self, vect_list, query_filter = None, topk = 5)


utilmy\db\qdrant\qdrant_example.py
-------------------------functions----------------------
get_data(filename = "startups.json")
main()
search_startup(q: str)

-------------------------methods----------------------
NeuralSearcher.__init__(self, collection_name)
NeuralSearcher.search(self, text: str)


utilmy\db\qdrant\test.py
-------------------------functions----------------------
get_data(filename = "startups.json")
main()



utilmy\db\qdrant\triplet.py
-------------------------functions----------------------
new_algo(df)



utilmy\deeplearning\autoencoder\keras_ae.py


utilmy\deeplearning\keras\loss_graph.py
-------------------------functions----------------------
help()
test_graph_loss()
test_adversarial()
create_fake_neighbor(x, max_neighbors)
map_func(x_batch, y_batch, neighbors, neighbor_weights)
create_graph_loss(max_neighbors = 2)
train_step(x, y, model, loss_fn, optimizer, nbr_features_layer = None, ### Graphregularizer = None, ## Graph) as tape_w)
test_step(x, y, model, loss_fn, nbr_features_layer = None, ### Graphregularizer = None, #### Graph)



utilmy\deeplearning\keras\loss_vq_vae2.py
-------------------------functions----------------------
encoder_Base(latent_dim)
get_vqvae_layer_hierarchical(latent_dim = 16, num_embeddings = 64)
plot_original_reconst_img(orig, rec)

-------------------------methods----------------------
Quantizer.__init__(self, number_of_embeddings, embedding_dimensions, beta = 0.25, **kwargs)
Quantizer.call(self, x)
Quantizer.get_code_indices(self, flattened_inputs)
VQ_VAE_Trainer_2.__init__(self, train_variance, latent_dim = 16, number_of_embeddings = 128, **kwargs)
VQ_VAE_Trainer_2.metrics(self)
VQ_VAE_Trainer_2.train_step(self, x)
PixelConvLayer.__init__(self, mask_type, **kwargs)
PixelConvLayer.build(self, input_shape)
PixelConvLayer.call(self, inputs)
ResidualBlock.__init__(self, filters, **kwargs)
ResidualBlock.call(self, inputs)


utilmy\deeplearning\keras\template_train.py
-------------------------functions----------------------
param_set()
params_set2()
pd_get_dummies(df, cols_cat, cat_dict:dict, only_onehot = True)
label_get_data()
train_step(x, model, y_label_list = None)
validation_step(x, model, y_label_list = None)



utilmy\deeplearning\keras\train_graph_loss.py
-------------------------functions----------------------
log(*s)
metric_accuracy(y_val, y_pred_head, class_dict)
metric_accuracy_2(y_test, y_pred, dd)
cal_loss_macro_soft_f1(y, y_hat)
plot_original_images(test_sample)
plot_reconstructed_images(model, test_sample)
valid_image_check(img_list, path = "", tag = "", y_labels = "", n_sample = 3, renorm = True)
save_best(model, model_dir2, valid_loss, best_loss, counter)
save_model_state(model, model_dir2)
train_stop(counter, patience)
make_encoder(n_outputs = 1)
make_classifier(class_dict)
plot_grid(images, title = '')
visualize_imgs(img_list, path, tag, y_labels, n_sample = None)
train_step(x, model, y_label_list = None)
train_step_2(x, model, y_label_list = None)
validation_step(x, model)
train_step(x, model, y_label_list = None)
validation_step(x, model)
make_decoder()
validation_step(x, model)
train_step(x, model, y_label_list = None)
test_step(x, y, model, loss_fn)

-------------------------methods----------------------
GraphDataGenerator.__init__(self, data_iter, graph_dict)
GraphDataGenerator.__len__(self)
GraphDataGenerator.__getitem__(self, idx)
GraphDataGenerator._map_func(self, index, x_batch, *y_batch)
LearningRateDecay.plot(self, epochs, title = "Learning Rate Schedule")
StepDecay.__init__(self, init_lr = 0.01, factor = 0.25, drop_every = 10)
StepDecay.__call__(self, epoch)
PolynomialDecay.__init__(self, max_epochs = 100, init_lr = 0.01, power = 1.0)
PolynomialDecay.__call__(self, epoch)


utilmy\deeplearning\keras\train_vqvae_loss.py
-------------------------functions----------------------
print_log(*s)
encoder_base(input_shape, latent_dim)
decoder_base(latent_dim, shape)
make_vqvae_encoder(input_shape, latent_dim)
make_vqvae_decoder(input_shape, latent_dim)
make_vqvae_classifier(class_dict)
make_encoder(n_outputs = 1)
perceptual_loss_function(x, x_recon, z_mean, z_logsigma, kl_weight = 0.00005, y_label_heads = None, y_pred_heads = None, clf_loss_fn = None)
apply_func(s, values)
plot_grid(images, title = '')
visualize_imgs(img_list, path, tag, y_labels, n_sample = None)
train_step(x, model, train_variance, y_label_list = None)
validation_step(x, model, train_variance)
metric_accuracy(y_val, y_pred_head, class_dict)
train_step(x, model, train_variance, y_label_list = None)
make_decoder()
clf_loss_crossentropy(y_true, y_pred)
validation_step(x, model, train_variance)
custom_loss(y_true, y_pred)
build_model_2(input_shape, num_classes)

-------------------------methods----------------------
Quantizer.__init__(self, number_of_embeddings, embedding_dimensions, beta = 0.25, **kwargs)
Quantizer.call(self, x)
Quantizer.get_code_indices(self, flattened_inputs)
VQ_VAE.__init__(self, latent_dim, class_dict, num_embeddings = 64, image_size = 64)
VQ_VAE.encode(self, x)
VQ_VAE.reparameterize(self, z_mean, z_logsigma)
VQ_VAE.decode(self, encoder_A_outputs, encoder_B_outputs, apply_sigmoid = False)
VQ_VAE.call(self, x, training = True, mask = None)
VQ_VAE.__init__(self, latent_dim, class_dict, num_embeddings = 64, image_size = 64)
VQ_VAE.encode(self, x)
VQ_VAE.reparameterize(self, z_mean, z_logsigma)
VQ_VAE.decode(self, encoder_A_outputs, encoder_B_outputs, apply_sigmoid = False)
VQ_VAE.call(self, x, training = True, mask = None)
CustomDataGenerator0.__init__(self, image_dir, label_path, class_dict, split = 'train', batch_size = 8, transforms = None)
CustomDataGenerator0.on_epoch_end(self)
CustomDataGenerator0.__len__(self)
CustomDataGenerator0.__getitem__(self, idx)
LearningRateDecay.plot(self, epochs, title = "Learning Rate Schedule")
StepDecay.__init__(self, init_lr = 0.01, factor = 0.25, drop_every = 10)
StepDecay.__call__(self, epoch)
PolynomialDecay.__init__(self, max_epochs = 100, init_lr = 0.01, power = 1.0)
PolynomialDecay.__call__(self, epoch)
SprinklesTransform.__init__(self, num_holes = 100, side_length = 10, always_apply = False, p = 1.0)
SprinklesTransform.apply(self, image, **params)
CustomDataGenerator.__init__(self, x, y, batch_size = 32, augmentations = None)
CustomDataGenerator.__len__(self)
CustomDataGenerator.__getitem__(self, idx)


utilmy\deeplearning\keras\util_dataloader_img.py
-------------------------functions----------------------
help()
test()
test1()
test2()
get_data_sample(batch_size, x_train, labels_val, labels_col)
pd_get_onehot_dict(df, labels_col:list, dfref = None, )
pd_merge_imgdir_onehotfeat(dfref, img_dir = "*.jpg", labels_col  =  [])
pd_to_onehot(dfref, labels_col  =  [])
_byte_feature(value)
_int64_feature(value)
_float_feature(value)
build_tfrecord(x, tfrecord_out_path, max_records)

-------------------------methods----------------------
DataGenerator_img.__init__(self, x, y, batch_size = 32, augmentations = None)
DataGenerator_img.__len__(self)
DataGenerator_img.__getitem__(self, idx)
DataGenerator_img_disk.__init__(self, img_dir, label_path, class_list, split = 'train', batch_size = 8, transforms = None)
DataGenerator_img_disk.on_epoch_end(self)
DataGenerator_img_disk.__len__(self)
DataGenerator_img_disk.__getitem__(self, idx)
SprinklesTransform.__init__(self, num_holes = 30, side_length = 5, always_apply = False, p = 1.0)
SprinklesTransform.apply(self, image, **params)
DataGenerator_img_disk2.__init__(self, image_dir, label_path, class_dict, split = 'train', batch_size = 8, transforms = None, shuffle = True)
DataGenerator_img_disk2._load_data(self, label_path)
DataGenerator_img_disk2.on_epoch_end(self)
DataGenerator_img_disk2.__len__(self)
DataGenerator_img_disk2.__getitem__(self, idx)


utilmy\deeplearning\keras\util_layers.py
-------------------------functions----------------------
help()
test_all()
test_resnetlayer()
make_classifier_multihead(label_name_ncount:dict = None, layers_dim = [128, 1024], tag = '1', cdim = 3, n_filters = 3)

-------------------------methods----------------------
DepthConvBlock.__init__(self, filters)
DepthConvBlock.call(self, inputs)
CNNBlock.__init__(self, filters, kernels, strides = 1, padding = 'valid', activation = None)
CNNBlock.call(self, input_tensor, training = True)
ResBlock.__init__(self, filters, kernels)
ResBlock.call(self, input_tensor, training = False)


utilmy\deeplearning\keras\util_loss.py
-------------------------functions----------------------
test_all()
test_loss1()
loss_clf_macro_soft_f1(y, y_hat)
loss_perceptual_function(x, x_recon, z_mean, z_logsigma, kl_weight = 0.00005, y_label_heads = None, y_pred_heads = None, clf_loss_fn = None, epoch = 1, percep_model = None, cc:dict = None)
loss_vae(x, output, z_logsigma, z_mean)
learning_rate_schedule_custom(mode = "step", epoch = 1, cc = None)
loss_schedule_custom(mode = "step", epoch = 1)

-------------------------methods----------------------
LearningRateDecay.plot(self, epochs, title = "Learning Rate Schedule", path = None)
StepDecay.__init__(self, init_lr = 0.01, factor = 0.25, drop_every = 5)
StepDecay.__call__(self, epoch)


utilmy\deeplearning\keras\util_models.py
-------------------------functions----------------------
test_all()
test_classactivation()
get_final_image(file_path, model_path, target_size)
make_efficientet(xdim, ydim, cdim)
test_DFC_VAE()
make_encoder(xdim = 256, ydim = 256, latent_dim = 10)
make_decoder(xdim, ydim, latent_dim)
make_classifier(class_dict, latent_dim = 10)
make_classifier_2(latent_dim, class_dict)

-------------------------methods----------------------
GradCAM.__init__(self, model, classIdx, layerName = None)
GradCAM.find_target_layer(self)
GradCAM.compute_heatmap(self, image, eps = 1e-8)
GradCAM.overlay_heatmap(self, heatmap, image, alpha = 0.5, colormap = cv2.COLORMAP_JET)
DFC_VAE.__init__(self, latent_dim, class_dict)
DFC_VAE.encode(self, x)
DFC_VAE.reparameterize(self, z_mean, z_logsigma)
DFC_VAE.decode(self, z, apply_sigmoid = False)
DFC_VAE.call(self, x, training = True, mask = None, y_label_list = None)


utilmy\deeplearning\keras\util_similarity.py


utilmy\deeplearning\keras\util_train.py
-------------------------functions----------------------
np_remove_duplicates(seq)
clean_duplicates(ll)
log3(*s)
log1(*s)
tf_compute_set(cc:dict)
log(*s)
config_save(cc, path)
os_path_copy(in_dir, path, ext = "*.py")
check_valid_image(img_list, path = "", tag = "", y_labels = "", n_sample = 3, renorm = True)
save_best_model(model, model_dir2, curr_loss, best_loss, counter, epoch, dd)
save_model_state(model, model_dir2)
train_stop(counter, patience)
model_reload(model_reload_name, cc, )
image_check(name, img, renorm = False)
get_custom_label_data()
pd_category_filter(df, category_map)
train_step_2(x, y, model, loss_fn, optimizer)
test_step(x, y, model, loss_fn)



utilmy\deeplearning\torch\sentence2.py
-------------------------functions----------------------
fit(model, data_pars = None, model_pars = None, compute_pars = None, out_pars = None, *args, **kw)
evaluate(model, session = None, data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(model, session = None, data_pars = None, out_pars = None, compute_pars = None, **kw)
reset_model()
save(model, session = None, save_pars = None)
load(load_pars = None)
get_dataset(data_pars = None, **kw)
fit2(model, data_pars = None, model_pars = None, compute_pars = None, out_pars = None, *args, **kw)
predict2(model, session = None, data_pars = None, out_pars = None, compute_pars = None, **kw)
get_dataset2(data_pars = None, model = None, **kw)
get_params(param_pars, **kw)
test(data_path = "dataset/", pars_choice = "test01", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, **kwargs)


utilmy\deeplearning\torch\sentences.py
-------------------------functions----------------------
log(*s)
test()
dataset_fake(dirdata)
dataset_fake2(dirdata = '')
dataset_download(dirout = '/content/sample_data/sent_tans/')
model_evaluate(model  = "modelname OR path OR model object", dirdata = './*.csv', dirout = './', cc:dict =  None, batch_size = 16, name = 'sts-test')
model_load(path_or_name_or_object)
model_save(model, path, reload = True)
model_setup_compute(model, use_gpu = 0, ngpu = 1, ncpu = 1, cc:dict = None)
pd_read(path_or_df = './myfile.csv', npool = 1, **kw)
load_evaluator(name = 'sts', path_or_df = "", dname = 'sts', cc:dict = None)
load_dataloader(name = 'sts', path_or_df  =  "", cc:dict =  None, npool = 4)
load_loss(model  = '', lossname  = 'cosinus', cc:dict =  None)
metrics_cosine_sim(sentence1  =  "sentence 1", sentence2  =  "sentence 2", model_id  =  "model name or path or object")
sentrans_train(modelname_or_path = 'distilbert-base-nli-mean-tokens', taskname = "classifier", lossname = "cosinus", datasetname  =  'sts', train_path = "train/*.csv", val_path   = "val/*.csv", eval_path  = "eval/*.csv", metricname = 'cosinus', dirout  = "mymodel_save/", cc:dict =  None)



utilmy\deeplearning\torch\util_train.py


utilmy\deeplearning\torch\zkeras_torch_sentence.py
-------------------------functions----------------------
test()
log(*s)
metric_evaluate(model, )fIn, delimiter = '\t', )test_samples = []) =  []):)
model_load(path)
model_save(path)
create_evaluator(dname = 'sts', dirin = '/content/sample_data/sent_tans/', cc:dict = None)
sentrans_train(modelname_or_path = "", taskname = "classifier", lossname = "", train_path = "train/*.csv", val_path = "val/*.csv", metricname = 'cosinus', dirout  = "mymodel_save/", cc:dict =  Nonecc)   #### can use cc.epoch   cc.lr{})cc.epoch = 3cc.lr = 1E-5cc.warmup = 100cc.n_sample  = 1000cc.batch_size=16cc.mode = 'cpu/gpu'cc.ncpu =5  dir_train )dftrain = dftrain[[ 'text1', 'text2', 'label'  ]].values  dir_train )dfval  =  dfval[[ 'text1', 'text2', 'label'  ]].valuesif lossname == 'cosinus' = = 'cosinus':  loss =if taskname == 'classifier ':)
build_model()

-------------------------methods----------------------
SentenceEncoder.__init__(self, num_labels = None)
SentenceEncoder.call(self, inputs, **kwargs)
ReRanker.__init__(self)
ReRanker.call(self, inputs, **kwargs)


utilmy\deeplearning\torch\__init__.py


utilmy\recsys\models\ease.py
-------------------------methods----------------------
EASE.__init__(self)
EASE._get_users_and_items(self, df)
EASE.fit(self, df, lambda_: float  =  0.5, implicit = True)
EASE.predict(self, train, users, items, k)


utilmy\recsys\zrecs\setup.py


utilmy\spark\conda\script.py


utilmy\spark\script\hadoopVersion.py


utilmy\spark\script\pysparkTest.py
-------------------------functions----------------------
inside(p)



utilmy\spark\src\utils.py
-------------------------functions----------------------
logger_setdefault()
log()
log2(*s)
log3(*s)
log()
log_sample(*s)
config_load(config_path:str)
spark_check(df:pyspark.sql.DataFrame, conf:dict = None, path:str = "", nsample:int = 10, save = True, verbose = True, returnval = False)

-------------------------methods----------------------
to_namespace.__init__(self, d)


utilmy\spark\src\util_hadoop.py
-------------------------functions----------------------
hdfs_down(from_dir = "", to_dir = "", verbose = False, n_pool = 1, **kw)



utilmy\spark\src\util_models.py
-------------------------functions----------------------
TimeSeriesSplit(df_m:pyspark.sql.DataFrame, splitRatio:float, sparksession:object)
Train(spark, df_m:pyspark.sql.DataFrame, features:list, regressor:str, path:str = None, conf_model:dict = None)
Predict(spark, df_m:pyspark.sql.DataFrame, features:list, regressor:str, path:str = None, conf_model:dict = None)
os_makedirs(path:str)



utilmy\spark\src\util_spark.py


utilmy\spark\src\__init__.py


utilmy\spark\tests\conftest.py
-------------------------functions----------------------
config()
spark_session(config: dict)



utilmy\spark\tests\test_common.py
-------------------------functions----------------------
assert_equal_spark_df_sorted(expected_sorted_df: DataFrame, actual_sorted_df: DataFrame, df_name: str)
assert_equal_spark_df(expected_sorted_df: DataFrame, actual_sorted_df: DataFrame, df_name: str)
assert_equal_spark_df_schema(expected_schema: [tuple], actual_schema: [tuple], df_name: str)



utilmy\spark\tests\test_functions.py
-------------------------functions----------------------
test_getall_families_from_useragent(spark_session: SparkSession)



utilmy\spark\tests\test_table_user_log.py
-------------------------functions----------------------
test_table_user_log_run(spark_session: SparkSession, config: dict)



utilmy\spark\tests\test_table_user_session_log.py
-------------------------functions----------------------
test_table_user_session_log_run(spark_session: SparkSession)
test_table_user_session_log(spark_session: SparkSession)
test_table_usersession_log_stats(spark_session: SparkSession, config: dict)



utilmy\spark\tests\test_table_user_session_stats.py
-------------------------functions----------------------
test_table_user_session_stats_run(spark_session: SparkSession)
test_table_user_session_stats(spark_session: SparkSession)
test_table_user_session_stats_ip(spark_session: SparkSession, config: dict)



utilmy\spark\tests\test_table_volume_predict.py
-------------------------functions----------------------
test_preprocess(spark_session: SparkSession, config: dict)



utilmy\spark\tests\test_utils.py
-------------------------functions----------------------
test_spark_check(spark_session: SparkSession, config: dict)



utilmy\spark\tests\__init__.py


utilmy\stats\hypothetical\aov.py


utilmy\stats\hypothetical\contingency.py


utilmy\stats\hypothetical\critical.py
-------------------------functions----------------------
chi_square_critical_value(alpha, dof)



utilmy\stats\hypothetical\descriptive.py
-------------------------functions----------------------
add_noise(cor, epsilon = None, m = None)
covar(x, y = None, method = None)



utilmy\stats\hypothetical\fa.py


utilmy\stats\hypothetical\gof.py


utilmy\stats\hypothetical\hypothesis.py


utilmy\stats\hypothetical\nonparametric.py


utilmy\stats\hypothetical\posthoc.py


utilmy\stats\hypothetical\_lib.py
-------------------------functions----------------------
_build_des_mat(*args, group = None)
_build_summary_matrix(x, y = None)
_rank(design_matrix)
_group_rank_sums(ranked_matrix)



utilmy\stats\hypothetical\__init__.py


utilmy\viz\zarchive\toptoolbar.py
-------------------------methods----------------------
TopToolbar.__init__(self)


utilmy\viz\zarchive\__init__.py


utilmy\webscraper\test\scraper_img.py


utilmy\webscraper\test\Scraper_INSTAGRAM.py
-------------------------functions----------------------
make_soup(url)



utilmy\webscraper\test\url_scraper.py
-------------------------methods----------------------
GlassDoor.parse(self, response)


utilmy\webscraper\test\vc_scraper.py
-------------------------methods----------------------
GlassDoor.start_requests(self)
GlassDoor.parse(self, response)


utilmy\zml\example\classifier_mlflow.py
-------------------------functions----------------------
global_pars_update(model_dict, data_name, config_name)
titanic_lightgbm()
pd_col_myfun(df = None, col = None, pars = {})
check()



utilmy\zml\example\test.py
-------------------------functions----------------------
os_get_function_name()
global_pars_update(model_dict, data_name, config_name)
titanic1(path_model_out = "")
data_profile(path_data_train = "", path_model = "", n_sample =  5000)
preprocess(config = None, nsample = None)
train(config = None, nsample = None)
check()
predict(config = None, nsample = None)
run_all()



utilmy\zml\example\test_automl.py
-------------------------functions----------------------
global_pars_update(model_dict, data_name, config_name)
config1()



utilmy\zml\example\test_features.py
-------------------------functions----------------------
global_pars_update(model_dict, data_name, config_name)
config1(path_model_out = "")
config2(path_model_out = "")
config4(path_model_out = "")
config9(path_model_out = "")
pd_col_amyfun(df: pd.DataFrame, col: list = None, pars: dict = None)
config3(path_model_out = "")



utilmy\zml\example\test_hyperopt.py
-------------------------functions----------------------
global_pars_update(model_dict, data_name, config_name)
post_process_fun(y)
pre_process_fun(y)
titanic1(path_model_out = "")
hyperparam(config_full = "", ntrials = 2, n_sample = 5000, debug = 1, path_output          =  "data/output/titanic1/", path_optuna_storage  =  'data/output/optuna_hyper/optunadb.db')
hyperparam_wrapper(config_full = "", ntrials = 2, n_sample = 5000, debug = 1, path_output          =  "data/output/titanic1/", path_optuna_storage  =  'data/output/optuna_hyper/optunadb.db', metric_name = 'accuracy_score', mdict_range = None)



utilmy\zml\example\test_keras_vaemdn.py
-------------------------functions----------------------
global_pars_update(model_dict, data_name, config_name)
config_sampler()



utilmy\zml\example\test_keras_vaemdn2.py
-------------------------functions----------------------
global_pars_update(model_dict, data_name, config_name)
config1()



utilmy\zml\example\test_mkeras.py
-------------------------functions----------------------
global_pars_update(model_dict, data_name, config_name)
config1()



utilmy\zml\example\test_mkeras_dense.py
-------------------------functions----------------------
global_pars_update(model_dict, data_name, config_name)
config1()



utilmy\zml\example\titanic_gefs.py
-------------------------functions----------------------
global_pars_update(model_dict, data_name, config_name)
config1()



utilmy\zml\example\zfraud.py
-------------------------functions----------------------
global_pars_update(model_dict, data_name, config_name)



utilmy\zml\source\prepro.py
-------------------------functions----------------------
log(*s)
log2(*s)
log3(*s)
log4(*s, n = 0, m = 1)
log4_pd(name, df, *s)
_pd_colnum(df, col, pars)
_pd_colnum_fill_na_median(df, col, pars)
prepro_load(prefix, pars)
prepro_save(prefix, pars, df_new, cols_new, prepro)
pd_col_atemplate(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_coly_clean(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_coly(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_colnum_normalize(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_colnum_quantile_norm(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_colnum_bin(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_colnum_binto_onehot(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_colcat_to_onehot(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_colcat_bin(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_colcross(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_coldate(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_colcat_encoder_generic(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_colcat_minhash(df: pd.DataFrame, col: list = None, pars: dict = None)
os_convert_topython_code(txt)
save_json(js, pfile, mode = 'a')
pd_col_genetic_transform(df: pd.DataFrame, col: list = None, pars: dict = None)
test()



utilmy\zml\source\prepro_rec.py
-------------------------functions----------------------
_preprocess_criteo(df, **kw)
_preprocess_movielens(df, **kw)



utilmy\zml\source\prepro_text.py
-------------------------functions----------------------
log(*s, n = 0, m = 1)
logs(*s)
log_pd(df, *s, n = 0, m = 1)
pd_coltext_clean(df, col, stopwords =  None, pars = None)
pd_coltext_wordfreq(df, col, stopwords, ntoken = 100)
nlp_get_stopwords()
pd_coltext(df, col, stopwords =  None, pars = None)
pd_coltext_universal_google(df, col, pars = {})



utilmy\zml\source\prepro_tseries.py
-------------------------functions----------------------
log(*s)
log2(*s)
log3(*s)
logd(*s, n = 0, m = 0)
pd_prepro_custom(df: pd.DataFrame, col: list = None, pars: dict = None)
pd_prepro_custom2(df: pd.DataFrame, cols: list = None, pars: dict = None)
pd_ts_date(df: pd.DataFrame, cols: list = None, pars: dict = None)
pd_ts_groupby(df: pd.DataFrame, cols: list = None, pars: dict = None)
pd_ts_onehot(df: pd.DataFrame, cols: list = None, pars: dict = None)
pd_ts_autoregressive(df: pd.DataFrame, cols: list = None, pars: dict = None)
pd_ts_rolling(df: pd.DataFrame, cols: list = None, pars: dict = None)
pd_ts_lag(df: pd.DataFrame, cols: list = None, pars: dict = None)
pd_ts_difference(df: pd.DataFrame, cols: list = None, pars: dict = None)
pd_ts_tsfresh_features(df: pd.DataFrame, cols: list = None, pars: dict = None)
pd_ts_deltapy_generic(df: pd.DataFrame, cols: list = None, pars: dict = None)
test_get_sampledata(url="https = "https://github.com/firmai/random-assets-two/raw/master/numpy/tsla.csv")
test_deltapy_all()
test_prepro_v1()
test_deltapy_get_method(df)
test_deltapy_all2()
m5_dataset()



utilmy\zml\source\run_feature_profile.py
-------------------------functions----------------------
log(*s, n = 0, m = 0)
run_profile(path_data = None, path_output = "data/out/ztmp/", n_sample = 5000)



utilmy\zml\source\run_hyperopt.py
-------------------------functions----------------------
log(*s)
run_hyper_optuna(obj_fun, pars_dict_init, pars_dict_range, engine_pars, ntrials = 3)
test_hyper()
test_hyper3()
test_hyper2()
eval_dict(src, dst = {})



utilmy\zml\source\run_inference.py
-------------------------functions----------------------
log(*s)
log2(*s)
log3(*s)
model_dict_load(model_dict, config_path, config_name, verbose = True)
map_model(model_name="model_sklearn = "model_sklearn:MyClassModel")
run_predict_batch(config_name, config_path, n_sample = -1, path_data = None, path_output = None, pars = {}, model_dict = None, return_mode = 'file')
predict(model_dict, dfX, cols_family, post_process_fun = None)
run_predict(config_name, config_path, n_sample = -1, path_data = None, path_output = None, pars = {}, model_dict = None, return_mode = 'file')
run_data_check(path_data, path_data_ref, path_model, path_output, sample_ratio = 0.5)



utilmy\zml\source\run_inpection.py
-------------------------functions----------------------
log(*s, n = 0, m = 0)
save_features(df, name, path)
model_dict_load(model_dict, config_path, config_name, verbose = True)



utilmy\zml\source\run_mlflow.py
-------------------------functions----------------------
register(run_name, params, metrics, signature, model_class, tracking_uri= "sqlite =  "sqlite:///local.db")



utilmy\zml\source\run_preprocess.py
-------------------------functions----------------------
log(*s)
log2(*s)
log3(*s)
log_pd(df, *s, n = 0, m = 1)
save_features(df, name, path = None)
load_features(name, path)
model_dict_load(model_dict, config_path, config_name, verbose = True)
preprocess_batch(path_train_X = "", path_train_y = "", path_pipeline_export = "", cols_group = None, n_sample = 5000, preprocess_pars = {}, path_features_store = None)
preprocess(path_train_X = "", path_train_y = "", path_pipeline_export = "", cols_group = None, n_sample = 5000, preprocess_pars = {}, path_features_store = None)
preprocess_inference(df, path_pipeline = "data/pipeline/pipe_01/", preprocess_pars = {}, cols_group = None)
preprocess_load(path_train_X = "", path_train_y = "", path_pipeline_export = "", cols_group = None, n_sample = 5000, preprocess_pars = {}, path_features_store = None)
run_preprocess(config_name, config_path, n_sample = 5000, mode = 'run_preprocess', model_dict = Nonemodel_dict, config_path, config_name, verbose = True)m = model_dict['global_pars']path_data         = m['path_data_preprocess']'path_data_prepro_X', path_data + "/features.zip") # ### Can be a list of zip or parquet files'path_data_prepro_y', path_data + "/target.zip")   # ### Can be a list of zip or parquet filespath_output          =  m['path_train_output']'path_pipeline', path_output + "/pipeline/" )'path_features_store', path_output + '/features_store/' )  #path_data_train replaced with path_output, because preprocessed files are stored there'path_check_out', path_output + "/check/" )path_output)"#### load input column family  ###################################################")cols_group = model_dict['data_pars']['cols_input_type']  ### the model config file"#### Preprocess  #################################################################")preprocess_pars = model_dict['model_pars']['pre_process_pars']if mode == "run_preprocess"  =  model_dict['data_pars']['cols_input_type']  ### the model config file"#### Preprocess  #################################################################")preprocess_pars = model_dict['model_pars']['pre_process_pars']if mode == "run_preprocess" :)



utilmy\zml\source\run_sampler.py
-------------------------functions----------------------
log(*s)
log2(*s)
log3(*s)
save_features(df, name, path)
model_dict_load(model_dict, config_path, config_name, verbose = True)
map_model(model_name)
train(model_dict, dfX, cols_family, post_process_fun)
run_train(config_name, config_path = "source/config_model.py", n_sample = 5000, mode = "run_preprocess", model_dict = None, return_mode = 'file', **kw)
transform(model_name, path_model, dfX, cols_family, model_dict)
run_transform(config_name, config_path, n_sample = 1, path_data = None, path_output = None, pars = {}, model_dict = None, return_mode = "")



utilmy\zml\source\run_train.py
-------------------------functions----------------------
log(*s)
log2(*s)
log3(*s)
save_features(df, name, path)
model_dict_load(model_dict, config_path, config_name, verbose = True)
map_model(model_name)
data_split(dfX, data_pars, model_path, colsX, coly)
train(model_dict, dfX, cols_family, post_process_fun)
cols_validate(model_dict)
run_train(config_name, config_path = "source/config_model.py", n_sample = 5000, mode = "run_preprocess", model_dict = None, return_mode = 'file', **kw)
run_model_check(path_output, scoring)
mlflow_register(dfXy, model_dict: dict, stats: dict, mlflow_pars:dict)



utilmy\zml\source\util.py
-------------------------functions----------------------
log(*s, n = 0, m = 1, **kw)
pd_to_scipy_sparse_matrix(df)
pd_to_keyvalue_dict(dfa, colkey =  [ "shop_id", "l2_genre_id" ], col_list = 'item_id', to_file = "")
create_appid(filename)
create_logfilename(filename)
create_uniqueid()
logger_setup(logger_name = None, log_file = None, formatter = 'FORMATTER_0', isrotate = False, isconsole_output = True, logging_level = 'info', )
logger_handler_console(formatter = None)
logger_handler_file(isrotate = False, rotate_time = "midnight", formatter = None, log_file_used = None)
logger_setup2(name = __name__, level = None)
test_log()
download_googledrive(file_list=[ {  "fileid" = [ {  "fileid": "1-K72L8aQPsl2qt_uBF-kzbai3TYG6Qg4", "path_target":  "data/input/download/test.json"}], **kw)
download_dtopbox(data_pars)
load_dataset_generator(data_pars)
tf_dataset(dataset_pars)

-------------------------methods----------------------
dict2.__init__(self, d)
dictLazy.__init__(self, *args, **kw)
dictLazy.__getitem__(self, key)
dictLazy.__iter__(self)
dictLazy.__len__(self)
logger_class.__init__(self, config_file = None, verbose = True)
logger_class.load_config(self, config_file_path = None)
logger_class.log(self, *s, level = 1)
logger_class.debug(self, *s, level = 1)
Downloader.__init__(self, url)
Downloader.clean_netloc(self)
Downloader.adjust_url(self)
Downloader._transform_github_url(self)
Downloader._transform_gdrive_url(self)
Downloader._transform_dropbox_url(self)
Downloader.get_filename(self, headers)
Downloader.download(self, filepath = '')


utilmy\zml\source\util_feature.py
-------------------------functions----------------------
log(*s, n = 0, m = 1, **kw)
log2(*s, **kw)
log3(*s, **kw)
os_get_function_name()
os_getcwd()
pa_read_file(path =   'folder_parquet/', cols = None, n_rows = 1000, file_start = 0, file_end = 100000, verbose = 1, )
pa_write_file(df, path =   'folder_parquet/', cols = None, n_rows = 1000, partition_cols = None, overwrite = True, verbose = 1, filesystem  =  'hdfs')
test_get_classification_data(name = None)
params_check(pars, check_list, name = "")
save_features(df, name, path = None)
load_features(name, path)
save_list(path, name_list, glob)
save(df, name, path = None)
load(name, path)
pd_read_file(path_glob = "*.pkl", ignore_index = True, cols = None, verbose = False, nrows = -1, concat_sort = True, n_pool = 1, drop_duplicates = None, col_filter = None, col_filter_val = None, **kw)
load_dataset(path_data_x, path_data_y = '', colid = "jobId", n_sample = -1)
fetch_spark_koalas(path_data_x, path_data_y = '', colid = "jobId", n_sample = -1)
fetch_dataset(url_dataset, path_target = None, file_target = None)
load_function_uri(uri_name="myfolder/myfile.py = "myfolder/myfile.py::myFunction")
metrics_eval(metric_list = ["mean_squared_error"], ytrue = None, ypred = None, ypred_proba = None, return_dict = False)
pd_stat_dataset_shift(dftrain, dftest, colused, nsample = 10000, buckets = 5, axis = 0)
pd_stat_datashift_psi(expected, actual, buckettype = 'bins', buckets = 10, axis = 0)
estimator_std_normal(err, alpha = 0.05, )
estimator_boostrap_bayes(err, alpha = 0.05, )
estimator_bootstrap(err, custom_stat = None, alpha = 0.05, n_iter = 10000)
test_heteroscedacity(y, y_pred, pred_value_only = 1)
test_normality(error, distribution = "norm", test_size_limit = 5000)
test_mutualinfo(error, Xtest, colname = None, bins = 5)
feature_importance_perm(clf, Xtrain, ytrain, cols, n_repeats = 8, scoring = 'neg_root_mean_squared_error', show_graph = 1)
feature_selection_multicolinear(df, threshold = 1.0)
feature_correlation_cat(df, colused)
pd_feature_generate_cross(df, cols, cols_cross_input = None, pct_threshold = 0.2, m_combination = 2)
pd_col_to_onehot(dfref, colname = None, colonehot = None, return_val = "dataframe,column")
pd_colcat_mergecol(df, col_list, x0, colid = "easy_id")
pd_colcat_tonum(df, colcat = "all", drop_single_label = False, drop_fact_dict = True)
pd_colcat_mapping(df, colname)
pd_colcat_toint(dfref, colname, colcat_map = None, suffix = None)
pd_colnum_tocat(df, colname = None, colexclude = None, colbinmap = None, bins = 5, suffix = "_bin", method = "uniform", na_value = -1, return_val = "dataframe,param", params={"KMeans_n_clusters" = {"KMeans_n_clusters": 8, "KMeans_init": 'k-means++', "KMeans_n_init": 10,"KMeans_max_iter": 300, "KMeans_tol": 0.0001, "KMeans_precompute_distances": 'auto',"KMeans_verbose": 0, "KMeans_random_state": None,"KMeans_copy_x": True, "KMeans_n_jobs": None, "KMeans_algorithm": 'auto'})
pd_colnum_normalize(df0, colname, pars, suffix = "_norm", return_val = 'dataframe,param')
pd_col_merge_onehot(df, colname)
pd_col_to_num(df, colname = None, default = np.nan)
pd_col_filter(df, filter_val = None, iscol = 1)
pd_col_fillna(dfref, colname = None, method = "frequent", value = None, colgroupby = None, return_val = "dataframe,param", )
pd_pipeline_apply(df, pipeline)
pd_stat_correl_pair(df, coltarget = None, colname = None)
pd_stat_pandas_profile(df, savefile = "report.html", title = "Pandas Profile")
pd_stat_distribution_colnum(df, nrows = 2000, verbose = False)
pd_stat_histogram(df, bins = 50, coltarget = "diff")
col_extractname(col_onehot)
col_remove(cols, colsremove, mode = "exact")
pd_colnum_tocat_stat(df, feature, target_col, bins, cuts = 0)
pd_stat_shift_trend_changes(df, feature, target_col, threshold = 0.03)
pd_stat_shift_trend_correlation(df, df_test, colname, target_col)
pd_stat_shift_changes(df, target_col, features_list = 0, bins = 10, df_test = 0)
np_conv_to_one_col(np_array, sep_char = "_")

-------------------------methods----------------------
dict2.__init__(self, d)


utilmy\zzarchive\py2to3\alldata.py


utilmy\zzarchive\py2to3\allmodule.py
-------------------------functions----------------------
pprint(table1, tablefmt = "simple")
pprint2(x)
str_convert_beforeprint(x)
str_to_utf8(x)
str_to_unicode(x, encoding = 'utf-8')



utilmy\zzarchive\py2to3\allmodule_fin.py


utilmy\zzarchive\py2to3\coke_functions.py
-------------------------functions----------------------
date_diffsecond(str_t1, str_t0, fmt='YYYY-MM-DD HH = 'YYYY-MM-DD HH:mm:SS')
date_diffstart(t)
date_diffend(t)
np_dict_tolist(dd)
np_dict_tostr_val(dd)
np_dict_tostr_key(dd)
day(s)
month(s)
year(s)
hour(s)
weekday(s, fmt = 'YYYY-MM-DD', i0 = 0, i1 = 10)
season(d)
daytime(d)
pd_date_splitall(df, coldate = 'purchased_at')



utilmy\zzarchive\py2to3\datanalysis.py
-------------------------functions----------------------
pd_filter_column(df_client_product, filter_val = [], iscol = 1)
pd_missing_show()
pd_validation_struct()
pd_checkpoint()
xl_setstyle(file1)
xl_val(ws, colj, rowi)
isnull(x)
xl_get_rowcol(ws, i0, j0, imax, jmax)
pd_stack_dflist(df_list)
xl_getschema(dirxl = "", filepattern = '*.xlsx', dirlevel = 1, outfile = '.xlsx')
str_to_unicode(x, encoding = 'utf-8')
csv_dtypes_getdict(df = None, csvfile = None)
csv_fast_processing()
csv_col_schema_toexcel(dircsv = "", filepattern = '*.csv', outfile = '.xlsx', returntable = 1, maxrow = 5000000, maxcol_pertable = 90, maxstrlen = 'U80')
csv_col_get_dict_categoryfreq(dircsv, filepattern = "*.csv", category_cols = [], maxline = -1, fileencoding = "utf-8")
csv_row_reduce_line(fromfile, tofile, condfilter, catval_tokeep, maxline = -1)
csv_analysis()
csv_row_reduce_line_manual(file_category, file_transact, file_reduced)
csv_row_mapreduce(dircsv = "", outfile = "", type_mapreduce = 'sum', chunk =  5000000)
csv_pivotable(dircsv = "", filepattern = '*.csv', fileh5 = '.h5', leftX = 'col0', topY = 'col2', centerZ = 'coli', mapreduce = 'sum', chunksize =  500000, tablename = 'df')
csv_bigcompute()
db_getdata()
db_sql()
db_meta_add(metadb, dbname, new_table = ('', [])
db_meta_find(ALLDB, query = '', filter_db = [], filter_table = [], filter_column = [])
col_study_getcategorydict_freq(catedict)
col_feature_importance(Xcol, Ytarget)
col_study_distribution_show(df, col_include = None, col_exclude = None, pars={'binsize' = {'binsize':20})
col_study_summary(Xmat = [0.0, 0.0], Xcolname = ['col1', 'col2'], Xcolselect = [9, 9], isprint = 0)
col_pair_plot(dfX, Xcolname_selectlist = None, dfY = None, Ycolname = None)
col_pair_correl(Xcol, Ytarget)
col_pair_interaction(Xcol, Ytarget)
plot_col_pair(dfX, Xcolname_selectlist = None, dfY = None, Ycolname = None)
tf_transform_catlabel_toint(Xmat)
tf_transform_pca(Xmat, dimpca = 2, whiten = True)
plot_distance_heatmap(Xmat_dist, Xcolname)
plot_cluster_2D(X_2dim, target_class, target_names)
plot_cluster_tsne(Xmat, Xcluster_label = None, metric = 'euclidean', perplexity = 50, ncomponent = 2, savefile = '', isprecompute = False, returnval = True)
plot_cluster_pca(Xmat, Xcluster_label = None, metric = 'euclidean', dimpca = 2, whiten = True, isprecompute = False, savefile = '', doreturn = 1)
plot_cluster_hiearchy(Xmat_dist, p = 30, truncate_mode = None, color_threshold = None, get_leaves = True, orientation = 'top', labels = None, count_sort = False, distance_sort = False, show_leaf_counts = True, no_plot = False, no_labels = False, leaf_font_size = None, leaf_rotation = None, leaf_label_func = None, show_contracted = False, link_color_func = None, ax = None, above_threshold_color = 'b')
plot_distribution_density(Xsample, kernel = 'gaussian', N = 10, bandwith = 1 / 10.0)
plot_Y(Yval, typeplot = '.b', tsize = None, labels = None, title = '', xlabel = '', ylabel = '', zcolor_label = '', figsize = (8, 6)
plot_XY(xx, yy, zcolor = None, tsize = None, labels = None, title = '', xlabel = '', ylabel = '', zcolor_label = '', figsize = (8, 6)
plot_XY_plotly(xx, yy, towhere = 'url')
plot_XY_seaborn(X, Y, Zcolor = None)
optim_is_pareto_efficient(Xmat_cost, epsilon =  0.01, ret_boolean = 1)
sk_model_auto_tpot(Xmat, y, outfolder = 'aaserialize/', model_type = 'regressor/classifier', train_size = 0.5, generation = 1, population_size = 5, verbosity = 2)
sk_params_search_best(Xmat, Ytarget, model1, param_grid={'alpha' = {'alpha':  np.linspace(0, 1, 5) }, method = 'gridsearch', param_search= {'scoretype' =  {'scoretype':'r2', 'cv':5, 'population_size':5, 'generations_number':3 })
sk_distribution_kernel_bestbandwidth(kde)
sk_distribution_kernel_sample(kde = None, n = 1)
sk_correl_rank(correl = [[1, 0], [0, 1]])
sk_error_r2(Ypred, y_true, sample_weight = None, multioutput = None)
sk_error_rmse(Ypred, Ytrue)
sk_cluster_distance_pair(Xmat, metric = 'jaccard')
sk_cluster(Xmat, metric = 'jaccard')
sk_cluster_algo_custom(Xmat, algorithm, args, kwds, returnval = 1)
sk_optim_de(obj_fun, bounds, maxiter = 1, name1 = '', solver1 = None, isreset = 1, popsize = 15)
sk_feature_importance(clfrf, feature_name)
sk_showconfusion(clfrf, X_train, Y_train, isprint = True)
sk_tree(Xtrain, Ytrain, nbtree, maxdepth, isprint1 = 1, njobs = 1)
sk_gen_ensemble_weight(vv, acclevel, maxlevel = 0.88)
sk_votingpredict(estimators, voting, ww, X_test)
sk_tree_get_ifthen(tree, feature_names, target_names, spacer_base = " ")

-------------------------methods----------------------
sk_model_template1.__init__(self, alpha = 0.5, low_y_cut = -0.09, high_y_cut = 0.09, ww0 = 0.95)
sk_model_template1.fit(self, X, Y = None)
sk_model_template1.predict(self, X, y = None, ymedian = None)
sk_model_template1.score(self, X, Ytrue = None, ymedian = None)
sk_stateRule.__init__(self, state, trigger, colname = [])
sk_stateRule.addrule(self, rulefun, name = '', desc = '')
sk_stateRule.eval(self, idrule, t, ktrig = 0)
sk_stateRule.help()


utilmy\zzarchive\py2to3\excel.py
-------------------------functions----------------------
get_workbook_name()
double_sum(x, y)
add_one(data)
matrix_mult(x, y)
npdot()



utilmy\zzarchive\py2to3\fast.py
-------------------------functions----------------------
day(s)
month(s)
year(s)
hour(s)
weekday(s)
season(d)
daytime(d)
fastStrptime(val, format)
drawdown_calc_fast(price)
std(x)
mean(x)
_compute_overlaps(u, v)
distance_jaccard2(u, v)
distance_jaccard(u, v)
distance_jaccard_X(X)
cosine(u, v)
rmse(y, yhat)
cross(vec1, vec2)
norm(vec)
log_exp_sum2(a, b)



utilmy\zzarchive\py2to3\fast_parallel.py
-------------------------functions----------------------
task_summary(tasks)
task_progress(tasks)
task_find_best(tasks, n_top = 5)
task_parallel_job_01(name, param, datadict)



utilmy\zzarchive\py2to3\filelock.py
-------------------------methods----------------------
FileLock.__init__(self, protected_file_path, timeout = None, delay = 1, lock_file_contents = None)
FileLock.locked(self)
FileLock.available(self)
FileLock.acquire(self, blocking = True)
FileLock.release(self)
FileLock.__enter__(self)
FileLock.__exit__(self, type, value, traceback)
FileLock.__del__(self)
FileLock.purge(self)


utilmy\zzarchive\py2to3\function_custom.py
-------------------------functions----------------------
mapping_calc_risk_elvis_v03(ss, tr, t, riskout)
mapping_calc_risk_v02(ss, tr, t, risk0)
mapping_calc_risk_v01(ss, tr, t, risk0)
mapping_risk_ww_v01(risk, wwmat, ww2)
mapping_calc_risk_v00(self, ss, tr, t, risk0)
getweight(ww, size = (9, 3)
fun_obj(vv, ext)



utilmy\zzarchive\py2to3\geospatial.py


utilmy\zzarchive\py2to3\global01.py


utilmy\zzarchive\py2to3\kagglegym.py
-------------------------functions----------------------
r_score(y_true, y_pred, sample_weight = None, multioutput = None)
make()

-------------------------methods----------------------
Observation.__init__(self, train, target, features)
Environment.__init__(self)
Environment.reset(self)
Environment.step(self, target)
Environment.__str__(self)


utilmy\zzarchive\py2to3\linux.py
-------------------------functions----------------------
load_session(name = 'test_20160815')
save_session(name = '')
isfloat(value)
isint(x)
aa_isanaconda()
aa_cleanmemory()
aa_getmodule_doc(module1, fileout = '')
np_interpolate_nan(y)
and1(x, y, x3 = None, x4 = None, x5 = None, x6 = None, x7 = None, x8 = None)
sortcol(arr, colid, asc = 1)
sort(arr, colid, asc = 1)
np_ma(vv, n)
np_cleanmatrix(m)
np_sortbycolumn(arr, colid, asc = True)
np_sortbycol(arr, colid, asc = True)
np_findfirst(item, vec)
np_find(item, vec)
find(item, vec)
findx(item, vec)
finds(itemlist, vec)
findhigher(x, vec)
findlower(x, vec)
np_find_minpos(values)
np_find_maxpos(values)
np_find_maxpos_2nd(numbers)
np_findlocalmax2(v, trig)
np_findlocalmin2(v, trig)
np_findlocalmax(v, trig)
np_findlocalmin(v, trig)
np_stack(v1, v2 = None, v3 = None, v4 = None, v5 = None)
np_uniquerows(a)
np_remove_zeros(vv, axis1 = 1)
np_sort(arr, colid, asc = True)
np_memory_array_adress(x)
sk_featureimportance(clfrf, feature_name)
sk_showconfusion(clfrf, X_train, Y_train, isprint = True)
sk_tree(Xtrain, Ytrain, nbtree, maxdepth, print1)
sk_gen_ensemble_weight(vv, acclevel, maxlevel = 0.88)
sk_votingpredict(estimators, voting, ww, X_test)
sk_tree_get_ifthen(tree, feature_names, target_names, spacer_base = " ")
pd_array_todataframe(price, symbols = None, date1 = None, dotranspose = False)
pd_date_intersection(qlist)
pd_resetindex(df)
pd_create_colmap_nametoid(df)
pd_dataframe_toarray(df)
pd_changeencoding(data, cols)
pd_createdf(val1, col1 = None, idx1 = None)
pd_insertcolumn(df, colname, vec)
pd_insertrows(df, rowval, index1 = None)
pd_replacevalues(df, matrix)
pd_storeadddf(df, dfname, dbfile='F = 'F:\temp_pandas.h5')
pd_storedumpinfo(dbfile='E = 'E:\_data\stock\intraday_google.h5')
pd_remove_row(df, row_list_index = [23, 45])
pd_extract_col_idx_val(df)
pd_split_col_idx_val(df)
pd_addcolumn(df1, name1 = 'new')
pd_removecolumn(df1, name1)
pd_save_vectopanda(vv, filenameh5)
pd_load_panda2vec(filenameh5, store_id = 'data')
pd_csv_topanda(filein1, filename, tablen = 'data')
pd_getpanda_tonumpy(filename, nsize, tablen = 'data')
pd_getrandom_tonumpy(filename, nbdim, nbsample, tablen = 'data')
sk_cluster_kmeans(x, nbcluster = 5, isplot = True)
datetime_tostring(datelist1)
date_remove_bdays(from_date, add_days)
date_add_bdays(from_date, add_days)
datestring_todatetime(datelist1, format1 =  "%Y%m%d")
datetime_toint(datelist1)
dateint_todatetime(datelist1)
date_diffindays(intdate1, intdate2)
date_finddateid(date1, dateref)
datestring_toint(datelist1)
date_now(i = 0)
date_as_float(dt)
datediff_inyear(startdate, enddate)
date_generatedatetime(start = "20100101", nbday = 10, end = "")
datetime_tostring(datelist1)
datestring_todatetime(datelist1, format1 =  "%Y%m%d")
datestring_toint(datelist1)
numexpr_vect_calc(filename, expr, i0 = 0, imax = 1000, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
numexpr_topanda(filename, expr, i0 = 0, imax = 1000, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
textvect_topanda(vv, fileout = "")
comoment(xx, yy, nsample, kx, ky)
acf(data)
plotsave(xx, yy, title1 = "")
plotshow(xx, yy, title1 = "")
compileVSsolution(dir1, flags1 = "", type1 = "devenv", compilerdir = "")
VS_start(self, version)
VS_build(self, lib_to_build)
set_rc_version(rcfile, target_version)



utilmy\zzarchive\py2to3\multiprocessfunc.py
-------------------------functions----------------------
multigbm_paralell_func(nbsimul, ww, voldt, drift, upper_cholesky, nbasset, n, price, type1 = 0, strike = 0, cp = 1)
func(val, lock)
multigbm_processfast7(nbsimul, s0, voldt, drift, upper_cholesky, nbasset, n, price)
bm_generator(bm, dt, n, type1)
merge(d2)
integratenp2(its, nchunk)
integratenp(its, nchunk)
integratene(its)
parzen_estimation(x_samples, point_x, h)
init2(d)
init_global1(l, r)
np_sin(value)
ne_sin(x)
res_shared2()
list_append(count, id, out_list)



utilmy\zzarchive\py2to3\portfolio.py
-------------------------functions----------------------
data_jpsector()
date_earningquater(t1)
date_is_3rdfriday(s)
date_option_expiry(date)
date_find_kday_fromintradaydate(kintraday, intradaydate, dailydate)
date_find_kintraday_fromdate(d1, intradaydate1, h1 = 9, m1 = 30)
date_find_intradateid(datetimelist, stringdate = ['20160420223000'])
datetime_convertzone1_tozone2(tt, fromzone = 'Japan', tozone = 'US/Eastern')
date_extract_dailyopenclosetime(spdateref1, market = 'us')
date_finddateid(date1, dateref)
date_alignfromdateref(array1, dateref)
_date_align(dateref, datei, tmax, closei)
date_align(array1, dateref)
min_withposition(values)
max_withposition(values)
_reshape(x)
_notnone(x)
plot_price(asset, y2 = None, y3 = None, y4 = None, y5 = None, sym = None, savename1 = '', tickperday = 20, date1 = None, graphsize = (10, 5)
plot_priceintraday(data)
plot_check(close, tt0i = 20140102, tt1i = 20160815, dateref = [], sym = [], tickperday = 120)
plot_pricedate(date1, sym1, asset1, sym2 = None, bsk1 = None, verticaldate = None, savename1 = '', graphsize = (10, 5)
generate_sepvertical(asset1, tt, tmax, start = None, datebar = None)
save_asset_tofile(file1, asset1, asset2 = None, asset3 = None, date1 = None, title1 = None)
load_asset_fromfile(file1)
array_todataframe(price, symbols = None, date1 = None)
pd_dataframe_toarray(df)
isfloat(value)
isint(x)
correlation_mat(Xmat, type1 = "robust", type2 = "correl")
correl_reducebytrigger(correl2, trigger)
sk_cov_fromcorrel(correl, ret_close1)
cointegration(x, y)
causality_y1_y2(price2, price1, maxlag)
reg_slope(close, dateref, tlag, type1 = 'elasticv')
rolling_cointegration(x, y)
regression(yreturn, xreturn, type1 = "elasticv")
regression_fixedsymbolstock(sym, ret_close2, tsstart, tsample, ret_spy, spyclose, regonly = True)
regression_getpricefromww(spyclose, ww01, regasset01, ret_close2, tstart, tlag = 1)
regression_allstocks_vs_riskfactors(symstock, pricestock, symriskfac, priceriskfac, nlaglist)
getdiff_fromquotes(close, timelag)
getret_fromquotes(close, timelag = 1)
getlogret_fromquotes(close, timelag = 1)
getprice_fromret(ret, normprice = 100)
price_normalize100(ret, normprice = 100)
price_normalize_1d(ret, normprice = 100, dtype1 =  np.float32)
norm_fast(y, ny)
correl_fast(xn, y, nx)
volhisto_fromret(retbsk, t, volrange, axis = 0)
volhisto_fromprice(price, t, volrange, axis = 0)
volhistorolling_fromprice(price, volrange)
rsk_calc_all_TA(df = 'panda_dataframe')
ta_highbandtrend1(close2, type1 = 0)
pd_transform_asset(q0, q1, type1 = "spread")
calcbasket_table(wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps =  0.000)
folio_createvolta_asset(close, vol = 0.12, volrange = 120, lev = 1.0)
folio_lowcorrelation(sym01, nstock, periodlist, dateref, close1, kbenchmark, badlist, costbppa = 0.02, showgraph = True)
folio_leverageetf(price, lev = 1.0, costpa = 0.0)
folio_inverseetf(price, costpa = 0.0)
folio_longshort_unit(long1, short1, ww = [1, -1], costpa = 0.0, tlag = 1, istable = 1, wwschedule = [])
folio_longshort_unitfixed(long1, short1, nn = [1, -1], costpa = 0.0, tlag = 1, istable = 1)
folio_longshort_pct(long1, short1, ww = [1, -1], costpa = 0.0)
folio_histogram(close)
folio_voltarget(bsk, targetvol = 0.11, volrange =  90, expocap = 1.5)
folio_volta(bsk, targetvol = 0.11, volrange =  90, expocap = 1.5)
folio_volta2(bsk, riskind, par, targetvol = 0.11, volrange =  90, cap = 1.5, floor = 0.0, costbp = 0.0005)
folio_fixedunitprice(price, fixedww, costpa = 0.0)
folio_fixedweightprice(price, fixedww, costpa = 0.0)
folio_fixedweightret(ret, fixedww)
folio_cost_turnover(wwall, bsk, dateref, costbp)
folio_riskpa(ret, targetvol = 0.1, volrange = 90, cap = 1.0)
folio_perfreport_schedule(sym, dateref, close, wwind, t0, scheduleperiod = "1monthend")
folio_concenfactor2(ww, masset = 12)
calcbasket_objext(RETURN, TMAX, riskind_i, wwmat, wwasset0, ww0, nbrange, criteria)
calc_ranktable(close2, symjp1, nlag, refindex, funeval, funargs)
similarity_correl(ret_close2, funargs)
np_similarity(x, y, wwerr = [], type1 = 0)
np_distance_l1(x, y, wwerr)
np_similarity(x, y, wwerr = [], type1 = 0)
np_distance_l1(x, y, wwerr)
get(close, timelag)

-------------------------methods----------------------
index.__init__(self, id1, sym, ww, tstart)
index.close(self)
index.updatehisto(self)
index.help(self)
index.__init__(self, id1, sym, ww, tstart)
index.calc_baskettable_pct(self, type1 = "table", showdetail = 0)
index._wwpct_rebal(self, wwpct_actual, t, trebal)
index._udpate_wwindpct(self, t, bskt, hedgecost, wwpct_actual, wwpct_th)
index.calc_baskettable_unit()
folioCalc.__init__(self, sym, close, dateref)
folioCalc.set_symclose(self, sym, close, dateref)
folioCalc.setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid = "spprice", lfun = None)
folioCalc._weightcalc_generic(self, wwvec, t)
folioCalc._weightcalc_regime(self, wwvec, wwextra, t)
folioCalc._regimecalc(self, t, wwextra)
folioCalc._weightcalc_constant(self, ww2, t)
folioCalc.getweight(self)
folioCalc.calc_baskettable(self, wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps = 0.000, showdetail = 0)
folioCalc.plot(self, wwvec = None, show1 = 1, tickperday = 60)
folioCalc.multiperiod_ww(self, t)
folioCalc.help(self)
folioRiskIndicator.__init__(self, sym, close, dateref)
folioRiskIndicator.set_symclose(self, sym, close, dateref)
folioRiskIndicator.setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid = "spprice", lfun = None)
folioRiskIndicator.calcrisk(self, wwvec = [], initval = 1)
folioRiskIndicator._weightcalc_generic(self, wwvec, t)
folioRiskIndicator._weightcalc_regime(self, wwvec, wwextra, t)
folioRiskIndicator._regimecalc(self, t, wwextra)
folioRiskIndicator.calc_optimal_weight(self, maxiter = 1, name1 = '', isreset = 1, popsize = 15)
folioOptimizationF.__init__(self, sym, close, dateref)
folioOptimizationF.set_symclose(self, sym, close, dateref)
folioOptimizationF.setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid = "spprice", lfun = None)
folioOptimizationF.calcbasket_obj2(self, wwvec)
folioOptimizationF.calcbasket_obj(self, wwvec)
folioOptimizationF._loss_obj(self, ww2, wwpenalty)
folioOptimizationF._weightcalc_generic(self, wwvec, t)
folioOptimizationF._weightcalc_regime(self, wwvec, wwextra, t)
folioOptimizationF._regimecalc(self, t, wwextra)
folioOptimizationF._weightcalc_constant(self, ww2, t)
folioOptimizationF.calc_optimal_weight(self, maxiter = 1, name1 = '', isreset = 1, popsize = 15)
folioOptimizationF.getweight(self)
folioOptimizationF.calc_baskettable(self, wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps =  0.000, showdetail = 0)
folioOptimizationF.plot(self, wwvec = None, show1 = 1, tickperday = 60)
folioOptimizationF._objective_criteria(self, bsk)
folioOptimizationF.multiperiod_ww(self, t)
folioOptimizationF.help(self)
folioOptimizationF.mapping_risk_ww(self, risk, wwmat, ww2 = self.wwasset0)
folioOptimizationF._mapping_calc_risk(self, ss, tr, t, risk0)
searchSimilarity.__init__(self, filejpstock=r'E = r'E:/_data/stock/daily/20160616/jp', sym01 = ['7203'], symname = ['Toyota'], startdate =  20150101, enddate = 20160601, pricetype = "close")
searchSimilarity.load_quotes_fromdb(self, picklefile = '')
searchSimilarity.__generate_return__(self, nlag)
searchSimilarity.__overweight__(self, px)
searchSimilarity.set_searchcriteria(self, name1 = '7203', date1 = 20160301, date2 = 20160601, nlag = 1, searchperiodstart = 20120101, typesearch = "pattern2", )
searchSimilarity.launch_search(self)
searchSimilarity.show_comparison_graph(self, maxresult = 20, show_only_different_time = True, fromid = 0, fromend =  0, filenameout = '')
searchSimilarity.staticmethod(self, x)
searchSimilarity.get_rankresult(self, filetosave = '')
searchSimilarity.export_results()


utilmy\zzarchive\py2to3\portfolio_withdate.py
-------------------------functions----------------------
data_jpsector()
date_earningquater(t1)
date_is_3rdfriday(s)
date_option_expiry(date)
date_find_kday_fromintradaydate(kintraday, intradaydate, dailydate)
date_find_kintraday_fromdate(d1, intradaydate1, h1 = 9, m1 = 30)
date_find_intradateid(datetimelist, stringdate = ['20160420223000'])
datetime_convertzone1_tozone2(tt, fromzone = 'Japan', tozone = 'US/Eastern')
date_extract_dailyopenclosetime(spdateref1, market = 'us')
datetime_tostring(tt)
datestring_todatetime(datelist1, format1 =  "%Y%m%d")
datetime_todate(tt)
datetime_toint(datelist1)
datetime_tointhour(datelist1)
dateint_tostring(datelist1, format1 = '%b-%y')
dateint_todatetime(datelist1)
datenumpy_todatetime(tt, islocaltime = True)
datetime_tonumpypdate(t, islocaltime = True)
date_diffindays(intdate1, intdate2)
date_finddateid(date1, dateref)
datestring_toint(datelist1)
date_as_float(dt)
date_todatetime(tlist)
date_removetimezone(datelist)
datediff_inyear(startdate, enddate)
date_generatedatetime(start = "20100101", nbday = 10, end = "")
date_add_bdays(from_date, add_days)
date_getspecificdate(datelist, datetype1 = "yearend", outputype1 = "intdate", includelastdate = True, includefirstdate = False, )
date_alignfromdateref(array1, dateref)
_date_align(dateref, datei, tmax, closei)
date_align(array1, dateref)
min_withposition(values)
max_withposition(values)
_reshape(x)
_notnone(x)
plot_price(asset, y2 = None, y3 = None, y4 = None, y5 = None, sym = None, savename1 = '', tickperday = 20, date1 = None, graphsize = (10, 5)
plot_priceintraday(data)
plot_check(close, tt0i = 20140102, tt1i = 20160815, dateref = [], sym = [], tickperday = 120)
plot_pricedate(date1, sym1, asset1, sym2 = None, bsk1 = None, verticaldate = None, savename1 = '', graphsize = (10, 5)
generate_sepvertical(asset1, tt, tmax, start = None, datebar = None)
save_asset_tofile(file1, asset1, asset2 = None, asset3 = None, date1 = None, title1 = None)
load_asset_fromfile(file1)
array_todataframe(price, symbols = None, date1 = None)
pd_dataframe_toarray(df)
isfloat(value)
isint(x)
correlation_mat(Xmat, type1 = "robust", type2 = "correl")
correl_reducebytrigger(correl2, trigger)
sk_cov_fromcorrel(correl, ret_close1)
cointegration(x, y)
causality_y1_y2(price2, price1, maxlag)
reg_slope(close, dateref, tlag, type1 = 'elasticv')
rolling_cointegration(x, y)
regression(yreturn, xreturn, type1 = "elasticv")
regression_fixedsymbolstock(sym, ret_close2, tsstart, tsample, ret_spy, spyclose, regonly = True)
regression_getpricefromww(spyclose, ww01, regasset01, ret_close2, tstart, tlag = 1)
regression_allstocks_vs_riskfactors(symstock, pricestock, symriskfac, priceriskfac, nlaglist)
getdiff_fromquotes(close, timelag)
getret_fromquotes(close, timelag = 1)
getlogret_fromquotes(close, timelag = 1)
getprice_fromret(ret, normprice = 100)
price_normalize100(ret, normprice = 100)
price_normalize_1d(ret, normprice = 100, dtype1 =  np.float32)
norm_fast(y, ny)
correl_fast(xn, y, nx)
volhisto_fromret(retbsk, t, volrange, axis = 0)
volhisto_fromprice(price, t, volrange, axis = 0)
volhistorolling_fromprice(price, volrange)
rsk_calc_all_TA(df = 'panda_dataframe')
ta_highbandtrend1(close2, type1 = 0)
pd_transform_asset(q0, q1, type1 = "spread")
calcbasket_table(wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps =  0.000)
folio_createvolta_asset(close, vol = 0.12, volrange = 120, lev = 1.0)
folio_lowcorrelation(sym01, nstock, periodlist, dateref, close1, kbenchmark, badlist, costbppa = 0.02, showgraph = True)
folio_leverageetf(price, lev = 1.0, costpa = 0.0)
folio_inverseetf(price, costpa = 0.0)
folio_longshort_unit(long1, short1, ww = [1, -1], costpa = 0.0, tlag = 1, istable = 1, wwschedule = [])
folio_longshort_unitfixed(long1, short1, nn = [1, -1], costpa = 0.0, tlag = 1, istable = 1)
folio_longshort_pct(long1, short1, ww = [1, -1], costpa = 0.0)
folio_histogram(close)
folio_voltarget(bsk, targetvol = 0.11, volrange =  90, expocap = 1.5)
folio_volta(bsk, targetvol = 0.11, volrange =  90, expocap = 1.5)
folio_volta2(bsk, riskind, par, targetvol = 0.11, volrange =  90, cap = 1.5, floor = 0.0, costbp = 0.0005)
folio_fixedunitprice(price, fixedww, costpa = 0.0)
folio_fixedweightprice(price, fixedww, costpa = 0.0)
folio_fixedweightret(ret, fixedww)
folio_cost_turnover(wwall, bsk, dateref, costbp)
folio_riskpa(ret, targetvol = 0.1, volrange = 90, cap = 1.0)
folio_perfreport_schedule(sym, dateref, close, wwind, t0, scheduleperiod = "1monthend")
folio_concenfactor2(ww, masset = 12)
calcbasket_objext(RETURN, TMAX, riskind_i, wwmat, wwasset0, ww0, nbrange, criteria)
calc_ranktable(close2, symjp1, nlag, refindex, funeval, funargs)
similarity_correl(ret_close2, funargs)
np_similarity(x, y, wwerr = [], type1 = 0)
np_distance_l1(x, y, wwerr)
np_similarity(x, y, wwerr = [], type1 = 0)
np_distance_l1(x, y, wwerr)
get(close, timelag)

-------------------------methods----------------------
index.__init__(self, id1, sym, ww, tstart)
index.close(self)
index.updatehisto(self)
index.help(self)
index.__init__(self, id1, sym, ww, tstart)
index.calc_baskettable_pct(self, type1 = "table", showdetail = 0)
index._wwpct_rebal(self, wwpct_actual, t, trebal)
index._udpate_wwindpct(self, t, bskt, hedgecost, wwpct_actual, wwpct_th)
index.calc_baskettable_unit()
folioCalc.__init__(self, sym, close, dateref)
folioCalc.set_symclose(self, sym, close, dateref)
folioCalc.setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid = "spprice", lfun = None)
folioCalc._weightcalc_generic(self, wwvec, t)
folioCalc._weightcalc_regime(self, wwvec, wwextra, t)
folioCalc._regimecalc(self, t, wwextra)
folioCalc._weightcalc_constant(self, ww2, t)
folioCalc.getweight(self)
folioCalc.calc_baskettable(self, wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps = 0.000, showdetail = 0)
folioCalc.plot(self, wwvec = None, show1 = 1, tickperday = 60)
folioCalc.multiperiod_ww(self, t)
folioCalc.help(self)
folioRiskIndicator.__init__(self, sym, close, dateref)
folioRiskIndicator.set_symclose(self, sym, close, dateref)
folioRiskIndicator.setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid = "spprice", lfun = None)
folioRiskIndicator.calcrisk(self, wwvec = [], initval = 1)
folioRiskIndicator._weightcalc_generic(self, wwvec, t)
folioRiskIndicator._weightcalc_regime(self, wwvec, wwextra, t)
folioRiskIndicator._regimecalc(self, t, wwextra)
folioRiskIndicator.calc_optimal_weight(self, maxiter = 1, name1 = '', isreset = 1, popsize = 15)
folioOptimizationF.__init__(self, sym, close, dateref)
folioOptimizationF.set_symclose(self, sym, close, dateref)
folioOptimizationF.setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid = "spprice", lfun = None)
folioOptimizationF.calcbasket_obj2(self, wwvec)
folioOptimizationF.calcbasket_obj(self, wwvec)
folioOptimizationF._loss_obj(self, ww2, wwpenalty)
folioOptimizationF._weightcalc_generic(self, wwvec, t)
folioOptimizationF._weightcalc_regime(self, wwvec, wwextra, t)
folioOptimizationF._regimecalc(self, t, wwextra)
folioOptimizationF._weightcalc_constant(self, ww2, t)
folioOptimizationF.calc_optimal_weight(self, maxiter = 1, name1 = '', isreset = 1, popsize = 15)
folioOptimizationF.getweight(self)
folioOptimizationF.calc_baskettable(self, wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps =  0.000, showdetail = 0)
folioOptimizationF.plot(self, wwvec = None, show1 = 1, tickperday = 60)
folioOptimizationF._objective_criteria(self, bsk)
folioOptimizationF.multiperiod_ww(self, t)
folioOptimizationF.help(self)
folioOptimizationF.mapping_risk_ww(self, risk, wwmat, ww2 = self.wwasset0)
folioOptimizationF._mapping_calc_risk(self, ss, tr, t, risk0)
searchSimilarity.__init__(self, filejpstock=r'E = r'E:/_data/stock/daily/20160616/jp', sym01 = ['7203'], symname = ['Toyota'], startdate =  20150101, enddate = 20160601, pricetype = "close")
searchSimilarity.load_quotes_fromdb(self, picklefile = '')
searchSimilarity.__generate_return__(self, nlag)
searchSimilarity.__overweight__(self, px)
searchSimilarity.set_searchcriteria(self, name1 = '7203', date1 = 20160301, date2 = 20160601, nlag = 1, searchperiodstart = 20120101, typesearch = "pattern2", )
searchSimilarity.launch_search(self)
searchSimilarity.show_comparison_graph(self, maxresult = 20, show_only_different_time = True, fromid = 0, fromend =  0, filenameout = '')
searchSimilarity.staticmethod(self, x)
searchSimilarity.get_rankresult(self, filetosave = '')
searchSimilarity.export_results()


utilmy\zzarchive\py2to3\report.py
-------------------------functions----------------------
map_show()
xl_create_pivot(infile, index_list = ["Manager", "Rep", "Product"], value_list = ["Price", "Quantity"])
xl_save_report(report, outfile)
xl_create_pdf()



utilmy\zzarchive\py2to3\rstatpy.py
-------------------------functions----------------------
stl(data, ns, np = None, nt = None, nl = None, isdeg = 0, itdeg = 1, ildeg = 1, nsjump = None, ntjump = None, nljump = None, ni = 2, no = 0, fulloutput = False)



utilmy\zzarchive\py2to3\utilgeo.py
-------------------------functions----------------------
df_to_geojson(df, col_properties, lat = 'latitude', lon = 'longitude')



utilmy\zzarchive\py2to3\util_min.py
-------------------------functions----------------------
os_wait_cpu(priority = 300, cpu_min = 50)
isexist(a)
isfloat(x)
isint(x)
a_isanaconda()
os_zip_checkintegrity(filezip1)
os_zipfile(folderin, folderzipname, iscompress = True)
os_zipfolder(dir_tozip = '/zdisks3/output', zipname = '/zdisk3/output.zip', dir_prefix = None, iscompress = True)
os_zipextractall(filezip_or_dir = "folder1/*.zip", tofolderextract = 'zdisk/test', isprint = 1)
os_folder_copy(src, dst, symlinks = False, pattern1 = "*.py", fun_file_toignore = None)
os_folder_create(directory)
os_folder_robocopy(from_folder = '', to_folder = '', my_log='H = 'H:/robocopy_log.txt')
os_file_replacestring1(findStr, repStr, filePath)
os_file_replacestring2(findstr, replacestr, some_dir, pattern = "*.*", dirlevel = 1)
os_file_getname(path)
os_file_getpath(path)
os_file_gettext(file1)
os_file_listall(dir1, pattern = "*.*", dirlevel = 1, onlyfolder = 0)
os_file_rename(some_dir, pattern = "*.*", pattern2 = "", dirlevel = 1)
os_print_tofile(vv, file1, mode1 = 'a')
a_get_pythonversion()
os_path_norm(pth)
os_path_change(path1)
os_path_current()
os_file_exist(file1)
os_file_size(file1)
os_file_read(file1)
os_file_mergeall(nfile, dir1, pattern1, deepness = 2)
os_path_append(p1, p2 = None, p3 = None, p4 = None)
os_split_dir_file(dirfile)
py_importfromfile(modulename, dir1)
py_memorysize(o, ids, hint = " deep_getsizeof(df_pd, set()
save(obj, folder = '/folder1/keyname', isabsolutpath = 0)
load(folder = '/folder1/keyname', isabsolutpath = 0)
save_test(folder = '/folder1/keyname', isabsolutpath = 0)
py_save_obj(obj, folder = '/folder1/keyname', isabsolutpath = 0)
py_load_obj(folder = '/folder1/keyname', isabsolutpath = 0, encoding1 = 'utf-8')
z_key_splitinto_dir_name(keyname)



utilmy\zzarchive\py2to3\util_ml.py
-------------------------functions----------------------
create_weight_variable(name, shape)
create_bias_variable(name, shape)
create_adam_optimizer(learning_rate, momentum)
tf_check()
parse_args(ppa = None, args =  {})
parse_args2(ppa = None)
tf_global_variables_initializer(sess = None)
visualize_result()

-------------------------methods----------------------
TextLoader.__init__(self, data_dir, batch_size, seq_length)
TextLoader.preprocess(self, input_file, vocab_file, tensor_file)
TextLoader.load_preprocessed(self, vocab_file, tensor_file)
TextLoader.create_batches(self)
TextLoader.next_batch(self)
TextLoader.reset_batch_pointer(self)


utilmy\zzarchive\py2to3\_HELP.py
-------------------------functions----------------------
os_compileVSsolution(dir1, flags1 = "", type1 = "devenv", compilerdir = "")
os_VS_build(self, lib_to_build)
set_rc_version(rcfile, target_version)
os_VS_start(self, version)
fun_cython(a)
fun_python(a)



utilmy\zzarchive\py2to3\__init__.py


utilmy\zzarchive\py3\util.py
-------------------------functions----------------------
session_load_function(name = 'test_20160815')
session_save_function(name = 'test')
py_save_obj_dill(obj1, keyname)
session_spyder_showall()
session_guispyder_save(filename)
session_guispyder_load(filename)
session_load(name = 'test_20160815')
session_save(name = 'test')
aa_unicode_ascii_utf8_issue()
isexist(a)
isfloat(x)
isint(x)
a_isanaconda()
a_run_ipython(cmd1)
a_autoreload()
a_get_platform()
a_start_log(id1 = '', folder = 'aaserialize/log/')
a_cleanmemory()
a_module_codesample(module_str = 'pandas')
a_module_doc(module_str = 'pandas')
a_module_generatedoc(module_str = "pandas", fileout = '')
a_info_conda_jupyter()
os_zipextractall(filezip_or_dir = "folder1/*.zip", tofolderextract = 'zdisk/test', isprint = 1)
os_folder_copy(src, dst, symlinks = False, pattern1 = "*.py", fun_file_toignore = None)
os_folder_create(directory)
os_folder_robocopy(from_folder = '', to_folder = '', my_log='H = 'H:/robocopy_log.txt')
os_file_replacestring1(findStr, repStr, filePath)
os_file_replacestring2(findstr, replacestr, some_dir, pattern = "*.*", dirlevel = 1)
os_file_getname(path)
os_file_getpath(path)
os_file_gettext(file1)
os_file_listall(dir1, pattern = "*.*", dirlevel = 1, onlyfolder = 0)
os_file_rename(some_dir, pattern = "*.*", pattern2 = "", dirlevel = 1)
os_gui_popup_show(txt)
os_print_tofile(vv, file1, mode1 = 'a')
os_path_norm(pth)
os_path_change(path1)
os_path_current()
os_file_exist(file1)
os_file_size(file1)
os_file_read(file1)
os_file_mergeall(nfile, dir1, pattern1, deepness = 2)
os_extracttext_allfile(nfile, dir1, pattern1 = "*.html", htmltag = 'p', deepness = 2)
os_path_append(p1, p2 = None, p3 = None, p4 = None)
os_split_dir_file(dirfile)
os_process_run(cmd_list = ['program', 'arg1', 'arg2'], capture_output = False)
os_process_2()
py_importfromfile(modulename, dir1)
py_memorysize(o, ids, hint = " deep_getsizeof(df_pd, set()
save(obj, folder = '/folder1/keyname', isabsolutpath = 0)
load(folder = '/folder1/keyname', isabsolutpath = 0)
save_test(folder = '/folder1/keyname', isabsolutpath = 0)
py_save_obj(obj1, keyname)
py_load_obj(folder = '/folder1/keyname', isabsolutpath = 0, encoding1 = 'utf-8')
z_key_splitinto_dir_name(keyname)
sql_getdate()
obj_getclass_of_method(meth)
obj_getclass_property(pfi)
print_topdf()
os_config_setfile(dict_params, outfile, mode1 = 'w+')
os_config_getfile(file1)
os_csv_process(file1)
read_funding_data(path)
read_funding_data(path)
read_funding_data(path)
find_fuzzy(xstring, list_string)
str_match_fuzzy(xstring, list_string)
str_parse_stringcalendar(cal)
str_make_unicode(input, errors = 'replace')
str_empty_string_array(x, y = 1)
str_empty_string_array_numpy(nx, ny = 1)
str_isfloat(value)
str_is_azchar(x)
str_is_az09char(x)
str_reindent(s, numSpaces)
str_split2(delimiters, string, maxsplit = 0)
str_split_pattern(sep2, ll, maxsplit = 0)
pd_str_isascii(x)
str_to_utf8(x)
str_to_unicode(x, encoding = 'utf-8')
web_restapi_toresp(apiurl1)
web_getrawhtml(url1)
web_importio_todataframe(apiurl1, isurl = 1)
web_getjson_fromurl(url)
web_gettext_fromurl(url, htmltag = 'p')
web_gettext_fromhtml(file1, htmltag = 'p')
web_getlink_fromurl(url)
web_send_email(FROM, recipient, subject, body, login1 = "mizenjapan@gmail.com", pss1 = "sophieelise237", server1 = "smtp.gmail.com", port1 = 465)
web_send_email_tls(FROM, recipient, subject, body, login1 = "mizenjapan@gmail.com", pss1 = "sophieelise237", server1 = "smtp.gmail.com", port1 = 465)
web_sendurl(url1)
np_minimize(fun_obj, x0 = [0.0], argext = (0, 0)
np_minimizeDE(fun_obj, bounds, name1, solver = None)
np_remove_NA_INF_2d(X)
np_addcolumn(arr, nbcol)
np_addrow(arr, nbrow)
np_int_tostr(i)
np_dictordered_create()
np_list_unique(seq)
np_list_tofreqdict(l1, wweight = [])
np_list_flatten(seq)
np_dict_tolist(dd, withkey = 0)
np_dict_tostr_val(dd)
np_dict_tostr_key(dd)
np_removelist(x0, xremove = [])
np_transform2d_int_1d(m2d, onlyhalf = False)
np_mergelist(x0, x1)
np_enumerate2(vec_1d)
np_pivottable_count(mylist)
np_nan_helper(y)
np_interpolate_nan(y)
np_and1(x, y, x3 = None, x4 = None, x5 = None, x6 = None, x7 = None, x8 = None)
np_sortcol(arr, colid, asc = 1)
np_sort(arr, colid, asc = 1)
np_ma(vv, n)
np_cleanmatrix(m)
np_torecarray(arr, colname)
np_sortbycolumn(arr, colid, asc = True)
np_sortbycol(arr, colid, asc = True)
min_kpos(arr, kth)
max_kpos(arr, kth)
np_findfirst(item, vec)
np_find(item, vec)
find(xstring, list_string)
findnone(vec)
findx(item, vec)
finds(itemlist, vec)
findhigher(x, vec)
findlower(x, vec)
np_find_minpos(values)
np_find_maxpos(values)
np_find_maxpos_2nd(numbers)
np_findlocalmax2(v, trig)
np_findlocalmin2(v, trig)
np_findlocalmax(v, trig)
np_findlocalmin(v, trig)
np_stack(v1, v2 = None, v3 = None, v4 = None, v5 = None)
np_uniquerows(a)
np_remove_zeros(vv, axis1 = 1)
np_sort(arr, colid, asc = 1)
np_memory_array_adress(x)
sql_create_dbengine(type1 = '', dbname = '', login = '', password = '', url = 'localhost', port = 5432)
sql_query(sqlr = 'SELECT ticker,shortratio,sector1_id, FROM stockfundamental', dbengine = None, output = 'df', dburl='sqlite = 'sqlite:///aaserialize/store/finviz.db')
sql_get_dbschema(dburl='sqlite = 'sqlite:///aapackage/store/yahoo.db', dbengine = None, isprint = 0)
sql_delete_table(name, dbengine)
sql_insert_excel(file1 = '.xls', dbengine = None, dbtype = '')
sql_insert_df(df, dbtable, dbengine, col_drop = ['id'], verbose = 1)
sql_insert_csv(csvfile, dbtable, dbengine, col_drop = [])
sql_insert_csv2(csvfile = '', dbtable = '', columns = [], dbengine = None, nrows =  10000)
sql_postgres_create_table(mytable = '', database = '', username = '', password = '')
sql_postgres_query_to_csv(sqlr = 'SELECT ticker,shortratio,sector1_id, FROM stockfundamental', csv_out = '')
sql_postgres_pivot()
sql_mysql_insert_excel()
sql_pivotable(dbcon, ss = 'select  ')
np_pivotable_create(table, left, top, value)
pd_info(df, doreturn = 1)
pd_info_memsize(df, memusage = 0)
pd_selectrow(df, **conditions)
pd_csv_randomread(filename, nsample = 10000, filemaxline = -1, dtype = None)
pd_array_todataframe(array, colname = None, index1 = None, dotranspose = False)
pd_dataframe_toarray(df)
pd_createdf(array1, col1 = None, idx1 = None)
pd_create_colmapdict_nametoint(df)
pd_extract_col_idx_val(df)
pd_extract_col_uniquevalue_tocsv(df, colname = '', csvfile = '')
pd_split_col_idx_val(df)
pd_splitdf_inlist(df, colid, type1 = "dict")
pd_find(df, regex_pattern = '*', col_restrict = [], isnumeric = False, doreturnposition = False)
pd_dtypes_totype2(df, columns = [], targetype = 'category')
pd_dtypes(df, columns = [], targetype = 'category')
pd_df_todict(df, colkey = 'table', excludekey = [''], onlyfirstelt =  True)
pd_applyfun_col(df, newcol, ff, use_colname = "all/[colname]")
pd_cleanquote(q)
pd_date_intersection(qlist)
pd_is_categorical(z)
pd_str_encoding_change(df, cols, fromenc = 'iso-8859-1', toenc = 'utf-8')
pd_str_unicode_tostr(df, targetype = str)
pd_dtypes_type1_totype2(df, fromtype = str, targetype = str)
pd_resetindex(df)
pd_insertdatecol(df_insider, format1="%Y-%m-%d %H = "%Y-%m-%d %H:%M:%S:%f")
pd_replacevalues(df, matrix)
pd_removerow(df, row_list_index = [23, 45])
pd_removecol(df1, name1)
pd_addcol(df1, name1 = 'new')
pd_insertcol(df, colname, vec)
pd_insertrow(df, rowval, index1 = None, isreset = 1)
pd_h5_cleanbeforesave(df)
pd_h5_addtable(df, tablename, dbfile='F = 'F:\temp_pandas.h5')
pd_h5_tableinfo(filenameh5, table)
pd_h5_dumpinfo(dbfile='E = 'E:\_data\stock\intraday_google.h5')
pd_h5_save(df, filenameh5='E = 'E:/_data/_data_outlier.h5', key = 'data')
pd_h5_load(filenameh5='E = 'E:/_data/_data_outlier.h5', table_id = 'data', exportype = "pandas", rowstart = -1, rowend = -1, cols = [])
pd_h5_fromcsv_tohdfs(dircsv = 'dir1/dir2/', filepattern = '*.csv', tofilehdfs = 'file1.h5', tablename = 'df', col_category = [], dtype0 = None, encoding = 'utf-8', chunksize =  2000000, mode = 'a', format = 'table', complib = None)
pd_np_toh5file(numpyarr, fileout = "file.h5", table1 = 'data')
date_allinfo()
date_convert(t1, fromtype, totype)
datetime_tostring(datelist1)
date_remove_bdays(from_date, add_days)
date_add_bdays(from_date, add_days)
datenumpy_todatetime(tt, islocaltime = True)
datetime_tonumpydate(t, islocaltime = True)
datestring_todatetime(datelist1, format1 =  "%Y%m%d")
datetime_toint(datelist1)
date_holiday()
date_add_bday(from_date, add_days)
dateint_todatetime(datelist1)
date_diffinday(intdate1, intdate2)
date_diffinyear(startdate, enddate)
date_diffinbday(intd2, intd1)
date_gencalendar(start = '2010-01-01', end = '2010-01-15', country = 'us')
date_finddateid(date1, dateref)
datestring_toint(datelist1)
date_now(i = 0)
date_nowtime(type1 = 'str', format1= "%Y-%m-%d %H =  "%Y-%m-%d %H:%M:%S:%f")
date_tofloat(dt)
date_generatedatetime(start = "20100101", nbday = 10, end = "")
np_numexpr_vec_calc(filename, expr, i0 = 0, imax = 1000, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
np_numexpr_tohdfs(filename, expr, i0 = 0, imax = 1000, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
np_comoment(xx, yy, nsample, kx, ky)
np_acf(data)
plot_XY(xx, yy, zcolor = None, tsize = None, title1 = '', xlabel = '', ylabel = '', figsize = (8, 6)
plot_heatmap(frame, ax = None, cmap = None, vmin = None, vmax = None, interpolation = 'nearest')
gc_map_dict_to_bq_schema(source_dict, schema, dest_dict)
aws_accesskey_get(access = '', key = '')
aws_conn_do(action = '', region = "ap-northeast-2")
aws_conn_getallregions(conn = None)
aws_conn_create(region = "ap-northeast-2", access = '', key = '')
aws_conn_getinfo(conn)
aws_s3_url_split(url)
aws_s3_getbucketconn(s3dir)
aws_s3_puto_s3(fromdir_file = 'dir/file.zip', todir = 'bucket/folder1/folder2')
aws_s3_getfrom_s3(froms3dir = 'task01/', todir = '', bucket_name = 'zdisk')
aws_s3_folder_printtall(bucket_name = 'zdisk')
aws_s3_file_read(filepath, isbinary = 1)
aws_ec2_python_script(script_path, args1, host)
aws_ec2_create_con(contype = 'sftp/ssh', host = 'ip', port = 22, username = 'ubuntu', keyfilepath = '', password = '', keyfiletype = 'RSA', isprint = 1)
aws_ec2_allocate_elastic_ip(instance_id, region = "ap-northeast-2")
googledrive_get()
googledrive_put()
googledrive_list()
os_processify_fun(func)
ztest_processify()
date_getspecificdate(datelist, datetype1 = "yearend", outputype1 = "intdate", includelastdate = True, includefirstdate = False, )

-------------------------methods----------------------
testclass.__init__(self, x)
testclass.z_autotest(self)
FundingRecord.parse(klass, row)
FundingRecord.__str__(self)
aws_ec2_ssh.__init__(self, hostname, username = 'ubuntu', key_file = None, password = None)
aws_ec2_ssh.command(self, cmd)
aws_ec2_ssh.put(self, localfile, remotefile)
aws_ec2_ssh.put_all(self, localpath, remotepath)
aws_ec2_ssh.get(self, remotefile, localfile)
aws_ec2_ssh.sftp_walk(self, remotepath)
aws_ec2_ssh.get_all(self, remotepath, localpath)
aws_ec2_ssh.write_command(self, text, remotefile)
aws_ec2_ssh.python_script(self, script_path, args1)
aws_ec2_ssh.command_list(self, cmdlist)
aws_ec2_ssh.listdir(self, remotedir)
aws_ec2_ssh.jupyter_kill(self)
aws_ec2_ssh.jupyter_start(self)
aws_ec2_ssh.cmd2(self, cmd1)
aws_ec2_ssh._help_ssh(self)


utilmy\zzarchive\storage\alldata.py


utilmy\zzarchive\storage\allmodule.py
-------------------------functions----------------------
aa_isanaconda()



utilmy\zzarchive\storage\benchmarktest.py
-------------------------functions----------------------
payoff1(pricepath)
payoffeuro1(st)
payoff2(pricepath)
payoffeuro1(st)



utilmy\zzarchive\storage\codeanalysis.py
-------------------------functions----------------------
wi(*args)
printinfile(vv, file2)
wi2(*args)
indent()
dedent()
describe_builtin(obj)
describe_func(obj, method = False)
describe_klass(obj)
describe(obj)
describe_builtin2(obj, name1)
describe_func2(obj, method = False, name1 = '')
describe_func3(obj, method = False, name1 = '')
describe_klass2(obj, name1 = '')
describe2(module, type1 = 0)
getmodule_doc(module1, file2 = '')



utilmy\zzarchive\storage\dbcheck.py


utilmy\zzarchive\storage\derivatives.py
-------------------------functions----------------------
loadbrownian(nbasset, step, nbsimul)
dN(d)
dN2d(x, y)
N(d)
d1f(St, K, t, T, r, d, vol)
d2f(St, K, t, T, r, d, vol)
bsbinarycall(S0, K, t, T, r, d, vol)
bscall(S0, K, t, T, r, d, vol)
bsput(S0, K, t, T, r, d, vol)
bs(S0, K, t, T, r, d, vol)
bsdelta(St, K, t, T, r, d, vol, cp1)
bsgamma(St, K, t, T, r, d, vol, cp)
bsstrikedelta(s0, K, t, T, r, d, vol, cp1)
bsstrikegamma(s0, K, t, T, r, d, vol)
bstheta(St, K, t, T, r, d, vol, cp)
bsrho(St, K, t, T, r, d, vol, cp)
bsvega(St, K, t, T, r, d, vol, cp)
bsdvd(St, K, t, T, r, d, vol, cp)
bsvanna(St, K, t, T, r, d, vol, cp)
bsvolga(St, K, t, T, r, d, vol, cp)
bsgammaspot(St, K, t, T, r, d, vol, cp)
gdelta(St, K, t, T, r, d, vol, pv)
ggamma(St, K, t, T, r, d, vol, pv)
gvega(St, K, t, T, r, d, vol, pv)
gtheta(St, K, t, T, r, d, vol, pv)
genmatrix(ni, nj, gg)
gensymmatrix(ni, nj, pp)
timegrid(timestep, maturityyears)
generateall_multigbm1(process, ww, s0, mu, vol, corrmatrix, timegrid, nbsimul, nproc = -1, type1 = -1, strike = 0.0, cp = 1)
logret_to_ret(log_returns)
logret_to_price(s0, log_ret)
brownian_logret(mu, vol, timegrid)
brownian_process(s0, vol, timegrid)
gbm_logret(mu, vol, timegrid)
gbm_process(s0, mu, vol, timegrid)
gbm_process_euro(s0, mu, vol, timegrid)
gbm_process2(s0, mu, vol, timegrid)
generateallprocess(process, params01, timegrid1, nbsimul)
generateallprocess_gbmeuro(process, params01, timegrid1, nbsimul)
getpv(discount, payoff, allpriceprocess)
multigbm_processfast(s0, voldt, drift, upper_cholesky, nbasset, n, kk)
getbrowniandata(nbasset, step, simulk)
multigbm_processfast2(s0, voldt, drift, upper_cholesky, nbasset, n, kk)
generateallmultigbmfast(process, s0, mu, vol, corrmatrix, timegrid, nbsimul, type1)
multigbm_processfast3(s0, voldt, drift, upper_cholesky, nbasset, n, kk)
generateallmultigbmfast2(process, s0, mu, vol, corrmatrix, timegrid, nbsimul, type1)
multibrownian_logret(mu, vol, corrmatrix, timegrid)
multigbm_logret(mu, vol, corrmatrix, timegrid)
multilogret_to_price(s0, log_ret)
multigbm_process(s0, voldt, drift, upper_cholesky, nbasset, n, kk)
generateallmultiprocess(process, s0, mu, vol, corrmatrix, timegrid, nbsimul)
jump_process(lamda, jumps_mu, jumps_vol, timegrid)
gbmjump_logret(s0, mu, vol, lamda, jump_mu, jump_vol, timegrid)
gbmjump_process(s0, mu, vol, lamda, jump_mu, jump_vol, timegrid)
lgnormalmoment1(ww, fft, vol, corr, tt)
lgnormalmoment2(ww, fft, vol, corr, tt)
lgnormalmoment3(ww, fft, vol, corr, tt)
lgnormalmoment4(ww, fft, vol, corr, tt)
solve_momentmatch3(ww, b0, fft, vol, corr, tt)
savebrownian(nbasset, step, nbsimul)
plot_greeks(function, greek)
plot_greeks(function, greek)
plot_values(function)
CRR_option_value(S0, K, T, r, vol, otype, M = 4)



utilmy\zzarchive\storage\dl_utils.py
-------------------------functions----------------------
save_weights(file, tuple_weights)
save_prediction(file, prediction)
log(msg, file = "")
logfile(msg, file)
log_p(msg, file = "")
init_weight(hidden1, hidden2, acti_type)
get_all_data(file)
get_batch_data(file, index, size)
get_xy(line)
file_len(fname)
feats_len(fname)



utilmy\zzarchive\storage\excel.py
-------------------------functions----------------------
get_workbook_name()
double_sum(x, y)
add_one(data)
matrix_mult(x, y)
npdot()



utilmy\zzarchive\storage\global01.py


utilmy\zzarchive\storage\installNewPackage.py


utilmy\zzarchive\storage\java.py
-------------------------functions----------------------
importJAR(path1 = "", path2 = "", path3 = "", path4 = "")
listallfile(some_dir, pattern = "*.*", dirlevel = 1)
importFolderJAR(dir1 = "", dirlevel = 1)
importFromMaven()
showLoadedClass()
inspectJAR(dir1)
loadSingleton(class1)
java_print(x)
compileJAVA(javafile)
writeText(text, filename)
compileJAVAtext(classname, javatxt, path1 = "")



utilmy\zzarchive\storage\multiprocessfunc.py
-------------------------functions----------------------
multigbm_paralell_func(nbsimul, ww, voldt, drift, upper_cholesky, nbasset, n, price, type1 = 0, strike = 0, cp = 1)
func(val, lock)
multigbm_processfast7(nbsimul, s0, voldt, drift, upper_cholesky, nbasset, n, price)
bm_generator(bm, dt, n, type1)
merge(d2)
integratenp2(its, nchunk)
integratenp(its, nchunk)
integratene(its)
parzen_estimation(x_samples, point_x, h)
init2(d)
init_global1(l, r)
np_sin(value)
ne_sin(x)
res_shared2()
list_append(count, id, out_list)



utilmy\zzarchive\storage\panda_util.py
-------------------------functions----------------------
excel_topandas(filein, fileout)
panda_toexcel()
panda_todabatase()
database_topanda()
sqlquery_topanda()
folder_topanda()
panda_tofolder()
numpy_topanda(vv, fileout = "", colname = "data")
panda_tonumpy(filename, nsize, tablen = 'data')
df_topanda(vv, filenameh5, colname = 'data')
load_frompanda(filenameh5, colname = "data")
csv_topanda(filein1, filename, tablen = 'data', lineterminator=",")
getrandom_tonumpy(filename, nbdim, nbsample, tablen = 'data')
numexpr_topanda(filename, i0 = 0, imax = 1000, expr, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
excel_topanda(filein, fileout)
array_toexcel(vv, wk, r1)subset = 'rownum', take_last=True)level=0))a) = True)level=0))a):)
unique_rows(a)
remove_zeros()
sort_array()



utilmy\zzarchive\storage\portfolio.py
-------------------------functions----------------------
data_jpsector()
date_earningquater(t1)
date_is_3rdfriday(s)
date_option_expiry(date)
date_find_kday_fromintradaydate(kintraday, intradaydate, dailydate)
date_find_kintraday_fromdate(d1, intradaydate1, h1 = 9, m1 = 30)
date_find_intradateid(datetimelist, stringdate = ['20160420223000'])
datetime_convertzone1_tozone2(tt, fromzone = 'Japan', tozone = 'US/Eastern')
date_extract_dailyopenclosetime(spdateref1, market = 'us')
datetime_tostring(datelist1)
datestring_todatetime(datelist1, format1 =  "%Y%m%d")
datetime_todate(tt)
datetime_toint(datelist1)
datetime_tointhour(datelist1)
dateint_tostring(datelist1, format1 = '%b-%y')
dateint_todatetime(datelist1)
datenumpy_todatetime(tt, islocaltime = True)
datetime_tonumpypdate(t, islocaltime = True)
date_diffindays(intdate1, intdate2)
date_finddateid(date1, dateref)
datestring_toint(datelist1)
date_as_float(dt)
datediff_inyear(startdate, enddate)
date_generatedatetime(start = "20100101", nbday = 10, end = "")
date_add_bdays(from_date, add_days)
date_getspecificdate(datelist, datetype1 = "yearend", outputype1 = "intdate", includelastdate = True, includefirstdate = False, )
date_alignfromdateref(array1, dateref)
_date_align(dateref, datei, tmax, closei)
date_align(array1, dateref)
min_withposition(values)
max_withposition(values)
_reshape(x)
_notnone(x)
plot_price(asset, y2 = None, y3 = None, y4 = None, y5 = None, sym = None, savename1 = '', tickperday = 20, date1 = None, graphsize = (10, 5)
plot_priceintraday(data)
plot_pricedate(date1, sym1, asset1, sym2 = None, bsk1 = None, verticaldate = None, savename1 = '', graphsize = (10, 5)
generate_sepvertical(asset1, tt, tmax, start = None, datebar = None)
save_asset_tofile(file1, asset1, asset2 = None, asset3 = None, date1 = None, title1 = None)
load_asset_fromfile(file1)
array_todataframe(price, symbols = None, date1 = None)
dataframe_toarray(df)
isfloat(value)
isint(x)
correlation_mat(matx, type1 = "robust", type2 = "correl")
correl_reducebytrigger(correl2, trigger)
sk_cov_fromcorrel(correl, ret_close1)
cointegration(x, y)
causality_y1_y2(price2, price1, maxlag)
rolling_cointegration(x, y)
regression(yreturn, xreturn, type1 = "elasticv")
regression_fixedsymbolstock(sym, ret_close2, tsstart, tsample, ret_spy, spyclose, regonly = True)
regression_getpricefromww(spyclose, ww01, regasset01, ret_close2, tstart, tlag = 1)
regression_allstocks_vs_riskfactors(symstock, pricestock, symriskfac, priceriskfac, nlaglist)
getdiff_fromquotes(close, timelag)
getret_fromquotes(close, timelag = 1)
getlogret_fromquotes(close, timelag = 1)
getprice_fromret(ret, normprice = 100)
price_normalize100(ret, normprice = 100)
price_normalize_1d(ret, normprice = 100, dtype1 =  np.float16)
norm_fast(y, ny)
correl_fast(xn, y, nx)
volhisto_fromret(retbsk, t, volrange, axis = 0)
volhisto_fromprice(price, t, volrange, axis = 0)
rsk_calc_all_TA(df = 'panda_dataframe')
ta_lowbandtrend1(close2, type1 = 0)
ta_highbandtrend1(close2, type1 = 0)
pd_transform_asset(q0, q1, type1 = "spread")
calcbasket_table(wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps =  0.000)
folio_lowcorrelation(sym01, nstock, periodlist, dateref, close1, kbenchmark, badlist, costbppa = 0.02, showgraph = True)
folio_inverseetf(price, costpa = 0.0)
folio_voltarget(bsk, targetvol = 0.11, volrange =  90, expocap = 1.5)
folio_volta(bsk, targetvol = 0.11, volrange =  90, expocap = 1.5)
folio_fixedweightprice(price, fixedww, costpa = 0.0)
folio_fixedweightret(ret, fixedww)
folio_cost_turnover(wwall, bsk, dateref)
folio_riskpa(ret, targetvol = 0.1, volrange = 90)
objective_criteria(bsk, criteria, date1 = None)
calcbasket_obj(wwvec, *data)
calc_optimal_weight(args, bounds, maxiter = 1)
fitness(p)
np_countretsign(x)
np_trendtest(x, alpha  =  0.05)
correl_rankbystock(stkid = [2, 5, 6], correl = [[1, 0], [0, 1]])
calc_print_correlrank(close2, symjp1, nlag, refindexname, toprank2 = 5, customnameid = [], customnameid2 = [])
calc_ranktable(close2, symjp1, nlag, refindex, funeval, funargs)
similarity_correl(ret_close2, funargs)
np_similarity(x, y, wwerr = [], type1 = 0)
np_distance_l1(x, y, wwerr)
imp_findticker(tickerlist, sym01, symname)
imp_close_dateref(sym01, sdate = 20100101, edate = 20160628, datasource = '', typeprice = "close")
imp_yahooticker(symbols, start = "20150101", end = "20160101", type1 = 1)
imp_errorticker(symbols, start = "20150101", end = "20160101")
imp_yahoo_financials_url(ticker_symbol, statement = "is", quarterly = False)
imp_yahoo_periodic_figure(soup, yahoo_figure)
imp_googleIntradayQuoteSave(name1, date1, inter, tframe, dircsv)
imp_googleQuoteSave(name1, date1, date2, dircsv)
imp_googleQuoteList(symbols, date1, date2, inter = 23400, tframe = 2000, dircsv = '', intraday1 = True)
pd_filterbydate(df, dtref = None, start='2016-06-06 00 = '2016-06-06 00:00:00', end='2016-06-14 00 = '2016-06-14 00:00:00', freq = '0d0h05min', timezone = 'Japan')
imp_panda_db_dumpinfo(dbfile='E = 'E:\_data\stock\intraday_google.h5')
imp_numpyclose_frompandas(dbfile, symlist = [], t0 = 20010101, t1 = 20010101, priceid = "close", maxasset = 2500, tmax2 = 2000)
imp_quotes_fromtxt(stocklist01, filedir='E = 'E:/_data/stock/daily/20160610/jp', startdate = 20150101, endate = 20160616)
imp_quotes_errordate(quotes, dateref)
imp_getcsvname(name1, date1, inter, tframe)
imp_quote_tohdfs(sym, qqlist, filenameh5, fromzone = 'Japan', tozone = 'UTC')
date_todatetime(tlist)
date_removetimezone(datelist)
imp_csvquote_topanda(file1, filenameh5, dfname = 'sym1', fromzone = 'Japan', tozone = 'UTC')
imp_panda_insertfoldercsv(dircsv, filepd= r'E =  r'E:\_data\stock\intraday_google.h5', fromtimezone = 'Japan', tozone = 'UTC')
imp_panda_checkquote(quotes)
imp_panda_getquote(filenameh5, dfname = "data")
imp_pd_merge_database(filepdfrom, filepdto)
imp_panda_getListquote(symbols, close1 = 'close', start='12/18/2015 00 = '12/18/2015 00:00:00+00:00', end='3/1/2016 00 = '3/1/2016 00:00:00+00:00', freq = '0d0h10min', filepd= 'E =  'E:\_data\stock\intraday_google.h5', tozone = 'Japan', fillna = True, interpo = True)
imp_panda_cleanquotes(df, datefilter)
imp_panda_storecopy()
imp_panda_removeDuplicate(filepd=  'E =   'E:\_data\stock\intraday_google.h5')
calc_statestock(close2, dateref, symfull)
imp_screening_addrecommend(string1, dbname = 'stock_recommend')
imp_finviz()
imp_finviz_news()
imp_finviz_financials()
get_price2book(symbol)

-------------------------methods----------------------
index.__init__(self, id1, sym, ww, tstart)
index.close()
index.updatehisto()
index.help()
index._statecalc(self)
index._objective_criteria(self, bsk)
index.calcbasket_obj(self, wwvec)
index.calc_optimal_weight(self, maxiter = 1)
index._weightcalc_generic(self, wwvec, t)
index._weightcalc_constant(self, ww2, t)
index._weightcalc_regime2(self, wwvec, t)
searchSimilarity.__init__(self, filejpstock=r'E = r'E:/_data/stock/daily/20160616/jp', sym01 = ['7203'], symname = ['Toyota'], startdate =  20150101, enddate = 20160601, pricetype = "close")
searchSimilarity.load_quotes_fromdb(self, picklefile = '')
searchSimilarity.__generate_return__(self, nlag)
searchSimilarity.__overweight__(self, px)
searchSimilarity.set_searchcriteria(self, name1 = '7203', date1 = 20160301, date2 = 20160601, nlag = 1, searchperiodstart = 20120101, typesearch = "pattern2", )
searchSimilarity.launch_search(self)
searchSimilarity.show_comparison_graph(self, maxresult = 20, show_only_different_time = True, fromid = 0, fromend =  0, filenameout = '')
searchSimilarity.staticmethod(self, x)
searchSimilarity.get_rankresult(self, filetosave = '')
searchSimilarity.export_results(self, filename)
Quote.__init__(self)
Quote.append(self, dt, open_, high, low, close, volume)
Quote.to_csv(self)
Quote.write_csv(self, filename)
Quote.read_csv(self, filename)
Quote.__repr__(self)
googleIntradayQuote.__init__(self, symbol, interval_seconds = 300, num_days = 5)
googleQuote.__init__(self, symbol, start_date, end_date = datetime.date.today()


utilmy\zzarchive\storage\rec_data.py
-------------------------functions----------------------
_get_movielens_path()
_download_movielens(dest_path)
_get_raw_movielens_data()
_parse(data)
_build_interaction_matrix(rows, cols, data)
_get_movie_raw_metadata()
get_movielens_item_metadata(use_item_ids)
get_dense_triplets(uids, pids, nids, num_users, num_items)
get_triplets(mat)
get_movielens_data()



utilmy\zzarchive\storage\rec_metrics.py
-------------------------functions----------------------
predict(model, uid, pids)
precision_at_k(model, ground_truth, k, user_features = None, item_features = None)
full_auc(model, ground_truth)



utilmy\zzarchive\storage\sobol.py
-------------------------functions----------------------
convert_csv2hd5f(filein1, filename)
getrandom_tonumpy(filename, nbdim, nbsample)
comoment(xx, yy, nsample, kx, ky)
acf(data)
getdvector(dimmax, istart, idimstart)
pathScheme_std(T, n, zz)
pathScheme_bb(T, n, zz)
pathScheme_(T, n, zz)
testdensity(nsample, totdim, bin01, Ti = -1)
plotdensity(nsample, totdim, bin01, tit0, Ti = -1)
testdensity2d(nsample, totdim, bin01, nbasset)
lognormal_process2d(a1, z1, a2, z2, k)
testdensity2d2(nsample, totdim, bin01, nbasset, process01 = lognormal_process2d, a1 = 0.25, a2 = 0.25, kk = 1)
call_process(a, z, k)
binary_process(a, z, k)
pricing01(totdim, nsample, a, strike, process01, aa = 0.25, itmax = -1, tt = 10)
plotdensity2(nsample, totdim, bin01, tit0, process01, vol = 0.25, tt = 5, Ti = -1)
Plot2D_random_show(dir1, title1, dimxmax, dimymax, dimstep, samplejump, nsamplegraph)
Plot2D_random_save(dir1, title1, dimxmax, dimymax, dimstep, samplejump, nsamplegraph, )
getoutlier_fromrandom(filename, jmax1, imax1, isamplejum, nsample, fileoutlier=   'E =    'E:\_data\_QUASI_SOBOL_gaussian_outlier.h5')
numexpr_vect_calc(filename, i0, imax, expr, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
getoutlier_fromrandom_fast(filename, jmax1, imax1, isamplejum, nsample, trigger1 = 0.28, fileoutlier=   'E =    'E:\_data\_QUASI_SOBOL_gaussian_outlier.h5')
outlier_clean(vv2)
overwrite_data(fileoutlier, vv2)
doublecheck_outlier(fileoutlier, ijump, nsample = 4000, trigger1 = 0.1, )
plot_outlier(fileoutlier, kk)
permute(yy, kmax)
permute2(xx, yy, kmax)



utilmy\zzarchive\storage\stateprocessor.py
-------------------------functions----------------------
sort(x, col, asc)
perf(close, t0, t1)
and2(tuple1)
ff(x, symfull = symfull)
gap(close, t0, t1, lag)
process_stock(stkstr, show1 = 1)
printn(ss, symfull = symfull, s1 = s1)
show(ll, s1 = s1)
get_treeselect(stk, s1 = s1, xnewdata = None, newsample = 5, show1 = 1, nbtree = 5, depthtree = 10)
store_patternstate(tree, sym1, theme, symfull = symfull)
load_patternstate(name1)
get_stocklist(clf, s11, initial, show1 = 1)



utilmy\zzarchive\storage\symbolicmath.py
-------------------------functions----------------------
spp()
print2(a0, a1 = '', a2 = '', a3 = '', a4 = '', a5 = '', a6 = '', a7 = '', a8 = '')
factorpoly(pp)
EEvarbrownian(ff1d)
EEvarbrownian2d(ff)
lagrangian2d(ll)
decomposecorrel(m1)
nn(x)
nn2(x, y, p)
dnn2(x, y, p)
dnn(x, y, p)
taylor2(ff, x0, n)
diffn(ff, x0, kk)
dN(x)
N(x)
d1f(St, K, t, T, r, d, vol)
d2f(St, K, t, T, r, d, vol)
d1xf(St, K, t, T, r, d, vol)
d2xf(St, K, t, T, r, d, vol)
bsbinarycall(s0, K, t, T, r, d, vol)
bscall(s0, K, t, T, r, d, vol)
bsput(s0, K, t, T, r, d, vol)
bs(s0, K, t, T, r, d, vol)
bsdelta(St, K, t, T, r, d, vol, cp1)
bsstrikedelta(s0, K, t, T, r, d, vol, cp1)
bsstrikegamma(s0, K, t, T, r, d, vol)
bsgamma(St, K, t, T, r, d, vol, cp)
bstheta(St, K, t, T, r, d, vol, cp)
bsrho(St, K, t, T, r, d, vol, cp)
bsvega(St, K, t, T, r, d, vol, cp)
bsdvd(St, K, t, T, r, d, vol, cp)
bsvanna(St, K, t, T, r, d, vol, cp)
bsvolga(St, K, t, T, r, d, vol, cp)
bsgammaspot(St, K, t, T, r, d, vol, cp)



utilmy\zzarchive\storage\technical_indicator.py
-------------------------functions----------------------
np_find(item, vec)
np_find_minpos(values)
np_find_maxpos(values)
date_earningquater(t1)
date_option_expiry(date)
linearreg(a, *args)
np_sortbycolumn(arr, colid, asc = True)
np_findlocalmax(v)
findhigher(item, vec)
findlower(item, vec)
np_findlocalmin(v)
supportmaxmin1(df1)
RET(df, n)
qearning_dist(df)
optionexpiry_dist(df)
nbtime_reachtop(df, n, trigger = 0.005)
nbday_high(df, n)
distance_day(df, tk, tkname)
distance(df, tk, tkname)
MA(df, n)
EMA(df, n)
MOM(df, n)
ROC(df, n)
ATR(df, n)
BBANDS(df, n)
PPSR(df)
STOK(df)
STO(df)
TRIX(df, n)
ADX(df, n, n_ADX)
MACD(df, n_fast, n_slow)
MassI(df)
Vortex(df, n)
KST(df, r1, r2, r3, r4, n1, n2, n3, n4)
RSI(df, n = 14)
RMI(df, n = 14, m = 10)
TSI(df, r, s)
ACCDIST(df, n)
Chaikin(df)
MFI(df, n)
OBV(df, n)
FORCE(df, n)
EOM(df, n)
CCI(df, n)
COPP(df, n)
KELCH(df, n)
ULTOSC(df)
DONCH(df, n)
STDDEV(df, n)
RWI(df, nn, nATR)
nbday_low(df, n)
nbday_high(df, n)



utilmy\zzarchive\storage\testmulti.py
-------------------------functions----------------------
mc01()
mc02()
serial(samples, x, widths)
multiprocess(processes, samples, x, widths)
test01()
random_tree(Data)
random_tree(Data)
test01()



utilmy\zzarchive\storage\theano_imdb.py
-------------------------functions----------------------
prepare_data(seqs, labels, maxlen = None)
get_dataset_file(dataset, default_dataset, origin)
load_data(path = "imdb.pkl", n_words = 100000, valid_portion = 0.1, maxlen = None, sort_by_len = True)



utilmy\zzarchive\storage\theano_lstm.py
-------------------------functions----------------------
numpy_floatX(data)
get_minibatches_idx(n, minibatch_size, shuffle = False)
get_dataset(name)
zipp(params, tparams)
unzip(zipped)
dropout_layer(state_before, use_noise, trng)
_p(pp, name)
init_params(options)
load_params(path, params)
init_tparams(params)
get_layer(name)
ortho_weight(ndim)
param_init_lstm(options, params, prefix = 'lstm')
lstm_layer(tparams, state_below, options, prefix = 'lstm', mask = None)
sgd(lr, tparams, grads, x, mask, y, cost)
adadelta(lr, tparams, grads, x, mask, y, cost)
rmsprop(lr, tparams, grads, x, mask, y, cost)
build_model(tparams, options)
pred_probs(f_pred_prob, prepare_data, data, iterator, verbose = False)
pred_error(f_pred, prepare_data, data, iterator, verbose = False)
train_lstm(dim_proj = 128, # word embeding dimension and LSTM number of hidden units.patience = 10, # Number of epoch to wait before early stop if no progressmax_epochs = 5000, # The maximum number of epoch to rundispFreq = 10, # Display to stdout the training progress every N updatesdecay_c = 0., # Weight decay for the classifier applied to the U weights.not used for adadelta and rmsprop)n_words = 10000, # Vocabulary sizeprobably need momentum and decaying learning rate).encoder = 'lstm', # TODO: can be removed must be lstm.saveto = 'lstm_model.npz', # The best model will be saved therevalidFreq = 370, # Compute the validation error after this number of update.saveFreq = 1110, # Save the parameters after every saveFreq updatesmaxlen = 100, # Sequence longer then this get ignoredbatch_size = 16, # The batch size during training.valid_batch_size = 64, # The batch size used for validation/test set.dataset = 'imdb', noise_std = 0., use_dropout = True, # if False slightly faster, but worst test errorreload_model = None, # Path to a saved model we want to start from.test_size = -1, # If >0, we keep only this number of test example.)



utilmy\zzarchive\zzarchive\zutil.py
-------------------------functions----------------------
session_load_function(name = "test_20160815")
session_save_function(name = "test")
py_save_obj_dill(obj1, keyname = "", otherfolder = 0)
aa_unicode_ascii_utf8_issue()
isfloat(x)
isint(x)
isanaconda()
a_run_ipython(cmd1)
py_autoreload()
os_platform()
a_start_log(id1 = "", folder = "aaserialize/log/")
a_cleanmemory()
a_info_conda_jupyter()
a_run_cmd(cmd1)
a_help()
print_object_tofile(vv, txt, file1="d = "d:/regression_output.py")
print_progressbar(iteration, total, prefix = "", suffix = "", decimals = 1, bar_length = 100)
os_zip_checkintegrity(filezip1)
os_zipfile(folderin, folderzipname, iscompress = True)
os_zipfolder(dir_tozip = "/zdisks3/output", zipname = "/zdisk3/output.zip", dir_prefix = True, iscompress=Trueimport shutil_ = iscompressdir_tozip = dir_tozip if dir_tozip[-1] != "/" else dir_tozip[ = Trueimport shutil_ = iscompressdir_tozip = dir_tozip if dir_tozip[-1] != "/" else dir_tozip[:-1]if dir_prefix:)
os_zipextractall(filezip_or_dir = "folder1/*.zip", tofolderextract = "zdisk/test", isprint = 1)
os_folder_copy(src, dst, symlinks = False, pattern1 = "*.py", fun_file_toignore = None)
os_folder_create(directory)
os_folder_robocopy(from_folder = "", to_folder = "", my_log="H = "H:/robocopy_log.txt")
os_file_replace(source_file_path, pattern, substring)
os_file_replacestring1(find_str, rep_str, file_path)
os_file_replacestring2(findstr, replacestr, some_dir, pattern = "*.*", dirlevel = 1)
os_file_getname(path)
os_file_getpath(path)
os_file_gettext(file1)
os_file_listall(dir1, pattern = "*.*", dirlevel = 1, onlyfolder = 0)
_os_file_search_fast(fname, texts = None, mode = "regex/str")
os_file_search_content(srch_pattern = None, mode = "str", dir1 = "", file_pattern = "*.*", dirlevel = 1)
os_file_rename(some_dir, pattern = "*.*", pattern2 = "", dirlevel = 1)
os_gui_popup_show(txt)
os_print_tofile(vv, file1, mode1 = "a")
os_path_norm(pth)
os_path_change(path1)
os_path_current()
os_file_exist(file1)
os_file_size(file1)
os_file_read(file1)
os_file_isame(file1, file2)
os_file_get_extension(file_path)
os_file_normpath(path)
os_folder_is_path(path_or_stream)
os_file_get_path_from_stream(maybe_stream)
os_file_try_to_get_extension(path_or_strm)
os_file_are_same_file_types(paths)
os_file_norm_paths(paths, marker = "*")
os_file_mergeall(nfile, dir1, pattern1, deepness = 2)
os_file_extracttext(output_file, dir1, pattern1 = "*.html", htmltag = "p", deepness = 2)
os_path_append(p1, p2 = None, p3 = None, p4 = None)
os_wait_cpu(priority = 300, cpu_min = 50)
os_split_dir_file(dirfile)
os_process_run(cmd_list, capture_output = False)
os_process_2()
py_importfromfile(modulename, dir1)
py_memorysize(o, ids, hint = " deep_getsizeof(df_pd, set()
save(obj, folder = "/folder1/keyname", isabsolutpath = 0)
load(folder = "/folder1/keyname", isabsolutpath = 0)
save_test(folder = "/folder1/keyname", isabsolutpath = 0)
py_save_obj(obj1, keyname = "", otherfolder = 0)
py_load_obj(folder = "/folder1/keyname", isabsolutpath = 0, encoding1 = "utf-8")
z_key_splitinto_dir_name(keyname)
os_config_setfile(dict_params, outfile, mode1 = "w+")
os_config_getfile(file1)
os_csv_process(file1)
pd_toexcel(df, outfile = "file.xlsx", sheet_name = "sheet1", append = 1, returnfile = 1)
pd_toexcel_many(outfile = "file1.xlsx", df1 = None, df2 = None, df3 = None, df4 = None, df5 = None, df6 = Nonedf1, outfile, sheet_name="df1")if df2 is not None = "df1")if df2 is not None:)
find_fuzzy(xstring, list_string)
str_match_fuzzy(xstring, list_string)
str_parse_stringcalendar(cal)
str_make_unicode(input_str, errors = "replace")
str_empty_string_array(x, y = 1)
str_empty_string_array_numpy(nx, ny = 1)
str_isfloat(value)
str_is_azchar(x)
str_is_az09char(x)
str_reindent(s, num_spaces)
str_split2(delimiters, string, maxsplit = 0)
str_split_pattern(sep2, ll, maxsplit = 0)
pd_str_isascii(x)
str_to_utf8(x)
str_to_unicode(x, encoding = "utf-8")
np_minimize(fun_obj, x0 = None, argext = (0, 0)
np_minimize_de(fun_obj, bounds, name1, maxiter = 10, popsize = 5, solver = None)
np_remove_na_inf_2d(x)
np_addcolumn(arr, nbcol)
np_addrow(arr, nbrow)
np_int_tostr(i)
np_dictordered_create()
np_list_unique(seq)
np_list_tofreqdict(l1, wweight = None)
np_list_flatten(seq)
np_dict_tolist(dd, withkey = 0)
np_dict_tostr_val(dd)
np_dict_tostr_key(dd)
np_removelist(x0, xremove = None)
np_transform2d_int_1d(m2d, onlyhalf = False)
np_mergelist(x0, x1)
np_enumerate2(vec_1d)
np_pivottable_count(mylist)
np_nan_helper(y)
np_interpolate_nan(y)
np_and1(x, y, x3 = None, x4 = None, x5 = None, x6 = None, x7 = None, x8 = None)
np_sortcol(arr, colid, asc = 1)
np_ma(vv, n)
np_cleanmatrix(m)
np_torecarray(arr, colname)
np_sortbycolumn(arr, colid, asc = True)
np_sortbycol(arr, colid, asc = True)
np_min_kpos(arr, kth)
np_max_kpos(arr, kth)
np_findfirst(item, vec)
np_find(item, vec)
find(xstring, list_string)
findnone(vec)
findx(item, vec)
finds(itemlist, vec)
findhigher(x, vec)
findlower(x, vec)
np_find_minpos(values)
np_find_maxpos(values)
np_find_maxpos_2nd(numbers)
np_findlocalmax2(v, trig)
np_findlocalmin2(v, trig)
np_findlocalmax(v, trig)
np_findlocalmin(v, trig)
np_stack(v1, v2 = None, v3 = None, v4 = None, v5 = None)
np_uniquerows(a)
np_remove_zeros(vv, axis1 = 1)
np_sort(arr, colid, asc = 1)
np_memory_array_adress(x)
np_pivotable_create(table, left, top, value)
pd_info(df, doreturn = 1)
pd_info_memsize(df, memusage = 0)
pd_row_findlast(df, colid = 0, emptyrowid = None)
pd_row_select(df, **conditions)
pd_csv_randomread(filename, nsample = 10000, filemaxline = -1, dtype = None)
pd_array_todataframe(array, colname = None, index1 = None, dotranspose = False)
pd_dataframe_toarray(df)
pd_createdf(array1, col1 = None, idx1 = None)
pd_create_colmapdict_nametoint(df)
pd_extract_col_idx_val(df)
pd_extract_col_uniquevalue_tocsv(df, colname = "", csvfile = "")
pd_split_col_idx_val(df)
pd_splitdf_inlist(df, colid, type1 = "dict")
pd_find(df, regex_pattern = "*", col_restrict = None, isnumeric = False, doreturnposition = False)
pd_dtypes_totype2(df, columns = ()
pd_dtypes(df, columns = ()
pd_df_todict2(df, colkey = "table", excludekey = ("", )
pd_df_todict(df, colkey = "table", excludekey = ("", )
pd_col_addfrom_dfmap(df, dfmap, colkey, colval, df_colused, df_colnew, exceptval = -1, inplace = Truedfmap, colkey = colkey, colval=colval)rowi) = colval)rowi):)
pd_applyfun_col(df, newcol, ff, use_colname = "all/[colname]")
pd_date_intersection(qlist)
pd_is_categorical(z)
pd_str_encoding_change(df, cols, fromenc = "iso-8859-1", toenc = "utf-8")
pd_str_unicode_tostr(df, targetype = str)
pd_dtypes_type1_totype2(df, fromtype = str, targetype = str)
pd_resetindex(df)
pd_insertdatecol(df, col, format1="%Y-%m-%d %H = "%Y-%m-%d %H:%M:%S:%f")
pd_replacevalues(df, matrix)
pd_removerow(df, row_list_index = (23, 45)
pd_removecol(df1, name1)
pd_insertrow(df, rowval, index1 = None, isreset = 1)
pd_h5_cleanbeforesave(df)
pd_h5_addtable(df, tablename, dbfile="F = "F:\temp_pandas.h5")
pd_h5_tableinfo(filenameh5, table)
pd_h5_dumpinfo(dbfile=r"E = r"E:\_data\stock\intraday_google.h5")
pd_h5_save(df, filenameh5="E = "E:/_data/_data_outlier.h5", key = "data")
pd_h5_load(filenameh5="E = "E:/_data/_data_outlier.h5", table_id = "data", exportype = "pandas", rowstart = -1, rowend = -1, ), )
pd_h5_fromcsv_tohdfs(dircsv = "dir1/dir2/", filepattern = "*.csv", tofilehdfs = "file1.h5", tablename = "df", ), dtype0 = None, encoding = "utf-8", chunksize = 2000000, mode = "a", form = "table", complib = None, )
pd_np_toh5file(numpyarr, fileout = "file.h5", table1 = "data")
date_allinfo()
datetime_tostring(datelist1)
date_remove_bdays(from_date, add_days)
date_add_bdays(from_date, add_days)
datenumpy_todatetime(tt, islocaltime = True)
datetime_tonumpydate(t, islocaltime = True)
datestring_todatetime(datelist1, format1 = "%Y%m%d")
datetime_toint(datelist1)
date_holiday()
date_add_bday(from_date, add_days)
dateint_todatetime(datelist1)
date_diffinday(intdate1, intdate2)
date_diffinbday(intd2, intd1)
date_gencalendar(start = "2010-01-01", end = "2010-01-15", country = "us")
date_finddateid(date1, dateref)
datestring_toint(datelist1)
date_now(i = 0)
date_nowtime(type1 = "str", format1="%Y-%m-%d %H = "%Y-%m-%d %H:%M:%S:%f")
date_generatedatetime(start = "20100101", nbday = 10, end = "")
np_numexpr_vec_calc()



utilmy\zzarchive\zzarchive\zutil_features.py
-------------------------functions----------------------
log(*s, n = 0, m = 1, **kw)
log2(*s, **kw)
log3(*s, **kw)
os_get_function_name()
os_getcwd()
pa_read_file(path =   'folder_parquet/', cols = None, n_rows = 1000, file_start = 0, file_end = 100000, verbose = 1, )
pa_write_file(df, path =   'folder_parquet/', cols = None, n_rows = 1000, partition_cols = None, overwrite = True, verbose = 1, filesystem  =  'hdfs')
test_get_classification_data(name = None)
params_check(pars, check_list, name = "")
save_features(df, name, path = None)
load_features(name, path)
save_list(path, name_list, glob)
save(df, name, path = None)
load(name, path)
pd_read_file(path_glob = "*.pkl", ignore_index = True, cols = None, verbose = False, nrows = -1, concat_sort = True, n_pool = 1, drop_duplicates = None, col_filter = None, col_filter_val = None, **kw)
load_dataset(path_data_x, path_data_y = '', colid = "jobId", n_sample = -1)
fetch_spark_koalas(path_data_x, path_data_y = '', colid = "jobId", n_sample = -1)
fetch_dataset(url_dataset, path_target = None, file_target = None)
load_function_uri(uri_name="myfolder/myfile.py = "myfolder/myfile.py::myFunction")
metrics_eval(metric_list = ["mean_squared_error"], ytrue = None, ypred = None, ypred_proba = None, return_dict = False)
pd_stat_dataset_shift(dftrain, dftest, colused, nsample = 10000, buckets = 5, axis = 0)
pd_stat_datashift_psi(expected, actual, buckettype = 'bins', buckets = 10, axis = 0)
feature_importance_perm(clf, Xtrain, ytrain, cols, n_repeats = 8, scoring = 'neg_root_mean_squared_error', show_graph = 1)
feature_selection_multicolinear(df, threshold = 1.0)
feature_correlation_cat(df, colused)
pd_feature_generate_cross(df, cols, cols_cross_input = None, pct_threshold = 0.2, m_combination = 2)
pd_col_to_onehot(dfref, colname = None, colonehot = None, return_val = "dataframe,column")
pd_colcat_mergecol(df, col_list, x0, colid = "easy_id")
pd_colcat_tonum(df, colcat = "all", drop_single_label = False, drop_fact_dict = True)
pd_colcat_mapping(df, colname)
pd_colcat_toint(dfref, colname, colcat_map = None, suffix = None)
pd_colnum_tocat(df, colname = None, colexclude = None, colbinmap = None, bins = 5, suffix = "_bin", method = "uniform", na_value = -1, return_val = "dataframe,param", params={"KMeans_n_clusters" = {"KMeans_n_clusters": 8, "KMeans_init": 'k-means++', "KMeans_n_init": 10,"KMeans_max_iter": 300, "KMeans_tol": 0.0001, "KMeans_precompute_distances": 'auto',"KMeans_verbose": 0, "KMeans_random_state": None,"KMeans_copy_x": True, "KMeans_n_jobs": None, "KMeans_algorithm": 'auto'})
pd_colnum_normalize(df0, colname, pars, suffix = "_norm", return_val = 'dataframe,param')
pd_col_merge_onehot(df, colname)
pd_col_to_num(df, colname = None, default = np.nan)
pd_col_filter(df, filter_val = None, iscol = 1)
pd_col_fillna(dfref, colname = None, method = "frequent", value = None, colgroupby = None, return_val = "dataframe,param", )
pd_pipeline_apply(df, pipeline)
pd_stat_correl_pair(df, coltarget = None, colname = None)
pd_stat_pandas_profile(df, savefile = "report.html", title = "Pandas Profile")
pd_stat_distribution_colnum(df, nrows = 2000, verbose = False)
pd_stat_histogram(df, bins = 50, coltarget = "diff")
col_extractname(col_onehot)
col_remove(cols, colsremove, mode = "exact")
pd_colnum_tocat_stat(df, feature, target_col, bins, cuts = 0)
pd_stat_shift_trend_changes(df, feature, target_col, threshold = 0.03)
pd_stat_shift_trend_correlation(df, df_test, colname, target_col)
pd_stat_shift_changes(df, target_col, features_list = 0, bins = 10, df_test = 0)
np_conv_to_one_col(np_array, sep_char = "_")

-------------------------methods----------------------
dict2.__init__(self, d)


utilmy\zzml\install\run_doc.py
-------------------------functions----------------------
os_package_root_path(add_path = "", n = 0)
get_recursive_files(folderPath, ext = '/*model*/*.py')



utilmy\zzml\install\run_pypi.py
-------------------------functions----------------------
ask(question, ans = 'yes')
pypi_upload()
update_version(path, n)
git_commit(message)
main(*args)

-------------------------methods----------------------
Version.__init__(self, major, minor, patch)
Version.__str__(self)
Version.__repr__(self)
Version.stringify(self)
Version.new_version(self, orig)
Version.parse(cls, string)


utilmy\zzml\mlmodels\benchmark.py
-------------------------functions----------------------
get_all_json_path(json_path)
config_model_list(folder = None)
metric_eval(actual = None, pred = None, metric_name = "mean_absolute_error")
benchmark_run(bench_pars = None, args = None, config_mode = "test")
cli_load_arguments(config_file = None)
main()



utilmy\zzml\mlmodels\data.py
-------------------------functions----------------------
tf_dataset(dataset_pars)
download_googledrive(file_list, **kw)
download_dtopbox(data_pars)
import_data_tch(name = "", mode = "train", node_id = 0, data_folder_root = "")
import_data_fromfile(**kw)
import_data_dask(**kw)
import_data(name = "", mode = "train", node_id = 0, data_folder_root = "")
get_dataset(data_pars)



utilmy\zzml\mlmodels\dataloader.py
-------------------------functions----------------------
pickle_dump(t, **kwargs)
_validate_data_info(self, data_info)
get_dataset_type(x)
split_xy_from_dict(out, **kwargs)
test_run_model()
test_single(arg)
test_dataloader(path = 'dataset/json/refactor/')
test_json_list(data_pars_list)
cli_load_arguments(config_file = None)
main()
pickle_load()
image_dir_load(path)
batch_generator(iterable, n = 1)
_check_output_shape(self, inter_output, shape, max_len)

-------------------------methods----------------------
DataLoader.__init__(self, data_pars)


utilmy\zzml\mlmodels\dataloader_test.py
-------------------------functions----------------------
pandas_split_xy(out, data_pars)
pandas_load_train_test(path, test_path, **args)
rename_target_to_y(out, data_pars)
load_npz(path)
split_xy_from_dict(out, data_pars)
split_timeseries_df(out, data_pars, length, shift)
gluon_append_target_string(out, data_pars)
identical_test_set_split(*args, test_size, **kwargs)
read_csvs_from_directory(path, files = None, **args)
tokenize_x(data, no_classes, max_words = None)
timeseries_split(*args, test_size = 0.2)
main()

-------------------------methods----------------------
SingleFunctionPreprocessor.__init__(self, func_dict)
SingleFunctionPreprocessor.compute(self, data)
SingleFunctionPreprocessor.get_data(self)


utilmy\zzml\mlmodels\distributed.py
-------------------------functions----------------------
get_all_json_path(json_path)
config_model_list(folder = None)



utilmy\zzml\mlmodels\distri_torch.py
-------------------------functions----------------------
load_arguments()
train(epoch)
metric_average(val, name)
test()



utilmy\zzml\mlmodels\metrics.py
-------------------------functions----------------------
log(*s, n = 0, m = 1)
metrics_eval(metric_list = ["mean_squared_error"], ytrue = None, ypred = None, ypred_proba = None, return_dict = 0)
test()



utilmy\zzml\mlmodels\models.py
-------------------------functions----------------------
module_env_build(model_uri = "", verbose = 0, do_env_build = 0)
module_load(model_uri = "", verbose = 0, env_build = 0)
module_load_full(model_uri = "", model_pars = None, data_pars = None, compute_pars = None, choice = None, **kwarg)
model_create(module, model_pars = None, data_pars = None, compute_pars = None, **kwarg)
fit(module, data_pars = None, compute_pars = None, out_pars = None, **kwarg)
predict(module, data_pars = None, compute_pars = None, out_pars = None, **kwarg)
evaluate(module, data_pars = None, compute_pars = None, out_pars = None, **kwarg)
get_params(module, params_pars, **kwarg)
metrics(module, data_pars = None, compute_pars = None, out_pars = None, **kwarg)
load(module, load_pars, **kwarg)
save(module, save_pars, **kwarg)
test_all(folder = None)
test(folder = None)
test_global(modelname)
test_api(model_uri = "model_xxxx/yyyy.py", param_pars = None)
test_module(model_uri = "model_xxxx/yyyy.py", param_pars = None, fittable  =  True)
config_get_pars(config_file, config_mode = "test")
config_generate_json(modelname, to_path = "ztest/new_model/")
config_init(to_path = ".")
config_model_list(folder = None)
fit_cli(arg)
predict_cli(arg)
test_cli(arg)
cli_load_arguments(config_file = None)
main()



utilmy\zzml\mlmodels\optim.py
-------------------------functions----------------------
optim(model_uri = "model_tf.1_lstm", hypermodel_pars = {}, model_pars = {}, data_pars = {}, compute_pars = {}, out_pars = {})
optim_optuna(model_uri = "model_tf.1_lstm.py", hypermodel_pars = {"engine_pars" =  {"engine_pars": {}}, model_pars       =  {}, data_pars        =  {}, compute_pars     =  {}, # only Model parsout_pars         =  {})
post_process_best(model, module, model_uri, model_pars_update, data_pars, compute_pars, out_pars)
test_json(path_json = "", config_mode = "test")
test_all()
optim_cli(arg)
cli()
cli_load_arguments(config_file = None)
main()



utilmy\zzml\mlmodels\pipeline.py
-------------------------functions----------------------
os_package_root_path(filepath, sublevel = 0, path_add = "")
log(*s, n = 0, m = 1)
pd_na_values(df, cols = None, default = 0.0, **kw)
generate_data(df, num_data = 0, means = [], cov = [[1, 0], [0, 1]])
drop_cols(df, cols = None, **kw)
pd_concat(df1, df2, colid1)
pipe_split(in_pars, out_pars, compute_pars, **kw)
pipe_merge(in_pars, out_pars, compute_pars = None, **kw)
pipe_load(df, **in_pars)
pipe_run_inference(pipe_list, in_pars, out_pars, compute_pars = None, checkpoint = True, **kw)
pipe_checkpoint(df, **kw)
load_model(path)
save_model(model, path)
get_params(choice = "", data_path = "dataset/", config_mode = "test", **kw)
test(data_path = "/dataset/", pars_choice = "colnum")

-------------------------methods----------------------
Pipe.__init__(self, pipe_list, in_pars, out_pars, compute_pars = None, **kw)
Pipe.run(self)
Pipe.get_fitted_pipe_list(self, key = "")
Pipe.get_checkpoint(self)
Pipe.get_model_path(self)


utilmy\zzml\mlmodels\preprocessor.py
-------------------------methods----------------------
MissingDataPreprocessorError.__init__(self)
PreprocessorNotFittedError.__init__(self)
Preprocessor.__init__(self, preprocessor_dict)
Preprocessor._interpret_preprocessor_dict(self, pars)
Preprocessor._name_outputs(self, names, outputs)
Preprocessor.fit_transform(self, data)
Preprocessor.transform(self, data)


utilmy\zzml\mlmodels\util.py
-------------------------functions----------------------
log(*s, n = 0, m = 0)
os_package_root_path(filepath = "", sublevel = 0, path_add = "")
os_file_current_path()
os_folder_copy(src, dst)
os_get_file(folder = None, block_list = [], pattern = r'*.py')
model_get_list(folder = None, block_list = [])
get_recursive_files(folderPath, ext = '/*model*/*.py')
get_recursive_files2(folderPath, ext)
get_recursive_files3(folderPath, ext)
get_model_uri(file)
json_norm(ddict)
path_norm(path = "")
path_norm_dict(ddict)
test_module(model_uri = "model_tf/1_lstm.py", data_path = "dataset/", pars_choice = "json", reset = True)
config_load_root()
config_path_pretrained()
config_path_dataset()
config_set(ddict2)
params_json_load(path, config_mode = "test", tlist =  [ "model_pars", "data_pars", "compute_pars", "out_pars"])
load_config(args, config_file, config_mode, verbose = 0)
val(x, xdefault)
env_conda_build(env_pars = None)
env_pip_requirement(env_pars = None)
env_pip_check(env_pars = None)
env_build(model_uri, env_pars)
tf_deprecation()
get_device_torch()
os_path_split(path)
load(args, config_file, config_mode, verbose = 0)
save(model = None, session = None, save_pars = None)
load_tf(load_pars = "")
save_tf(model = None, sess = None, save_pars =  None)
load_tch(load_pars)
save_tch(model = None, optimizer = None, save_pars = None)
save_tch_checkpoint(model, optimiser, save_pars)
load_tch_checkpoint(model, optimiser, load_pars)
load_pkl(load_pars)
save_pkl(model = None, session = None, save_pars = None)
load_keras(load_pars, custom_pars = None)
save_keras(model = None, session = None, save_pars = None, )
save_gluonts(model = None, session = None, save_pars = None)
load_gluonts(load_pars = None)
load_function(package = "mlmodels.util", name = "path_norm")
load_function_uri(uri_name = "path_norm")
load_callable_from_uri(uri)
load_callable_from_dict(function_dict, return_other_keys = False)
os_folder_getfiles(folder, ext, dirlevel  =  -1, mode = "fullpath")

-------------------------methods----------------------
to_namespace.__init__(self, adict)
to_namespace.get(self, key)
Model_empty.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zzml\mlmodels\util_json.py


utilmy\zzml\mlmodels\util_log.py
-------------------------functions----------------------
create_appid(filename)
create_logfilename(filename)
create_uniqueid()
logger_setup(logger_name = None, log_file = None, formatter = FORMATTER_1, isrotate = False, isconsole_output = True, logging_level = logging.DEBUG, )
logger_handler_console(formatter = None)
logger_handler_file(isrotate = False, rotate_time = "midnight", formatter = None, log_file_used = None)
logger_setup2(name = __name__, level = None)
printlog(s = "", s1 = "", s2 = "", s3 = "", s4 = "", s5 = "", s6 = "", s7 = "", s8 = "", s9 = "", s10 = "", app_id = "", logfile = None, iswritelog = True, )
writelog(m = "", f = None)
load_arguments(config_file = None, arg_list = None)



utilmy\zzml\mlmodels\ztest.py
-------------------------functions----------------------
os_bash(cmd)
log_separator(space = 140)
log_info_repo(arg = None)
to_logfile(prefix = "", dateformat='+%Y-%m-%d_%H = '+%Y-%m-%d_%H:%M:%S,%3N')
os_file_current_path()
os_system(cmd, dolog = 1, prefix = "", dateformat='+%Y-%m-%d_%H = '+%Y-%m-%d_%H:%M:%S,%3N')
json_load(path)
log_remote_start(arg = None)
log_remote_push(arg = None)



utilmy\zzml\mlmodels\_version.py
-------------------------functions----------------------
get_keywords()
get_config()
register_vcs_handler(vcs, method)
run_command(commands, args, cwd = None, verbose = False, hide_stderr = False, env = None)
versions_from_parentdir(parentdir_prefix, root, verbose)
git_get_keywords(versionfile_abs)
git_versions_from_keywords(keywords, tag_prefix, verbose)
git_pieces_from_vcs(tag_prefix, root, verbose, run_command = run_command)
plus_or_dot(pieces)
render_pep440(pieces)
render_pep440_pre(pieces)
render_pep440_post(pieces)
render_pep440_old(pieces)
render_git_describe(pieces)
render_git_describe_long(pieces)
render(pieces)
get_versions()



utilmy\zzml\pullrequest\aa_mycode_test.py
-------------------------functions----------------------
os_file_current_path()
test(arg = None)



utilmy\codeparser\project_graph\project_graph\__init__.py


utilmy\codeparser\project_graph\tests\goodnight.py
-------------------------functions----------------------
sleep_five_seconds()



utilmy\codeparser\project_graph\tests\script_test_case_1.py
-------------------------functions----------------------
foo()
bar()



utilmy\codeparser\project_graph\tests\test_performance_graph.py
-------------------------functions----------------------
test_toplvl()
test_lowlvl()



utilmy\codeparser\project_graph\tests\__init__.py


utilmy\recsys\zrecs\recommenders\__init__.py


utilmy\recsys\zrecs\tests\conftest.py
-------------------------functions----------------------
output_notebook()
kernel_name()
path_notebooks()
tmp(tmp_path_factory)
spark(tmp_path_factory, app_name = "Sample", url = "local[*]")
sar_settings()
header()
pandas_dummy(header)
pandas_dummy_timestamp(pandas_dummy, header)
train_test_dummy_timestamp(pandas_dummy_timestamp)
demo_usage_data(header, sar_settings)
demo_usage_data_spark(spark, demo_usage_data, header)
criteo_first_row()
notebooks()
test_specs_ncf()
python_dataset_ncf(test_specs_ncf)
test_specs()
affinity_matrix(test_specs)
deeprec_resource_path()
mind_resource_path(deeprec_resource_path)
deeprec_config_path()



utilmy\recsys\zrecs\tests\__init__.py


utilmy\recsys\zrecs\tools\databricks_install.py
-------------------------functions----------------------
create_egg(), local_eggname = "Recommenders.egg", overwrite = False, )
dbfs_file_exists(api_client, dbfs_path)
prepare_for_operationalization(cluster_id, api_client, dbfs_path, overwrite, spark_version)



utilmy\recsys\zrecs\tools\generate_conda_file.py


utilmy\recsys\zrecs\tools\generate_requirements_txt.py


utilmy\recsys\zrecs\tools\__init__.py


utilmy\spark\src\afpgrowth\main.py


utilmy\spark\src\functions\GetFamiliesFromUserAgent.py
-------------------------functions----------------------
getall_families_from_useragent(ua_string)



utilmy\spark\src\tables\table_predict_session_length.py
-------------------------functions----------------------
run(spark:SparkSession, config_path: str = 'config.yaml', mode:str = 'train,pred')
preprocess(spark, conf, check = True)



utilmy\spark\src\tables\table_predict_url_unique.py
-------------------------functions----------------------
run(spark:SparkSession, config_path: str = 'config.yaml', mode:str = 'train,pred')
preprocess(spark, conf, check = True)



utilmy\spark\src\tables\table_predict_volume.py
-------------------------functions----------------------
run(spark:SparkSession, config_path: str = 'config.yaml')
preprocess(spark, conf, check = True)
model_train(df:object, conf_model:dict, verbose:bool = True)
model_predict(df:pd.DataFrame, conf_model:dict, verbose:bool = True)



utilmy\spark\src\tables\table_user_log.py
-------------------------functions----------------------
run(spark:SparkSession, config_name:str)
create_userid(userlogDF:pyspark.sql.DataFrame)



utilmy\spark\src\tables\table_user_session_log.py
-------------------------functions----------------------
run(spark:SparkSession, config_name = 'config.yaml')



utilmy\spark\src\tables\table_user_session_stats.py
-------------------------functions----------------------
run(spark:SparkSession, config_name: str = 'config.yaml')



utilmy\stats\hypothetical\docs\setup.py


utilmy\stats\hypothetical\tests\test_aov.py
-------------------------functions----------------------
test_data()
multivariate_test_data()

-------------------------methods----------------------
TestAnovaOneWay.test_anova_oneway(self)
TestManovaOneWay.test_manova_oneway(self)


utilmy\stats\hypothetical\tests\test_contingency.py
-------------------------methods----------------------
TestChiSquareContingency.test_chi_square_contingency(self)
TestChiSquareContingency.test_chi_square_contingency_no_continuity(self)
TestChiSquareContingency.test_chi_square_contingency_no_expected(self)
TestChiSquareContingency.test_chi_square_exceptions(self)
TestCochranQ.test_cochranq(self)
TestCochranQ.test_cochranq_exceptions(self)
TestMcNemarTest.test_mcnemartest_exceptions(self)
TestTableMargins.test_table_margins(self)
TestTableMargins.test_margins_exceptions(self)
TestTableMargins.test_expected_frequencies(self)
TestTableMargins.test_expected_frequencies_exceptions(self)


utilmy\stats\hypothetical\tests\test_critical_values.py
-------------------------methods----------------------
TestChiSquareCritical.test_critical_values(self)
TestChiSquareCritical.test_exceptions(self)
TestUCritical.test_critical_values(self)
TestUCritical.test_exceptions(self)
TestWCritical.test_critical_values(self)
TestWCritical.test_exceptions(self)
TestRCritical.test_critical_values(self)
TestRCritical.test_exceptions(self)


utilmy\stats\hypothetical\tests\test_descriptive.py
-------------------------methods----------------------
TestCorrelationCovariance.test_naive_covariance(self)
TestCorrelationCovariance.test_shifted_covariance(self)
TestCorrelationCovariance.test_two_pass_covariance(self)
TestCorrelationCovariance.test_covar_no_method(self)
TestCorrelationCovariance.test_pearson(self)
TestCorrelationCovariance.test_spearman(self)
TestVariance.test_var_corrected_two_pass(self)
TestVariance.test_var_textbook_one_pass(self)
TestVariance.test_var_standard_two_pass(self)
TestVariance.test_var_youngs_cramer(self)
TestVariance.test_stddev(self)
TestVariance.test_var_cond(self)
TestVariance.test_errors(self)
TestKurtosis.test_exceptions(self)
TestKurtosis.test_kurtosis(self)
TestSkewness.test_exceptions(self)
TestSkewness.test_skewness(self)
TestMeanAbsoluteDeviation.test_exceptions(self)
TestMeanAbsoluteDeviation.test_mean_deviation(self)


utilmy\stats\hypothetical\tests\test_factor_analysis.py


utilmy\stats\hypothetical\tests\test_gof.py
-------------------------methods----------------------
TestChiSquare.test_chisquaretest(self)
TestChiSquare.test_chisquaretest_arr(self)
TestChiSquare.test_chisquaretest_continuity(self)
TestChiSquare.test_chisquare_no_exp(self)
TestChiSquare.test_chisquare_exceptions(self)
TestJarqueBera.test_jarquebera(self)
TestJarqueBera.test_jarquebera_exceptions(self)


utilmy\stats\hypothetical\tests\test_hypothesis.py
-------------------------functions----------------------
test_data()
test_multiclass_data()

-------------------------methods----------------------
TestBinomial.test_binomial_twosided(self)
TestBinomial.test_binomial_less(self)
TestBinomial.test_binomial_greater(self)
TestBinomial.test_binomial_no_continuity(self)
TestBinomial.test_binomial_no_continuity_greater(self)
TestBinomial.test_binomial_no_continuity_less(self)
TestBinomial.test_binomial_exceptions(self)
Test_tTest.test_two_sample_welch_test(self)
Test_tTest.test_two_sample_students_test(self)
Test_tTest.test_one_sample_test(self)
Test_tTest.test_paired_sample_test(self)
Test_tTest.test_alternatives(self)
Test_tTest.test_ttest_exceptions(self)


utilmy\stats\hypothetical\tests\test_internal.py
-------------------------functions----------------------
test_array()
test_build_design_matrix()
test_build_matrix()



utilmy\stats\hypothetical\tests\test_nonparametric.py
-------------------------functions----------------------
test_data()
multivariate_test_data()
plants_test_data()
test_tie_correction()

-------------------------methods----------------------
TestFriedmanTest.test_friedman_test(self)
TestMannWhitney.test_mann_whitney(self)
TestMannWhitney.test_exceptions(self)
TestWilcoxon.test_wilcoxon_one_sample(self)
TestWilcoxon.test_wilcoxon_multi_sample(self)
TestWilcoxon.test_exceptions(self)
TestKruskalWallis.test_kruskal_wallis(self)
TestKruskalWallis.test_exceptions(self)
TestSignTest.test_sign_test(self)
TestSignTest.test_sign_test_less(self)
TestSignTest.test_sign_test_greater(self)
TestSignTest.test_sign_test_exceptions(self)
TestMedianTest.test_mediantest(self)
TestMedianTest.test_median_ties_above(self)
TestMedianTest.test_median_ties_ignore(self)
TestMedianTest.test_median_continuity(self)
TestMedianTest.test_median_no_continuity(self)
TestMedianTest.test_median_exceptions(self)
TestRunsTest.test_runs_test_small_sample(self)
TestRunsTest.test_runs_test_large_sample(self)
TestVanDerWaerden.test_van_der_waerden(self)
TestWaldWolfowitz.test_wald_wolfowitz(self)


utilmy\stats\hypothetical\tests\test_posthoc.py
-------------------------functions----------------------
test_tukeytest()

-------------------------methods----------------------
TestGamesHowell.test_games_howell(self)


utilmy\templates\templist\pypi_package\run_pipy.py
-------------------------functions----------------------
get_current_githash()
update_version(path, n = 1)
git_commit(message)
ask(question, ans = 'yes')
pypi_upload()
main(*args)

-------------------------methods----------------------
Version.__init__(self, major, minor, patch)
Version.__str__(self)
Version.__repr__(self)
Version.stringify(self)
Version.new_version(self, orig)
Version.parse(cls, string)


utilmy\templates\templist\pypi_package\setup.py
-------------------------functions----------------------
get_current_githash()



utilmy\zml\example\classifier\classifier_adfraud.py
-------------------------functions----------------------
global_pars_update(model_dict, data_name, config_name)
check()



utilmy\zml\example\classifier\classifier_airline.py
-------------------------functions----------------------
global_pars_update(model_dict, data_name, config_name)
airline_lightgbm(path_model_out = "")
check()



utilmy\zml\example\classifier\classifier_bankloan.py
-------------------------functions----------------------
global_pars_update(model_dict, data_name, config_name)
bank_lightgbm()
check()



utilmy\zml\example\classifier\classifier_cardiff.py
-------------------------functions----------------------
global_pars_update(model_dict, data_name, config_name)
cardif_lightgbm(path_model_out = "")
check()



utilmy\zml\example\classifier\classifier_income.py
-------------------------functions----------------------
global_pars_update(model_dict, data_name, config_name)
income_status_lightgbm(path_model_out = "")
check()



utilmy\zml\example\classifier\classifier_multi.py
-------------------------functions----------------------
global_pars_update(model_dict, data_name, config_name)
multi_lightgbm()
check()



utilmy\zml\example\classifier\classifier_optuna.py
-------------------------functions----------------------
global_pars_update(model_dict, data_name, config_name)
titanic_lightoptuna()
check()



utilmy\zml\example\classifier\classifier_sentiment.py
-------------------------functions----------------------
global_pars_update(model_dict, data_name, config_name)
os_get_function_name()
sentiment_lightgbm(path_model_out = "")
data_profile(path_data_train = "", path_model = "", n_sample =  5000)
preprocess(config = None, nsample = None)
train(config = None, nsample = None)
check()
predict(config = None, nsample = None)
run_all()
sentiment_elasticnetcv(path_model_out = "")
sentiment_bayesian_pyro(path_model_out = "")



utilmy\zml\example\click\online_shopping.py
-------------------------functions----------------------
global_pars_update(model_dict, data_name, config_name)
online_lightgbm()
check()



utilmy\zml\example\click\outlier_predict.py
-------------------------functions----------------------
os_get_function_name()
global_pars_update(model_dict, data_name, config_name)
titanic_pyod(path_model_out = "")
data_profile(path_data_train = "", path_model = "", n_sample =  5000)
check()



utilmy\zml\example\click\test_online_shopping.py
-------------------------functions----------------------
global_pars_update(model_dict, data_name, config_name)
online_lightgbm()
pd_col_myfun(df = None, col = None, pars = {})
check()



utilmy\zml\example\regress\regress_airbnb.py
-------------------------functions----------------------
global_pars_update(model_dict, data_name, config_name)
y_norm(y, inverse = True, mode = 'boxcox')
airbnb_lightgbm(path_model_out = "")
airbnb_elasticnetcv(path_model_out = "")



utilmy\zml\example\regress\regress_boston.py
-------------------------functions----------------------
global_pars_update(model_dict, data_name, config_name)
y_norm(y, inverse = True, mode = 'boxcox')
boston_lightgbm(path_model_out = "")
boston_causalnex(path_model_out = "")



utilmy\zml\example\regress\regress_house.py
-------------------------functions----------------------
global_pars_update(model_dict, data_name, config_name)
y_norm(y, inverse = True, mode = 'boxcox')
house_price_lightgbm(path_model_out = "")
house_price_elasticnetcv(path_model_out = "")
data_profile()
preprocess()
train()
check()
predict()
run_all()



utilmy\zml\example\regress\regress_salary.py
-------------------------functions----------------------
global_pars_update(model_dict, data_name, config_name)
y_norm(y, inverse = True, mode = 'boxcox')
salary_lightgbm(path_model_out = "")
salary_elasticnetcv(path_model_out = "")
salary_bayesian_pyro(path_model_out = "")
salary_glm(path_model_out = "")
check()



utilmy\zml\example\svd\benchmark_mf.py
-------------------------functions----------------------
linear_surplus_confidence_matrix(B, alpha)
log_surplus_confidence_matrix(B, alpha, epsilon)
iter_rows(S)
recompute_factors(Y, S, lambda_reg, dtype = 'float32')
recompute_factors_bias(Y, S, lambda_reg, dtype = 'float32')
factorize(S, num_factors, lambda_reg = 1e-5, num_iterations = 20, init_std = 0.01, verbose = False, dtype = 'float32', recompute_factors = recompute_factors, *args, **kwargs)
time_reps(func, params, reps)
scipy_svd(A, K)
sklearn_randomized_svd(A, k)
sklearn_truncated_randomized_svd(A, k)
sklearn_truncated_arpack_svd(A, k)
sparsesvd_svd(A, k)
gensim_svd(A, k)
wmf(A, k)
implicit_mf(A, k)
nmf_1(A, k)
nmf_2(A, k)
nmf_3(A, k)
nmf_4(A, k)
nmf_5(A, k)
daal4py_svd(A, k)
daal4py_als(A, k)
time_ns()

-------------------------methods----------------------
ImplicitMF.__init__(self, counts, num_factors = 40, num_iterations = 30, reg_param = 0.8)
ImplicitMF.train_model(self)
ImplicitMF.iteration(self, user, fixed_vecs)


utilmy\zml\example\svd\benchmark_mf0.py
-------------------------functions----------------------
linear_surplus_confidence_matrix(B, alpha)
log_surplus_confidence_matrix(B, alpha, epsilon)
iter_rows(S)
recompute_factors(Y, S, lambda_reg, dtype = 'float32')
recompute_factors_bias(Y, S, lambda_reg, dtype = 'float32')
factorize(S, num_factors, lambda_reg = 1e-5, num_iterations = 20, init_std = 0.01, verbose = False, dtype = 'float32', recompute_factors = recompute_factors, *args, **kwargs)
time_reps(func, params, reps)
scipy_svd(A, K)
sklearn_randomized_svd(A, k)
sklearn_truncated_randomized_svd(A, k)
sklearn_truncated_arpack_svd(A, k)
sparsesvd_svd(A, k)
gensim_svd(A, k)
wmf(A, k)
implicit_mf(A, k)
nmf_1(A, k)
nmf_2(A, k)
nmf_3(A, k)
nmf_4(A, k)
nmf_5(A, k)
time_ns()

-------------------------methods----------------------
ImplicitMF.__init__(self, counts, num_factors = 40, num_iterations = 30, reg_param = 0.8)
ImplicitMF.train_model(self)
ImplicitMF.iteration(self, user, fixed_vecs)


utilmy\zml\example\tseries\tseries_m5sales.py
-------------------------functions----------------------
pd_col_tocat(df, nan_cols, colcat)
pd_merge(df_list, cols_join)
train(input_path, n_experiments  =  3, colid  =  None, coly  =  None)
featurestore_meta_update(featnames, filename, colcat)
featurestore_get_filelist_fromcolname(selected_cols, colid)
featurestore_generate_feature(dir_in, dir_out, my_fun_features, features_group_name, input_raw_path  =  None, auxiliary_csv_path  =  None, coldrop  =  None, index_cols  =  None, merge_cols_mapping  =  None, colcat  =  None, colid = None, coly  =  None, coldate  =  None, max_rows  =  5, step_wise_saving  =  False)
featurestore_filter_features(mode  = "random", colid  =  None, coly  =  None)
featurestore_get_filename(file_name, path)
featurestore_get_feature_fromcolname(path, selected_cols, colid)
pd_tsfresh_m5data_sales(df_sales, dir_out, features_group_name, drop_cols, df_calendar, index_cols, merge_cols_mapping, id_cols)
pd_tsfresh_m5data(df_sales, dir_out, features_group_name, drop_cols, df_calendar, index_cols, merge_cols_mapping, id_cols)
pd_ts_tsfresh(df, input_raw_path, dir_out, features_group_name, auxiliary_csv_path, drop_cols, index_cols, merge_cols_mapping, cat_cols  =  None, id_cols  =  None, dep_col  =  None, coldate  =  None, max_rows  =  10)
custom_get_colsname(colid, coly)
custom_rawdata_merge(out_path = 'out/', max_rows = 10)
custom_generate_feature_all(input_path  =  data_path, out_path = ".", input_raw_path  = ".", auxiliary_csv_path  =  None, coldrop  =  None, colindex  =  None, merge_cols_mapping  =  None, coldate  =  None, colcat  =  None, colid  =  None, coly  =  None, max_rows  =  10)
run_train(input_path  = "data/input/tseries/tseries_m5/raw", out_path = data_path, do_generate_raw = True, do_generate_feature = True, do_train = False, max_rows  =  10)

-------------------------methods----------------------
FeatureStore.__init__(self)


utilmy\zml\example\tseries\tseries_retail.py
-------------------------functions----------------------
pd_col_tocat(df, nan_cols, colcat)
pd_merge(df_list, cols_join)
train(input_path, n_experiments  =  3, colid  =  None, coly  =  None)
featurestore_meta_update(featnames, filename, colcat)
featurestore_get_filelist_fromcolname(selected_cols, colid)
featurestore_generate_feature(dir_in, dir_out, my_fun_features, features_group_name, input_raw_path  =  None, auxiliary_csv_path  =  None, coldrop  =  None, index_cols  =  None, merge_cols_mapping  =  None, colcat  =  None, colid = None, coly  =  None, coldate  =  None, max_rows  =  5, step_wise_saving  =  False)
featurestore_filter_features(mode  = "random", colid  =  None, coly  =  None)
featurestore_get_filename(file_name, path)
featurestore_get_feature_fromcolname(path, selected_cols, colid)
custom_get_colsname(colid, coly)
custom_rawdata_merge(out_path = 'out/', max_rows = 10)
custom_generate_feature_all(input_path  =  data_path, out_path = ".", input_raw_path  = ".", auxiliary_csv_path  =  None, coldrop  =  None, colindex  =  None, merge_cols_mapping  =  None, coldate  =  None, colcat  =  None, colid  =  None, coly  =  None, max_rows  =  10)
run_train(input_path  = "data/input/tseries/retail/raw", out_path = data_path, do_generate_raw = True, do_generate_feature = True, do_train = False, max_rows  =  10)

-------------------------methods----------------------
FeatureStore.__init__(self)


utilmy\zml\example\tseries\tseries_sales.py
-------------------------functions----------------------
pd_col_tocat(df, nan_cols, colcat)
pd_merge(df_list, cols_join)
featurestore_meta_update(featnames, filename, colcat)
featurestore_get_filelist_fromcolname(selected_cols, colid)
featurestore_generate_feature(dir_in, dir_out, my_fun_features, features_group_name, input_raw_path  =  None, auxiliary_csv_path  =  None, coldrop  =  None, index_cols  =  None, merge_cols_mapping  =  None, colcat  =  None, colid = None, coly  =  None, coldate  =  None, max_rows  =  5, step_wise_saving  =  False)
featurestore_filter_features(mode  = "random", colid  =  None, coly  =  None)
featurestore_get_filename(file_name, path)
featurestore_get_feature_fromcolname(path, selected_cols, colid)
custom_get_colsname(colid, coly)
custom_rawdata_merge(out_path = 'out/', max_rows = 10)
custom_generate_feature_all(input_path  =  data_path, out_path = ".", input_raw_path  = ".", auxiliary_csv_path  =  None, coldrop  =  None, colindex  =  None, merge_cols_mapping  =  None, coldate  =  None, colcat  =  None, colid  =  None, coly  =  None, max_rows  =  10)
run_generate_train_data(input_path  = "data/input/tseries/retail/raw", out_path = data_path, do_generate_raw = True, do_generate_feature = True, do_train = False, max_rows  =  10)

-------------------------methods----------------------
FeatureStore.__init__(self)


utilmy\zml\source\bin\column_encoder.py
-------------------------methods----------------------
OneHotEncoderRemoveOne.__init__(self, n_values = None, categorical_features = None, categories = "auto", sparse = True, dtype = np.float64, handle_unknown = "error", )
OneHotEncoderRemoveOne.transform(self, X, y = None)
MinHashEncoder.__init__(self, n_components, ngram_range = (2, 4)
MinHashEncoder.get_unique_ngrams(self, string, ngram_range)
MinHashEncoder.minhash(self, string, n_components, ngram_range)
MinHashEncoder.fit(self, X, y = None)
MinHashEncoder.transform(self, X)
PasstroughEncoder.__init__(self, passthrough = True)
PasstroughEncoder.fit(self, X, y = None)
PasstroughEncoder.transform(self, X)


utilmy\zml\source\bin\__init__.py


utilmy\zml\source\models\dataset.py
-------------------------functions----------------------
pack_features_vector(features, labels)
log(*s)
test1()
pack_features_vector(features, labels)
get_dataset_split_for_model_petastorm(Xtrain, ytrain = None, pars:dict = None)
eval_dict(src, dst = {})
pack_features_vector(features, labels)
fIt_(dataset_url, training_iterations, batch_size, evaluation_interval)
train_and_test(dataset_url, training_iterations, batch_size, evaluation_interval)
main()
tensorflow_hello_world(dataset_url='file = 'file:///tmp/external_dataset')
pytorch_hello_world(dataset_url='file = 'file:///tmp/external_dataset')
python_hello_world(dataset_url='file = 'file:///tmp/external_dataset')

-------------------------methods----------------------
dictEval.__init__(self)
dictEval.reset(self)
dictEval.eval_dict(self, src, dst = {})
dictEval.tf_dataset_create(self, key2, path_pattern, batch_size = 32, **kw)
dictEval.pandas_create(self, key2, path, )


utilmy\zml\source\models\keras_deepctr.py
-------------------------functions----------------------
log(*s)
log2(*s)
log3(*s)
init(*kw, **kwargs)
reset()
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
eval(data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(Xpred = None, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)
save(path = None, save_weight = False)
load_model(path = "", load_weight = False)
load_info(path = "")
preprocess(prepro_pars)
get_dataset(data_pars = None, task_type = "train", **kw)
get_xy_random2(X, y, cols_family = {})
get_xy_random(X, y, cols_family = {})
get_xy_fd(use_neg = False, hash_flag = False, use_session = False)
get_xy_dataset(data_sample = None)
test(config = '')
test_helper(model_name, model_pars, data_pars, compute_pars)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, **kwargs)


utilmy\zml\source\models\keras_widedeep.py
-------------------------functions----------------------
log(*s)
log2(*s)
log3(*s)
init(*kw, **kwargs)
reset()
WideDeep_sparse(model_pars2)
fit(data_pars = None, compute_pars = None, out_pars = None)
predict(Xpred = None, data_pars = None, compute_pars = None, out_pars = None)
save(path = None, info = None)
load_model(path = "")
model_summary(path = "ztmp/")
get_dataset_split(data_pars = None, task_type = "train", **kw)
get_dataset_split_for_model_pandastuple(Xtrain, ytrain = None, data_pars = None, )
get_dataset_split_for_model_petastorm(Xtrain, ytrain = None, pars:dict = None)
get_dataset_split_for_model_tfsparse(Xtrain, ytrain = None, pars:dict = None)
test(config = '', n_sample  =  100)
test2(config = '')
test_helper(model_pars, data_pars, compute_pars)
zz_WideDeep_dense(model_pars2)
zz_get_dataset(data_pars = None, task_type = "train", **kw)
zz_input_template_feed_keras_model(Xtrain, cols_type_received, cols_ref, **kw)
zz_get_dataset2(data_pars = None, task_type = "train", **kw)
ModelCustom2()
zz_get_dataset_tuple_keras(pattern, batch_size, mode = tf.estimator.ModeKeys.TRAIN, truncate = None)
zz_Modelsparse2()

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, )
tf_FeatureColumns.__init__(self, dataframe = None)
tf_FeatureColumns.df_to_dataset(self, dataframe, target, shuffle = True, batch_size = 32)
tf_FeatureColumns.df_to_dataset_dense(self, dataframe, target, shuffle = True, batch_size = 32)
tf_FeatureColumns.split_sparse_data(self, df, shuffle_train = False, shuffle_test = False, shuffle_val = False, batch_size = 32, test_split = 0.2, colnum = [], colcat = [])
tf_FeatureColumns.data_to_tensorflow_split(self, df, target, model = 'sparse', shuffle_train = False, shuffle_test = False, shuffle_val = False, batch_size = 32, test_split = 0.2, colnum = [], colcat = [])
tf_FeatureColumns.data_to_tensorflow(self, df, target, model = 'sparse', shuffle_train = False, shuffle_test = False, shuffle_val = False, batch_size = 32, test_split = 0.2, colnum = [], colcat = [])
tf_FeatureColumns.numeric_columns(self, columnsName)
tf_FeatureColumns.bucketized_columns(self, columnsBoundaries)
tf_FeatureColumns.categorical_columns(self, indicator_column_names, colcat_nunique = None, output = False)
tf_FeatureColumns.hashed_columns(self, hashed_columns_dict)
tf_FeatureColumns.crossed_feature_columns(self, columns_crossed, nameOfLayer, bucket_size = 10)
tf_FeatureColumns.embeddings_columns(self, coldim_dict)
tf_FeatureColumns.transform_output(self, featureColumn)
tf_FeatureColumns.get_features(self)


utilmy\zml\source\models\keras_widedeep_dense.py
-------------------------functions----------------------
log(*s)
log2(*s)
log3(*s)
init(*kw, **kwargs)
reset()
Modelcustom(n_wide_cross, n_wide, n_deep, n_feat = 8, m_EMBEDDING = 10, loss = 'mse', metric  =  'mean_squared_error')
get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
get_dataset(Xtrain, cols_type_received, cols_ref)
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
evaluate(Xy_pred = None, data_pars = None, compute_pars = {}, out_pars = {}, **kw)
predict(Xpred = None, data_pars = None, compute_pars = {}, out_pars = {}, **kw)
save(path = None, info = None)
load_model(path = "")
load_info(path = "")
test(config = '')
test_helper(model_pars, data_pars, compute_pars)
get_dataset2(data_pars = None, task_type = "train", **kw)
get_dataset_tuple_keras(Xtrain, cols_type_received, cols_ref, **kw)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zml\source\models\model_bayesian_numpyro.py
-------------------------functions----------------------
log(*s)
log2(*s)
log3(*s)
init(*kw, **kwargs)
reset()
log(*s)
init(*kw, **kwargs)
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(Xpred = None, data_pars = {}, compute_pars = None, out_pars = {}, **kw)
reset()
save(path = None, info = None)
load_model(path = "")
load_info(path = "")
preprocess(prepro_pars)
get_dataset(data_pars = None, task_type = "train", **kw)
get_params(param_pars = {}, **kw)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zml\source\models\model_bayesian_pyro.py
-------------------------functions----------------------
log(*s)
log2(*s)
log3(*s)
init(*kw, **kwargs)
reset()
model_class_loader(m_name = 'BayesianRegression', class_list:list = None)
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(Xpred = None, data_pars = {}, compute_pars = None, out_pars = {}, **kw)
save(path = None, info = None)
load_model(path = "")
load_info(path = "")
get_dataset(data_pars = None, task_type = "train", **kw)
y_norm(y, inverse = True, mode = 'boxcox')
test_dataset_regress_fake(nrows = 500)
test(nrows = 500)

-------------------------methods----------------------
BayesianRegression.__init__(self, X_dim:int = 17, y_dim:int = 1)
BayesianRegression.forward(self, x, y = None)
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zml\source\models\model_encoder.py
-------------------------functions----------------------
log(*s)
log2(*s)
log3(*s)
init(*kw, **kwargs)
reset()
log(*s)
log2(*s)
log3(*s)
init(*kw, **kwargs)
fit(data_pars: dict = None, compute_pars: dict = None, out_pars: dict = None, **kw)
transform(Xpred = None, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)
decode(Xpred = None, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)
predict(Xpred = None, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)
reset()
save(path = None, info = None)
load_model(path = "")
load_info(path = "")
get_dataset_tuple(Xtrain, cols_type_received, cols_ref, split = False)
get_dataset(Xtrain, cols_type_received, cols_ref, split = False)
test_dataset_classi_fake(nrows = 500)
test(nrows = 500)
test_helper(model_pars, data_pars, compute_pars)
pd_export(df, col, pars)
pd_autoencoder(df, col, pars)
pd_covariate_shift_adjustment()

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zml\source\models\model_gefs.py
-------------------------functions----------------------
log(*s)
log2(*s)
log3(*s)
init(*kw, **kwargs)
reset()
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
eval(data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(Xpred = None, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)
save(path = None, info = None)
load_model(path = "")
load_info(path = "")
get_dataset(data_pars = None, task_type = "train", **kw)
test(n_sample  =  100)
test_helper(model_pars, data_pars, compute_pars)
pd_colcat_get_catcount(df, colcat, coly, continuous_ids = None)
is_continuous(v_array)
test2()
get_dummies(data)
learncats(data, classcol = None, continuous_ids = [])
gef_is_continuous(data)
gef_get_stats(data, ncat = None)
gef_normalize_data(data, maxv, minv)
gef_standardize_data(data, mean, std)
train_test_split2(data, ncat, train_ratio = 0.7, prep = 'std')
test_converion()
train_test_split(data, ncat, train_ratio = 0.7, prep = 'std')
adult(data)
australia(data)
bank(data)
credit(data)
electricity(data)
segment(data)
german(data)
vowel(data)
cmc(data)
get_data(data_pars = None, task_type = "train", **kw)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zml\source\models\model_numpyro.py
-------------------------functions----------------------
log(*s)
log2(*s)
log3(*s)
init(*kw, **kwargs)
reset()
require_fitted(f)
metrics(y: pd.Series, yhat: pd.Series)

-------------------------methods----------------------
NumpyEncoder.default(self, obj)
BaseModel.link(x)
BaseModel.likelihood_func(self, yhat)
BaseModel.__init__(self, rng_seed: int  =  None)
BaseModel.__repr__(self)
BaseModel.split_rand_key(self, n: int  =  1)
BaseModel.transform(cls, df: pd.DataFrame)
BaseModel.model(self, df: pd.DataFrame)
BaseModel.fit(self, df: pd.DataFrame, sampler: str  =  "NUTS", rng_key: np.ndarray  =  None, sampler_kwargs: typing.Dict[str, typing.Any]  =  None, **mcmc_kwargs, )
BaseModel.predict(self, df: pd.DataFrame, ci: bool  =  False, ci_interval: float  =  0.9, aggfunc: typing.Union[str, typing.Callable]  =  "mean", )
BaseModel.sample_posterior_predictive(self, df: pd.DataFrame, hdpi: bool  =  False, hdpi_interval: float  =  0.9, rng_key: np.ndarray  =  None, )
BaseModel.num_samples(self)
BaseModel.num_chains(self)
BaseModel.samples_flat(self)
BaseModel.samples_df(self)
BaseModel.from_dict(cls, data: typing.Dict[str, typing.Any], **model_kw)
BaseModel.to_json(self)
BaseModel.preprocess_config_dict(cls, config: dict)
BaseModel.metrics(self, df: pd.DataFrame, aggerrs: bool  =  True)
BaseModel.grouped_metrics(self, df: pd.DataFrame, groupby: typing.Union[str, typing.List[str]], aggfunc: typing.Callable  =  onp.sum, aggerrs: bool  =  True, )
BaseModel.formula(self)
Normal.link(x)
Normal.likelihood_func(self, yhat)
Poisson.link(x)
Poisson.likelihood_func(self, yhat)
Bernoulli.link(x)
Bernoulli.likelihood_func(self, probs)
ShabadooException.__str__(self)
NotFittedError.__init__(self, func = None)
AlreadyFittedError.__init__(self, model)
IncompleteModel.__init__(self, model, attribute)
IncompleteFeature.__init__(self, name, key)
IncompleteSamples.__init__(self, name)


utilmy\zml\source\models\model_outlier.py
-------------------------functions----------------------
log(*s)
log2(*s)
log3(*s)
init(*kw, **kwargs)
reset()
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(Xpred = None, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)
save(path = None, info = None)
load_model(path = "")
load_info(path = "")
get_dataset_split_for_model_pandastuple(Xtrain, ytrain = None, data_pars = None, )
get_dataset2(data_pars = None, task_type = "train", **kw)
get_dataset(Xtrain, ytrain = None, data_pars = None, )

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zml\source\models\model_sampler.py
-------------------------functions----------------------
log(*s)
log2(*s)
log3(*s)
init(*kw, **kwargs)
reset()
fit(data_pars: dict = None, compute_pars: dict = None, out_pars: dict = None, **kw)
eval(data_pars = None, compute_pars = None, out_pars = None, **kw)
transform(Xpred = None, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)
predict(Xpred = None, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)
save(path = None, info = None)
load_model(path = "")
load_info(path = "")
get_dataset_tuple(Xtrain, cols_type_received, cols_ref, split = False)
get_dataset(Xtrain, cols_type_received, cols_ref, split = False)
test()
test2(n_sample  =  1000)
test_helper(model_pars, data_pars, compute_pars)
zz_pd_sample_imblearn(df = None, col = None, pars = None)
zz_pd_augmentation_sdv(df, col = None, pars = {})
zz_pd_covariate_shift_adjustment()
zz_test()

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zml\source\models\model_sklearn.py
-------------------------functions----------------------
log(*s)
log2(*s)
log3(*s)
init(*kw, **kwargs)
reset()
model_automl()
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(Xpred = None, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)
save(path = None, info = None)
load_model(path = "")
load_info(path = "")
get_dataset_split_for_model_pandastuple(Xtrain, ytrain = None, data_pars = None, )
get_dataset2(data_pars = None, task_type = "train", **kw)
get_dataset(Xtrain, ytrain = None, data_pars = None, )
get_params_sklearn(deep = False)
get_params(deep = False)
test(n_sample           =  1000)
zz_eval(data_pars = None, compute_pars = None, out_pars = None, **kw)
zz_preprocess(prepro_pars)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zml\source\models\model_tseries.py
-------------------------functions----------------------
log(*s)
log2(*s)
log3(*s)
init(*kw, **kwargs)
reset()
LighGBM_recursive(lightgbm_pars= {'objective' =  {'objective':'quantile', 'alpha': 0.5}, forecaster_pars = {'window_length' =  {'window_length': 4})
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(Xpred = None, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)
predict_forward(Xpred = None, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)
save(path = None, info = None)
load_model(path = "")
load_info(path = "")
get_dataset(data_pars = None, task_type = "train", **kw)
test_dataset_tseries(nrows = 10000, coly = None, coldate = None, colcat = None)
time_train_test_split(df, test_size  =  0.4, cols = None, coltime  = "time_key", sort = True, minsize = 5, n_sample = 5, verbose = False)
test(nrows = 10000, coly = None, coldate = None, colcat = None)
test2(nrows = 1000, file_path = None, coly = None, coldate = None, colcat = None)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zml\source\models\model_vaem.py
-------------------------functions----------------------
log(*s)
log2(*s)
log3(*s)
init(*kw, **kwargs)
reset()
save(path = '', info = None)
load_model(path = "")
load_info(path = "")
load_data(filePath, categories, cat_col, num_cols, discrete_cols, targetCol, nsample, delimiter)
encode2(data_decode, list_discrete, records_d, fast_plot)
decode2(data_decode, scaling_factor, list_discrete, records_d, plot = False)
save_model2(model, output_dir)
p_vae_active_learning(Data_train_compressed, Data_train, mask_train, Data_test, mask_test_compressed, mask_test, cat_dims, dim_flt, dic_var_type, args, list_discrete, records_d, estimation_method = 1)
train_p_vae(stage, x_train, Data_train, mask_train, epochs, latent_dim, cat_dims, dim_flt, batch_size, p, K, iteration, list_discrete, records_d, args)
test()

-------------------------methods----------------------
Model_custom.__init__(self)
Model_custom.fit(self,filePath, categories,cat_cols,num_cols,discrete_cols,targetCol,nsample  =  -1,delimiter=',',plot=False)
Model_custom.encode(self)
Model_custom.decode(self)


utilmy\zml\source\models\model_vaemdn.py
-------------------------functions----------------------
log(*s)
log2(*s)
log3(*s)
init(*kw, **kwargs)
reset()
sampling(args)
VAEMDN(model_pars)
AUTOENCODER_BASIC(X_input_dim, loss_type = "CosineSimilarity", lr = 0.01, epsilon = 1e-3, decay = 1e-4, optimizer = 'adam', encodingdim  =  50, dim_list = "50,25,10")
AUTOENCODER_MULTIMODAL(input_shapes = [10], hidden_dims = [128, 64, 8], output_activations = ['sigmoid', 'relu'], loss  =  ['bernoulli_divergence', 'poisson_divergence'], optimizer = 'adam')
fit(data_pars = None, compute_pars = None, out_pars = None, model_class = 'VAEMDN', **kw)
encode(Xpred = None, data_pars = None, compute_pars = {}, out_pars = {}, model_class = 'VAEMDN', **kw)
decode(Xpred = None, data_pars = None, compute_pars = {}, out_pars = {}, index  =  0, **kw)
predict(Xpred = None, data_pars = None, compute_pars = {}, out_pars = {}, model_class = 'VAEMDN', **kw)
get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
get_dataset(Xtrain, cols_type_received, cols_ref)
get_label(encoder, x_train, dummy_train, class_num = 5, batch_size = 256)
save(path = None, info = None)
load_model(path = "", model_class = 'VAEMDN')
load_info(path = "")
test_dataset_correlation(n_rows = 100)
test(n_rows = 100)
test2(n_sample           =  1000)
test3(n_sample  =  1000)
test_helper(model_pars, data_pars, compute_pars)
benchmark(config = '', dmin = 5, dmax = 6)
test4()

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zml\source\models\optuna_lightgbm.py
-------------------------functions----------------------
log(*s)
log2(*s)
log3(*s)
os_makedirs(dir_or_file)
init(*kw, **kwargs)
reset()
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(Xpred = None, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)
save(path = None, info = None)
load_model(path = "")
load_info(path = "")
get_dataset(data_pars = None, task_type = "train", **kw)
test_dataset_classi_fake(nrows = 500)
test(nrows = 500)
test_helper(model_pars, data_pars, compute_pars)
benchmark()
benchmark_helper(train_df, test_df)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zml\source\models\torch_ease.py
-------------------------functions----------------------
log(*s)
log2(*s)
log3(*s)
os_makedirs(dir_or_file)
init(*kw, **kwargs)
reset()
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(Xpred = None, data_pars = None, compute_pars = {}, out_pars = {}, **kw)
save(path = None, info = None)
load_info(path = "")
test_dataset_goodbooks(nrows = 1000)
train_test_split2(df, coly)
get_dataset(data_pars = None, task_type = "train", **kwargs)
test(nrows = 1000)
test_helper(model_pars, data_pars, compute_pars)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zml\source\models\torch_rectorch.py
-------------------------functions----------------------
log(*s)
log2(*s)
log3(*s)
init(*kw, **kwargs)
reset()
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(Xpred = None, data_pars = None, compute_pars = {}, out_pars = {}, **kw)
get_dataset(data_pars = None, task_type = "train")
save(path = None, info = None)
load_info(path = "")
make_rand_sparse_dataset(n_rows = 1000, )
test(n_sample           =  1000)
test_helper(model_pars, data_pars, compute_pars)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zml\source\models\torch_rvae.py
-------------------------functions----------------------
log(*s)
log2(*s)
log3(*s)
init(*kw, **kwargs)
reset()
get_dataset(data_pars, task_type = "train")
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
encode(Xpred = None, data_pars: dict = {}, compute_pars: dict = {}, out_pars: dict = {}, **kw)
decode(Xpred = None, data_pars: dict = {}, compute_pars: dict = {}, out_pars: dict = {}, **kw)
compute_metrics(model, X, dataset_obj, args, epoch, losses_save, logit_pi_prev, X_clean, target_errors, mode)
predict(Xpred = None, data_pars = None, compute_pars = {}, out_pars = {}, **kw)
eval(Xpred = None, data_pars: dict = {}, compute_pars: dict = {}, out_pars: dict = {}, **kw)
save(path = None, info = None)
load_info(path = "")
test(nrows = 1000)
test_helper(m)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, global_pars = None)
RVAE.__init__(self, args)
RVAE._get_dataset_obj(self)
RVAE.fit(self)
RVAE.save(self)
RVAE._save_to_csv(self, X_data, X_data_clean, target_errors, attributes, losses_save, dataset_obj, path_output, args, epoch, mode = 'train')
RVAE.get_inputs(self, x_data, one_hot_categ = False, masking = False, drop_mask = [], in_aux_samples = [])
RVAE.encode(self, x_data, one_hot_categ = False, masking = False, drop_mask = [], in_aux_samples = [])
RVAE.sample_normal(self, q_params_z, eps = None)
RVAE.reparameterize(self, q_params, eps_samples = None)
RVAE.decode(self, z)
RVAE.predict(self, x_data, n_epoch = None, one_hot_categ = False, masking = False, drop_mask = [], in_aux_samples = [])
RVAE.forward(self, x_data, n_epoch = None, one_hot_categ = False, masking = False, drop_mask = [], in_aux_samples = [])
RVAE.loss_function(self, input_data, p_params, q_params, q_samples, clean_comp_only = False, data_eval_clean = False)


utilmy\zml\source\models\torch_tabular.py
-------------------------functions----------------------
log(*s)
log2(*s)
log3(*s)
init(*kw, **kwargs)
reset()
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(Xpred = None, data_pars: dict = {}, compute_pars: dict = {}, out_pars: dict = {}, **kw)
save(path = None, info = None)
load_model(path = "")
load_info(path = "")
get_dataset_tuple(Xtrain, cols_type_received, cols_ref = None)
get_dataset(Xtrain, cols_type_received, cols_ref = None)
train_test_split2(df, coly)
test(n_sample  =  100)
test3(n_sample  =  100)
test_helper(m, X_valid)
test2(nrows = 10000)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zml\source\models\util_models.py
-------------------------functions----------------------
log(*s)
test_dataset_classifier_covtype(nrows = 500)
test_dataset_regress_fake(nrows = 500)
test_dataset_classi_fake(nrows = 500)
test_dataset_petfinder(nrows = 1000)
tf_data_create_sparse(cols_type_received:dict =  {'cols_sparse' : ['col1', 'col2'], 'cols_num'    : ['cola', 'colb']}, cols_ref:list =   [ 'col_sparse', 'col_num'  ], Xtrain:pd.DataFrame = None, **kw)
tf_data_pandas_to_dataset(training_df, colsX, coly)
tf_data_file_to_dataset(pattern, batch_size, mode = tf.estimator.ModeKeys.TRAIN, truncate = None)



utilmy\zml\source\utils\metrics.py


utilmy\zml\source\utils\util.py
-------------------------functions----------------------
os_make_dirs(filename)
save_all(variable_list, folder, globals_main = None)
save(variable_list, folder, globals_main = None)
load(filename = "/folder1/keyname", isabsolutpath = 0, encoding1 = "utf-8")
create_appid(filename)
create_logfilename(filename)
create_uniqueid()
logger_setup(logger_name = None, log_file = None, formatter = FORMATTER_1, isrotate = False, isconsole_output = True, logging_level = logging.DEBUG, )
logger_handler_console(formatter = None)
logger_handler_file(isrotate = False, rotate_time = "midnight", formatter = None, log_file_used = None)
logger_setup2(name = __name__, level = None)
printlog(s = "", s1 = "", s2 = "", s3 = "", s4 = "", s5 = "", s6 = "", s7 = "", s8 = "", s9 = "", s10 = "", app_id = "", logfile = None, iswritelog = True, )
writelog(m = "", f = None)
load_arguments(config_file = None, arg_list = None)
sk_tree_get_ifthen(tree, feature_names, target_names, spacer_base = " ")



utilmy\zml\source\utils\util_autofeature.py
-------------------------functions----------------------
create_model_name(save_folder, model_name)
optim_(modelname = "model_dl.1_lstm.py", pars =  {}, df  =  None, optim_engine = "optuna", optim_method = "normal/prune", save_folder = "model_save/", log_folder = "logs/", ntrials = 2)
optim_optuna(modelname = "model_dl.1_lstm.py", pars =  {}, df  =  None, optim_method = "normal/prune", save_folder = "/mymodel/", log_folder = "", ntrials = 2)
load_arguments(config_file =  None)
data_loader(file_name = 'dataset/GOOG-year.csv')
test_all()
test_fast()



utilmy\zml\source\utils\util_automl.py
-------------------------functions----------------------
import_(abs_module_path, class_name = None)
model_auto_tpot(df, colX, coly, outfolder = "aaserialize/", model_type = "regressor/classifier", train_size = 0.5, generation = 1, population_size = 5, verbosity = 2, )
model_auto_mlbox(filepath= [ "train.csv", "test.csv" ],colX=None, coly=None,do="predict",outfolder="aaserialize/",model_type="regressor/classifier",params={ "csv_seprator"  =  [ "train.csv", "test.csv" ],colX=None, coly=None,do="predict",outfolder="aaserialize/",model_type="regressor/classifier",params={ "csv_seprator" : ",", "train_size" : 0.5, "score_metric" : "accuracy","n_folds": 3, "n_step": 10},param_space =  {'est__strategy':{"search":"choice",                         "space":["LightGBM"]},'est__n_estimators':{"search":"choice",                     "space":[150]},'est__colsample_bytree':{"search":"uniform",                "space":[0.8,0.95]},'est__subsample':{"search":"uniform",                       "space":[0.8,0.95]},'est__max_depth':{"search":"choice",                        "space":[5,6,7,8,9]},'est__learning_rate':{"search":"choice",                    "space":[0.07]}},generation=1,population_size=5,verbosity=2,)
model_auto_automlgs(filepath= [ "train.csv", "test.csv" ],colX=None, coly=None,do="predict",outfolder="aaserialize/",model_type="regressor/classifier",params={ "csv_seprator"  =  [ "train.csv", "test.csv" ],colX=None, coly=None,do="predict",outfolder="aaserialize/",model_type="regressor/classifier",params={ "csv_seprator" : ",", "train_size" : 0.5, "score_metric" : "accuracy","n_folds": 3, "n_step": 10},param_space =  {'est__strategy':{"search":"choice",                         "space":["LightGBM"]},'est__n_estimators':{"search":"choice",                     "space":[150]},'est__colsample_bytree':{"search":"uniform",                "space":[0.8,0.95]},'est__subsample':{"search":"uniform",                       "space":[0.8,0.95]},'est__max_depth':{"search":"choice",                        "space":[5,6,7,8,9]},'est__learning_rate':{"search":"choice",                    "space":[0.07]}},generation=1,population_size=5,verbosity=2,)

-------------------------methods----------------------
dict2.__init__(self, d)


utilmy\zml\source\utils\util_credit.py
-------------------------functions----------------------
ztest()
pd_num_segment_limit(df, col_score = "scoress", coldefault = "y", ntotal_default = 491, def_list = None, nblock = 20.0)
fun_get_segmentlimit(x, l1)
np_drop_duplicates(l1)
model_logistic_score(clf, df1, cols, coltarget, outype = "score")
split_train_test(X, y, split_ratio = 0.8)
split_train(X, y, split_ratio = 0.8)
split_train2(df1, ntrain = 10000, ntest = 100000, colused = None, coltarget = None, nratio = 0.4)



utilmy\zml\source\utils\util_csv.py
-------------------------functions----------------------
xl_setstyle(file1)
xl_val(ws, colj, rowi)
xl_get_rowcol(ws, i0, j0, imax, jmax)
csv_dtypes_getdict(df = None, csvfile = None)
csv_fast_processing()
csv_col_schema_toexcel(dircsv = "", filepattern = "*.csv", outfile = ".xlsx", returntable = 1, maxrow = 5000000, maxcol_pertable = 90, maxstrlen = "U80", )
csv_col_get_dict_categoryfreq(dircsv, filepattern = "*.csv", category_cols = [], maxline = -1, fileencoding = "utf-8")
csv_row_reduce_line(fromfile, tofile, condfilter, catval_tokeep, header = True, maxline = -1)
csv_analysis()
csv_row_reduce_line_manual(file_category, file_transact, file_reduced)
csv_row_mapreduce(dircsv = "", outfile = "", type_mapreduce = "sum", nrow = 1000000, chunk = 5000000)
csv_pivotable(dircsv = "", filepattern = "*.csv", fileh5 = ".h5", leftX = "col0", topY = "col2", centerZ = "coli", mapreduce = "sum", chunksize = 500000, tablename = "df", )
csv_bigcompute()
db_getdata()
db_sql()
db_meta_add("", []), schema = None, df_table_uri = None, df_table_columns = None)
db_meta_find(ALLDB, query = "", filter_db = [], filter_table = [], filter_column = [])
str_to_unicode(x, encoding = "utf-8")
isnull(x)



utilmy\zml\source\utils\util_date.py
-------------------------functions----------------------
pd_datestring_split(dfref, coldate, fmt="%Y-%m-%d %H = "%Y-%m-%d %H:%M:%S", return_val = "split")
datestring_todatetime(datelist, fmt="%Y-%m-%d %H = "%Y-%m-%d %H:%M:%S")
datetime_tostring(datelist, fmt="%Y-%m-%d %H = "%Y-%m-%d %H:%M:%S")
datetime_tointhour(datelist)
datetime_toint(datelist)
datetime_to_milisec(datelist)
datetime_weekday(datelist)
datetime_weekday_fast(dateval)
datetime_quarter(datetimex)
dateime_daytime(datetimex)
datenumpy_todatetime(tt, islocaltime = True)
datetime_tonumpydate(t, islocaltime = True)
np_dict_tolist(dd)
np_dict_tostr_val(dd)
np_dict_tostr_key(dd)



utilmy\zml\source\utils\util_deep.py
-------------------------functions----------------------
tf_to_dot(graph)



utilmy\zml\source\utils\util_import.py
-------------------------methods----------------------
dict2.__init__(self, d)


utilmy\zml\source\utils\util_metric.py
-------------------------functions----------------------
mean_reciprocal_rank(rs)
r_precision(r)
precision_at_k(r, k)
average_precision(r)
mean_average_precision(rs)
dcg_at_k(r, k, method = 0)
ndcg_at_k(r, k, method = 0)



utilmy\zml\source\utils\util_model.py
-------------------------functions----------------------
import_(abs_module_path, class_name = None)
pd_dim_reduction(df, colname, colprefix = "colsvd", method = "svd", dimpca = 2, model_pretrain = None, return_val = "dataframe,param", )
model_lightgbm_kfold(df, colname = None, num_folds = 2, stratified = False, colexclude = None, debug = False)
model_catboost_classifier(Xtrain, Ytrain, Xcolname = None, pars={"learning_rate" = {"learning_rate": 0.1, "iterations": 1000, "random_seed": 0, "loss_function": "MultiClass", }, isprint = 0, )
sk_score_get(name = "r2")
sk_params_search_best(clf, X, y, 0, 1, 5)}, method = "gridsearch", param_search={"scorename" = {"scorename": "r2", "cv": 5, "population_size": 5, "generations_number": 3}, )
sk_error(ypred, ytrue, method = "r2", sample_weight = None, multioutput = None)
sk_cluster(Xmat, method = "kmode", ), kwds={"metric" = {"metric": "euclidean", "min_cluster_size": 150, "min_samples": 3}, isprint = 1, preprocess={"norm" = {"norm": False}, )
sk_model_ensemble_weight(model_list, acclevel, maxlevel = 0.88)
sk_model_votingpredict(estimators, voting, ww, X_test)
sk_showconfusion(Y, Ypred, isprint = True)
sk_showmetrics(y_test, ytest_pred, ytest_proba, target_names = ["0", "1"], return_stat = 0)
sk_metric_roc_optimal_cutoff(ytest, ytest_proba)
sk_metric_roc_auc(y_test, ytest_pred, ytest_proba)
sk_metric_roc_auc_multiclass(n_classes = 3, y_test = None, y_test_pred = None, y_predict_proba = None)
sk_model_eval_regression(clf, istrain = 1, Xtrain = None, ytrain = None, Xval = None, yval = None)
sk_model_eval_classification(clf, istrain = 1, Xtrain = None, ytrain = None, Xtest = None, ytest = None)
sk_metrics_eval(clf, Xtest, ytest, cv = 1, metrics = ["f1_macro", "accuracy", "precision_macro", "recall_macro"])
sk_model_eval(clf, istrain = 1, Xtrain = None, ytrain = None, Xval = None, yval = None)
sk_feature_impt(clf, colname, model_type = "logistic")
sk_feature_selection(clf, method = "f_classif", colname = None, kbest = 50, Xtrain = None, ytrain = None)
sk_feature_evaluation(clf, df, kbest = 30, colname_best = None, dfy = None)
sk_feature_prior_shift()
sk_feature_concept_shift(df)
sk_feature_covariate_shift(dftrain, dftest, colname, nsample = 10000)
sk_model_eval_classification_cv(clf, X, y, test_size = 0.5, ncv = 1, method = "random")

-------------------------methods----------------------
dict2.__init__(self, d)
model_template1.__init__(self, alpha = 0.5, low_y_cut = -0.09, high_y_cut = 0.09, ww0 = 0.95)
model_template1.fit(self, X, Y = None)
model_template1.predict(self, X, y = None, ymedian = None)
model_template1.score(self, X, Ytrue = None, ymedian = None)


utilmy\zml\source\utils\util_optim.py
-------------------------functions----------------------
create_model_name(save_folder, model_name)
optim(modelname = "model_dl.1_lstm.py", pars =  {}, df  =  None, optim_engine = "optuna", optim_method = "normal/prune", save_folder = "model_save/", log_folder = "logs/", ntrials = 2)
optim_optuna(modelname = "model_dl.1_lstm.py", pars =  {}, df  =  None, optim_method = "normal/prune", save_folder = "/mymodel/", log_folder = "", ntrials = 2)
load_arguments(config_file =  None)
data_loader(file_name = 'dataset/GOOG-year.csv')
test_all()
test_fast()



utilmy\zml\source\utils\util_pipeline.py
-------------------------functions----------------------
pd_pipeline(bin_cols, text_col, X, y)
pd_grid_search(full_pipeline, X, y)



utilmy\zml\source\utils\util_plot.py
-------------------------functions----------------------
pd_colnum_tocat_stat(input_data, feature, target_col, bins, cuts = 0)
plot_univariate_plots(data, target_col, features_list = 0, bins = 10, data_test = 0)
plot_univariate_histogram(feature, data, target_col, bins = 10, data_test = 0)
pd_stat_distribution_trend_correlation(grouped, grouped_test, feature, target_col)
plot_col_univariate(input_data, feature, target_col, trend_correlation = None)
plotbar(df, colname, figsize = (20, 10)
plotxy(12, 10), title = "feature importance", savefile = "myfile.png")
plot_col_distribution(df, col_include = None, col_exclude = None, pars={"binsize" = {"binsize": 20})
plot_pair(df, Xcolname = None, Ycoltarget = None)
plot_distance_heatmap(Xmat_dist, Xcolname)
plot_cluster_2D(X_2dim, target_class, target_names)
plot_cluster_tsne(Xmat, Xcluster_label = None, metric = "euclidean", perplexity = 50, ncomponent = 2, savefile = "", isprecompute = False, returnval = True, )
plot_cluster_pca(Xmat, Xcluster_label = None, metric = "euclidean", dimpca = 2, whiten = True, isprecompute = False, savefile = "", doreturn = 1, )
plot_cluster_hiearchy(Xmat_dist, p = 30, truncate_mode = None, color_threshold = None, get_leaves = True, orientation = "top", labels = None, count_sort = False, distance_sort = False, show_leaf_counts = True, do_plot = 1, no_labels = False, leaf_font_size = None, leaf_rotation = None, leaf_label_func = None, show_contracted = False, link_color_func = None, ax = None, above_threshold_color = "b", annotate_above = 0, )
plot_distribution_density(Xsample, kernel = "gaussian", N = 10, bandwith = 1 / 10.0)
plot_Y(Yval, typeplot = ".b", tsize = None, labels = None, title = "", xlabel = "", ylabel = "", zcolor_label = "", 8, 6), dpi = 75, savefile = "", color_dot = "Blues", doreturn = 0, )
plot_XY(xx, yy, zcolor = None, tsize = None, labels = None, title = "", xlabel = "", ylabel = "", zcolor_label = "", 8, 6), dpi = 75, savefile = "", color_dot = "Blues", doreturn = 0, )
plot_XY_plotly(xx, yy, towhere = "url")
plot_XY_seaborn(X, Y, Zcolor = None)
plot_cols_with_NaNs(df, nb_to_show)
plot_col_correl_matrix(df, cols, annot = True, size = 30)
plot_col_correl_target(df, cols, coltarget, nb_to_show = 10, ascending = False)
plot_plotly()



utilmy\zml\source\utils\util_sql.py
-------------------------functions----------------------
sql_create_dbengine(type1 = "", dbname = "", login = "", password = "", url = "localhost", port = 5432)
sql_query(sqlr = "SELECT ticker,shortratio,sector1_id, FROM stockfundamental", dbengine = None, output = "df", dburl="sqlite = "sqlite:///aaserialize/store/finviz.db", )
sql_get_dbschema(dburl="sqlite = "sqlite:///aapackage/store/yahoo.db", dbengine = None, isprint = 0)
sql_delete_table(name, dbengine)
sql_insert_excel(file1 = ".xls", dbengine = None, dbtype = "")
sql_insert_df(df, dbtable, dbengine, col_drop = ["id"], verbose = 1)
sql_insert_csv(csvfile, dbtable, dbengine, col_drop = [])
sql_insert_csv2(csvfile = "", dbtable = "", columns = [], dbengine = None, nrows = 10000)
sql_postgres_create_table(mytable = "", database = "", username = "", password = "")
sql_postgres_query_to_csv(sqlr = "SELECT ticker,shortratio,sector1_id, FROM stockfundamental", csv_out = "")
sql_postgres_pivot()
sql_mysql_insert_excel()
sql_pivotable(dbcon, ss = "select  ")



utilmy\zml\source\utils\util_stat.py
-------------------------functions----------------------
np_conditional_entropy(x, y)
np_correl_cat_cat_cramers_v(x, y)
np_correl_cat_cat_theils_u(x, y)
np_correl_cat_num_ratio(cat_array, num_array)
pd_num_correl_associations(df, colcat = None, mark_columns = False, theil_u = False, plot = True, return_results = False, **kwargs)
stat_hypothesis_test_permutation(df, variable, classes, repetitions)
np_transform_pca(X, dimpca = 2, whiten = True)
sk_distribution_kernel_bestbandwidth(X, kde)
sk_distribution_kernel_sample(kde = None, n = 1)

-------------------------methods----------------------
dict2.__init__(self, d)


utilmy\zml\source\utils\util_text.py
-------------------------functions----------------------
get_stopwords(lang)
coltext_stemporter(text)
coltext_lemmatizer(text)
coltext_stemmer(text, sep = " ")
coltext_stopwords(text, stopwords = None, sep = " ")
pd_coltext_fillna(df, colname, val = "")
pd_coltext_clean(dfref, colname, stopwords)
pd_coltext_clean_advanced(dfref, colname, fromword, toword)
pd_coltext_wordfreq(df, coltext, sep = " ")
pd_fromdict(ddict, colname)
pd_coltext_encoder(df)
pd_coltext_countvect(df, coltext, word_tokeep = None, word_minfreq = 1, return_val = "dataframe,param")
pd_coltext_tdidf(df, coltext, word_tokeep = None, word_minfreq = 1, return_val = "dataframe,param")
pd_coltext_minhash(dfref, colname, n_component = 2, model_pretrain_dict = None, return_val = "dataframe,param")
pd_coltext_hashing(df, coltext, n_features = 20)
pd_coltext_tdidf_multi(df, coltext, coltext_freq, ntoken = 100, word_tokeep_dict = None, stopwords = None, return_val = "dataframe,param", )



utilmy\zml\source\utils\util_text_embedding.py
-------------------------functions----------------------
test_MDVEncoder()

-------------------------methods----------------------
NgramNaiveFisherKernel.__init__(self, 2, 4), categories = "auto", dtype = np.float64, handle_unknown = "ignore", hashing_dim = None, n_prototypes = None, random_state = None, n_jobs = None, )
NgramNaiveFisherKernel.fit(self, X, y = None)
NgramNaiveFisherKernel.transform(self, X)
NgramNaiveFisherKernel._ngram_presence_fisher_kernel(self, strings, cats)
NgramNaiveFisherKernel._ngram_presence_fisher_kernel2(self, strings, cats)
PretrainedWord2Vec.__init__(self, n_components = None, language = "english", model_path = None, bert_args={'bert_model' = {'bert_model': None, 'bert_dataset_name': None, 'oov': 'sum', 'ctx': None})
PretrainedWord2Vec.fit(self, X, y = None)
PretrainedWord2Vec.transform(self, X)
PretrainedBert.fit(self, X, y = None)
PretrainedBert.transform(self, X: list)
PretrainedGensim.fit(self, X, y = None)
PretrainedGensim.transform(self, X: dict)
PretrainedGensim.__word_forms(self, word)
PretrainedGensim.__get_word_embedding(self, word, model)
PretrainedFastText.__init__(self, n_components, language = "english")
PretrainedFastText.fit(self, X, y = None)
PretrainedFastText.transform(self, X)
AdHocIndependentPDF.__init__(self, fisher_kernel = True, dtype = np.float64, ngram_range = (2, 4)
AdHocIndependentPDF.fit(self, X, y = None)
AdHocIndependentPDF.transform(self, X)
NgramsMultinomialMixture.__init__(self, n_topics = 10, max_iters = 100, fisher_kernel = True, beta_init_type = None, max_mean_change_tol = 1e-5, 2, 4), )
NgramsMultinomialMixture._get_most_frequent(self, X)
NgramsMultinomialMixture._max_mean_change(self, last_beta, beta)
NgramsMultinomialMixture._e_step(self, D, unqD, X, unqX, theta, beta)
NgramsMultinomialMixture._m_step(self, D, _doc_topic_posterior)
NgramsMultinomialMixture.fit(self, X, y = None)
NgramsMultinomialMixture.transform(self, X)
AdHocNgramsMultinomialMixture.__init__(self, n_iters = 10, fisher_kernel = True, ngram_range = (2, 4)
AdHocNgramsMultinomialMixture._e_step(self, D, unqD, X, unqX, theta, beta)
AdHocNgramsMultinomialMixture._m_step(self, D, _doc_topic_posterior)
AdHocNgramsMultinomialMixture.fit(self, X, y = None)
AdHocNgramsMultinomialMixture.transform(self, X)
MDVEncoder.__init__(self, clf_type)
MDVEncoder.fit(self, X, y = None)
MDVEncoder.transform(self, X)
PasstroughEncoder.__init__(self, passthrough = True)
PasstroughEncoder.fit(self, X, y = None)
PasstroughEncoder.transform(self, X)
ColumnEncoder.__init__(self, encoder_name, reduction_method = None, 2, 4), categories = "auto", dtype = np.float64, handle_unknown = "ignore", clf_type = None, n_components = None, )
ColumnEncoder._get_most_frequent(self, X)
ColumnEncoder.get_feature_names(self)
ColumnEncoder.fit(self, X, y = None)
ColumnEncoder.transform(self, X)
DimensionalityReduction.__init__(self, method_name = None, n_components = None, column_names = None)
DimensionalityReduction.fit(self, X, y = None)
DimensionalityReduction.transform(self, X)


utilmy\zml\source\utils\ztest.py
-------------------------methods----------------------
dict2.__init__(self, d)


utilmy\zml\source\utils\__init__.py


utilmy\zzarchive\storage\aapackagedev\random.py
-------------------------functions----------------------
convert_csv2hd5f(filein1, filename)
getrandom_tonumpy(filename, nbdim, nbsample)
comoment(xx, yy, nsample, kx, ky)
acf(data)
getdvector(dimmax, istart, idimstart)
pathScheme_std(T, n, zz)
pathScheme_bb(T, n, zz)
pathScheme_(T, n, zz)
testdensity(nsample, totdim, bin01, Ti = -1)
plotdensity(nsample, totdim, bin01, tit0, Ti = -1)
testdensity2d(nsample, totdim, bin01, nbasset)
lognormal_process2d(a1, z1, a2, z2, k)
testdensity2d2(nsample, totdim, bin01, nbasset, process01 = lognormal_process2d, a1 = 0.25, a2 = 0.25, kk = 1)
call_process(a, z, k)
binary_process(a, z, k)
pricing01(totdim, nsample, a, strike, process01, aa = 0.25, itmax = -1, tt = 10)
plotdensity2(nsample, totdim, bin01, tit0, process01, vol = 0.25, tt = 5, Ti = -1)
Plot2D_random_show(dir1, title1, dimxmax, dimymax, dimstep, samplejump, nsamplegraph)
Plot2D_random_save(dir1, title1, dimxmax, dimymax, dimstep, samplejump, nsamplegraph, )
getoutlier_fromrandom(filename, jmax1, imax1, isamplejum, nsample, fileoutlier=   'E =    'E:\_data\_QUASI_SOBOL_gaussian_outlier.h5')
numexpr_vect_calc(filename, i0 = 0, imax = 1000, expr, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
getoutlier_fromrandom_fast(filename, jmax1, imax1, isamplejum, nsample, trigger1 = 0.28, fileoutlier=   'E =    'E:\_data\_QUASI_SOBOL_gaussian_outlier.h5')
outlier_clean(vv2)
overwrite_data(fileoutlier, vv2)
doublecheck_outlier(fileoutlier, ijump, nsample = 4000, trigger1 = 0.1, )fileoutlier=   'E =    'E:\_data\_QUASI_SOBOL_gaussian_outlier.h5'fileoutlier, 'data')    #from filevv5 =  pdf.values   #to numpy vectordel pdfistartx= 0; istarty= 0nsample= 4000trigger1=  0.1crrmax = 250000kk=0(crrmax, 4), dtype = 'int')  #empty listvv5)[0]0, kkmax1, 1) :  #Decrasing: dimy0 to dimmindimx =  vv5[kk, 0];   dimy =  vv5[kk, 1]y0= dimy * ijump + istartyym= dimy* ijump + nsample + istartyyyu1= yy1[y0 =  dimy * ijump + istartyym= dimy* ijump + nsample + istartyyyu1= yy1[y0:ym];   yyu2= yy2[y0:ym];   yyu3= yy3[y0:ym]x0= dimx * ijump + istartxxm= dimx* ijump + nsample + istartxxxu1= yy1[x0:xm];   xxu2= yy2[x0:xm];   xxu3= yy3[x0:xm]"sum( xxu3 * yyu1)") / (nsample) # X3.Y moments"sum( xxu1 * yyu3)") / (nsample)"sum( xxu2 * yyu2)") / (nsample)abs(c22) > trigger1)  :)
plot_outlier(fileoutlier, kk)fileoutlier, 'data')    #from filevv =  df.values   #to numpy vectordel dfxx= vv[kk, 0]yy =  vv[kk, 1]xx, yy, s = 1 )[00, 1000, 00, 1000])nsample)+'sampl D_'+str(dimx)+' X D_'+str(dimy)tit1)'_img/'+tit1+'_outlier.jpg', dpi = 100))yy, kmax))
permute(yy, kmax)
permute2(xx, yy, kmax)



utilmy\zzarchive\storage\aapackage_gen\codeanalysis.py
-------------------------functions----------------------
wi(*args)
printinfile(vv, file1)
wi2(*args)
indent()
dedent()
describe_builtin(obj)
describe_func(obj, method = False)
describe_klass(obj)
describe(obj)
describe_builtin2(obj, name1)
describe_func2(obj, method = False, name1 = '')
describe_klass2(obj, name1 = '')
describe2(module)
getmodule_doc(module1, file1 = 'moduledoc.txt')



utilmy\zzarchive\storage\aapackage_gen\global01.py


utilmy\zzarchive\storage\aapackage_gen\util.py
-------------------------functions----------------------
getmodule_doc(module1, fileout = '')



utilmy\zzml\docs\source\conf.py


utilmy\zzml\mlmodels\example\arun_hyper.py


utilmy\zzml\mlmodels\example\arun_model.py


utilmy\zzml\mlmodels\example\benchmark_timeseries_m4.py
-------------------------functions----------------------
benchmark_m4()



utilmy\zzml\mlmodels\example\benchmark_timeseries_m5.py
-------------------------functions----------------------
gluonts_create_dynamic(df_dynamic, submission = 1, single_pred_length = 28, submission_pred_length = 10, n_timeseries = 1, transpose = 1)
gluonts_create_static(df_static, submission = 1, single_pred_length = 28, submission_pred_length = 10, n_timeseries = 1, transpose = 1)
gluonts_create_timeseries(df_timeseries, submission = 1, single_pred_length = 28, submission_pred_length = 10, n_timeseries = 1, transpose = 1)
create_startdate(date = "2011-01-29", freq = "1D", n_timeseries = 1)
gluonts_create_dataset(train_timeseries_list, start_dates_list, train_dynamic_list, train_static_list, freq = "D")
pandas_to_gluonts_multiseries(df_timeseries, df_dynamic, df_static, pars = None)
plot_prob_forecasts(ts_entry, forecast_entry, path, sample_id, inline = True)



utilmy\zzml\mlmodels\example\lightgbm_glass.py


utilmy\zzml\mlmodels\example\vision_mnist.py


utilmy\zzml\mlmodels\model_gluon\fb_prophet.py
-------------------------functions----------------------
get_dataset(data_pars)
get_params(param_pars = {}, **kw)
fit(model = None, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)
predict(model = None, model_pars = None, sess = None, data_pars = None, compute_pars = None, out_pars = None, **kwargs)
save(model = None, session = None, save_pars = {})
load(load_pars = {}, **kw)
metrics_plot(metrics_params)
test(data_path = "dataset/", pars_choice = "test0", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zzml\mlmodels\model_gluon\gluonts_model.py
-------------------------functions----------------------
get_params(choice = "", data_path = "dataset/timeseries/", config_mode = "test", **kw)
get_dataset2(data_pars)
get_dataset(data_pars)
get_dataset_pandas_multi(data_pars)
get_dataset_gluonts(data_pars)
get_dataset_pandas_single(data_pars)
fit(model, sess = None, data_pars = None, model_pars = None, compute_pars = None, out_pars = None, session = None, **kwargs)
predict(model, sess = None, data_pars = None, compute_pars = None, out_pars = None, **kw)
evaluate(model, sess = None, data_pars = None, compute_pars = None, out_pars = None, **kw)
metrics(ypred, data_pars, compute_pars = None, out_pars = None, **kw)
evaluate(model, sess = None, data_pars = None, compute_pars = None, out_pars = None, **kw)
save(model, path)
load(path)
plot_prob_forecasts(ypred, out_pars = None)
plot_predict(item_metrics, out_pars = None)
test_single(data_path = "dataset/", choice = "", config_mode = "test")
test(data_path = "dataset/", choice = "", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, **kwargs)


utilmy\zzml\mlmodels\model_gluon\gluonts_model_old.py
-------------------------functions----------------------
get_params(choice = "", data_path = "dataset/timeseries/", config_mode = "test", **kw)
get_dataset(data_pars)
fit(model, sess = None, data_pars = None, model_pars = None, compute_pars = None, out_pars = None, session = None, **kwargs)
predict(model, sess = None, data_pars = None, compute_pars = None, out_pars = None, **kw)
metrics(ypred, data_pars, compute_pars = None, out_pars = None, **kw)
evaluate(ypred, data_pars, compute_pars = None, out_pars = None, **kw)
save(model, path)
load(path)
plot_prob_forecasts(ypred, out_pars = None)
plot_predict(item_metrics, out_pars = None)
test_single(data_path = "dataset/", choice = "", config_mode = "test")
test(data_path = "dataset/", choice = "", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, **kwargs)


utilmy\zzml\mlmodels\model_gluon\gluon_automl.py
-------------------------functions----------------------
path_setup(out_folder = "", sublevel = 0, data_path = "dataset/")
_config_process(config)
get_params(choice = "", data_path = "dataset/", config_mode = "test", **kw)
test(data_path = "dataset/", pars_choice = "json")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, compute_pars = None)


utilmy\zzml\mlmodels\model_gluon\util.py
-------------------------functions----------------------
_config_process(data_path, config_mode = "test")
fit(model, sess = None, data_pars = None, model_pars = None, compute_pars = None, out_pars = None, session = None, **kwargs)
predict(model, sess = None, data_pars = None, compute_pars = None, out_pars = None, **kwargs)
metrics(ypred, data_pars, compute_pars = None, out_pars = None, **kwargs)
save(model, path)
load(path)
get_dataset(data_pars)
plot_prob_forecasts(ypred, out_pars = None)
plot_predict(item_metrics, out_pars = None)

-------------------------methods----------------------
Model_empty.__init__(self, model_pars = None, compute_pars = None)


utilmy\zzml\mlmodels\model_gluon\util_autogluon.py
-------------------------functions----------------------
os_package_root_path(filepath, sublevel = 0, path_add = "")
log(*s, n = 0, m = 1)
_get_dataset_from_aws(**kw)
import_data_fromfile(**kw)
get_dataset(**kw)
fit(model, data_pars = None, model_pars = None, compute_pars = None, out_pars = None, session = None, **kwargs)
predict(model, data_pars, compute_pars = None, out_pars = None, **kwargs)
metrics(model, ypred, ytrue, data_pars, compute_pars = None, out_pars = None, **kwargs)
save(model, out_pars)
load(path)

-------------------------methods----------------------
Model_empty.__init__(self, model_pars = None, compute_pars = None)


utilmy\zzml\mlmodels\model_keras\armdn.py


utilmy\zzml\mlmodels\model_keras\Autokeras.py
-------------------------functions----------------------
get_config_file()
get_params(param_pars = None, **kw)
get_dataset_imbd(data_pars)
get_dataset_titanic(data_pars)
get_dataset(data_pars)
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kwargs)
predict(model, session = None, data_pars = None, compute_pars = None, out_pars = None)
evaluate(model, data_pars = None, compute_pars = None, out_pars = None)
save(model, session = None, save_pars = None, config_mode = "test")
load(load_pars, config_mode = "test")
test_single(data_path = "dataset/", pars_choice = "json", config_mode = "test")
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, out_pars = None)


utilmy\zzml\mlmodels\model_keras\charcnn.py
-------------------------functions----------------------
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
evaluate(model, session = None, data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(model, session = None, data_pars = None, out_pars = None, compute_pars = None, **kw)
reset_model()
save(model = None, save_pars = None, session = None)
load(load_pars = None)
str_to_indexes(s)
tokenize(data, num_of_classes = 4)
get_dataset(data_pars = None, **kw)
get_params(param_pars = {}, **kw)
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zzml\mlmodels\model_keras\charcnn_zhang.py
-------------------------functions----------------------
fit(model, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)
evaluate(model, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)
predict(model, sess = None, data_pars = {}, out_pars = {}, compute_pars = {}, **kw)
reset_model()
save(model = None, session = None, save_pars = {})
load(load_pars = {})
get_dataset(data_pars = None, **kw)
get_params(param_pars = {}, **kw)
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zzml\mlmodels\model_keras\deepctr.py
-------------------------functions----------------------
_preprocess_criteo(df, **kw)
_preprocess_movielens(df, **kw)
get_dataset(data_pars = None, **kw)
fit(model, session = None, compute_pars = None, data_pars = None, out_pars = None, **kwargs)
predict(model, session = None, compute_pars = None, data_pars = None, out_pars = None, **kwargs)
metrics(ypred, ytrue = None, session = None, compute_pars = None, data_pars = None, out_pars = None, **kwargs)
reset_model()
path_setup(out_folder = "", sublevel = 0, data_path = "dataset/")
_config_process(config)
config_load(data_path, file_default, config_mode)
get_params(choice = "", data_path = "dataset/", config_mode = "test", **kwargs)
test(data_path = "dataset/", pars_choice = 0, **kwargs)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, **kwargs)


utilmy\zzml\mlmodels\model_keras\namentity_crm_bilstm.py
-------------------------functions----------------------
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
evaluate(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(model, sess = None, data_pars = None, out_pars = None, compute_pars = None, **kw)
reset_model()
save(model = None, session = None, save_pars = None)
load(load_pars)
get_dataset(data_pars)
get_params(param_pars = {}, **kw)
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, **kwargs)


utilmy\zzml\mlmodels\model_keras\preprocess.py
-------------------------functions----------------------
os_package_root_path(filepath, sublevel = 0, path_add = "")
log(*s, n = 0, m = 1)
_preprocess_criteo(df, **kw)
_preprocess_movielens(df, **kw)
_preprocess_none(df, **kw)
get_dataset(**kw)
test(data_path = "dataset/", pars_choice = 0)



utilmy\zzml\mlmodels\model_keras\textcnn.py
-------------------------functions----------------------
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
evaluate(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(model, sess = None, data_pars = None, out_pars = None, compute_pars = None, **kw)
reset_model()
save(model = None, session = None, save_pars = {})
load(load_pars = {})
get_dataset(data_pars = None, **kw)
get_params(param_pars = {}, **kw)
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zzml\mlmodels\model_keras\util.py
-------------------------functions----------------------
os_package_root_path(filepath, sublevel = 0, path_add = "")
log(*s, n = 0, m = 1)
_config_process(data_path, config_mode = "test")
get_dataset(**kw)
fit(model, data_pars = None, model_pars = None, compute_pars = None, out_pars = None, session = None, **kwargs)
predict(model, data_pars, compute_pars = None, out_pars = None, **kwargs)
metrics(ypred, data_pars, compute_pars = None, out_pars = None, **kwargs)
save(model, path)
load(path)

-------------------------methods----------------------
Model_empty.__init__(self, model_pars = None, compute_pars = None)


utilmy\zzml\mlmodels\model_tch\matchZoo.py
-------------------------functions----------------------
get_task(model_pars, task)
get_glove_embedding_matrix(term_index, dimension)
get_data_loader(model_name, preprocessor, preprocess_pars, raw_data)
get_config_file()
get_raw_dataset(data_info, **args)
get_dataset(_model, preprocessor, _preprocessor_pars, data_pars)
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kwargs)
predict(model, session = None, data_pars = None, compute_pars = None, out_pars = None)
evaluate(model, data_pars = None, compute_pars = None, out_pars = None)
save(model, session = None, save_pars = None)
load(load_pars)
get_params(param_pars = None, **kw)
test_train(data_path, pars_choice, model_name)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, out_pars = None)


utilmy\zzml\mlmodels\model_tch\textcnn.py
-------------------------functions----------------------
_train(m, device, train_itr, optimizer, epoch, max_epoch)
_valid(m, device, test_itr)
_get_device()
get_config_file()
get_data_file()
analyze_datainfo_paths(data_info)
split_train_valid(data_info, **args)
clean_str(string)
create_tabular_dataset(data_info, **args)
create_data_iterator(batch_size, tabular_train, tabular_valid, d)
evaluate(model, session = None, data_pars = None, compute_pars = None, out_pars = None, **kwargs)
fit(model, sess = None, data_pars = None, compute_pars = None, out_pars = None, **kwargs)
get_dataset(data_pars = None, out_pars = None, **kwargs)
predict(model, session = None, data_pars = None, compute_pars = None, out_pars = None, return_ytrue = 1)
save(model, session = None, save_pars = None)
load(load_pars =  None)
get_params(param_pars = None, **kw)
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
TextCNN.__init__(self, model_pars = None, data_pars = None, compute_pars = None, **kwargs)
TextCNN.rebuild_embed(self, vocab_built)
TextCNN.forward(self, x)
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zzml\mlmodels\model_tch\torchhub.py
-------------------------functions----------------------
_train(m, device, train_itr, criterion, optimizer, epoch, max_epoch, imax = 1)
_valid(m, device, test_itr, criterion, imax = 1)
_get_device()
get_config_file()
get_params(param_pars = None, **kw)
get_dataset(data_pars = None, **kw)
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kwargs)
predict(model, session = None, data_pars = None, compute_pars = None, out_pars = None, imax  =  1, return_ytrue = 1)
evaluate(model, data_pars = None, compute_pars = None, out_pars = None)
save(model, session = None, save_pars = None)
load(load_pars)
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")
test2(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, out_pars = None)


utilmy\zzml\mlmodels\model_tch\transformer_sentence.py
-------------------------functions----------------------
fit(model, data_pars = None, model_pars = None, compute_pars = None, out_pars = None, *args, **kw)
evaluate(model, session = None, data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(model, session = None, data_pars = None, out_pars = None, compute_pars = None, **kw)
reset_model()
save(model, session = None, save_pars = None)
load(load_pars = None)
get_dataset(data_pars = None, **kw)
fit2(model, data_pars = None, model_pars = None, compute_pars = None, out_pars = None, *args, **kw)
predict2(model, session = None, data_pars = None, out_pars = None, compute_pars = None, **kw)
get_dataset2(data_pars = None, model = None, **kw)
get_params(param_pars, **kw)
test(data_path = "dataset/", pars_choice = "test01", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, **kwargs)


utilmy\zzml\mlmodels\model_tch\util_data.py


utilmy\zzml\mlmodels\model_tch\util_transformer.py
-------------------------functions----------------------
convert_example_to_feature(example_row, pad_token = 0, sequence_a_segment_id = 0, sequence_b_segment_id = 1, cls_token_segment_id = 1, pad_token_segment_id = 0, mask_padding_with_zero = True, sep_token_extra = False)
convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_mode, cls_token_at_end = False, sep_token_extra = False, pad_on_left = False, cls_token = '[CLS]', sep_token = '[SEP]', pad_token = 0, sequence_a_segment_id = 0, sequence_b_segment_id = 1, cls_token_segment_id = 1, pad_token_segment_id = 0, mask_padding_with_zero = True, ) - 2))
_truncate_seq_pair(tokens_a, tokens_b, max_length)

-------------------------methods----------------------
InputExample.__init__(self, guid, text_a, text_b = None, label = None)
InputFeatures.__init__(self, input_ids, input_mask, segment_ids, label_id)
DataProcessor.get_train_examples(self, data_dir)
DataProcessor.get_dev_examples(self, data_dir)
DataProcessor.get_labels(self)
DataProcessor._read_tsv(cls, input_file, quotechar = None)
BinaryProcessor.get_train_examples(self, data_dir)
BinaryProcessor.get_dev_examples(self, data_dir)
BinaryProcessor.get_labels(self)
BinaryProcessor._create_examples(self, lines, set_type)
TransformerDataReader.__init__(self, **args)
TransformerDataReader.compute(self, input_tmp)
TransformerDataReader.get_data(self)


utilmy\zzml\mlmodels\model_tf\1_lstm.py
-------------------------functions----------------------
init(*kw, **kwargs)
fit(data_pars = None, compute_pars = None, out_pars = None, **kwarg)
evaluate(data_pars = None, compute_pars = None, out_pars = None)
metrics(data_pars = None, compute_pars = None, out_pars = None)
predict(data_pars = None, compute_pars = None, out_pars = None, get_hidden_state = False, init_value = None)
reset_model()
save(save_pars = None)
load(load_pars = None)
get_dataset(data_pars = None)
get_params(param_pars = {}, **kw)
test(data_path = "dataset/", pars_choice = "test01", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, **kwargs)


utilmy\zzml\mlmodels\model_tf\util.py
-------------------------functions----------------------
os_module_path()
os_file_path(data_path)
os_package_root_path(filepath, sublevel = 0, path_add = "")
batch_invert_permutation(permutations)
batch_gather(values, indices)
one_hot(length, index)
set_root_dir()



utilmy\zzml\mlmodels\preprocess\generic.py
-------------------------functions----------------------
log2(*v, **kw)
torch_datasets_wrapper(sets, args_list  =  None, **args)
pandas_reader(task, path, colX, coly, path_eval = None, train_size = 0.8)
tf_dataset_download(data_info, **args)
get_dataset_torch(data_info, **args)
get_dataset_keras(data_info, **args)
get_model_embedding(data_info, **args)
get_model_embedding(data_info, **args)
text_create_tabular_dataset(path_train, path_valid, lang = 'en', pretrained_emb = 'glove.6B.300d')
create_kerasDataloader()
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Custom_DataLoader.__init__(self, dataset = None, batch_size = -1, shuffle = True, drop_last = False)
Custom_DataLoader.__iter__(self)
pandasDataset.__init__(self, root = "", train = True, transform = None, target_transform = None, download = False, data_info = {}, **args)
pandasDataset.__len__(self)
pandasDataset.shuffle(self, frac = 1.0, random_state = 123)
pandasDataset.get_data(self)
NumpyDataset.__init__(self, root = "", train = True, transform = None, target_transform = None, download = False, data_info = {}, **args)
NumpyDataset.__getitem__(self, index)
NumpyDataset.__len__(self)
NumpyDataset.get_data(self)


utilmy\zzml\mlmodels\preprocess\generic_old.py
-------------------------functions----------------------
torch_datasets_wrapper(sets, args_list  =  None, **args)
load_function(uri_name = "path_norm")
get_dataset_torch(data_pars)
get_dataset_keras(data_pars)
get_model_embedding(model_pars, data_pars)
text_create_tabular_dataset(path_train, path_valid, lang = 'en', pretrained_emb = 'glove.6B.300d')
create_kerasDataloader()
tf_dataset_download(data_pars)
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
NumpyDataset.__init__(self, root = "", train = True, transform = None, target_transform = None, download = False, data_pars = None)


utilmy\zzml\mlmodels\preprocess\image.py
-------------------------functions----------------------
torch_transform_mnist()
torchvision_dataset_MNIST_load(path, **args)
torch_transform_data_augment(fixed_scale  =  256, train  =  False)
torch_transform_generic(fixed_scale  =  256, train  =  False)



utilmy\zzml\mlmodels\preprocess\tabular_keras.py
-------------------------functions----------------------
get_xy_fd_din(hash_flag = False)
get_xy_fd_dien(use_neg = False, hash_flag = False)
get_xy_fd_dsin(hash_flag = False)
gen_sequence(dim, max_len, sample_size)
get_test_data(sample_size = 1000, embedding_size = 4, sparse_feature_num = 1, dense_feature_num = 1, sequence_feature = None, classification = True, include_length = False, hash_flag = False, prefix = '', use_group = False)
layer_test(layer_cls, kwargs = {}, input_shape = None, input_dtype = None, input_data = None, expected_output = None, expected_output_dtype = None, fixed_batch_size = False, supports_masking = False)
has_arg(fn, name, accept_all = False)
check_model(model, model_name, x, y, check_model_io = True)



utilmy\zzml\mlmodels\preprocess\text_keras.py
-------------------------functions----------------------
_remove_long_seq(maxlen, seq, label)

-------------------------methods----------------------
Preprocess_namentity.__init__(self, max_len, **args)
Preprocess_namentity.compute(self, df)
Preprocess_namentity.get_data(self)
IMDBDataset.__init__(self, *args, **kwargs)
IMDBDataset.compute(self, data)
IMDBDataset.get_data(self)


utilmy\zzml\mlmodels\preprocess\text_torch.py
-------------------------functions----------------------
test_pandas_fillna(data, **args)
test_onehot_sentences(data, max_len)
test_word_count(data)
test_word_categorical_labels_per_sentence(data, max_len)
clean_str(string)
imdb_spacy_tokenizer(text, lang = "en")



utilmy\zzml\mlmodels\preprocess\timeseries.py
-------------------------functions----------------------
save_to_file(path, data)
gluonts_dataset_to_pandas(dataset_name_list = ["m4_hourly", "m4_daily", "m4_weekly", "m4_monthly", "m4_quarterly", "m4_yearly", ])
gluonts_to_pandas(ds)
pandas_to_gluonts(df, pars = None)
test_gluonts()
gluonts_create_dynamic(df_dynamic, submission = True, single_pred_length = 28, submission_pred_length = 10, n_timeseries = 1, transpose = 1)
gluonts_create_static(df_static, submission = 1, single_pred_length = 28, submission_pred_length = 10, n_timeseries = 1, transpose = 1)
gluonts_create_timeseries(df_timeseries, submission = 1, single_pred_length = 28, submission_pred_length = 10, n_timeseries = 1, transpose = 1)
create_startdate(date = "2011-01-29", freq = "1D", n_timeseries = 1)
gluonts_create_dataset(train_timeseries_list, start_dates_list, train_dynamic_list, train_static_list, freq = "D")
pandas_to_gluonts_multiseries(df_timeseries, df_dynamic, df_static, pars={'submission' = {'submission':True, 'single_pred_length':28, 'submission_pred_length':10, 'n_timeseries':1, 'start_date':"2011-01-29", 'freq':"1D"})
test_gluonts2()
tofloat(x)
pd_load(path)
pd_interpolate(df, cols, pars={"method" = {"method": "linear", "limit_area": "inside"  })
pd_clean_v1(df, cols = None, pars = None)
pd_reshape(test, features, target, pred_len, m_feat)
pd_clean(df, cols = None, pars = None)
time_train_test_split2(df, **kw)
time_train_test_split(df, **kw)
preprocess_timeseries_m5(data_path = None, dataset_name = None, pred_length = 10, item_id = None)
benchmark_m4()
preprocess_timeseries_m5b()

-------------------------methods----------------------
Preprocess_nbeats.__init__(self, backcast_length, forecast_length)
Preprocess_nbeats.compute(self, df)
Preprocess_nbeats.get_data(self)
SklearnMinMaxScaler.__init__(self, **args)
SklearnMinMaxScaler.compute(self, df)
SklearnMinMaxScaler.get_data(self)


utilmy\zzml\mlmodels\preprocess\ztemp.py
-------------------------functions----------------------
get_loader(fix_length, vocab_threshold, batch_size)
pandas_dataset()
custom_dataset()
text_dataloader()
pickle_load(file)
image_dir_load(path)
batch_generator(iterable, n = 1)

-------------------------methods----------------------
MNIST.train_labels(self)
MNIST.test_labels(self)
MNIST.train_data(self)
MNIST.test_data(self)
MNIST.__init__(self, root, train = True, transform = None, target_transform = None, download = False)
MNIST.__getitem__(self, index)
MNIST.__len__(self)
MNIST.raw_folder(self)
MNIST.processed_folder(self)
MNIST.class_to_idx(self)
MNIST._check_exists(self)
MNIST.download(self)
MNIST.extra_repr(self)
DataLoader.__init__(self, data_pars)
DataLoader.compute(self)
DataLoader.__getitem__(self, key)
DataLoader._interpret_input_pars(self, input_pars)
DataLoader._load_data(self, loader)
DataLoader._interpret_output(self, output, intermediate_output)
DataLoader.get_data(self, intermediate = False)
DataLoader._name_outputs(self, names, outputs)
DataLoader._split_data(self)


utilmy\zzml\mlmodels\template\00_template_keras.py
-------------------------functions----------------------
os_package_root_path(filepath, sublevel = 0, path_add = "")
log(*s, n = 0, m = 1)
_preprocess_XXXX(df, **kw)
get_dataset(**kw)
fit(model, session = None, data_pars = None, model_pars = None, compute_pars = None, out_pars = None, **kwargs)
predict(model, data_pars, compute_pars = None, out_pars = None, **kwargs)
metrics(ypred, data_pars, compute_pars = None, out_pars = None, **kwargs)
reset_model()
save(model, path)
load(path)
path_setup(out_folder = "", sublevel = 0, data_path = "dataset/")
get_params(choice = 0, data_path = "dataset/", **kw)
test(data_path = "dataset/", pars_choice = 0)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, **kwargs)
Model_empty.__init__(self, model_pars = None, compute_pars = None)


utilmy\zzml\mlmodels\template\model_xxx.py
-------------------------functions----------------------
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
evaluate(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(model, sess = None, data_pars = None, compute_pars = None, out_pars = None, **kw)
reset_model()
save(model = None, session = None, save_pars = {})
load(load_pars = {})
get_dataset(data_pars = None, **kw)
get_params(param_pars = {}, **kw)
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zzml\mlmodels\utils\bayesian.py
-------------------------functions----------------------
X_transform(dfXy, colsX)
pd_filter(df, filter_dict = None)
demand(price, i0 = 17)
score_fun2(price, cost, units, alpha = 0.3)
unit_fun01(price)
season_remove(x)
season_remove(x)
get_l2_item(df, item_id)
score_fun(price, cost, units, alpha = 0.3)
unit_fun(price)
unit_fun01(price)
unit_fun02(ii = 6990003, t = 0, price = 0, verbose = False)
cost_total(vprice, unit_fun, verbose = False)
optim_de(cost_class, n_iter = 10, time_list = None, pop_size = 20, date0 = "20200507")
price_normalize(vprice)
logic(x1, x2, )
exp_(x1)
covariate_01(ds)
objective(trial)
objective(trial)
objective(trial)
my_funcs(df)
pd_trim(dfi)
train_split_time(df, test_period  =  40, cols = None, coltime  = "time_key", minsize = 5)
pd_show_file(path = "*y-porder_2020* ")
generate_X_item(df, prefix_col  = "")
to_json_highcharts(df, cols, coldate, fpath, verbose = False)
generate_report(path_model)
histo(dfi, path_save = None, nbin = 20.0)
generate_metrics(path, cola = "porder_s2")
pd_to_onehot(df, colnames, map_dict = None, verbose = 1)
generate_itemid_stats(price_dir = "")
pd_col_flatten(cols)

-------------------------methods----------------------
cost_class.__init__(self, dim = 0)
cost_class.fitness(self, x)
cost_class.get_bounds(self)
cost_class.get_name(self)
cost_class.get_extra_info()
sphere_function.__init__(self, dim)
sphere_function.fitness(self, x)
sphere_function.get_bounds(self)
sphere_function.get_name(self)
sphere_function.get_extra_info()
item._init__(self, shop_id = None, item_id = None)
item.elastic(self, window = "1m", date"", model = "default")
item.forecast(start = "", end = "", model = "", model_date = "")


utilmy\zzml\mlmodels\utils\model_v1.py
-------------------------functions----------------------
log(*s)
init(*kw, **kwargs)
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
fit_metrics(data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(data_pars = None, compute_pars = None, out_pars = None, **kw)
reset()
save(path = None, info = None)
load_model(path = "")
load_info(path = "")
preprocess(prepro_pars)
get_dataset(data_pars = None, task_type = "train", **kw)
get_params(param_pars = {}, **kw)
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zzml\mlmodels\utils\parse.py
-------------------------functions----------------------
cli_load_arguments(config_file = None)
extract_args(txt, outfile)



utilmy\zzml\mlmodels\utils\predict.py
-------------------------functions----------------------
cli_load_argument(config_file = None)



utilmy\zzml\mlmodels\utils\test_dataloader.py
-------------------------functions----------------------
pandas_split_xy(out, data_pars)
pandas_load_train_test(path, test_path, **args)
rename_target_to_y(out, data_pars)
load_npz(path)
split_xy_from_dict(out, **kwargs)
split_timeseries_df(out, data_pars, length, shift)
gluon_append_target_string(out, data_pars)
identical_test_set_split(*args, test_size, **kwargs)
read_csvs_from_directory(path, files = None, **args)
tokenize_x(data, no_classes, max_words = None)
timeseries_split(*args, test_size = 0.2)

-------------------------methods----------------------
SingleFunctionPreprocessor.__init__(self, func_dict)
SingleFunctionPreprocessor.compute(self, data)
SingleFunctionPreprocessor.get_data(self)


utilmy\zzml\mlmodels\utils\train.py
-------------------------functions----------------------
cli_load_argument(config_file = None)
create_metrics_summary(path_model, im = 40, verbose = True)
create_mae_summary(path, path_modelgroup, tag = "", ytarget = "porder_s2", agg_level =  None, verbose = True)
pd_check_na(name, dfXy, verbose  =  False, debug = False, train_path = "ztmp/")
train_enhance(dfi, colsref, ytarget, n_sample = 5)
add_dates(df)



utilmy\zzml\mlmodels\utils\ztest_structure.py
-------------------------functions----------------------
get_recursive_files(folderPath, ext = '/*model*/*.py')
log(*s, n = 0, m = 1)
os_package_root_path(filepath, sublevel = 0, path_add = "")
os_file_current_path()
model_get_list(folder = None, block_list = [])
find_in_list(x, llist)
code_check(sign_list = None, model_list = None)
main()



utilmy\codeparser\project_graph\tests\sub_dir\script_test_case_2.py
-------------------------functions----------------------
foo()
bar()



utilmy\codeparser\project_graph\tests\sub_dir\__init__.py


utilmy\recsys\zrecs\docs\source\conf.py


utilmy\recsys\zrecs\examples\06_benchmarks\benchmark_utils.py
-------------------------functions----------------------
prepare_training_als(train, test)
train_als(params, data)
prepare_metrics_als(train, test)
predict_als(model, test)
recommend_k_als(model, test, train, top_k = DEFAULT_K, remove_seen = True)
prepare_training_svd(train, test)
train_svd(params, data)
predict_svd(model, test)
recommend_k_svd(model, test, train, top_k = DEFAULT_K, remove_seen = True)
prepare_training_fastai(train, test)
train_fastai(params, data)
prepare_metrics_fastai(train, test)
predict_fastai(model, test)
recommend_k_fastai(model, test, train, top_k = DEFAULT_K, remove_seen = True)
prepare_training_ncf(train, test)
train_ncf(params, data)
recommend_k_ncf(model, test, train, top_k = DEFAULT_K, remove_seen = True)
prepare_training_cornac(train, test)
recommend_k_cornac(model, test, train, top_k = DEFAULT_K, remove_seen = True)
train_bpr(params, data)
train_bivae(params, data)
prepare_training_sar(train, test)
train_sar(params, data)
recommend_k_sar(model, test, train, top_k = DEFAULT_K, remove_seen = True)
prepare_training_lightgcn(train, test)
train_lightgcn(params, data)
recommend_k_lightgcn(model, test, train, top_k = DEFAULT_K, remove_seen = True)
rating_metrics_pyspark(test, predictions)
ranking_metrics_pyspark(test, predictions, k = DEFAULT_K)
rating_metrics_python(test, predictions)
ranking_metrics_python(test, predictions, k = DEFAULT_K)



utilmy\recsys\zrecs\recommenders\datasets\amazon_reviews.py
-------------------------functions----------------------
data_preprocessing(reviews_file, meta_file, train_file, valid_file, test_file, user_vocab, item_vocab, cate_vocab, sample_rate = 0.01, valid_num_ngs = 4, test_num_ngs = 9, is_history_expanding = True, )
_create_vocab(train_file, user_vocab, item_vocab, cate_vocab)
_negative_sampling_offline(instance_input_file, valid_file, test_file, valid_neg_nums = 4, test_neg_nums = 49)
_data_generating(input_file, train_file, valid_file, test_file, min_sequence = 1)
_data_generating_no_history_expanding(input_file, train_file, valid_file, test_file, min_sequence = 1)
_create_item2cate(instance_file)
_get_sampled_data(instance_file, sample_rate)
_meta_preprocessing(meta_readfile)
_reviews_preprocessing(reviews_readfile)
_create_instance(reviews_file, meta_file)
_data_processing(input_file)
download_and_extract(name, dest_path)
_download_reviews(name, dest_path)
_extract_reviews(file_path, zip_path)



utilmy\recsys\zrecs\recommenders\datasets\cosmos_cli.py
-------------------------functions----------------------
find_collection(client, dbid, id)
read_collection(client, dbid, id)
read_database(client, id)
find_database(client, id)



utilmy\recsys\zrecs\recommenders\datasets\covid_utils.py
-------------------------functions----------------------
load_pandas_df(azure_storage_account_name = "azureopendatastorage", azure_storage_sas_token = "", container_name = "covid19temp", metadata_filename = "metadata.csv", )
remove_duplicates(df, cols)
remove_nan(df, cols)
clean_dataframe(df)
retrieve_text(entry, container_name, azure_storage_account_name = "azureopendatastorage", azure_storage_sas_token = "", )
get_public_domain_text(df, container_name, azure_storage_account_name = "azureopendatastorage", azure_storage_sas_token = "", )



utilmy\recsys\zrecs\recommenders\datasets\criteo.py
-------------------------functions----------------------
load_pandas_df(size = "sample", local_cache_path = None, header = DEFAULT_HEADER)
load_spark_df(spark, size = "sample", header = DEFAULT_HEADER, local_cache_path = None, dbfs_datapath="dbfs = "dbfs:/FileStore/dac", dbutils = None, )
download_criteo(size = "sample", work_directory = ".")
extract_criteo(size, compressed_file, path = None)
get_spark_schema(header = DEFAULT_HEADER)



utilmy\recsys\zrecs\recommenders\datasets\download_utils.py
-------------------------functions----------------------
maybe_download(url, filename = None, work_directory = ".", expected_bytes = None)
download_path(path = None)
unzip_file(zip_src, dst_dir, clean_zip_file = False)



utilmy\recsys\zrecs\recommenders\datasets\mind.py
-------------------------functions----------------------
download_mind(size = "small", dest_path = None)
extract_mind(train_zip, valid_zip, train_folder = "train", valid_folder = "valid", clean_zip_file = True, )
read_clickhistory(path, filename)
_newsample(nnn, ratio)
get_train_input(session, train_file_path, npratio = 4)
get_valid_input(session, valid_file_path)
get_user_history(train_history, valid_history, user_history_path)
_read_news(filepath, news_words, news_entities, tokenizer)
get_words_and_entities(train_news, valid_news)
download_and_extract_glove(dest_path)
generate_embeddings(data_path, news_words, news_entities, train_entities, valid_entities, max_sentence = 10, word_embedding_dim = 100, )
load_glove_matrix(path_emb, word_dict, word_embedding_dim)
word_tokenize(sent)



utilmy\recsys\zrecs\recommenders\datasets\movielens.py
-------------------------functions----------------------
load_pandas_df(size = "100k", header = None, local_cache_path = None, title_col = None, genres_col = None, year_col = None, )
load_item_df(size = "100k", local_cache_path = None, movie_col = DEFAULT_ITEM_COL, title_col = None, genres_col = None, year_col = None, )
_load_item_df(size, item_datapath, movie_col, title_col, genres_col, year_col)
load_spark_df(spark, size = "100k", header = None, schema = None, local_cache_path = None, dbutils = None, title_col = None, genres_col = None, year_col = None, )
_get_schema(header, schema)
_maybe_download_and_extract(size, dest_path)
download_movielens(size, dest_path)
extract_movielens(size, rating_path, item_path, zip_path)

-------------------------methods----------------------
_DataFormat.__init__(self, sep, path, has_header = False, item_sep = None, item_path = None, item_has_header = False, )
_DataFormat.separator(self)
_DataFormat.path(self)
_DataFormat.has_header(self)
_DataFormat.item_separator(self)
_DataFormat.item_path(self)
_DataFormat.item_has_header(self)


utilmy\recsys\zrecs\recommenders\datasets\pandas_df_utils.py
-------------------------functions----------------------
user_item_pairs(user_df, item_df, user_col = DEFAULT_USER_COL, item_col = DEFAULT_ITEM_COL, user_item_filter_df = None, shuffle = True, seed = None, )
filter_by(df, filter_by_df, filter_by_cols)
negative_feedback_sampler(df, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_label = DEFAULT_LABEL_COL, col_feedback = "feedback", ratio_neg_per_user = 1, pos_value = 1, neg_value = 0, seed = 42, )
has_columns(df, columns)
has_same_base_dtype(df_1, df_2, columns = None)
lru_cache_df(maxsize, typed = False)

-------------------------methods----------------------
LibffmConverter.__init__(self, filepath = None)
LibffmConverter.fit(self, df, col_rating = DEFAULT_RATING_COL)
LibffmConverter.transform(self, df)
LibffmConverter.fit_transform(self, df, col_rating = DEFAULT_RATING_COL)
LibffmConverter.get_params(self)
PandasHash.__init__(self, pandas_object)
PandasHash.__eq__(self, other)
PandasHash.__hash__(self)


utilmy\recsys\zrecs\recommenders\datasets\python_splitters.py
-------------------------functions----------------------
python_random_split(data, ratio = 0.75, seed = 42)
_do_stratification(data, ratio = 0.75, min_rating = 1, filter_by = "user", is_random = True, seed = 42, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_timestamp = DEFAULT_TIMESTAMP_COL, )
python_chrono_split(data, ratio = 0.75, min_rating = 1, filter_by = "user", col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_timestamp = DEFAULT_TIMESTAMP_COL, )
python_stratified_split(data, ratio = 0.75, min_rating = 1, filter_by = "user", col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, seed = 42, )
numpy_stratified_split(X, ratio = 0.75, seed = 42)



utilmy\recsys\zrecs\recommenders\datasets\spark_splitters.py
-------------------------functions----------------------
spark_random_split(data, ratio = 0.75, seed = 42)
_do_stratification_spark(data, ratio = 0.75, min_rating = 1, filter_by = "user", is_partitioned = True, is_random = True, seed = 42, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_timestamp = DEFAULT_TIMESTAMP_COL, )
spark_chrono_split(data, ratio = 0.75, min_rating = 1, filter_by = "user", col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_timestamp = DEFAULT_TIMESTAMP_COL, no_partition = False, )
spark_stratified_split(data, ratio = 0.75, min_rating = 1, filter_by = "user", col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, seed = 42, )
spark_timestamp_split(data, ratio = 0.75, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_timestamp = DEFAULT_TIMESTAMP_COL, )



utilmy\recsys\zrecs\recommenders\datasets\sparse.py
-------------------------methods----------------------
AffinityMatrix.__init__(self, df, items_list = None, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_pred = DEFAULT_PREDICTION_COL, save_path = None, )
AffinityMatrix._gen_index(self)
AffinityMatrix.gen_affinity_matrix(self)
AffinityMatrix.map_back_sparse(self, X, kind)


utilmy\recsys\zrecs\recommenders\datasets\split_utils.py
-------------------------functions----------------------
process_split_ratio(ratio)
min_rating_filter_pandas(data, min_rating = 1, filter_by = "user", col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, )
min_rating_filter_spark(data, min_rating = 1, filter_by = "user", col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, )
_get_column_name(name, col_user, col_item)
split_pandas_data_with_ratios(data, ratios, seed = 42, shuffle = False)



utilmy\recsys\zrecs\recommenders\datasets\wikidata.py
-------------------------functions----------------------
get_session(session = None)
find_wikidata_id(name, limit = 1, session = None)
query_entity_links(entity_id, session = None)
read_linked_entities(data)
query_entity_description(entity_id, session = None)
search_wikidata(names, extras = None, describe = True, verbose = False)



utilmy\recsys\zrecs\recommenders\datasets\__init__.py


utilmy\recsys\zrecs\recommenders\evaluation\python_evaluation.py
-------------------------functions----------------------
_check_column_dtypes(func)
merge_rating_true_pred(rating_true, rating_pred, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_prediction = DEFAULT_PREDICTION_COL, )
rmse(rating_true, rating_pred, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_prediction = DEFAULT_PREDICTION_COL, )
mae(rating_true, rating_pred, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_prediction = DEFAULT_PREDICTION_COL, )
rsquared(rating_true, rating_pred, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_prediction = DEFAULT_PREDICTION_COL, )
exp_var(rating_true, rating_pred, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_prediction = DEFAULT_PREDICTION_COL, )
auc(rating_true, rating_pred, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_prediction = DEFAULT_PREDICTION_COL, )
logloss(rating_true, rating_pred, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_prediction = DEFAULT_PREDICTION_COL, )
merge_ranking_true_pred(rating_true, rating_pred, col_user, col_item, col_rating, col_prediction, relevancy_method, k = DEFAULT_K, threshold = DEFAULT_THRESHOLD, )
precision_at_k(rating_true, rating_pred, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_prediction = DEFAULT_PREDICTION_COL, relevancy_method = "top_k", k = DEFAULT_K, threshold = DEFAULT_THRESHOLD, )
recall_at_k(rating_true, rating_pred, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_prediction = DEFAULT_PREDICTION_COL, relevancy_method = "top_k", k = DEFAULT_K, threshold = DEFAULT_THRESHOLD, )
ndcg_at_k(rating_true, rating_pred, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_prediction = DEFAULT_PREDICTION_COL, relevancy_method = "top_k", k = DEFAULT_K, threshold = DEFAULT_THRESHOLD, )
map_at_k(rating_true, rating_pred, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_prediction = DEFAULT_PREDICTION_COL, relevancy_method = "top_k", k = DEFAULT_K, threshold = DEFAULT_THRESHOLD, )
get_top_k_items(dataframe, col_user = DEFAULT_USER_COL, col_rating = DEFAULT_RATING_COL, k = DEFAULT_K)
_check_column_dtypes_diversity_serendipity(func)
_check_column_dtypes_novelty_coverage(func)
_get_pairwise_items(df, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, )
_get_cosine_similarity(train_df, item_feature_df = None, item_sim_measure = DEFAULT_ITEM_SIM_MEASURE, col_item_features = DEFAULT_ITEM_FEATURES_COL, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_sim = DEFAULT_SIMILARITY_COL, )
_get_cooccurrence_similarity(train_df, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_sim = DEFAULT_SIMILARITY_COL, )
_get_item_feature_similarity(item_feature_df, col_item_features = DEFAULT_ITEM_FEATURES_COL, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_sim = DEFAULT_SIMILARITY_COL, )
_get_intralist_similarity(train_df, reco_df, item_feature_df = None, item_sim_measure = DEFAULT_ITEM_SIM_MEASURE, col_item_features = DEFAULT_ITEM_FEATURES_COL, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_sim = DEFAULT_SIMILARITY_COL, )
user_diversity(train_df, reco_df, item_feature_df = None, item_sim_measure = DEFAULT_ITEM_SIM_MEASURE, col_item_features = DEFAULT_ITEM_FEATURES_COL, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_sim = DEFAULT_SIMILARITY_COL, col_relevance = None, )
diversity(train_df, reco_df, item_feature_df = None, item_sim_measure = DEFAULT_ITEM_SIM_MEASURE, col_item_features = DEFAULT_ITEM_FEATURES_COL, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_sim = DEFAULT_SIMILARITY_COL, col_relevance = None, )
historical_item_novelty(train_df, reco_df, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, )
novelty(train_df, reco_df, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL)
user_item_serendipity(train_df, reco_df, item_feature_df = None, item_sim_measure = DEFAULT_ITEM_SIM_MEASURE, col_item_features = DEFAULT_ITEM_FEATURES_COL, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_sim = DEFAULT_SIMILARITY_COL, col_relevance = None, )
user_serendipity(train_df, reco_df, item_feature_df = None, item_sim_measure = DEFAULT_ITEM_SIM_MEASURE, col_item_features = DEFAULT_ITEM_FEATURES_COL, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_sim = DEFAULT_SIMILARITY_COL, col_relevance = None, )
serendipity(train_df, reco_df, item_feature_df = None, item_sim_measure = DEFAULT_ITEM_SIM_MEASURE, col_item_features = DEFAULT_ITEM_FEATURES_COL, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_sim = DEFAULT_SIMILARITY_COL, col_relevance = None, )
catalog_coverage(train_df, reco_df, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL)
distributional_coverage(train_df, reco_df, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL)



utilmy\recsys\zrecs\recommenders\evaluation\spark_evaluation.py
-------------------------functions----------------------
_get_top_k_items(dataframe, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_prediction = DEFAULT_PREDICTION_COL, k = DEFAULT_K, )
_get_relevant_items_by_threshold(dataframe, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_prediction = DEFAULT_PREDICTION_COL, threshold = DEFAULT_THRESHOLD, )
_get_relevant_items_by_timestamp(dataframe, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_timestamp = DEFAULT_TIMESTAMP_COL, col_prediction = DEFAULT_PREDICTION_COL, k = DEFAULT_K, )

-------------------------methods----------------------
SparkRatingEvaluation.__init__(self, rating_true, rating_pred, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_prediction = DEFAULT_PREDICTION_COL, )
SparkRatingEvaluation.rmse(self)
SparkRatingEvaluation.mae(self)
SparkRatingEvaluation.rsquared(self)
SparkRatingEvaluation.exp_var(self)
SparkRankingEvaluation.__init__(self, rating_true, rating_pred, k = DEFAULT_K, relevancy_method = "top_k", col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_rating = DEFAULT_RATING_COL, col_prediction = DEFAULT_PREDICTION_COL, threshold = DEFAULT_THRESHOLD, )
SparkRankingEvaluation._calculate_metrics(self)
SparkRankingEvaluation.precision_at_k(self)
SparkRankingEvaluation.recall_at_k(self)
SparkRankingEvaluation.ndcg_at_k(self)
SparkRankingEvaluation.map_at_k(self)
SparkDiversityEvaluation.__init__(self, train_df, reco_df, item_feature_df = None, item_sim_measure = DEFAULT_ITEM_SIM_MEASURE, col_user = DEFAULT_USER_COL, col_item = DEFAULT_ITEM_COL, col_relevance = None, )
SparkDiversityEvaluation._get_pairwise_items(self, df)
SparkDiversityEvaluation._get_cosine_similarity(self, n_partitions = 200)
SparkDiversityEvaluation._get_cooccurrence_similarity(self, n_partitions)
SparkDiversityEvaluation.sim_cos(v1, v2)
SparkDiversityEvaluation._get_item_feature_similarity(self, n_partitions)
SparkDiversityEvaluation._get_intralist_similarity(self, df)
SparkDiversityEvaluation.user_diversity(self)
SparkDiversityEvaluation.diversity(self)
SparkDiversityEvaluation.historical_item_novelty(self)
SparkDiversityEvaluation.novelty(self)
SparkDiversityEvaluation.user_item_serendipity(self)
SparkDiversityEvaluation.user_serendipity(self)
SparkDiversityEvaluation.serendipity(self)
SparkDiversityEvaluation.catalog_coverage(self)
SparkDiversityEvaluation.distributional_coverage(self)


utilmy\recsys\zrecs\recommenders\evaluation\__init__.py


utilmy\recsys\zrecs\recommenders\models\__init__.py


utilmy\recsys\zrecs\recommenders\tuning\parameter_sweep.py
-------------------------functions----------------------
generate_param_grid(params)



utilmy\recsys\zrecs\recommenders\tuning\__init__.py


utilmy\recsys\zrecs\recommenders\utils\constants.py


utilmy\recsys\zrecs\recommenders\utils\general_utils.py
-------------------------functions----------------------
invert_dictionary(dictionary)
get_physical_memory()
get_number_processors()



utilmy\recsys\zrecs\recommenders\utils\gpu_utils.py
-------------------------functions----------------------
get_number_gpus()
get_gpu_info()
clear_memory_all_gpus()
get_cuda_version(unix_path = DEFAULT_CUDA_PATH_LINUX)
get_cudnn_version()



utilmy\recsys\zrecs\recommenders\utils\k8s_utils.py
-------------------------functions----------------------
qps_to_replicas(target_qps, processing_time, max_qp_replica = 1, target_utilization = 0.7)
replicas_to_qps(num_replicas, processing_time, max_qp_replica = 1, target_utilization = 0.7)
nodes_to_replicas(n_cores_per_node, n_nodes = 3, cpu_cores_per_replica = 0.1)



utilmy\recsys\zrecs\recommenders\utils\notebook_memory_management.py
-------------------------functions----------------------
start_watching_memory()
stop_watching_memory()
watch_memory()
pre_run_cell()



utilmy\recsys\zrecs\recommenders\utils\notebook_utils.py
-------------------------functions----------------------
is_jupyter()
is_databricks()



utilmy\recsys\zrecs\recommenders\utils\plot.py
-------------------------functions----------------------
line_graph(values, labels, x_guides = None, x_name = None, y_name = None, x_min_max = None, y_min_max = None, legend_loc = None, subplot = None, 5, 5), )



utilmy\recsys\zrecs\recommenders\utils\python_utils.py
-------------------------functions----------------------
exponential_decay(value, max_val, half_life)
jaccard(cooccurrence)
lift(cooccurrence)
get_top_k_scored_items(scores, top_k, sort_top_k = False)
binarize(a, threshold)
rescale(data, new_min = 0, new_max = 1, data_min = None, data_max = None)



utilmy\recsys\zrecs\recommenders\utils\spark_utils.py
-------------------------functions----------------------
start_or_get_spark(app_name = "Sample", url = "local[*]", memory = "10g", config = None, packages = None, jars = None, repository = None, )



utilmy\recsys\zrecs\recommenders\utils\tf_utils.py
-------------------------functions----------------------
pandas_input_fn_for_saved_model(df, feat_name_type)
pandas_input_fn(df, feat_name_type)
_dataset(x, y = None, batch_size = 128, num_epochs = 1, shuffle = False, seed = None)
build_optimizer(name, lr = 0.001, **kwargs)
export_model(model, train_input_fn, eval_input_fn, tf_feat_cols, base_dir)
evaluation_log_hook(estimator, logger, true_df, y_col, eval_df, every_n_iter = 10000, model_dir = None, batch_size = 256, eval_fns = None, **eval_kwargs)

-------------------------methods----------------------
_TrainLogHook.__init__(self, estimator, logger, true_df, y_col, eval_df, every_n_iter = 10000, model_dir = None, batch_size = 256, eval_fns = None, **eval_kwargs)
_TrainLogHook.begin(self)
_TrainLogHook.before_run(self, run_context)
_TrainLogHook.after_run(self, run_context, run_values)
_TrainLogHook.end(self, session)
_TrainLogHook._log(self, tag, value)
MetricsLogger.__init__(self)
MetricsLogger.log(self, metric, value)
MetricsLogger.get_log(self)


utilmy\recsys\zrecs\recommenders\utils\timer.py
-------------------------methods----------------------
Timer.__init__(self)
Timer.__enter__(self)
Timer.__exit__(self, *args)
Timer.__str__(self)
Timer.start(self)
Timer.stop(self)
Timer.interval(self)


utilmy\recsys\zrecs\recommenders\utils\__init__.py


utilmy\recsys\zrecs\tests\ci\run_pytest.py
-------------------------functions----------------------
create_arg_parser()



utilmy\recsys\zrecs\tests\ci\submit_azureml_pytest.py
-------------------------functions----------------------
setup_workspace(workspace_name, subscription_id, resource_group, cli_auth, location)
setup_persistent_compute_target(workspace, cluster_name, vm_size, max_nodes)
create_run_config(cpu_cluster, docker_proc_type, conda_env_file)
create_experiment(workspace, experiment_name)
submit_experiment_to_azureml(test, test_folder, test_markers, junitxml, run_config, experiment)
create_arg_parser()



utilmy\recsys\zrecs\tests\integration\__init__.py


utilmy\recsys\zrecs\tests\smoke\__init__.py


utilmy\recsys\zrecs\tests\unit\__init__.py


utilmy\spark\src\tables\__pycache__\table_predict_session_length.py
-------------------------functions----------------------
run(spark:object, config_name: str = 'config.yaml')
preprocess_dummy(spark:object)
preprocess(spark:object)
holt_winters_time_series_udf(data)
Difference(df, inputCol, outputCol)
Forecast(df, forecast_days, nLags, \timeSeriesColumn, regressor, sparksession)

-------------------------methods----------------------
LagGather.__init__(self)
LagGather.setLagLength(self, nLags)
LagGather.setInputCol(self, colname)
LagGather.transform(self, df)
LagGather.getFeatureNames(self)


utilmy\spark\src\tables\__pycache__\table_predict_url_unique.py
-------------------------functions----------------------
run(spark:object, config_name: str = 'config.yaml')
preprocess_dummy(spark:object)
fun_tseries_predict(groupkeys, df)
preprocess(spark:object)



utilmy\templates\templist\pypi_package\mygenerator\dataset.py
-------------------------functions----------------------
dataset_build_meta_mnist(path: Optional[Union[str, pathlib.Path]]  =  None, get_image_fn = None, meta = None, image_suffix = "*.png", **kwargs, )

-------------------------methods----------------------
NlpDataset.__init__(self, meta: pd.DataFrame)
NlpDataset.__len__(self)
NlpDataset.get_sample(self, idx: int)
NlpDataset.get_text_only(self, idx: int)
PhoneNlpDataset.__init__(self, size: int  =  1)
PhoneNlpDataset.__len__(self)
PhoneNlpDataset.get_phone_number(self, idx, islocal = False)
ImageDataset.__init__(self, path: Optional[Union[str, pathlib.Path]]  =  None, get_image_fn = None, meta = None, image_suffix = "*.png", **kwargs, )
ImageDataset.__len__(self)
ImageDataset.get_image_only(self, idx: int)
ImageDataset.get_sample(self, idx: int)
ImageDataset.get_label_list(self, label: Any)
ImageDataset.read_image(self, filepath_or_buffer: Union[str, io.BytesIO])
ImageDataset.save(self, path: str, prefix: str  =  "img", suffix: str  =  "png", nrows: int  =  -1)


utilmy\templates\templist\pypi_package\mygenerator\pipeline.py
-------------------------functions----------------------
run_generate_numbers_sequence(sequence: str, min_spacing: int  =  1, max_spacing: int  =  10, image_width: int  =  280, ### image_widthoutput_path: str  =  "./", config_file: str  =  "config/config.yaml", )
run_generate_phone_numbers(num_images: int  =  10, min_spacing: int  =  1, max_spacing: int  =  10, image_width: int  =  280, output_path: str  =  "./", config_file: str  =  "config/config.yaml", )



utilmy\templates\templist\pypi_package\mygenerator\transform.py
-------------------------methods----------------------
ImageTransform.__init__(self)
ImageTransform.transform(self, ds: dataset.ImageDataset)
ImageTransform.fit(self, ds: dataset.ImageDataset)
ImageTransform.fit_transform(self, ds: dataset.ImageDataset)
CharToImages.__init__(self, font: dataset.ImageDataset)
CharToImages.transform(self, ds: dataset.NlpDataset)
CharToImages.fit(self, ds: dataset.NlpDataset)
CharToImages.fit_transform(self, ds: dataset.NlpDataset)
RemoveWhitePadding.transform(self, ds: dataset.ImageDataset)
RemoveWhitePadding.transform_sample(self, image: np.ndarray)
CombineImagesHorizontally.__init__(self, padding_range: Tuple[int, int], combined_width: int)
CombineImagesHorizontally.transform(self, ds: dataset.ImageDataset)
CombineImagesHorizontally.transform_sample(self, image_list: List[np.ndarray], 1, 1), combined_width = 10, min_image_width = 2, validate = True, )
ScaleImage.__init__(self, width: Optional[int]  =  None, height: Optional[int]  =  None, inter = cv2.INTER_AREA)
ScaleImage.transform(self, ds: dataset.ImageDataset)
ScaleImage.transform_sample(self, image, width = None, height = None, inter = cv2.INTER_AREA)
TextToImage.__init__(self, font_dir: Union[str, pathlib.Path], spacing_range: Tuple[int, int], image_width: int)
TextToImage.transform(self, ds: dataset.NlpDataset)
TextToImage.fit(self, ds: dataset.NlpDataset)
TextToImage.fit_transform(self, ds: dataset.NlpDataset)


utilmy\templates\templist\pypi_package\mygenerator\utils.py
-------------------------functions----------------------
log(*s)
log2(*s)
logw(*s)
loge(*s)
logger_setup()
config_load(config_path: Optional[Union[str, pathlib.Path]]  =  None)
dataset_donwload(url, path_target)
dataset_get_path(cfg: dict)
os_extract_archive(file_path, path = ".", archive_format = "auto")
to_file(s, filep)



utilmy\templates\templist\pypi_package\mygenerator\util_exceptions.py
-------------------------functions----------------------
log(*s)
log2(*s)
logw(*s)
loge(*s)
logger_setup()
config_load(config_path: Optional[Union[str, pathlib.Path]]  =  None)
dataset_donwload(url, path_target)
dataset_get_path(cfg: dict)
os_extract_archive(file_path, path = ".", archive_format = "auto")
to_file(s, filep)



utilmy\templates\templist\pypi_package\mygenerator\util_image.py
-------------------------functions----------------------
padding_generate(paddings_number: int  =  1, min_padding: int  =  1, max_padding: int  =  1)
image_merge(image_list, n_dim, padding_size, max_height, total_width)
image_remove_extra_padding(img, inverse = False, removedot = True)
image_resize(image, width = None, height = None, inter = cv2.INTER_AREA)
image_read(filepath_or_buffer: Union[str, io.BytesIO])



utilmy\templates\templist\pypi_package\mygenerator\validate.py
-------------------------functions----------------------
image_padding_validate(final_image, min_padding, max_padding)
image_padding_load(img_path, threshold = 15)
image_padding_get(img, threshold = 0, inverse = True)
run_image_padding_validate(min_spacing: int  =  1, max_spacing: int  =  1, image_width: int  =  5, input_path: str  =  "", inverse_image: bool  =  True, config_file: str  =  "default", **kwargs, )



utilmy\templates\templist\pypi_package\mygenerator\__init__.py


utilmy\templates\templist\pypi_package\tests\conftest.py


utilmy\templates\templist\pypi_package\tests\test_common.py


utilmy\templates\templist\pypi_package\tests\test_dataset.py
-------------------------functions----------------------
test_image_dataset_get_label_list()
test_image_dataset_len()
test_image_dataset_get_sampe()
test_image_dataset_get_image_only()
test_nlp_dataset_len()



utilmy\templates\templist\pypi_package\tests\test_import.py
-------------------------functions----------------------
test_import()



utilmy\templates\templist\pypi_package\tests\test_pipeline.py
-------------------------functions----------------------
test_generate_phone_numbers(tmp_path)



utilmy\templates\templist\pypi_package\tests\test_transform.py
-------------------------functions----------------------
test_chars_to_images_transform()
test_combine_images_horizontally_transform()
test_scale_image_transform()
create_font_files(font_dir)
test_text_to_image_transform(tmp_path)



utilmy\templates\templist\pypi_package\tests\test_util_image.py
-------------------------functions----------------------
create_blank_image(width, height, rgb_color = (0, 0, 0)
test_image_merge()
test_image_remove_extra_padding()
test_image_resize()
test_image_read(tmp_path)



utilmy\templates\templist\pypi_package\tests\test_validate.py
-------------------------functions----------------------
test_image_padding_get()



utilmy\templates\templist\pypi_package\tests\__init__.py


utilmy\zml\source\bin\auto_feature_AFEM\AFE.py
-------------------------functions----------------------
timer(func)

-------------------------methods----------------------
BasePath.__init__(self, pathstype, name = None)
BasePath.getpathname(self)
BasePath.getpathentities(self)
BasePath.getpathstype(self)
BasePath.getinversepathstype(self)
BasePath._inversepathstype(self)
BasePath.getlastentityid(self)
Path.__init__(self, pathstype, df, firstindex, start_time_index, lastindex, last_time_index, name = None, start_part_id = None)
Path.getfirstkey(self)
Path.getlastkey(self)
Path.getstarttimeindex(self)
Path.getlasttimeindex(self)
Path.getpathdetail(self)
Path.getstartpartname(self)
EntitySet.__init__(self, name)
EntitySet.draw(self)
EntitySet.entity_from_dataframe(self, entity_id, dataframe, index, time_index = None, variable_types = None)
EntitySet.addrelationship(self, entityA, entityB, keyA, keyB)
EntitySet.search_path(self, targetnode, maxdepth, max_famous_son)
EntitySet._search_path(self, shortpaths, targetnode, maxdepth, max_famous_son)
EntitySet._pathstype(self, paths)
EntitySet.collectiontransform(self, path, target)
EntitySet.getentity(self, entityid)
Entity.__init__(self, entity_id, dataframe, index, time_index = None, variable_types = None)
Entity.getcolumns(self, columns)
Entity.getfeattype(self, featname)
Entity.getfeatname(self)
Entity.merge(self, features, path, how = 'right')
Function.__init__(self, arg)
Generator.__init__(self, es)
Generator.reload_data(self, es)
Generator.layer(self, path, start_part = None, start_part_id = None)
Generator.layers(self, paths, start_part = None, start_part_id = None)
Generator.defaultfunc(self, path)
Generator._layer(self, path, start_part = None, start_part_id = None)
Generator.pathfilter(self, path, function, start_part = None, start_part_id = None)
Generator.aggregate(self, path, function, iftimeuse  =  True, winsize = 'all', lagsize = 'last')
Generator.add_compute_series(self, compute_series, start_part = None)
Generator.pathcompute(self, cs, ngroups = 'auto', njobs = 1)
Generator.collect_agg(self, inputs)
Generator.layer_sequencal_agg(self, path, es, ngroups  =  None, njobs = 1)
Generator.transform(self, path, featurenames, function)
Generator.singlepathcompunation(self, pathstype, targetfeatures, functionset)
Generator.pathcompunation(self, pathsfunc)


utilmy\zml\source\bin\auto_feature_AFEM\__init__.py


utilmy\zml\source\bin\deltapy\extract.py
-------------------------functions----------------------
set_property(key, value)
abs_energy(x)
cid_ce(x, normalize)
mean_abs_change(x)
_roll(a, shift)
mean_second_derivative_central(x)
variance_larger_than_standard_deviation(x)
var_index(time, param = var_index_param)
symmetry_looking(x, param=[{"r" = [{"r": 0.2}])
has_duplicate_max(x)
partial_autocorrelation(x, param=[{"lag" = [{"lag": 1}])
augmented_dickey_fuller(x, param=[{"attr" = [{"attr": "teststat"}])
gskew(x)
stetson_mean(x, param = stestson_param)
length(x)
count_above_mean(x)
get_length_sequences_where(x)
longest_strike_below_mean(x)
wozniak(magnitude, param = woz_param)
last_location_of_maximum(x)
fft_coefficient(x, param = [{"coeff" =  [{"coeff": 10, "attr": "real"}])
ar_coefficient(x, param=[{"coeff" = [{"coeff": 5, "k": 5}])
index_mass_quantile(x, param=[{"q" = [{"q": 0.3}])
number_cwt_peaks(x, param = cwt_param)
spkt_welch_density(x, param=[{"coeff" = [{"coeff": 5}])
linear_trend_timewise(x, param= [{"attr" =  [{"attr": "pvalue"}])
c3(x, lag = 3)
binned_entropy(x, max_bins = 10)
_embed_seq(X, Tau, D)
svd_entropy(epochs, param = svd_param)
_hjorth_mobility(epochs)
hjorth_complexity(epochs)
_estimate_friedrich_coefficients(x, m, r)
max_langevin_fixed_point(x, r = 3, m = 30)
willison_amplitude(X, param = will_param)
percent_amplitude(x, param  = perc_param)
cad_prob(cads, param = cad_param)
zero_crossing_derivative(epochs, param = zero_param)
detrended_fluctuation_analysis(epochs)
_embed_seq(X, Tau, D)
fisher_information(epochs, param = fisher_param)
higuchi_fractal_dimension(epochs, param = hig_param)
petrosian_fractal_dimension(epochs)
hurst_exponent(epochs)
_embed_seq(X, Tau, D)
largest_lyauponov_exponent(epochs, param = lyaup_param)
whelch_method(data, param = whelch_param)
find_freq(serie, param = freq_param)
flux_perc(magnitude)
range_cum_s(magnitude)
structure_func(time, param = struct_param)
kurtosis(x)
stetson_k(x)



utilmy\zml\source\bin\deltapy\interact.py
-------------------------functions----------------------
lowess(df, cols, y, f = 2. / 3., iter = 3)
autoregression(df, drop = None, settings={"autoreg_lag" = {"autoreg_lag":4})
muldiv(df, feature_list)
decision_tree_disc(df, cols, depth = 4)
quantile_normalize(df, drop)
haversine_distance(row, lon = "Open", lat = "Close")
tech(df)
genetic_feat(df, num_gen = 20, num_comp = 10)



utilmy\zml\source\bin\deltapy\mapper.py
-------------------------functions----------------------
pca_feature(df, memory_issues = False, mem_iss_component = False, variance_or_components = 0.80, n_components = 5, drop_cols = None, non_linear = True)
cross_lag(df, drop = None, lags = 1, components = 4)
a_chi(df, drop = None, lags = 1, sample_steps = 2)
encoder_dataset(df, drop = None, dimesions = 20)
lle_feat(df, drop = None, components = 4)
feature_agg(df, drop = None, components = 4)
neigh_feat(df, drop, neighbors = 6)



utilmy\zml\source\bin\deltapy\transform.py
-------------------------functions----------------------
infer_seasonality(train, index = 0)
robust_scaler(df, drop = None, quantile_range = (25, 75)
standard_scaler(df, drop)
fast_fracdiff(x, cols, d)
outlier_detect(data, col, threshold = 1, method = "IQR")
windsorization(data, col, para, strategy = 'both')
operations(df, features)
initial_trend(series, slen)
initial_seasonal_components(series, slen)
triple_exponential_smoothing(df, cols, slen, alpha, beta, gamma, n_preds)
naive_dec(df, columns, freq = 2)
bkb(df, cols)
butter_lowpass(cutoff, fs = 20, order = 5)
butter_lowpass_filter(df, cols, cutoff, fs = 20, order = 5)
instantaneous_phases(df, cols)
kalman_feat(df, cols)
perd_feat(df, cols)
fft_feat(df, cols)
harmonicradar_cw(df, cols, fs, fc)
saw(df, cols)
modify(df, cols)
multiple_rolling(df, windows  =  [1, 2], functions = ["mean", "std"], columns = None)
multiple_lags(df, start = 1, end = 3, columns = None)
prophet_feat(df, cols, date, freq, train_size = 150)



utilmy\zml\source\bin\deltapy\__init__.py


utilmy\zml\source\bin\hunga_bunga\classification.py
-------------------------functions----------------------
run_all_classifiers(x, y, small  =  True, normalize_x  =  True, n_jobs = cpu_count()

-------------------------methods----------------------
HungaBungaClassifier.__init__(self, brain = False, test_size  =  0.2, n_splits  =  5, random_state = None, upsample = True, scoring = None, verbose = False, normalize_x  =  True, n_jobs  = cpu_count()
HungaBungaClassifier.fit(self, x, y)
HungaBungaClassifier.predict(self, x)


utilmy\zml\source\bin\hunga_bunga\core.py
-------------------------functions----------------------
upsample_indices_clf(inds, y)
cv_clf(x, y, test_size  =  0.2, n_splits  =  5, random_state = None, doesUpsample  =  True)
cv_reg(x, test_size  =  0.2, n_splits  =  5, random_state = None)
timeit(klass, params, x, y)
main_loop(models_n_params, x, y, isClassification, test_size  =  0.2, n_splits  =  5, random_state = None, upsample = True, scoring = None, verbose = True, n_jobs  = cpu_count()

-------------------------methods----------------------
GridSearchCVProgressBar._get_param_iterator(self)
RandomizedSearchCVProgressBar._get_param_iterator(self)


utilmy\zml\source\bin\hunga_bunga\params.py


utilmy\zml\source\bin\hunga_bunga\regression.py
-------------------------functions----------------------
gen_reg_data(x_mu = 10., x_sigma = 1., num_samples = 100, num_features = 3, y_formula = sum, y_sigma = 1.)
run_all_regressors(x, y, small  =  True, normalize_x  =  True, n_jobs = cpu_count()

-------------------------methods----------------------
HungaBungaRegressor.__init__(self, brain = False, test_size  =  0.2, n_splits  =  5, random_state = None, upsample = True, scoring = None, verbose = False, normalize_x  =  True, n_jobs  = cpu_count()
HungaBungaRegressor.fit(self, x, y)
HungaBungaRegressor.predict(self, x)


utilmy\zml\source\bin\hunga_bunga\__init__.py
-------------------------methods----------------------
HungaBungaZeroKnowledge.__init__(self, brain = False, test_size  =  0.2, n_splits  =  5, random_state = None, upsample = True, scoring = None, verbose = True, normalize_x  =  True, n_jobs  = cpu_count()
HungaBungaZeroKnowledge.fit(self, X, y)
HungaBungaZeroKnowledge.predict(self, x)


utilmy\zml\source\models\akeras\armdn.py


utilmy\zml\source\models\akeras\Autokeras.py
-------------------------functions----------------------
get_config_file()
get_params(param_pars = None, **kw)
get_dataset_imbd(data_pars)
get_dataset_titanic(data_pars)
get_dataset(data_pars)
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kwargs)
predict(model, session = None, data_pars = None, compute_pars = None, out_pars = None)
evaluate(model, data_pars = None, compute_pars = None, out_pars = None)
save(model, session = None, save_pars = None, config_mode = "test")
load(load_pars, config_mode = "test")
test_single(data_path = "dataset/", pars_choice = "json", config_mode = "test")
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, out_pars = None)


utilmy\zml\source\models\akeras\charcnn.py
-------------------------functions----------------------
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
evaluate(model, session = None, data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(model, session = None, data_pars = None, out_pars = None, compute_pars = None, **kw)
reset_model()
save(model = None, save_pars = None, session = None)
load(load_pars = None)
str_to_indexes(s)
tokenize(data, num_of_classes = 4)
get_dataset(data_pars = None, **kw)
get_params(param_pars = {}, **kw)
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zml\source\models\akeras\charcnn_zhang.py
-------------------------functions----------------------
fit(model, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)
evaluate(model, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)
predict(model, sess = None, data_pars = {}, out_pars = {}, compute_pars = {}, **kw)
reset_model()
save(model = None, session = None, save_pars = {})
load(load_pars = {})
get_dataset(data_pars = None, **kw)
get_params(param_pars = {}, **kw)
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zml\source\models\akeras\deepctr.py
-------------------------functions----------------------
_preprocess_criteo(df, **kw)
_preprocess_movielens(df, **kw)
get_dataset(data_pars = None, **kw)
fit(model, session = None, compute_pars = None, data_pars = None, out_pars = None, **kwargs)
predict(model, session = None, compute_pars = None, data_pars = None, out_pars = None, **kwargs)
metrics(ypred, ytrue = None, session = None, compute_pars = None, data_pars = None, out_pars = None, **kwargs)
reset_model()
path_setup(out_folder = "", sublevel = 0, data_path = "dataset/")
_config_process(config)
config_load(data_path, file_default, config_mode)
get_params(choice = "", data_path = "dataset/", config_mode = "test", **kwargs)
test(data_path = "dataset/", pars_choice = 0, **kwargs)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, **kwargs)


utilmy\zml\source\models\akeras\namentity_crm_bilstm.py
-------------------------functions----------------------
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
evaluate(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(model, sess = None, data_pars = None, out_pars = None, compute_pars = None, **kw)
reset_model()
save(model = None, session = None, save_pars = None)
load(load_pars)
get_dataset(data_pars)
get_params(param_pars = {}, **kw)
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, **kwargs)


utilmy\zml\source\models\akeras\preprocess.py
-------------------------functions----------------------
os_package_root_path(filepath, sublevel = 0, path_add = "")
log(*s, n = 0, m = 1)
_preprocess_criteo(df, **kw)
_preprocess_movielens(df, **kw)
_preprocess_none(df, **kw)
get_dataset(**kw)
test(data_path = "dataset/", pars_choice = 0)



utilmy\zml\source\models\akeras\textcnn.py
-------------------------functions----------------------
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
evaluate(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(model, sess = None, data_pars = None, out_pars = None, compute_pars = None, **kw)
reset_model()
save(model = None, session = None, save_pars = {})
load(load_pars = {})
get_dataset(data_pars = None, **kw)
get_params(param_pars = {}, **kw)
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zml\source\models\akeras\util.py
-------------------------functions----------------------
os_package_root_path(filepath, sublevel = 0, path_add = "")
log(*s, n = 0, m = 1)
_config_process(data_path, config_mode = "test")
get_dataset(**kw)
fit(model, data_pars = None, model_pars = None, compute_pars = None, out_pars = None, session = None, **kwargs)
predict(model, data_pars, compute_pars = None, out_pars = None, **kwargs)
metrics(ypred, data_pars, compute_pars = None, out_pars = None, **kwargs)
save(model, path)
load(path)

-------------------------methods----------------------
Model_empty.__init__(self, model_pars = None, compute_pars = None)


utilmy\zml\source\models\akeras\__init__.py


utilmy\zml\source\models\atorch\matchZoo.py
-------------------------functions----------------------
get_task(model_pars, task)
get_glove_embedding_matrix(term_index, dimension)
get_data_loader(model_name, preprocessor, preprocess_pars, raw_data)
get_config_file()
get_raw_dataset(data_info, **args)
get_dataset(_model, preprocessor, _preprocessor_pars, data_pars)
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kwargs)
predict(model, session = None, data_pars = None, compute_pars = None, out_pars = None)
evaluate(model, data_pars = None, compute_pars = None, out_pars = None)
save(model, session = None, save_pars = None)
load(load_pars)
get_params(param_pars = None, **kw)
test_train(data_path, pars_choice, model_name)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, out_pars = None)


utilmy\zml\source\models\atorch\textcnn.py
-------------------------functions----------------------
_train(m, device, train_itr, optimizer, epoch, max_epoch)
_valid(m, device, test_itr)
_get_device()
get_config_file()
get_data_file()
analyze_datainfo_paths(data_info)
split_train_valid(data_info, **args)
clean_str(string)
create_tabular_dataset(data_info, **args)
create_data_iterator(batch_size, tabular_train, tabular_valid, d)
evaluate(model, session = None, data_pars = None, compute_pars = None, out_pars = None, **kwargs)
fit(model, sess = None, data_pars = None, compute_pars = None, out_pars = None, **kwargs)
get_dataset(data_pars = None, out_pars = None, **kwargs)
predict(model, session = None, data_pars = None, compute_pars = None, out_pars = None, return_ytrue = 1)
save(model, session = None, save_pars = None)
load(load_pars =  None)
get_params(param_pars = None, **kw)
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
TextCNN.__init__(self, model_pars = None, data_pars = None, compute_pars = None, **kwargs)
TextCNN.rebuild_embed(self, vocab_built)
TextCNN.forward(self, x)
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zml\source\models\atorch\torchhub.py
-------------------------functions----------------------
_train(m, device, train_itr, criterion, optimizer, epoch, max_epoch, imax = 1)
_valid(m, device, test_itr, criterion, imax = 1)
_get_device()
get_config_file()
get_params(param_pars = None, **kw)
get_dataset(data_pars = None, **kw)
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kwargs)
predict(model, session = None, data_pars = None, compute_pars = None, out_pars = None, imax  =  1, return_ytrue = 1)
evaluate(model, data_pars = None, compute_pars = None, out_pars = None)
save(model, session = None, save_pars = None)
load(load_pars)
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")
test2(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, out_pars = None)


utilmy\zml\source\models\atorch\torch_ctr.py
-------------------------functions----------------------
log(*s)
init(*kw, **kwargs)
customModel()
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(Xpred = None, data_pars = {}, compute_pars = None, out_pars = {}, **kw)
reset()
save(path = None, info = None)
load_model(path = "")
load_info(path = "")
preprocess(prepro_pars)
get_dataset(data_pars = None, task_type = "train", **kw)
get_params(param_pars = {}, **kw)
test(config = '')
test2(config = '')

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zml\source\models\atorch\transformer_sentence.py
-------------------------functions----------------------
fit(model, data_pars = None, model_pars = None, compute_pars = None, out_pars = None, *args, **kw)
evaluate(model, session = None, data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(model, session = None, data_pars = None, out_pars = None, compute_pars = None, **kw)
reset_model()
save(model, session = None, save_pars = None)
load(load_pars = None)
get_dataset(data_pars = None, **kw)
fit2(model, data_pars = None, model_pars = None, compute_pars = None, out_pars = None, *args, **kw)
predict2(model, session = None, data_pars = None, out_pars = None, compute_pars = None, **kw)
get_dataset2(data_pars = None, model = None, **kw)
get_params(param_pars, **kw)
test(data_path = "dataset/", pars_choice = "test01", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, **kwargs)


utilmy\zml\source\models\atorch\util_data.py


utilmy\zml\source\models\atorch\util_transformer.py
-------------------------functions----------------------
convert_example_to_feature(example_row, pad_token = 0, sequence_a_segment_id = 0, sequence_b_segment_id = 1, cls_token_segment_id = 1, pad_token_segment_id = 0, mask_padding_with_zero = True, sep_token_extra = False)
convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_mode, cls_token_at_end = False, sep_token_extra = False, pad_on_left = False, cls_token = '[CLS]', sep_token = '[SEP]', pad_token = 0, sequence_a_segment_id = 0, sequence_b_segment_id = 1, cls_token_segment_id = 1, pad_token_segment_id = 0, mask_padding_with_zero = True, ) - 2))
_truncate_seq_pair(tokens_a, tokens_b, max_length)

-------------------------methods----------------------
InputExample.__init__(self, guid, text_a, text_b = None, label = None)
InputFeatures.__init__(self, input_ids, input_mask, segment_ids, label_id)
DataProcessor.get_train_examples(self, data_dir)
DataProcessor.get_dev_examples(self, data_dir)
DataProcessor.get_labels(self)
DataProcessor._read_tsv(cls, input_file, quotechar = None)
BinaryProcessor.get_train_examples(self, data_dir)
BinaryProcessor.get_dev_examples(self, data_dir)
BinaryProcessor.get_labels(self)
BinaryProcessor._create_examples(self, lines, set_type)
TransformerDataReader.__init__(self, **args)
TransformerDataReader.compute(self, input_tmp)
TransformerDataReader.get_data(self)


utilmy\zml\source\models\atorch\__init__.py


utilmy\zml\source\models\repo\functions.py
-------------------------functions----------------------
sampling(args)
get_dataset(state_num = 10, time_len = 50000, signal_dimension = 15, CNR = 1, window_len = 11, half_window_len = 5)
get_model(original_dim, class_num = 5, intermediate_dim = 64, intermediate_dim_2 = 16, latent_dim = 3, batch_size = 256, Lambda1 = 1, Lambda2 = 200, Alpha = 0.075)
fit(vae, x_train, epochs = 1, batch_size = 256)
save(model)
load(model, path)
test(self, encoder, x_train, dummy_train, class_num = 5, batch_size = 256)



utilmy\zml\source\models\repo\model_rec.py
-------------------------functions----------------------
log(*s)
log2(*s)
log3(*s)
os_makedirs(dir_or_file)
init(*kw, **kwargs)
reset()
get_dataset(data_pars, task_type = "train")
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(Xpred = None, data_pars = None, compute_pars = {}, out_pars = {}, **kw)
eval(Xpred = None, data_pars: dict = {}, compute_pars: dict = {}, out_pars: dict = {}, **kw)
save(path = None, info = None)
load_info(path = "")
get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
get_dataset2(data_pars = None, task_type = "train", **kw)
train_test_split2(df, coly)
test(n_sample           =  1000)
test_helper(model_pars, data_pars, compute_pars)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, global_pars = None)


utilmy\zml\source\models\repo\model_rec_ease.py
-------------------------functions----------------------
log(*s)
log2(*s)
log3(*s)
os_makedirs(dir_or_file)
init(*kw, **kwargs)
reset()
get_dataset(data_pars, task_type = "train")
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(Xpred = None, data_pars = None, compute_pars = {}, out_pars = {}, **kw)
eval(Xpred = None, data_pars: dict = {}, compute_pars: dict = {}, out_pars: dict = {}, **kw)
save(path = None, info = None)
load_info(path = "")
get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
get_dataset2(data_pars = None, task_type = "train", **kw)
train_test_split2(df, coly)
get_dataset_sampler(data_pars)
init_dataset(data_pars)
test(n_sample           =  1000)
test_helper(model_pars, data_pars, compute_pars)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, global_pars = None)


utilmy\zml\source\models\ztmp2\keras_widedeep_2.py
-------------------------functions----------------------
log(*s)
log2(*s)
init(*kw, **kwargs)
Modelcustom(n_wide_cross, n_wide, n_deep, n_feat = 8, m_EMBEDDING = 10, loss = 'mse', metric  =  'mean_squared_error')
get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
get_dataset(Xtrain, cols_type_received, cols_ref)
ModelCustom2()
input_template_feed_keras(Xtrain, cols_type_received, cols_ref, **kw)
get_dataset_tuple_keras(pattern, batch_size, mode = tf.estimator.ModeKeys.TRAIN, truncate = None)
get_dataset2(data_pars = None, task_type = "train", **kw)
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(Xpred = None, data_pars = None, compute_pars = {}, out_pars = {}, **kw)
reset()
save(path = None, info = None)
load_model(path = "")
load_info(path = "")
test(config = '')
test_helper(model_pars, data_pars, compute_pars)
Modelsparse2()

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zml\source\models\ztmp2\keras_widedeep_old.py
-------------------------functions----------------------
log(*s)
init(*kw, **kwargs)
Modelcustom(n_wide_cross, n_wide, n_feat = 8, m_EMBEDDING = 10, loss = 'mse', metric  =  'mean_squared_error')
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
eval(data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(Xpred = None, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)
reset()
save(path = None)
load_model(path = "")
load_info(path = "")
preprocess(prepro_pars)
get_dataset(data_pars = None, task_type = "train", **kw)
test(config = '')
get_dataset2(data_pars = None, task_type = "train", **kw)
get_params_sklearn(deep = False)
get_params(deep = False)
test_helper(model_pars, data_pars, compute_pars)
test2(config = '')

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zml\source\models\ztmp2\modelsVaem.py
-------------------------functions----------------------
log(*s)
log2(*s)
init(*kw, **kwargs)
load_data(filePath, categories, cat_col, num_cols, discrete_cols, targetCol, nsample, delimiter)
encode2(data_decode, list_discrete, records_d, fast_plot)
decode2(data_decode, scaling_factor, list_discrete, records_d, plot = False)
save_model2(model, output_dir)
p_vae_active_learning(Data_train_compressed, Data_train, mask_train, Data_test, mask_test_compressed, mask_test, cat_dims, dim_flt, dic_var_type, args, list_discrete, records_d, estimation_method = 1)
train_p_vae(stage, x_train, Data_train, mask_train, epochs, latent_dim, cat_dims, dim_flt, batch_size, p, K, iteration, list_discrete, records_d, args)
reset()
save(model, output_dir)
load_model(path = "")
load_info(path = "")

-------------------------methods----------------------
Model.__init__(self)
Model.fit(self,filePath, categories,cat_cols,num_cols,discrete_cols,targetCol,nsample  =  -1,delimiter=',',plot=False)
Model.decode(self)
Model.encode(self)


utilmy\zml\source\models\ztmp2\model_functions_3.py
-------------------------functions----------------------
log(*s)
log2(*s)
init(*kw, **kwargs)
sampling(args)
get_model(model_pars)
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(Xpred = None, data_pars = None, compute_pars = {}, out_pars = {}, **kw)
get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
get_dataset(Xtrain, cols_type_received, cols_ref)
get_label(encoder, x_train, dummy_train, class_num = 5, batch_size = 256)
reset()
save(path = None, info = None)
load_model(path = "")
load_info(path = "")
get_mydata_correl(data_pars)
test()

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zml\source\models\ztmp2\model_vaem.py
-------------------------functions----------------------
log(*s)
log2(*s)
init(*kw, **kwargs)
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(Xpred = None, data_pars = None, compute_pars = {}, out_pars = {}, **kw)
get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
get_dataset(Xtrain, cols_type_received, cols_ref)
reset()
save(path = None, info = None)
load_model(path = "")
load_info(path = "")
test(config = '')
test_helper(model_pars, data_pars, compute_pars)
load_dataset()seed  =  3000"./data/bank/bankmarketing_train.csv")bank_raw.info())label_column="y")matrix1, ["job"])matrix1, ["marital"])matrix1, ["education"])matrix1, ["default"])matrix1, ["housing"])matrix1, ["loan"])matrix1, ["contact"])matrix1, ["month"])matrix1, ["day_of_week"])matrix1, ["poutcome"])matrix1, ["y"])(matrix1.values).astype(float))[0, :]max_Data  =  0.7min_Data = 0.3[0, 1, 2, 3, 4, 5, 6, 7])[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])[8, 9])np.in1d(list_flt, list_discrete).nonzero()[0])list_cat)list_flt)>0 and len(list_cat)>0)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zml\source\models\ztmp2\model_vaem3.py
-------------------------functions----------------------
log(*s)
log2(*s)
init(*kw, **kwargs)
load_data(filePath, categories, cat_col, num_cols, discrete_cols, targetCol, nsample, delimiter)
encode2(data_decode, list_discrete, records_d, fast_plot)
decode2(data_decode, scaling_factor, list_discrete, records_d, plot = False, args = None)
save_model2(model, output_dir)
p_vae_active_learning(Data_train_comp, Data_train, mask_train, Data_test, mask_test_comp, mask_test, cat_dims, dim_flt, dic_var_type, args, list_discrete, records_d, estimation_method = 1)
train_p_vae(stage, x_train, Data_train, mask_train, epochs, latent_dim, cat_dims, dim_flt, batch_size, p, K, iteration, list_discrete, records_d, args)
reset()
save(model, output_dir)
load_model(path = "")
load_info(path = "")

-------------------------methods----------------------
Model.__init__(self)
Model.fit(self, p)
Model.encode(self, plot = False, args = None)
Model.decode(self, plot = False, args = None)


utilmy\zml\source\models\ztmp2\torch_rvae2.py
-------------------------functions----------------------
log(*s)
log2(*s)
init(*kw, **kwargs)
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(Xpred = None, data_pars: dict = {}, compute_pars: dict = {}, out_pars: dict = {}, **kw)
reset()
save(path = None, info = None)
load_model(path = "")
load_info(path = "")
get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
get_dataset(Xtrain, cols_type_received, cols_ref)
test_dataset_1(nrows = 1000)
test(nrows = 1000)
test2(nrow = 10000)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zml\source\models\ztmp2\torch_tabular2.py
-------------------------functions----------------------
log(*s)
log2(*s)
init(*kw, **kwargs)
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(Xpred = None, data_pars: dict = {}, compute_pars: dict = {}, out_pars: dict = {}, **kw)
reset()
save(path = None, info = None)
load_model(path = "")
load_info(path = "")
get_dataset2(data_pars = None, task_type = "train", **kw)
get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
get_dataset(data_pars = None, task_type = "train", **kw)
test_dataset_covtype(nrows = 1000)
test(nrows = 1000)
test3()
test2(nrows = 10000)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zzarchive\storage\aapackage_gen\34\global01.py


utilmy\zzarchive\storage\aapackage_gen\34\util34.py
-------------------------functions----------------------
getmodule_doc(module1, fileout = '')



utilmy\zzarchive\storage\aapackage_gen\34\Working Copy of util34.py
-------------------------functions----------------------
getmodule_doc(module1, fileout = '')



utilmy\zzarchive\storage\aapackage_gen\old\util27.py
-------------------------functions----------------------
getmodule_doc(module1, fileout = '')



utilmy\zzarchive\storage\aapackage_gen\old\util34.py
-------------------------functions----------------------
getmodule_doc(module1, fileout = '')



utilmy\zzarchive\storage\aapackage_gen\old\utils27.py
-------------------------functions----------------------
numexpr_vect_calc(filename, i0 = 0, imax = 1000, expr, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
numexpr_topanda(filename, i0 = 0, imax = 1000, expr, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
convertcsv_topanda(filein1, filename, tablen = 'data')
getpanda_tonumpy(filename, nsize, tablen = 'data')
getrandom_tonumpy(filename, nbdim, nbsample, tablen = 'data')
comoment(xx, yy, nsample, kx, ky)
acf(data)
unique_rows(a)
remove_zeros(vv, axis1 = 1)
sort_array(vv)
save_topanda(vv, filenameh5)
load_frompanda(filenameh5)
plotsave(xx, yy, title1 = "")
plotshow(xx, yy, title1 = "")
parsePDF(url)



utilmy\zzarchive\storage\aapackage_gen\old\utils34.py
-------------------------functions----------------------
numexpr_vect_calc(filename, i0 = 0, imax = 1000, expr, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
numexpr_topanda(filename, i0 = 0, imax = 1000, expr, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
convertcsv_topanda(filein1, filename, tablen = 'data')
getpanda_tonumpy(filename, nsize, tablen = 'data')
getrandom_tonumpy(filename, nbdim, nbsample, tablen = 'data')
comoment(xx, yy, nsample, kx, ky)
acf(data)
unique_rows(a)
remove_zeros(vv, axis1 = 1)
sort_array(vv)
save_topanda(vv, filenameh5)
load_frompanda(filenameh5)
plotsave(xx, yy, title1 = "")
plotshow(xx, yy, title1 = "")
parsePDF(url)



utilmy\zzarchive\storage\aapackage_gen\old\Working Copy of util34.py
-------------------------functions----------------------
getmodule_doc(module1, fileout = '')



utilmy\zzml\install\zconda\zold\distri_model_tch.py
-------------------------functions----------------------
model_create(modelname = "", params = None, modelonly = 1)
model_instance(name = "net", params = {})

-------------------------methods----------------------
Net.__init__(self)
Net.forward(self, x)


utilmy\zzml\mlmodels\example\custom_model\1_lstm.py
-------------------------functions----------------------
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kwarg)
evaluate(model, sess = None, data_pars = None, compute_pars = None, out_pars = None)
metrics(model, sess = None, data_pars = None, compute_pars = None, out_pars = None)
predict(model, sess = None, data_pars = None, compute_pars = None, out_pars = None, get_hidden_state = False, init_value = None)
reset_model()
save(model, session = None, save_pars = None)
load(load_pars = None)
get_dataset(data_pars = None)
get_params(param_pars = {}, **kw)
test(data_path = "dataset/", pars_choice = "test01", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, **kwargs)


utilmy\zzml\mlmodels\model_gluon\raw\gluon_prophet.py
-------------------------functions----------------------
get_params(choice = "", data_path = "dataset/", config_mode = "test", **kw)
test(data_path = "dataset/", choice = "")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zzml\mlmodels\model_keras\old\01_deepctr.py
-------------------------functions----------------------
_preprocess_criteo(df, **kw)
_preprocess_movielens(df, **kw)
get_dataset(data_pars = None, **kw)
fit(model, session = None, compute_pars = None, data_pars = None, out_pars = None, **kwargs)
predict(model, session = None, compute_pars = None, data_pars = None, out_pars = None, **kwargs)
metrics(ypred, ytrue = None, session = None, compute_pars = None, data_pars = None, out_pars = None, **kwargs)
reset_model()
path_setup(out_folder = "", sublevel = 0, data_path = "dataset/")
_config_process(config)
config_load(data_path, file_default, config_mode)
get_params(choice = "", data_path = "dataset/", config_mode = "test", **kwargs)
test(data_path = "dataset/", pars_choice = 0, **kwargs)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, **kwargs)


utilmy\zzml\mlmodels\model_keras\old\02_cnn.py
-------------------------functions----------------------
os_package_root_path(filepath, sublevel = 0, path_add = "")
log(*s, n = 0, m = 1)
get_dataset(data_pars, **kw)
fit(model, data_pars = None, model_pars = None, compute_pars = None, out_pars = None, session = None, **kwargs)
predict(model, session = None, data_pars = None, compute_pars = None, out_pars = None, **kwargs)
metrics(ypred, model, session = None, model_pars = None, data_pars = None, compute_pars = None, out_pars = None, **kwargs)
save(model = None, session = None, save_pars = {})
load(load_pars = {})
get_params(choice = 0, data_path = "dataset/", **kw)
test2(data_path = "dataset/", out_path = "keras/keras.png", reset = True)
test(data_path = "dataset/", out_path = "keras/keras.png", reset = True)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, compute_pars = None, data_pars = None)


utilmy\zzml\mlmodels\model_keras\old\armdn.py


utilmy\zzml\mlmodels\model_keras\old\Autokeras.py
-------------------------functions----------------------
get_config_file()
get_params(param_pars = None, **kw)
get_dataset_imbd(data_pars)
get_dataset_titanic(data_pars)
get_dataset_auto_mpg(data_pars)
get_dataset(data_pars)
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kwargs)
predict(model, session = None, data_pars = None, compute_pars = None, out_pars = None)
evaluate(model, session = None, data_pars = None, compute_pars = None, out_pars = None)
save(model, session = None, save_pars = None)
load(load_pars)
test_single(data_path = "dataset/", pars_choice = "json", config_mode = "test")
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, out_pars = None)
Model_keras_empty.__init__(self, model_pars = None, data_pars = None, compute_pars = None, out_pars = None)


utilmy\zzml\mlmodels\model_keras\old\charcnn.py
-------------------------functions----------------------
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
evaluate(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(model, session = None, data_pars = None, out_pars = None, compute_pars = None, **kw)
reset_model()
save(model = None, save_pars = None, session = None)
load(load_pars = None)
get_dataset(data_pars = None, **kw)
get_params(param_pars = {}, **kw)
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zzml\mlmodels\model_keras\old\charcnn_zhang.py
-------------------------functions----------------------
fit(model, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)
evaluate(model, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)
predict(model, sess = None, data_pars = {}, out_pars = {}, compute_pars = {}, **kw)
reset_model()
save(model = None, session = None, save_pars = {})
load(load_pars = {})
get_dataset(data_pars = None, **kw)
get_params(param_pars = {}, **kw)
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zzml\mlmodels\model_keras\old\namentity_crm_bilstm.py
-------------------------functions----------------------
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
evaluate(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(model, sess = None, data_pars = None, out_pars = None, compute_pars = None, **kw)
reset_model()
save(model = None, session = None, save_pars = None)
load(load_pars)
get_dataset(data_pars, **kw)
_preprocess_test(data_pars, **kw)
get_params(param_pars = {}, **kw)
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, **kwargs)


utilmy\zzml\mlmodels\model_keras\old\nbeats.py
-------------------------functions----------------------
main()
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
evaluate(model, session = None, data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(model, session = None, data_pars = None, out_pars = None, compute_pars = None, **kw)
reset_model()
save(model = None, save_pars = None, session = None)
load(load_pars = None)
get_dataset(data_pars = None, **kw)
get_params(param_pars = {}, **kw)
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zzml\mlmodels\model_keras\old\textcnn.py
-------------------------functions----------------------
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
evaluate(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(model, sess = None, data_pars = None, out_pars = None, compute_pars = None, **kw)
reset_model()
save(model = None, session = None, save_pars = {})
load(load_pars = {})
get_dataset(data_pars = None, **kw)
get_params(param_pars = {}, **kw)
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zzml\mlmodels\model_keras\old\textvae.py
-------------------------functions----------------------
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
evaluate(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(model, sess = None, data_pars = None, compute_pars = None, out_pars = None, **kw)
reset_model()
save(model = None, session = None, save_pars = {})
load(load_pars = {})
get_dataset(data_pars = None, **kw)
get_params(param_pars = {}, **kw)
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zzml\mlmodels\model_keras\raw\no_03_textcnn.py
-------------------------functions----------------------
log(*s, n = 0, m = 1)
os_module_path()
path_setup(out_folder = "", sublevel = 0, data_path = "dataset/")
get_params(choice = "", data_path = "./dataset/", config_mode = "test", **kw)
get_pre_train_word2vec(model, index2word, vocab_size)
fit(model, Xtrain, ytrain, compute_pars = None, **kw)
metrics(ytrue, ypred, data_pars = None, out_pars = None, **kw)
predict(model, Xtest, ytest, data_pars = None, out_pars = None, compute_pars = None, **kw)
reset_model()
save(model, path)
load(path)
test(data_path = "dataset/", pars_choice = "json", reset = True)
test2(data_path = "dataset/", pars_choice = "json", reset = True)

-------------------------methods----------------------
data_loader.__init__(self, data_pars = None)
data_loader.clean_str(self, string)
data_loader.load_data_and_labels(self)
data_loader.as_matrix(self, sequences, max_len, index2word)
data_loader.Generate_data(self, data_pars = None)
data_provider.__init__(self, data_loader, data_pars = None)
data_provider.get_dataset(self, **kw)
Model.__init__(self, embedding_matrix = None, vocab_size = None, model_pars = None)
Model.model(self)


utilmy\zzml\mlmodels\model_keras\todo\02_cnn.py
-------------------------functions----------------------
os_package_root_path(filepath, sublevel = 0, path_add = "")
log(*s, n = 0, m = 1)
get_dataset(data_pars, **kw)
fit(model, data_pars = None, model_pars = None, compute_pars = None, out_pars = None, session = None, **kwargs)
predict(model, session = None, data_pars = None, compute_pars = None, out_pars = None, **kwargs)
metrics(ypred, model, session = None, model_pars = None, data_pars = None, compute_pars = None, out_pars = None, **kwargs)
save(model = None, session = None, save_pars = {})
load(load_pars = {})
get_params(choice = 0, data_path = "dataset/", **kw)
test2(data_path = "dataset/", out_path = "keras/keras.png", reset = True)
test(data_path = "dataset/", out_path = "keras/keras.png", reset = True)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, compute_pars = None, data_pars = None)


utilmy\zzml\mlmodels\model_keras\todo\Autokeras.py
-------------------------functions----------------------
get_config_file()
get_params(param_pars = None, **kw)
get_dataset_imbd(data_pars)
get_dataset_titanic(data_pars)
get_dataset_auto_mpg(data_pars)
get_dataset(data_pars)
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kwargs)
predict(model, session = None, data_pars = None, compute_pars = None, out_pars = None)
evaluate(model, session = None, data_pars = None, compute_pars = None, out_pars = None)
save(model, session = None, save_pars = None)
load(load_pars)
test_single(data_path = "dataset/", pars_choice = "json", config_mode = "test")
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, out_pars = None)
Model_keras_empty.__init__(self, model_pars = None, data_pars = None, compute_pars = None, out_pars = None)


utilmy\zzml\mlmodels\model_keras\todo\charcnn.py
-------------------------functions----------------------
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
evaluate(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(model, session = None, data_pars = None, out_pars = None, compute_pars = None, **kw)
reset_model()
save(model = None, save_pars = None, session = None)
load(load_pars = None)
get_dataset(data_pars = None, **kw)
get_params(param_pars = {}, **kw)
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zzml\mlmodels\model_keras\todo\charcnn_zhang.py
-------------------------functions----------------------
fit(model, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)
evaluate(model, data_pars = {}, compute_pars = {}, out_pars = {}, **kw)
predict(model, sess = None, data_pars = {}, out_pars = {}, compute_pars = {}, **kw)
reset_model()
save(model = None, session = None, save_pars = {})
load(load_pars = {})
get_dataset(data_pars = None, **kw)
get_params(param_pars = {}, **kw)
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zzml\mlmodels\model_keras\todo\namentity_crm_bilstm.py
-------------------------functions----------------------
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
evaluate(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(model, sess = None, data_pars = None, out_pars = None, compute_pars = None, **kw)
reset_model()
save(model = None, session = None, save_pars = None)
load(load_pars)
get_dataset(data_pars, **kw)
_preprocess_test(data_pars, **kw)
get_params(param_pars = {}, **kw)
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, **kwargs)


utilmy\zzml\mlmodels\model_keras\todo\nbeats.py
-------------------------functions----------------------
main()
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
evaluate(model, session = None, data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(model, session = None, data_pars = None, out_pars = None, compute_pars = None, **kw)
reset_model()
save(model = None, save_pars = None, session = None)
load(load_pars = None)
get_dataset(data_pars = None, **kw)
get_params(param_pars = {}, **kw)
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zzml\mlmodels\model_keras\todo\textcnn.py
-------------------------functions----------------------
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
evaluate(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(model, sess = None, data_pars = None, out_pars = None, compute_pars = None, **kw)
reset_model()
save(model = None, session = None, save_pars = {})
load(load_pars = {})
get_dataset(data_pars = None, **kw)
get_params(param_pars = {}, **kw)
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zzml\mlmodels\model_keras\todo\textvae.py
-------------------------functions----------------------
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
evaluate(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(model, sess = None, data_pars = None, compute_pars = None, out_pars = None, **kw)
reset_model()
save(model = None, session = None, save_pars = {})
load(load_pars = {})
get_dataset(data_pars = None, **kw)
get_params(param_pars = {}, **kw)
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zzml\mlmodels\model_rank\dev\LambdaRank.py
-------------------------functions----------------------
train(start_epoch = 0, additional_epoch = 100, lr = 0.0001, optim = "adam", leaky_relu = False, ndcg_gain_in_train = "exp2", sigma = 1.0, double_precision = False, standardize = False, small_dataset = False, debug = False, )

-------------------------methods----------------------
LambdaRank.__init__(self, net_structures, leaky_relu = False, sigma = 1.0, double_precision = False)
LambdaRank.forward(self, input1)
LambdaRank.dump_param(self)


utilmy\zzml\mlmodels\model_rank\dev\load_mslr.py
-------------------------functions----------------------
get_time()

-------------------------methods----------------------
DataLoader.__init__(self, path)
DataLoader.get_num_pairs(self)
DataLoader.get_num_sessions(self)
DataLoader._load_mslr(self)
DataLoader._parse_feature_and_label(self, df)
DataLoader.generate_query_pairs(self, df, qid)
DataLoader.generate_query_pair_batch(self, df = None, batchsize = 2000)
DataLoader.generate_query_batch(self, df, batchsize = 100000)
DataLoader.generate_batch_per_query(self, df = None)
DataLoader.load(self)
DataLoader.train_scaler_and_transform(self)
DataLoader.apply_scaler(self, scaler)


utilmy\zzml\mlmodels\model_rank\dev\metrics.py
-------------------------methods----------------------
DCG.__init__(self, k = 10, gain_type = 'exp2')
DCG.evaluate(self, targets)
DCG._get_gain(self, targets)
DCG._get_discount(self, k)
DCG._make_discount(n)
NDCG.__init__(self, k = 10, gain_type = 'exp2')
NDCG.evaluate(self, targets)
NDCG.maxDCG(self, targets)


utilmy\zzml\mlmodels\model_rank\dev\RankNet.py
-------------------------functions----------------------
train_rank_net(start_epoch = 0, additional_epoch = 100, lr = 0.0001, optim = "adam", train_algo = SUM_SESSION, double_precision = False, standardize = False, small_dataset = False, debug = False)
get_train_inference_net(train_algo, num_features, start_epoch, double_precision)
baseline_pairwise_training_loop(epoch, net, loss_func, optimizer, train_loader, batch_size = 100000, precision = torch.float32, device = "cpu", debug = False)
factorized_training_loop(epoch, net, loss_func, optimizer, train_loader, batch_size = 200, sigma = 1.0, training_algo = SUM_SESSION, precision = torch.float32, device = "cpu", debug = False)
eval_model(inference_model, device, df_valid, valid_loader)
load_from_ckpt(ckpt_file, epoch, model)

-------------------------methods----------------------
RankNet.__init__(self, net_structures, double_precision = False)
RankNet.forward(self, input1)
RankNet.dump_param(self)
RankNetPairs.__init__(self, net_structures, double_precision = False)
RankNetPairs.forward(self, input1, input2)


utilmy\zzml\mlmodels\model_rank\dev\utils.py
-------------------------functions----------------------
get_device()
get_ckptdir(net_name, net_structure, sigma = None)
save_to_ckpt(ckpt_file, epoch, model, optimizer, lr_scheduler)
load_train_vali_data(data_fold, small_dataset = False)
init_weights(m)
eval_cross_entropy_loss(model, device, loader, phase = "Eval", sigma = 1.0)
eval_ndcg_at_k(inference_model, device, df_valid, valid_loader, batch_size, k_list, phase = "Eval")
str2bool(v)
get_args_parser()



utilmy\zzml\mlmodels\model_sklearn\model_lightgbm\model.py
-------------------------functions----------------------
init(*kw, **kwargs)
fit(data_pars = None, compute_pars = None, out_pars = None, **kw)
evaluate(data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(data_pars = None, compute_pars = None, out_pars = None, **kw)
reset()
save(path = None, info = {})
load(path = "")
load_info(path = "")
get_dataset(data_pars = None, **kw)
get_dataset2(data_pars = None, **kw)
get_params(param_pars = {}, **kw)
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zzml\mlmodels\model_sklearn\model_sklearn\model.py
-------------------------functions----------------------
json_parse(js)
init(*kw, **kwargs)
fit(data_pars = None, compute_pars = None, out_pars = None, verbose = False, **kw)
evaluate(data_pars = None, compute_pars = None, out_pars = None, verbose = False, **kw)
predict(data_pars = None, compute_pars = None, out_pars = None, verbose = False, **kw)
reset()
save(path = None, info = {})
load(path = "")
load_info(path = "")
get_dataset(data_pars = None, **kw)
get_params(param_pars = {}, **kw)
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")
preprocessor_ram(data_pars = None, task_type = "train", **kw)
get_dataset2(data_pars = None, **kw)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zzml\mlmodels\model_sklearn\model_sklearn\myprocessor.py
-------------------------functions----------------------
json_parse(js)
process()
get_dataset(data_pars = None, **kw)
get_params(param_pars = {}, **kw)
test()
get_dataset2(data_pars = None, **kw)



utilmy\zzml\mlmodels\model_tch\old\03_nbeats_dataloader.py
-------------------------functions----------------------
Model(model_pars, data_pars, compute_pars)
get_dataset(data_pars)
data_generator(x_full, y_full, bs)
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
fit_simple(net, optimiser, data_generator, on_save_callback, device, data_pars, out_pars, max_grad_steps = 500, )
predict(model, sess, data_pars = None, compute_pars = None, out_pars = None, **kw)
evaluate(model, data_pars, compute_pars, out_pars)
plot(net, x, target, backcast_length, forecast_length, grad_step, out_path = "./")
plot_model(net, x, target, grad_step, data_pars, disable_plot = False)
plot_predict(x_test, y_test, p, data_pars, compute_pars, out_pars)
save_checkpoint(model, optimiser, grad_step, CHECKPOINT_NAME = "mycheckpoint")
load_checkpoint(model, optimiser, CHECKPOINT_NAME = "nbeats-fiting-checkpoint.th")
save(model, optimiser, grad_step, CHECKPOINT_NAME = "mycheckpoint")
load(model, optimiser, CHECKPOINT_NAME = "nbeats-fiting-checkpoint.th")
get_params(param_pars, **kw)
test(data_path = "dataset/milk.csv")



utilmy\zzml\mlmodels\model_tch\old\matchzoo_models.py
-------------------------functions----------------------
get_task(model_pars)
get_glove_embedding_matrix(term_index, dimension)
get_data_loader(model_name, preprocessor, preprocess_pars, raw_data)
get_config_file()
get_raw_dataset(data_pars, task)
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kwargs)
predict(model, session = None, data_pars = None, compute_pars = None, out_pars = None)
evaluate(model, data_pars = None, compute_pars = None, out_pars = None)
save(model, session = None, save_pars = None)
load(load_pars)
get_params(param_pars = None, **kw)
test_train(data_path, pars_choice, model_name)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, out_pars = None)


utilmy\zzml\mlmodels\model_tch\old\mlp.py
-------------------------methods----------------------
Model.__init__(self)
Model.forward(self, x)


utilmy\zzml\mlmodels\model_tch\old\nbeats.py
-------------------------functions----------------------
get_data(data_pars)
get_dataset(**kw)
data_generator(x_full, y_full, bs)
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
fit_simple(net, optimiser, data_generator, on_save_callback, device, data_pars, out_pars, max_grad_steps = 500)
predict(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
plot(net, x, target, backcast_length, forecast_length, grad_step, out_path = "./")
plot_model(net, x, target, grad_step, data_pars, disable_plot = False)
plot_predict(x_test, y_test, p, data_pars, compute_pars, out_pars)
save_checkpoint(model, optimiser, grad_step, CHECKPOINT_NAME = "mycheckpoint")
load_checkpoint(model, optimiser, CHECKPOINT_NAME = 'nbeats-fiting-checkpoint.th')
save(model, optimiser, grad_step, CHECKPOINT_NAME = "mycheckpoint")
load(model, optimiser, CHECKPOINT_NAME = 'nbeats-fiting-checkpoint.th')
get_params(param_pars, **kw)
test(choice = "json", data_path = "nbeats.json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zzml\mlmodels\model_tch\old\pplm.py
-------------------------functions----------------------
path_setup(out_folder = "", sublevel = 0, data_path = "dataset/")
generate(cond_text, bag_of_words, discrim = None, class_label = -1)
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(model, sess = None, data_pars = None, compute_pars = None, out_pars = None, **kw)
get_dataset(data_pars = None, **kw)
get_params(param_pars = None, **kw)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None)


utilmy\zzml\mlmodels\model_tch\old\pytorch_vae.py
-------------------------functions----------------------
get_params(param_pars = None, **kw)
get_dataset(data_pars = None, **kw)
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kwargs)
predict(model, session = None, data_pars = None, compute_pars = None, out_pars = None, imax  =  1, return_ytrue = 1)
evaluate(model, data_pars = None, compute_pars = None, out_pars = None)
save(model, session = None, save_pars = None)
load(load_pars)
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, out_pars = None)


utilmy\zzml\mlmodels\model_tch\old\textcnn.py
-------------------------functions----------------------
_train(m, device, train_itr, optimizer, epoch, max_epoch)
_valid(m, device, test_itr)
_get_device()
get_config_file()
get_data_file()
split_train_valid(path_data, path_train, path_valid, frac = 0.7)
clean_str(string)
create_tabular_dataset(path_train, path_valid, lang = 'en', pretrained_emb = 'glove.6B.300d')
create_data_iterator(tr_batch_size, val_batch_size, tabular_train, tabular_valid, d)
evaluate(model, session = None, data_pars = None, compute_pars = None, out_pars = None, **kwargs)
fit(model, sess = None, data_pars = None, compute_pars = None, out_pars = None, **kwargs)
get_dataset(data_pars = None, out_pars = None, **kwargs)
predict(model, session = None, data_pars = None, compute_pars = None, out_pars = None, return_ytrue = 1)
save(model, session = None, save_pars = None)
load(load_pars =  None)
get_params(param_pars = None, **kw)
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")

-------------------------methods----------------------
TextCNN.__init__(self, model_pars = None, data_pars = None, compute_pars = None, **kwargs)
TextCNN.rebuild_embed(self, vocab_built)
TextCNN.forward(self, x)
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zzml\mlmodels\model_tch\old\torchhub.py
-------------------------functions----------------------
_train(m, device, train_itr, criterion, optimizer, epoch, max_epoch, imax = 1)
_valid(m, device, test_itr, criterion, imax = 1)
_get_device()
get_config_file()
get_params(param_pars = None, **kw)
get_dataset(data_pars = None, **kw)
fit(model, data_pars = None, compute_pars = None, out_pars = None, **kwargs)
predict(model, session = None, data_pars = None, compute_pars = None, out_pars = None, imax  =  1, return_ytrue = 1)
evaluate(model, data_pars = None, compute_pars = None, out_pars = None)
save(model, session = None, save_pars = None)
load(load_pars)
test(data_path = "dataset/", pars_choice = "json", config_mode = "test")
test2(data_path = "dataset/", pars_choice = "json", config_mode = "test")
get_dataset_mnist_torch(data_pars)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, out_pars = None)


utilmy\zzml\mlmodels\model_tch\old\transformer_classifier.py
-------------------------functions----------------------
_preprocess_XXXX(df, **kw)
load_and_cache_examples(task, tokenizer, evaluate = False)
get_dataset(task, tokenizer, evaluate = False)
fit(train_dataset, model, tokenizer)
get_mismatched(labels, preds)
get_eval_report(labels, preds)
metrics(task_name, preds, labels)
evaluate(model, tokenizer, model_pars, data_pars, out_pars, compute_pars, prefix = "")
reset_model()
save(model = None, session = None, save_pars = {})
load(task, tokenizer, evaluate = False)
get_params(param_pars = {}, **kw)
test(data_path, model_pars, data_pars, compute_pars, out_pars, pars_choice = 0)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None)


utilmy\zzml\mlmodels\model_tch\old\transformer_sentence.py
-------------------------functions----------------------
fit(model, data_pars = None, model_pars = None, compute_pars = None, out_pars = None, *args, **kw)
evaluate(model, session = None, data_pars = None, compute_pars = None, out_pars = None, **kw)
predict(model, session = None, data_pars = None, out_pars = None, compute_pars = None, **kw)
reset_model()
save(model, session = None, save_pars = None)
load(load_pars = None)
get_dataset(data_pars = None, **kw)
fit2(model, data_pars = None, model_pars = None, compute_pars = None, out_pars = None, *args, **kw)
predict2(model, session = None, data_pars = None, out_pars = None, compute_pars = None, **kw)
get_dataset2(data_pars = None, model = None, **kw)
get_params(param_pars, **kw)
test(data_path = "dataset/", pars_choice = "test01", config_mode = "test")

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None, compute_pars = None, **kwargs)


utilmy\zzml\mlmodels\model_tch\zdocs\transformer_classifier2.py
-------------------------functions----------------------
_preprocess_XXXX(df, **kw)
get_dataset(data_pars = None, **kw)
fit(model, data_pars = None, model_pars = {}, compute_pars = None, out_pars = None, *args, **kw)
predict(model, sess = None, data_pars = None, out_pars = None, compute_pars = None, **kw)
get_mismatched(labels, preds)
get_eval_report(labels, preds)
metrics(task_name, preds, labels)
evaluate(model, tokenizer, prefix = "")
reset_model()
save(model, out_pars)
load(out_pars = None)
path_setup(out_folder = "", sublevel = 0, data_path = "dataset/")
get_params(choice = 0, data_path = "dataset/", **kw)
metrics_evaluate()
test(data_path = "dataset/", pars_choice = 0)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, data_pars = None)
Model_empty.__init__(self, model_pars = None, compute_pars = None)


utilmy\zzml\mlmodels\model_tf\raw\10_encoder_vanilla.py
-------------------------functions----------------------
reducedimension(input_, dimension = 2, learning_rate = 0.01, hidden_layer = 256, epoch = 20)
fit(model, data_frame)
predict(model, sess, data_frame)
test(filename = "dataset/GOOG-year.csv")

-------------------------methods----------------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size, forget_bias = 0.1, epoch = 500, timestep = 5, )
Model.build_model(self)


utilmy\zzml\mlmodels\model_tf\raw\11_bidirectional_vanilla.py
-------------------------functions----------------------
fit(model, data_frame)
predict(model, sess, data_frame, get_hidden_state = False, init_value_forward = None, init_value_backward = None, )
test(filename = "dataset/GOOG-year.csv")

-------------------------methods----------------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size, forget_bias = 0.1, epoch = 500, timestep = 5, )


utilmy\zzml\mlmodels\model_tf\raw\12_vanilla_2path.py
-------------------------functions----------------------
fit(model, data_frame)
predict(model, sess, data_frame, get_hidden_state = False, init_value_forward = None, init_value_backward = None, )
test(filename = "dataset/GOOG-year.csv")

-------------------------methods----------------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size, forget_bias = 0.1, epoch = 500, timestep = 5, )


utilmy\zzml\mlmodels\model_tf\raw\13_lstm_seq2seq.py
-------------------------functions----------------------
fit(model, data_frame)
predict(model, sess, data_frame, get_hidden_state = False, init_value = None)
test(filename = "dataset/GOOG-year.csv")

-------------------------methods----------------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size, forget_bias = 0.1, epoch = 500, timestep = 5, )


utilmy\zzml\mlmodels\model_tf\raw\14_lstm_attention.py
-------------------------functions----------------------
fit(model, data_frame)
predict(model, sess, data_frame, get_hidden_state = False, init_value = None)
test(filename = "dataset/GOOG-year.csv")

-------------------------methods----------------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size, forget_bias = 0.1, attention_size = 10, epoch = 500, timestep = 5, )


utilmy\zzml\mlmodels\model_tf\raw\15_lstm_seq2seq_attention.py
-------------------------functions----------------------
fit(model, data_frame)
predict(model, sess, data_frame, get_hidden_state = False, init_value = None)
test(filename = "dataset/GOOG-year.csv")

-------------------------methods----------------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size, forget_bias = 0.1, attention_size = 10, epoch = 500, timestep = 5, )


utilmy\zzml\mlmodels\model_tf\raw\16_lstm_seq2seq_bidirectional.py
-------------------------functions----------------------
fit(model, data_frame)
predict(model, sess, data_frame, get_hidden_state = False, init_value_forward = None, init_value_backward = None, )
test(filename = "dataset/GOOG-year.csv")

-------------------------methods----------------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size, forget_bias = 0.1, epoch = 500, timestep = 5, )


utilmy\zzml\mlmodels\model_tf\raw\17_lstm_seq2seq_bidirectional_attention.py
-------------------------functions----------------------
fit(model, data_frame)
predict(model, sess, data_frame, get_hidden_state = False, init_value_forward = None, init_value_backward = None, )
test(filename = "dataset/GOOG-year.csv")

-------------------------methods----------------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size, forget_bias = 0.1, attention_size = 10, epoch = 500, timestep = 5, )


utilmy\zzml\mlmodels\model_tf\raw\18_lstm_attention_scaleddot.py
-------------------------functions----------------------
fit(model, data_frame)
predict(model, sess, data_frame, get_hidden_state = False, init_value = None)
test(filename = "dataset/GOOG-year.csv")

-------------------------methods----------------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size, seq_len, forget_bias = 0.1, epoch = 500, )


utilmy\zzml\mlmodels\model_tf\raw\19_lstm_dilated.py
-------------------------functions----------------------
contruct_cells(hidden_structs)
rnn_reformat(x, input_dims, n_steps)
dilated_rnn(cell, inputs, rate, states, scope = "default")
multi_dilated_rnn(cells, inputs, dilations, states)
fit(model, data_frame)
predict(model, sess, data_frame, get_hidden_state = False, init_value = None)
test(filename = "dataset/GOOG-year.csv")

-------------------------methods----------------------
Model.__init__(self, steps, dimension_input, dimension_output, learning_rate = 0.001, hidden_structs = [20], dilations = [1, 1, 1, 1], epoch = 500, )


utilmy\zzml\mlmodels\model_tf\raw\20_only_attention.py
-------------------------functions----------------------
sinusoidal_positional_encoding(inputs, num_units, zero_pad = False, scale = False)
fit(model, data_frame)
predict(model, sess, data_frame)
test(filename = "dataset/GOOG-year.csv")

-------------------------methods----------------------
Model.__init__(self, seq_len, learning_rate, dimension_input, dimension_output, epoch = 100)


utilmy\zzml\mlmodels\model_tf\raw\21_multihead_attention.py
-------------------------functions----------------------
embed_seq(inputs, vocab_size = None, embed_dim = None, zero_pad = False, scale = False)
learned_positional_encoding(inputs, embed_dim, zero_pad = False, scale = False)
layer_norm(inputs, epsilon = 1e-8)
pointwise_feedforward(inputs, num_units = [None, None], activation = None)
fit(model, data_frame)
predict(model, sess, data_frame)
test(filename = "dataset/GOOG-year.csv")

-------------------------methods----------------------
Model.__init__(self, dimension_input, dimension_output, seq_len, learning_rate, num_heads = 8, 1, 6), epoch = 1, )
Model.multihead_attn(self, inputs, masks)
Model.window_mask(self, h_w)


utilmy\zzml\mlmodels\model_tf\raw\22_lstm_bahdanau.py
-------------------------functions----------------------
fit(model, data_frame)
predict(model, sess, data_frame, get_hidden_state = False, init_value = None)
test(filename = "dataset/GOOG-year.csv")

-------------------------methods----------------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size, forget_bias = 0.1, attention_size = 10, epoch = 100, timestep = 5, )


utilmy\zzml\mlmodels\model_tf\raw\23_lstm_luong.py
-------------------------functions----------------------
fit(model, data_frame)
predict(model, sess, data_frame, get_hidden_state = False, init_value = None)
test(filename = "dataset/GOOG-year.csv")

-------------------------methods----------------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size, forget_bias = 0.1, attention_size = 10, epoch = 100, timestep = 5, )


utilmy\zzml\mlmodels\model_tf\raw\24_lstm_luong_bahdanau.py
-------------------------functions----------------------
fit(model, data_frame)
predict(model, sess, data_frame, get_hidden_state = False, init_value = None)
test(filename = "dataset/GOOG-year.csv")

-------------------------methods----------------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size, forget_bias = 0.1, attention_size = 10, epoch = 1, timestep = 5, )


utilmy\zzml\mlmodels\model_tf\raw\25_dnc.py
-------------------------functions----------------------
fit(model, data_frame)
predict(model, sess, data_frame, get_hidden_state = False, init_value = None)
test(filename = "dataset/GOOG-year.csv")

-------------------------methods----------------------
Model.__init__(self, learning_rate, size, size_layer, output_size, epoch, timestep, access_config, controller_config, clip_value, )


utilmy\zzml\mlmodels\model_tf\raw\26_lstm_residual.py
-------------------------functions----------------------
fit(model, data_frame)
predict(model, sess, data_frame, get_hidden_state = False, init_value = None)
test(filename = "dataset/GOOG-year.csv")

-------------------------methods----------------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size, epoch = 1, timestep = 5)


utilmy\zzml\mlmodels\model_tf\raw\27_byte_net.py
-------------------------functions----------------------
layer_normalization(x, epsilon = 1e-8)
conv1d(input_, output_channels, dilation = 1, filter_width = 1, causal = False)
bytenet_residual_block(input_, dilation, layer_no, residual_channels, filter_width, causal = True)
fit(model, data_frame)
predict(model, sess, data_frame)
test(filename = "dataset/GOOG-year.csv")

-------------------------methods----------------------
Model.__init__(self, size, output_size, channels, encoder_dilations, encoder_filter_width, learning_rate = 0.001, beta1 = 0.5, epoch = 1, timestep = 5, )


utilmy\zzml\mlmodels\model_tf\raw\28_attention_is_all_you_need.py
-------------------------functions----------------------
layer_norm(inputs, epsilon = 1e-8)
multihead_attn(queries, keys, q_masks, k_masks, future_binding, num_units, num_heads)
pointwise_feedforward(inputs, hidden_units, activation = None)
learned_position_encoding(inputs, mask, embed_dim)
sinusoidal_position_encoding(inputs, mask, repr_dim)
label_smoothing(inputs, epsilon = 0.1)
fit(model, data_frame)
predict(model, sess, data_frame)
test(filename = "dataset/GOOG-year.csv")

-------------------------methods----------------------
Model.__init__(self, size_layer, embedded_size, learning_rate, size, output_size, num_blocks = 2, num_heads = 8, min_freq = 50, epoch = 1, timestep = 5, )


utilmy\zzml\mlmodels\model_tf\raw\29_fairseq.py
-------------------------functions----------------------
encoder_block(inp, n_hidden, filter_size)
decoder_block(inp, n_hidden, filter_size)
glu(x)
layer(inp, conv_block, kernel_width, n_hidden, residual = None)
fit(model, data_frame)
predict(model, sess, data_frame)
test(filename = "dataset/GOOG-year.csv")

-------------------------methods----------------------
Model.__init__(self, n_layers, size, output_size, emb_size, n_hidden, n_attn_heads, learning_rate, epoch = 1, timestep = 5, )


utilmy\zzml\mlmodels\model_tf\raw\2_encoder_lstm.py
-------------------------functions----------------------
reducedimension(input_, dimension = 2, learning_rate = 0.01, hidden_layer = 256, epoch = 20, sess = None)
fit(model, data_frame)
predict(model, sess, data_frame)
test(filename = "dataset/GOOG-year.csv")

-------------------------methods----------------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size, forget_bias = 0.1, epoch = 5, timestep = 5, )
Model.build_model(self)


utilmy\zzml\mlmodels\model_tf\raw\3_bidirectional_lstm.py
-------------------------functions----------------------
fit(model, data_frame)
predict(model, sess, data_frame, get_hidden_state = False, init_value_forward = None, init_value_backward = None, )
test(filename = "dataset/GOOG-year.csv")

-------------------------methods----------------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size, forget_bias = 0.1, epoch = 500, timestep = 5, )


utilmy\zzml\mlmodels\model_tf\raw\4_lstm_2path.py
-------------------------functions----------------------
fit(model, data_frame)
predict(model, sess, data_frame, get_hidden_state = False, init_value_forward = None, init_value_backward = None, )
test(filename = "dataset/GOOG-year.csv")

-------------------------methods----------------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size, forget_bias = 0.1, timestep = 5, epoch = 10, )


utilmy\zzml\mlmodels\model_tf\raw\50lstm attention.py
-------------------------functions----------------------
softmax_activation(x)

-------------------------methods----------------------
AttentionModel.__init__(self, x, y, layer_1_rnn_units, attn_dense_nodes = 0, epochs = 100, batch_size = 128, shared_attention_layer = True, chg_yield = False, float_type = 'float32', 0.00001, '00001'), window = 52, predict = 1)
AttentionModel.delete_model(self)
AttentionModel.load_model(self)
AttentionModel.save_model(self)
AttentionModel.set_learning(self, learning)
AttentionModel.make_shared_layers(self)
AttentionModel.build_attention_rnn(self)
AttentionModel.fit_model(self)
AttentionModel.calculate_attentions(self, x_data)
AttentionModel.heatmap(self, data, title_supplement = None)


utilmy\zzml\mlmodels\model_tf\raw\5_gru.py
-------------------------functions----------------------
fit(model, data_frame)
predict(model, sess, data_frame, get_hidden_state = False, init_value = None)
test(filename = "dataset/GOOG-year.csv")

-------------------------methods----------------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size, forget_bias = 0.1, epoch = 1, timestep = 5, )


utilmy\zzml\mlmodels\model_tf\raw\6_encoder_gru.py
-------------------------functions----------------------
reducedimension(input_, dimension = 2, learning_rate = 0.01, hidden_layer = 256, epoch = 20)
fit(model, data_frame)
predict(model, sess, data_frame)
test(filename = "dataset/GOOG-year.csv")

-------------------------methods----------------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size, forget_bias = 0.1, timestep = 5, epoch = 1, )
Model.build_model(self)


utilmy\zzml\mlmodels\model_tf\raw\7_bidirectional_gru.py
-------------------------functions----------------------
fit(model, data_frame)
predict(model, sess, data_frame, get_hidden_state = False, init_value_forward = None, init_value_backward = None, )
test(filename = "dataset/GOOG-year.csv")

-------------------------methods----------------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size, forget_bias = 0.1, epoch = 500, timestep = 5, )


utilmy\zzml\mlmodels\model_tf\raw\8_gru_2path.py
-------------------------functions----------------------
fit(model, data_frame)
predict(model, sess, data_frame, get_hidden_state = False, init_value_forward = None, init_value_backward = None, )
test(filename = "dataset/GOOG-year.csv")

-------------------------methods----------------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size, forget_bias = 0.1, epoch = 500, timestep = 5, )


utilmy\zzml\mlmodels\model_tf\raw\9_vanilla.py
-------------------------functions----------------------
fit(model, data_frame)
predict(model, sess, data_frame, get_hidden_state = False, init_value = None)
test(filename = "dataset/GOOG-year.csv")

-------------------------methods----------------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size, forget_bias = 0.1, epoch = 500, timestep = 5, )


utilmy\zzml\mlmodels\model_tf\raw\access.py
-------------------------functions----------------------
_erase_and_write(memory, address, reset_weights, values)

-------------------------methods----------------------
MemoryAccess.__init__(self, memory_size = 128, word_size = 20, num_reads = 1, num_writes = 1, name = "memory_access")
MemoryAccess._build(self, inputs, prev_state)
MemoryAccess._read_inputs(self, inputs)
MemoryAccess._write_weights(self, inputs, memory, usage)
MemoryAccess._read_weights(self, inputs, memory, prev_read_weights, link)
MemoryAccess.state_size(self)
MemoryAccess.output_size(self)


utilmy\zzml\mlmodels\model_tf\raw\addressing.py
-------------------------functions----------------------
_vector_norms(m)
weighted_softmax(activations, strengths, strengths_op)

-------------------------methods----------------------
CosineWeights.__init__(self, num_heads, word_size, strength_op = tf.nn.softplus, name = "cosine_weights")
CosineWeights._build(self, memory, keys, strengths)
TemporalLinkage.__init__(self, memory_size, num_writes, name = "temporal_linkage")
TemporalLinkage._build(self, write_weights, prev_state)
TemporalLinkage.directional_read_weights(self, link, prev_read_weights, forward)
TemporalLinkage._link(self, prev_link, prev_precedence_weights, write_weights)
TemporalLinkage._precedence_weights(self, prev_precedence_weights, write_weights)
Freeness.__init__(self, memory_size, name = "freeness")
Freeness._build(self, write_weights, free_gate, read_weights, prev_usage)
Freeness.write_allocation_weights(self, usage, write_gates, num_writes)
Freeness._usage_after_write(self, prev_usage, write_weights)
Freeness._usage_after_read(self, prev_usage, free_gate, read_weights)
Freeness._allocation(self, usage)
Freeness.state_size(self)


utilmy\zzml\mlmodels\model_tf\raw\autoencoder.py
-------------------------functions----------------------
reducedimension(input_, dimension = 2, learning_rate = 0.01, hidden_layer = 256, epoch = 20)



utilmy\zzml\mlmodels\model_tf\raw\convert_ipny_cli.py
-------------------------functions----------------------
scan(data_file)
convert_topython(source_files, data_file, out_dir)
check(file_list, dump = False)
Run()



utilmy\zzml\mlmodels\model_tf\raw\dnc.py
-------------------------methods----------------------
DNC.__init__(self, access_config, controller_config, output_size, clip_value = None, name = "dnc")
DNC._clip_if_enabled(self, x)
DNC._build(self, inputs, prev_state)
DNC.initial_state(self, batch_size, dtype = tf.float32)
DNC.state_size(self)
DNC.output_size(self)


utilmy\zzml\mlmodels\model_tf\rl\0_template_rl.py
-------------------------functions----------------------
val(x, y)
fit(model, df, do_action, state_initial = None, reward_initial = None, params = None)
predict(model, sess, df, do_action = None, params =  params)
do_action_example(action_dict)

-------------------------methods----------------------
to_name.__init__(self, adict)
Model.__init__(self, history, params = {})
Agent.__init__(self, history, do_action, params = {})
Agent.predict_action(self, inputs)
Agent.get_predicted_action(self, sequence)
Agent.get_state(self, t, state = None, history = None, reward = None)
Agent.discount_rewards(self, r)
Agent.run_sequence(self, history, do_action, params)
Agent.train(self, n_iters = 1, n_log_freq = 1, state_initial = None, reward_initial = None)


utilmy\zzml\mlmodels\model_tf\rl\1.turtle-agent.py
-------------------------functions----------------------
buy_stock(real_movement, signal, initial_money = 10000, max_buy = 20, max_sell = 20)



utilmy\zzml\mlmodels\model_tf\rl\10.duel-q-learning-agent.py
-------------------------methods----------------------
Agent.__init__(self, state_size, window_size, trend, skip, batch_size)
Agent.act(self, state)
Agent.get_state(self, t)
Agent.replay(self, batch_size)
Agent.buy(self, initial_money)
Agent.train(self, iterations, checkpoint, initial_money)


utilmy\zzml\mlmodels\model_tf\rl\11.double-duel-q-learning-agent.py
-------------------------methods----------------------
Model.__init__(self, input_size, output_size, layer_size, learning_rate)
Agent.__init__(self, state_size, window_size, trend, skip)
Agent._assign(self)
Agent._memorize(self, state, action, reward, new_state, done)
Agent._select_action(self, state)
Agent._construct_memories(self, replay)
Agent.predict(self, inputs)
Agent.get_predicted_action(self, sequence)
Agent.get_state(self, t)
Agent.buy(self, initial_money)
Agent.train(self, iterations, checkpoint, initial_money)


utilmy\zzml\mlmodels\model_tf\rl\12.duel-recurrent-q-learning-agent.py
-------------------------methods----------------------
Agent.__init__(self, state_size, window_size, trend, skip)
Agent._memorize(self, state, action, reward, new_state, dead, rnn_state)
Agent._construct_memories(self, replay)
Agent.get_state(self, t)
Agent.buy(self, initial_money)
Agent.train(self, iterations, checkpoint, initial_money)


utilmy\zzml\mlmodels\model_tf\rl\13.double-duel-recurrent-q-learning-agent.py
-------------------------methods----------------------
Model.__init__(self, input_size, output_size, layer_size, learning_rate, name)
Agent.__init__(self, state_size, window_size, trend, skip)
Agent._assign(self, from_name, to_name)
Agent._memorize(self, state, action, reward, new_state, dead, rnn_state)
Agent._select_action(self, state)
Agent._construct_memories(self, replay)
Agent.get_state(self, t)
Agent.buy(self, initial_money)
Agent.train(self, iterations, checkpoint, initial_money)


utilmy\zzml\mlmodels\model_tf\rl\14.actor-critic-agent.py
-------------------------methods----------------------
Actor.__init__(self, name, input_size, output_size, size_layer)
Critic.__init__(self, name, input_size, output_size, size_layer, learning_rate)
Agent.__init__(self, state_size, window_size, trend, skip)
Agent._assign(self, from_name, to_name)
Agent._memorize(self, state, action, reward, new_state, dead)
Agent._select_action(self, state)
Agent._construct_memories_and_train(self, replay)
Agent.get_state(self, t)
Agent.buy(self, initial_money)
Agent.train(self, iterations, checkpoint, initial_money)


utilmy\zzml\mlmodels\model_tf\rl\15.actor-critic-duel-agent.py
-------------------------methods----------------------
Actor.__init__(self, name, input_size, output_size, size_layer)
Critic.__init__(self, name, input_size, output_size, size_layer, learning_rate)
Agent.__init__(self, state_size, window_size, trend, skip)
Agent._assign(self, from_name, to_name)
Agent._memorize(self, state, action, reward, new_state, dead)
Agent._select_action(self, state)
Agent._construct_memories_and_train(self, replay)
Agent.get_state(self, t)
Agent.buy(self, initial_money)
Agent.train(self, iterations, checkpoint, initial_money)


utilmy\zzml\mlmodels\model_tf\rl\16.actor-critic-recurrent-agent.py
-------------------------methods----------------------
Actor.__init__(self, name, input_size, output_size, size_layer)
Critic.__init__(self, name, input_size, output_size, size_layer, learning_rate)
Agent.__init__(self, state_size, window_size, trend, skip)
Agent._assign(self, from_name, to_name)
Agent._memorize(self, state, action, reward, new_state, dead, rnn_state)
Agent._select_action(self, state)
Agent._construct_memories_and_train(self, replay)
Agent.get_state(self, t)
Agent.buy(self, initial_money)
Agent.train(self, iterations, checkpoint, initial_money)


utilmy\zzml\mlmodels\model_tf\rl\17.actor-critic-duel-recurrent-agent.py
-------------------------methods----------------------
Actor.__init__(self, name, input_size, output_size, size_layer)
Critic.__init__(self, name, input_size, output_size, size_layer, learning_rate)
Agent.__init__(self, state_size, window_size, trend, skip)
Agent._assign(self, from_name, to_name)
Agent._memorize(self, state, action, reward, new_state, dead, rnn_state)
Agent._select_action(self, state)
Agent._construct_memories_and_train(self, replay)
Agent.get_state(self, t)
Agent.buy(self, initial_money)
Agent.train(self, iterations, checkpoint, initial_money)


utilmy\zzml\mlmodels\model_tf\rl\18.curiosity-q-learning-agent.py
-------------------------methods----------------------
Agent.__init__(self, state_size, window_size, trend, skip)
Agent._memorize(self, state, action, reward, new_state, done)
Agent.get_state(self, t)
Agent.predict(self, inputs)
Agent.get_predicted_action(self, sequence)
Agent._select_action(self, state)
Agent._construct_memories(self, replay)
Agent.buy(self, initial_money)
Agent.train(self, iterations, checkpoint, initial_money)


utilmy\zzml\mlmodels\model_tf\rl\19.recurrent-curiosity-q-learning-agent.py
-------------------------methods----------------------
Agent.__init__(self, state_size, window_size, trend, skip)
Agent._memorize(self, state, action, reward, new_state, done, rnn_state)
Agent.get_state(self, t)
Agent._construct_memories(self, replay)
Agent.buy(self, initial_money)
Agent.train(self, iterations, checkpoint, initial_money)


utilmy\zzml\mlmodels\model_tf\rl\2.moving-average-agent.py
-------------------------functions----------------------
buy_stock(real_movement, signal, initial_money = 10000, max_buy = 20, max_sell = 20)



utilmy\zzml\mlmodels\model_tf\rl\20.duel-curiosity-q-learning-agent.py
-------------------------methods----------------------
Agent.__init__(self, state_size, window_size, trend, skip)
Agent._memorize(self, state, action, reward, new_state, done)
Agent.get_state(self, t)
Agent.predict(self, inputs)
Agent.get_predicted_action(self, sequence)
Agent._select_action(self, state)
Agent._construct_memories(self, replay)
Agent.buy(self, initial_money)
Agent.train(self, iterations, checkpoint, initial_money)


utilmy\zzml\mlmodels\model_tf\rl\21.neuro-evolution-agent.py
-------------------------functions----------------------
relu(X)
softmax(X)
feed_forward(X, nets)

-------------------------methods----------------------
neuralnetwork.__init__(self, id_, hidden_size = 128)
NeuroEvolution.__init__(self, population_size, mutation_rate, model_generator, state_size, window_size, trend, skip, initial_money, )
NeuroEvolution._initialize_population(self)
NeuroEvolution.mutate(self, individual, scale = 1.0)
NeuroEvolution.inherit_weights(self, parent, child)
NeuroEvolution.crossover(self, parent1, parent2)
NeuroEvolution.get_state(self, t)
NeuroEvolution.act(self, p, state)
NeuroEvolution.buy(self, individual)
NeuroEvolution.calculate_fitness(self)
NeuroEvolution.evolve(self, generations = 20, checkpoint = 5)


utilmy\zzml\mlmodels\model_tf\rl\22.neuro-evolution-novelty-search-agent.py
-------------------------functions----------------------
relu(X)
softmax(X)
feed_forward(X, nets)

-------------------------methods----------------------
neuralnetwork.__init__(self, id_, hidden_size = 128)
NeuroEvolution.__init__(self, population_size, mutation_rate, model_generator, state_size, window_size, trend, skip, initial_money, )
NeuroEvolution._initialize_population(self)
NeuroEvolution._memorize(self, q, i, limit)
NeuroEvolution.mutate(self, individual, scale = 1.0)
NeuroEvolution.inherit_weights(self, parent, child)
NeuroEvolution.crossover(self, parent1, parent2)
NeuroEvolution.get_state(self, t)
NeuroEvolution.act(self, p, state)
NeuroEvolution.buy(self, individual)
NeuroEvolution.calculate_fitness(self)
NeuroEvolution.evaluate(self, individual, backlog, pop, k = 4)
NeuroEvolution.evolve(self, generations = 20, checkpoint = 5)


utilmy\zzml\mlmodels\model_tf\rl\3.signal-rolling-agent.py
-------------------------functions----------------------
buy_stock(real_movement, delay = 5, initial_state = 1, initial_money = 10000, max_buy = 20, max_sell = 20)



utilmy\zzml\mlmodels\model_tf\rl\4.policy-gradient-agent_old.py
-------------------------functions----------------------
fit(model, df, do_action)
predict(model, sess, df, do_action)
do_action_example(action_dict)
test(filename =  'dataset/GOOG-year.csv')

-------------------------methods----------------------
Model.__init__(self, state_size, window_size, trend, skip, iterations, initial_reward)
Agent.__init__(self, state_size, window_size, trend, skip)
Agent.predict(self, inputs)
Agent.get_state(self, t, reward_state = None)
Agent.discount_rewards(self, r)
Agent.get_predicted_action(self, sequence)
Agent.predict_sequence(self, trend_input, do_action, param = None)
Agent.buy(self, initial_money)
Agent.train(self, iterations, checkpoint, initial_money)
to_name.__init__(self, adict)


utilmy\zzml\mlmodels\model_tf\rl\4_policy-gradient-agent.py
-------------------------functions----------------------
fit(model, dftrain, params = {})
predict(model, sess, dftest, params = {})
test(filename =  'dataset/GOOG-year.csv')

-------------------------methods----------------------
Model.__init__(self, state_size, window_size, trend, skip, iterations, initial_reward, checkpoint  =  10)
Agent.__init__(self, state_size, window_size, trend, skip)
Agent.predict(self, inputs)
Agent.get_state(self, t, reward_state = None)
Agent.discount_rewards(self, r)
Agent.get_predicted_action(self, sequence)
Agent.predict_sequence(self, pars, trend_history = None)
Agent.train(self, iterations, checkpoint, initial_money)
to_name.__init__(self, adict)


utilmy\zzml\mlmodels\model_tf\rl\5_q-learning-agent.py
-------------------------functions----------------------
fit(model, dftrain, params = {})
predict(model, sess, dftest, params = {})
test(filename =  '../dataset/GOOG-year.csv')

-------------------------methods----------------------
Model.__init__(self, state_size, window_size, trend, skip, iterations, initial_reward, checkpoint  =  10)
Agent.__init__(self, state_size, window_size, trend, skip, batch_size)
Agent.act(self, state)
Agent.get_state(self, t)
Agent.replay(self, batch_size)
Agent.predict_sequence(self, pars, trend_history = None)
Agent.train(self, iterations, checkpoint, initial_money)
to_name.__init__(self, adict)


utilmy\zzml\mlmodels\model_tf\rl\6_evolution-strategy-agent.py
-------------------------functions----------------------
get_imports()
fit(model, dftrain, params = {})
predict(model, sess, dftest, params = {})
test(filename =  '../dataset/GOOG-year.csv')

-------------------------methods----------------------
Deep_Evolution_Strategy.__init__(self, weights, reward_function, population_size, sigma, learning_rate)
Deep_Evolution_Strategy._get_weight_from_population(self, weights, population)
Deep_Evolution_Strategy.get_weights(self)
Deep_Evolution_Strategy.train(self, epoch = 100, print_every = 1)
Model.__init__(self, input_size, layer_size, output_size, window_size, skip, initial_money, iterations = 500, checkpoint = 10)
Model.predict(self, inputs)
Model.get_weights(self)
Model.set_weights(self, weights)
Agent.__init__(self, model, window_size, trend, skip, initial_money)
Agent.act(self, sequence)
Agent.get_state(self, t)
Agent.get_reward(self, weights)
Agent.fit(self, iterations, checkpoint)
Agent.run_sequence(self, df_test)
to_name.__init__(self, adict)


utilmy\zzml\mlmodels\model_tf\rl\7.double-q-learning-agent.py
-------------------------functions----------------------
fit(model, dftrain, params = {})
predict(model, sess, dftest, params = {})
test(filename =  '../dataset/GOOG-year.csv')

-------------------------methods----------------------
Model.__init__(self, window_size, trend, skip, iterations, initial_reward, checkpoint  =  10)
QModel.__init__(self, input_size, output_size, layer_size, learning_rate)
Agent.__init__(self, state_size, window_size, trend, skip)
Agent._assign(self)
Agent._memorize(self, state, action, reward, new_state, done)
Agent._select_action(self, state)
Agent._construct_memories(self, replay)
Agent.predict(self, inputs)
Agent.get_predicted_action(self, sequence)
Agent.get_state(self, t)
Agent.run_sequence(self, initial_money)
Agent.train(self, iterations, checkpoint, initial_money)


utilmy\zzml\mlmodels\model_tf\rl\8.recurrent-q-learning-agent.py
-------------------------methods----------------------
Agent.__init__(self, state_size, window_size, trend, skip)
Agent._memorize(self, state, action, reward, new_state, dead, rnn_state)
Agent._construct_memories(self, replay)
Agent.get_state(self, t)
Agent.buy(self, initial_money)
Agent.train(self, iterations, checkpoint, initial_money)


utilmy\zzml\mlmodels\model_tf\rl\9.double-recurrent-q-learning-agent.py
-------------------------methods----------------------
Model.__init__(self, input_size, output_size, layer_size, learning_rate, name)
Agent.__init__(self, state_size, window_size, trend, skip)
Agent._assign(self, from_name, to_name)
Agent._memorize(self, state, action, reward, new_state, dead, rnn_state)
Agent._select_action(self, state)
Agent._construct_memories(self, replay)
Agent.get_state(self, t)
Agent.buy(self, initial_money)
Agent.train(self, iterations, checkpoint, initial_money)


utilmy\zzml\mlmodels\model_tf\rl\updated-NES-google.py
-------------------------functions----------------------
f(w)
get_state(data, t, n)
act(model, sequence)

-------------------------methods----------------------
Deep_Evolution_Strategy.__init__(self, weights, reward_function, population_size, sigma, learning_rate)
Deep_Evolution_Strategy._get_weight_from_population(self, weights, population)
Deep_Evolution_Strategy.get_weights(self)
Deep_Evolution_Strategy.train(self, epoch = 100, print_every = 1)
Model.__init__(self, input_size, layer_size, output_size)
Model.predict(self, inputs)
Model.get_weights(self)
Model.set_weights(self, weights)
Agent.__init__(self, model, money, max_buy, max_sell, close, window_size, skip)
Agent.act(self, sequence)
Agent.get_reward(self, weights)
Agent.fit(self, iterations, checkpoint)
Agent.buy(self)


utilmy\zzml\mlmodels\preprocess\keras_dataloader\dataloader.py
-------------------------functions----------------------
default_collate_fn(samples)

-------------------------methods----------------------
DataGenerator.__init__(self, dataset: Dataset, collate_fn = default_collate_fn, batch_size = 32, shuffle = True, num_workers = 0, replacement: bool  =  False, )
DataGenerator.__getitem__(self, index)
DataGenerator.on_epoch_end(self)
DataGenerator.__len__(self)


utilmy\zzml\mlmodels\preprocess\keras_dataloader\dataset.py
-------------------------methods----------------------
Dataset.__init__(self, dtype = 'float32')
Dataset.__getitem__(self, index)
Dataset.__len__(self)
Dataset.__add__(self, other)
ConcatDataset.cumsum(sequence)
ConcatDataset.__init__(self, datasets)
ConcatDataset.__len__(self)
ConcatDataset.__getitem__(self, idx)
ConcatDataset.cummulative_sizes(self)


utilmy\zzml\mlmodels\template\zarchive\gluonts_model.py
-------------------------functions----------------------
get_params(choice = 0, data_path = "dataset/", **kw)
test2(data_path = "dataset/", out_path = "GLUON/gluon.png", reset = True)
test(data_path = "dataset/", out_path = "GLUON/gluon.png", reset = True)

-------------------------methods----------------------
Model.__init__(self, model_pars = None, compute_pars = None)


utilmy\zzml\mlmodels\template\zarchive\model_tf_sequential.py
-------------------------functions----------------------
os_module_path()
os_file_path(data_path)
os_package_root_path(filepath, sublevel = 0, path_add = "")
log(*s, n = 0, m = 1)
fit(model, data_pars, out_pars = None, compute_pars = None, **kwargs)
metrics(model, sess = None, data_pars = None, out_pars = None)
predict(model, sess, data_pars = None, out_pars = None, compute_pars = None, get_hidden_state = False, init_value = None)
reset_model()
get_dataset(data_pars = None)
get_pars(choice = "test", **kwargs)
test(data_path = "dataset/GOOG-year.csv", out_path = "", reset = True)
test2(data_path = "dataset/GOOG-year.csv")

-------------------------methods----------------------
Model.__init__(self, epoch = 5, learning_rate = 0.001, num_layers = 2, size = None, size_layer = 128, output_size = None, forget_bias = 0.1, timestep = 5, )
