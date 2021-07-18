### Code Structure
```
### Files
    mygenerator/dataset.py                           : Dataset related Classes (Img, Text)
 
    mygenerator/transform.py                         : Transformation on Dataset (Image or Text_to_Image)

    mygenerator/pipeline.py                          : Pipeline of Transformations (ETL Type).

    mygenerator/validate.py                          : Code to validate the padding, Image format rules



    mygenerator/util_image.py                        : Functional API for image Transformations.

    mygenerator/utils.py                             : Functonal API for logs, config file,...



    mygenerator/cli/cli_generate_numbers_sequence.py : CLI Interface
    mygenerator/cli/cli_generate_phone_numbers.py    : CLI Interface
    mygenerator/cli/cli_validate_phone_sequence.py   : CLI Interface

#### Data
    ztmpp/phone/   : Output of Phone Numbers
    ztmpp/seq/     : Output of number generation




#### Test Code
    tests/   Pytest code
    .github/CI_build_release.yml : CI using Github Actions

    .github/CI_test_deep.yml : CI using Github Actions, large scale generation : > 10k 



#### Configuration:
    config/config.yaml             : Full size running
    setup.py  : Python config for spark-submit
    pylinrc   : Code Checker






#### Script :
    run_exanple.sh :  Some sample bash scripts.

```





### Code Design:
```
  We use an hybrid design between lightweight OO approach, backed by functional design:
     A bit like pytorch (Dataset, Dataloader) and scikit-learn (fit, fit_transform).     
  Functional part aims to simplify/accelerate compute.
  OO helps to integrate various data access, interfaces.   
  
  3 main blocks :
  
     Dataset  : Wrapper to access raw data.
     Transform: Dataset --> Dataset
     Pipeline : sequence of transforms

     Common functional api (image, text, ...)


  Repo is split in 3 interface, sub-libraries


  ## CLI Interface
      in CLI folder


  ## Object Interface
      Dataset : Lazy Class wrapper for dataset access:
                MetaData + raw data path.
                Object Does NOT store the data, but access on disk/fast storage.
                Can be extend to handle S3 buckets,... any filessystem storage
                This is to wrap the access to data (ie similiar to Dataset in Pytorch).    
                   
                    
      Transform : Class wrapper to operate on Dataset Class.
                  Follow sklearn API style: fit, fit_transform, transform
                  Take a Dataset --> returns a Dataset                            
                  This is similar than Dataloader and Transform in pytorch
                  or Transform in Apache Spark.
                   
                  Main benefits :
                       Easier to handle on disk dataset, in memory dataset, Tflow format,...
                       Paralleization can be done in the background.
                       Drawback is wrapping over function is not good for performance...
                       
                                                
  
      Pipeline : ETL/Functional type for sequence of Transformations on Dataset.

      
      Technically, Dataset does not load the full data in memory, 
      but sequentially execute during action process (ie save on disk), reducing memory footprint.



  ## Functional Interface:
     util_image.py : Mostly re-usable functions for image processing, packaging potential high perf. runtime.
                     Image cropping
     
     utils.py : logging, cofnig loader ..
  
     
  ## Phone Numbers are represented by :
           NlpDataset, generating fake numbers (ie no physical storage),
           allowing validation of phone numbers,...
 
  ## Image of numbers (MNIST) by :
          ImageDataset,  allowing various transformations to be plugged.   
     
 
```








### Running Tree
```
This is to highlight the execution sequence when using CLI:
It would help to optimize the code design (ie reduce bottleneck).

cli_generate_phone_numbers --num_images 100 --min_spacing 3 --max_spacing 5 --image_width 300 --output_path ztmpp/output/phone/ --config_file default       



25.323 <module>  <string>:1                                                                                                                                                        
   [8 frames hidden]  <string>, runpy, importlib, <built-in>                                                                                                                       
      25.321 _run_code  runpy.py:62                                                                                                                                                
      └─ 25.321 <module>  mygenerator\cli\cli_generate_phone_numbers.py:1  
                                                                                                              
         ├─ 24.275 run_cli  mygenerator\cli\cli_generate_phone_numbers.py:5                                                                                                        
         │  └─ 24.272 run_generate_phone_numbers  mygenerator\pipeline.py:52
         
                  ### Lazy Operation                                                                                                       
         │     ├─ 23.530 save  mygenerator\dataset.py:149                                                                                                                          
         │     │  ├─ 22.356 get_sample  mygenerator\dataset.py:128                                                                                                                 
         │     │  │  └─ 22.256 get_image_only  mygenerator\dataset.py:120                                                                                                          
         │     │  │     └─ 22.251 _get_image_fn  mygenerator\transform.py:167                                                                                                      
         │     │  │        ├─ 19.818 get_image_only  mygenerator\dataset.py:120                                                                                                    
         │     │  │        │  └─ 19.817 _get_image_fn  mygenerator\transform.py:78                                                                                                 
         │     │  │        │     ├─ 18.728 get_label_list  mygenerator\dataset.py:133                                                                                              
         │     │  │        │     │  └─ 18.515 [self]                                                                                                                               
         │     │  │        │     └─ 0.932 get_image_only  mygenerator\dataset.py:120                                                                                               
         │     │  │        │        └─ 0.930 _get_image_fn  mygenerator\transform.py:121                                                                                           
         │     │  │        │           ├─ 0.641 get_image_only  mygenerator\dataset.py:120                                                                                         
         │     │  │        │           │  └─ 0.462 __getitem__  pandas\core\indexing.py:864                                                                                        
         │     │  │        │           │        [260 frames hidden]  pandas, <built-in>, numpy, abc, _weak...                                                                      
         │     │  │        │           └─ 0.286 transform_sample  mygenerator\transform.py:129                                                                                     
         │     │  │        │              └─ 0.275 image_remove_extra_padding  mygenerator\util_image.py:62                                                                        
         │     │  │        └─ 2.431 transform_sample  mygenerator\transform.py:176                                                                                                 
         │     │  │           └─ 2.366 image_padding_validate  mygenerator\validate.py:12                                                                                          
         │     │  │              └─ 2.363 image_padding_get  mygenerator\validate.py:61                                                                                            
         │     │  │                 ├─ 1.464 sum  <__array_function__ internals>:2                                                                                                 
         │     │  │                 │     [18 frames hidden]  <__array_function__ internals>, numpy...                                                                             
         │     │  │                 └─ 0.899 [self]                                                                                                                                
         │     │  └─ 1.159 imwrite  <built-in>:0                                                                                                                                   
         │     │        [2 frames hidden]  <built-in> 
         
                                                                                                                                      
         │     ├─ 0.361 __init__  mygenerator\transform.py:311                                                                                                                     
         │     │  └─ 0.358 __init__  mygenerator\dataset.py:78
         
                                                                                                                              
         │     └─ 0.325 dataset_get_path  mygenerator\utils.py:113                                                                                                                 
         │        └─ 0.323 glob  glob.py:9                                                                                                                                         
         │              [42 frames hidden]  glob, ntpath, <built-in>, fnmatch
         
                                                                                                               
         └─ 1.044 <module>  mygenerator\pipeline.py:1                                                                                                                              
            ├─ 0.522 <module>  mygenerator\dataset.py:2                                                                                                                            
            │  └─ 0.505 <module>  mygenerator\util_image.py:2                                                                                                                      
            │     └─ 0.472 <module>  skimage\morphology\__init__.py:1                                                                                                              
            │           [711 frames hidden]  skimage, scipy, <built-in>, inspect, ...                                                                                              
            └─ 0.422 <module>  pandas\__init__.py:3                                                                                                                                
                  [1103 frames hidden]  pandas, zipfile, <built-in>, gzip, ty...                                                                                                   
                                                                                                                                                                                                                 
                                                     



```


