
[![Build and Test , Package PyPI](https://github.com/arita37/myutil/actions/workflows/build%20and%20release.yml/badge.svg)](https://github.com/arita37/myutil/actions/workflows/build%20and%20release.yml)

[     https://pypi.org/project/utilmy/#history ](https://pypi.org/project/utilmy/#history)


# myutil
    One liner utilities


# HELP !!!

   https://github.com/arita37/myutil/blob/main/docs/doc_index.py

   https://github.com/arita37/myutil/issues/127
   

# Looking for contributors

   This package is looking for contributors. 
    

# myutil
    https://pypi.org/project/utilmy/#history




# Install for dev
    git clone  https://github.com/arita37/myutil.git
    git checkout newbranch
    cd myutil
    pip install -e .

    
    

# Usage
 
  https://colab.research.google.com/drive/12rpbgH3WYcQq3jtl9vzEYeVdu9a9GOM_?usp=sharing
 
  https://colab.research.google.com/drive/1NYQZrfAPqbuLCt9yhVROLMRJM-RrFYWr#scrollTo=Rrho08zYe6Gj

  https://colab.research.google.com/drive/1NYQZrfAPqbuLCt9yhVROLMRJM-RrFYWr#scrollTo=2zMKv6MXOJJu


 ```
   #### Save current python session on disk.
   from utilmy import Session
   sess = Session("ztmp/session")
   
   
   aabb = 'ok'
   
   
   
   ### Save Python sesison
   sess.save('mysess', globals(),)
   sess.show()
   
   
   ### Reload session
   del aabb
   sess.load('mysess', )
   print(aabb)
   
   
   
 ```
 
 






# Misc
 ```


REQUIRED_PKGS = [
    # We use numpy>=1.17 to have np.random.Generator (Dataset shuffling)
    "numpy>=1.17",
    # Backend and serialization.
    # Minimum 3.0.0 to support mix of struct and list types in parquet, and batch iterators of parquet data
    # pyarrow 4.0.0 introduced segfault bug, see: https://github.com/huggingface/datasets/pull/2268
    "pyarrow>=1.0.0,!=4.0.0",
    # For smart caching dataset processing
    "dill",
    # For performance gains with apache arrow
    "pandas",
    # for downloading datasets over HTTPS
    "requests>=2.19.0",
    # progress bars in download and scripts
    "tqdm>=4.27",
    # dataclasses for Python versions that don't have it
    "dataclasses;python_version<'3.7'",
    # for fast hashing
    "xxhash",
    # for better multiprocessing
    "multiprocess",
    # to get metadata of optional dependencies such as torch or tensorflow for Python versions that don't have it
    "importlib_metadata;python_version<'3.8'",
    # to save datasets locally or on any filesystem
    # minimum 2021.05.0 to have the AbstractArchiveFileSystem
    "fsspec>=2021.05.0",
    # To get datasets from the Datasets Hub on huggingface.co
    "huggingface_hub<0.1.0",
    # Utilities from PyPA to e.g., compare versions
    "packaging",
]

BENCHMARKS_REQUIRE = [
    "numpy==1.18.5",
    "tensorflow==2.3.0",
    "torch==1.6.0",
    "transformers==3.0.2",
]



 #### Generate automatic docs/stats
 docs markdown --repo_url  https://github.com/arita37/spacefusion.git   --out_dir docs/
 
 docs all      --repo_dir yourpath  --out_dir docs/ 
 
 
 
 
 
```
