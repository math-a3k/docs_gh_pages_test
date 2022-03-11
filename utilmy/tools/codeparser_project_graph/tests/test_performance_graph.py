import pytest
import os

def test_toplvl():
    """function test_toplvl
    Args:
    Returns:
        
    """
    n_files_before = len(os.listdir("./tests"))
    os.system("project_graph tests/script_test_case_1.py")
    n_files_after = len(os.listdir("./tests"))
    assert n_files_before == n_files_after - 1, "No png file output."

def test_lowlvl():
    """function test_lowlvl
    Args:
    Returns:
        
    """
    n_files_before = len(os.listdir("./tests/sub_dir"))
    os.system("project_graph tests/sub_dir/script_test_case_2.py")
    n_files_after = len(os.listdir("./tests/sub_dir"))
    assert n_files_before == n_files_after - 1, "No png file output."