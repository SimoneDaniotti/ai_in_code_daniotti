import pandas as pd
import networkx as nx
import numpy as np
from pickle_file import save_obj, load_obj
import jsonlines
from tqdm import tqdm
from collections import defaultdict, Counter
import random
from CoLoc_class import CoLoc #import the CoLoc class
import scipy

data_path = 'datasets/'
data_path_save = data_path + 'temp_lib_co/'


def get_bayesian_pmi_edgelist_from_occurrence(co_occurrrence, p_value = 0.05):
    co_df = pd.DataFrame({"c1":[a[0] for a in co_occurrrence] + [a[1] for a in co_occurrrence], "c2":[a[1] for a in co_occurrrence] + [a[0] for a in co_occurrrence], "cc":[a[2] for a in co_occurrrence] + [a[2] for a in co_occurrrence]})
    
    dft = co_df.pivot(index = 'c1', columns = 'c2', values = 'cc')
    dft = dft.fillna(0)
    
    #Q = CoLoc(df_q, prior = 'uniform', nr_prior_obs = np.size(df_q))
    Q = CoLoc(dft)

    df_Q = Q.make_sigPMIpci(p_value)

    df_index = df_Q.index
    df_columns = df_Q.columns

    res = scipy.sparse.coo_matrix(df_Q.fillna(0).values)

    df_res = pd.DataFrame({'c1':df_columns[res.col], 'c2':df_index[res.row], 'cc':res.data})

    df_edgelist = df_res[df_res['cc'] > 0]

    edgelist_sig = [(c1,c2,cc) for c1,c2,cc in zip(df_edgelist['c1'], df_edgelist['c2'], df_edgelist['cc']) if c1 != c2]

    #df_variance = Q.make_stdPMIpci()

    return edgelist_sig, Q



##! community_cooccurrence matrix
def build_library_pmi(lib_matrix, C_dict, p_value = 0.05):

    lib_temp_dict = {i:l for l,i in C_dict.items()}
    lib_std = [lib_temp_dict[i] for i in range(len(C_dict))]
    cc_cooccurrence = []
    for i1 in range(np.shape(lib_matrix)[0]):
        for i2 in range(i1+1, np.shape(lib_matrix)[1]):
            l1 = lib_std[i1]
            l2 = lib_std[i2]
            if lib_matrix[i1,i2] > 0:
                cc_cooccurrence.append((l1,l2,lib_matrix[i1,i2]))

    print(f"p value: {p_value}")
    cc_pmi, Q =get_bayesian_pmi_edgelist_from_occurrence(cc_cooccurrence, p_value)
    
    cc_pmi_matrix = np.zeros((len(C_dict), len(C_dict)))
    for ccp in cc_pmi:
        cc_pmi_matrix[C_dict[ccp[0]], C_dict[ccp[1]]] = ccp[2]
        cc_pmi_matrix[C_dict[ccp[1]], C_dict[ccp[0]]] = ccp[2]

    
    return cc_pmi, cc_pmi_matrix, Q




def get_bayesian_pmi_edgelist_from_occurrence_zscore(co_occurrrence, p_value = 0.05):
    co_df = pd.DataFrame({"c1":[a[0] for a in co_occurrrence] + [a[1] for a in co_occurrrence], "c2":[a[1] for a in co_occurrrence] + [a[0] for a in co_occurrrence], "cc":[a[2] for a in co_occurrrence] + [a[2] for a in co_occurrrence]})
    
    dft = co_df.pivot(index = 'c1', columns = 'c2', values = 'cc')
    dft = dft.fillna(0)
    
    #Q = CoLoc(df_q, prior = 'uniform', nr_prior_obs = np.size(df_q))
    Q = CoLoc(dft)

    df_Q = Q.make_PMIpci()

    df_index = df_Q.index
    df_columns = df_Q.columns

    res = scipy.sparse.coo_matrix(df_Q.fillna(0).values)

    df_res = pd.DataFrame({'c1':df_columns[res.col], 'c2':df_index[res.row], 'cc':res.data})

    df_edgelist = df_res[df_res['cc'] > 0]

    edgelist_mean = [(c1,c2,cc) for c1,c2,cc in zip(df_edgelist['c1'], df_edgelist['c2'], df_edgelist['cc']) if c1 != c2]
    edgelist_bool = defaultdict(bool)
    for cc in edgelist_mean:
        edgelist_bool[(cc[0],cc[1])] = True

    df_Qstd = Q.make_stdPMIpci()

    df_index = df_Qstd.index
    df_columns = df_Qstd.columns

    res = scipy.sparse.coo_matrix(df_Qstd.fillna(0).values)

    df_res = pd.DataFrame({'c1':df_columns[res.col], 'c2':df_index[res.row], 'cc':res.data})

    df_edgelist = df_res[df_res['cc'] > 0]

    edgelist_std = [(c1,c2,cc) for c1,c2,cc in zip(df_edgelist['c1'], df_edgelist['c2'], df_edgelist['cc']) if edgelist_bool[(c1, c2)]]


    #df_variance = Q.make_stdPMIpci()

    return edgelist_mean, edgelist_std, Q

##! community_cooccurrence matrix
def build_library_pmi_zscore(lib_matrix, C_dict, p_value = 0.05):

    lib_temp_dict = {i:l for l,i in C_dict.items()}
    lib_std = [lib_temp_dict[i] for i in range(len(C_dict))]
    cc_cooccurrence = []
    for i1 in range(np.shape(lib_matrix)[0]):
        for i2 in range(i1+1, np.shape(lib_matrix)[1]):
            l1 = lib_std[i1]
            l2 = lib_std[i2]
            if lib_matrix[i1,i2] > 0:
                cc_cooccurrence.append((l1,l2,lib_matrix[i1,i2]))

    print(f"p value: {p_value}")
    cc_pmi, cc_std, Q =get_bayesian_pmi_edgelist_from_occurrence_zscore(cc_cooccurrence, p_value)
    
    cc_pmi_matrix = np.zeros((len(C_dict), len(C_dict)))
    for ccp in cc_pmi:
        cc_pmi_matrix[C_dict[ccp[0]], C_dict[ccp[1]]] = ccp[2]
        cc_pmi_matrix[C_dict[ccp[1]], C_dict[ccp[0]]] = ccp[2]

    cc_std_matrix = np.zeros((len(C_dict), len(C_dict)))
    for ccz in cc_std:
        cc_std_matrix[C_dict[ccz[0]], C_dict[ccz[1]]] = ccz[2]
        cc_std_matrix[C_dict[ccz[1]], C_dict[ccz[0]]] = ccz[2]

    
    return cc_pmi, cc_pmi_matrix, cc_std, cc_std_matrix, Q

import pandas as pd
import numpy as np
import ast
import os
import duckdb as db
from tqdm import tqdm
from pickle_file import load_obj, save_obj
from path_file import data_path, data_path_save

import csv

##! 0: imported libraries; 
##! 1: functions; 
##! 2: library in functions;
##! 3: library in script;
##! 4: classes;
##! 5: library in classes


def get_code_dict(df):

    code_df = df.sort_values(['committer_date'], ascending = False)

    code_dict = {}
    pylist_dict = {}

    for a, f, fn in zip(code_df.change_type, code_df.filesource, code_df.file_name):
        if a != 'DELETE' and not pd.isna(f):
            if fn not in code_dict:
                code_dict[fn] = f
                pylist_dict[fn] = True

    return code_dict, pylist_dict


import ast
import typing

class LibraryVisitor_full(ast.NodeVisitor):
    def __init__(self, libraries, aliases):
        self.libraries = libraries
        self.aliases = aliases
        self.function_libraries = {}
        self.class_libraries = {}
        self.global_libraries = set()
        self.all_functions = set()
        self.all_classes = set()
        self.function_stack = []
        self.current_class = None

    def _add_library(self, used_lib):
        if not used_lib:
            return
        if self.function_stack:
            self.function_libraries[self.function_stack[-1]].add(used_lib)
        elif self.current_class:
            self.class_libraries[self.current_class].add(used_lib)
        else:
            self.global_libraries.add(used_lib)

    def visit_FunctionDef(self, node):
        # Handle function name and stack
        if self.current_class:
            function_name = f"{self.current_class}++++++++{node.name}"
        else:
            function_name = node.name
        self.function_stack.append(function_name)
        self.function_libraries[self.function_stack[-1]] = set()
        self.all_functions.add(self.function_stack[-1])
        
        # New: Process function return type annotation
        if node.returns:
            self._add_library(self._find_used_library_from_annotation(node.returns))
        
        # New: Process function arguments with type annotations
        self.visit(node.args)
        
        # Visit the function body
        for item in node.body:
            self.visit(item)
        
        self.function_stack.pop()

    def visit_arguments(self, node):
        for arg in node.args:
            if arg.annotation:
                self._add_library(self._find_used_library_from_annotation(arg.annotation))
        for arg in node.kwonlyargs:
            if arg.annotation:
                self._add_library(self._find_used_library_from_annotation(arg.annotation))
        if node.vararg and node.vararg.annotation:
            self._add_library(self._find_used_library_from_annotation(node.vararg.annotation))
        if node.kwarg and node.kwarg.annotation:
            self._add_library(self._find_used_library_from_annotation(node.kwarg.annotation))

    def visit_ClassDef(self, node):
        self.current_class = node.name
        self.class_libraries[self.current_class] = set()
        self.all_classes.add(node.name)
        
        self.generic_visit(node)
        
        self.current_class = None

    def visit_Call(self, node):
        used_lib = self._find_used_library_from_execution(node.func)
        self._add_library(used_lib)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        used_lib = self._find_used_library_from_execution(node)
        self._add_library(used_lib)
        self.generic_visit(node)
        
    def visit_AnnAssign(self, node):
        if node.annotation:
            self._add_library(self._find_used_library_from_annotation(node.annotation))
        self.generic_visit(node)

    def _find_used_library_from_execution(self, node: ast.expr) -> typing.Optional[str]:
        parts = []
        current_node = node
        while isinstance(current_node, ast.Attribute):
            parts.append(current_node.attr)
            current_node = current_node.value
        
        if isinstance(current_node, ast.Name):
            root_name = current_node.id
            if root_name in self.aliases:
                root_name = self.aliases[root_name]
            
            full_path = ".".join([root_name] + list(reversed(parts)))
            if any(full_path.startswith(lib) for lib in self.libraries):
                return full_path
        return None

    def _find_used_library_from_annotation(self, node: ast.expr) -> typing.Optional[str]:
        if isinstance(node, ast.Name):
            name = node.id
            if name in self.aliases:
                return self.aliases[name]
            if name in self.libraries:
                return name
            return None
        
        if isinstance(node, ast.Attribute):
            base_path = self._find_used_library_from_annotation(node.value)
            if base_path:
                return f"{base_path}.{node.attr}"
            return None
            
        if isinstance(node, ast.Subscript):
            return self._find_used_library_from_annotation(node.value)
            
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            for element in node.elts:
                lib = self._find_used_library_from_annotation(element)
                if lib:
                    return lib # Only returns the first one found
        return None

def analyze_python_file_full(code_string):
    try:
        tree = ast.parse(code_string)
    except SyntaxError:
        return ("GRAMMAR ERROR!", None, None, None, None, None)

    all_libraries = set()
    library_aliases = {}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                all_libraries.add(alias.name)
                if alias.asname:
                    library_aliases[alias.asname] = alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                all_libraries.add(node.module)
                parts = node.module.split('.')
                for i in range(1, len(parts) + 1):
                    all_libraries.add('.'.join(parts[:i]))
            for alias in node.names:
                if alias.asname:
                    library_aliases[alias.asname] = f"{node.module}.{alias.name}"
                else:
                    library_aliases[alias.name] = f"{node.module}.{alias.name}"

    visitor = LibraryVisitor_full(all_libraries, library_aliases)
    visitor.visit(tree)

    def filter_libs(libs_set):
        to_remove = set()
        for lib1 in libs_set:
            for lib2 in libs_set:
                if lib1 != lib2 and lib2.startswith(lib1 + '.'):
                    to_remove.add(lib1)
        return libs_set - to_remove

    final_global_libs = filter_libs(visitor.global_libraries)
    
    final_func_libs = {}
    for func, libs in visitor.function_libraries.items():
        filtered_libs = filter_libs(libs)
        if filtered_libs:
            final_func_libs[func] = sorted(list(filtered_libs))

    final_class_libs = {}
    for cls, libs in visitor.class_libraries.items():
        filtered_libs = filter_libs(libs)
        if filtered_libs:
            final_class_libs[cls] = sorted(list(filtered_libs))
            
    return (
        sorted(list(all_libraries)), 
        sorted(list(visitor.all_functions)), 
        final_func_libs, 
        sorted(list(final_global_libs)),
        sorted(list(visitor.all_classes)),
        final_class_libs
    )

def analyze_code_dict_full(code_dict):
    library_results_temp = {}
    for p, c in code_dict.items():
        analysis_output = analyze_python_file_full(c)
        if analysis_output[0]!="GRAMMAR ERROR!":
            library_results_temp[p] = analysis_output

    return library_results_temp

def match_lists_original(A,B):

    matched_A = [a for a in A if len([b for b in B if a.startswith(b + '.')]) > 0]
    
    return {l for l in matched_A if l!=None}


def recheck_libraries_full(library_results_temp, pylist_dict):
    library_results = {}
    for scpt, rs in library_results_temp.items():
        library_list = [l for l in rs[0] if not pylist_dict.get(l+'.py', False)]
        
        function_library_list = {f:[l for l in lst if not pylist_dict.get(l+'.py', False)] for f, lst in rs[2].items()}
        function_library_list_matched = {f:match_lists_original(v, library_list) for f,v in function_library_list.items()}
        
        global_library_list = [l for l in rs[3] if not pylist_dict.get(l+'.py', False)]
        global_library_list_matched = match_lists_original(global_library_list, library_list)

        class_library_list = {f:[l for l in lst if not pylist_dict.get(l+'.py', False)] for f, lst in rs[5].items()}
        class_library_list_matched = {f:match_lists_original(v, library_list) for f,v in class_library_list.items()}

        library_results[scpt] = (library_list, rs[1], function_library_list_matched, global_library_list_matched, rs[4], class_library_list_matched)
        
    return library_results


def match_lists_root(A,B):

    matched_B = [b for b in B if len([a for a in A if a.startswith(b + '.')] > 0)]
    
    return {l for l in matched_B if l!=None}


def get_libraries_root(library_results_temp):
    library_results = {}
    for scpt, rs in library_results_temp.items():
        library_set = set([l.split('.')[0] for l in rs[0]])
        library_list = list(library_set)
        
        function_library_list = {f:list(set([l.split('.')[0] for l in lst]).intersection(library_set)) for f, lst in rs[2].items()}
        
        global_library_list = list(set([l.split('.')[0] for l in rs[3]]).intersection(library_set))

        class_library_list = {f:list(set([l.split('.')[0] for l in lst]).intersection(library_set)) for f, lst in rs[5].items()}

        library_results[scpt] = (library_list, rs[1], function_library_list, global_library_list, rs[4], class_library_list)
        
    return library_results

##Read and Load files

#import pickle5 as pickle
import pickle
def save_obj(obj, name, data_path_save = 'obj/'):
    with open(data_path_save + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        

def load_obj(name, data_path_load = 'obj/'):
    with open(data_path_load + name + '.pkl', 'rb') as f:
        return pickle.load(f)
