#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nltk.parse.dependencygraph import *
from ufo.word import *

def build_graph(sent, tmpf='tmpdep.txt'):
    '''
    Build a graph for the sentence sent using the class DependencyGraph from NLTK
    Return also the list of pos occurring in the doc ie the one token treelets
    '''
    pos_treelet = []
    o = open(tmpf, 'w')
    for wrd in sent:
        if not isinstance(wrd, Multiword): 
            form = wrd.form
            pos = wrd.upos
            head = wrd.head #int
            deprel = 'root' if head == 0 else wrd.deprel
            pos_treelet.append(pos)
            o.write('\t'.join([form, pos, str(head), deprel])+'\n')
    o.close()

    depgraph = DependencyGraph()
    g = depgraph.load(tmpf, cell_separator='\t', top_relation_label='root')
    if len(g) != 1:
        print(open(tmp).read())
        sys.exit("More than one graph?")
    # dotg = g[0].to_dot()#should allow to visualize the graph with graphivz
    return g[0], pos_treelet

 
def get_three_tok_treelets(dg):
    '''
    Three tokens treelets are either:
    - one node dominating two nodes  DEP1 -> r1 -> NODE <- r2 <- Dep2
    - a chain: one node dominating another node that dominates another node NODE -> r1 -> Dep1 -> r2 -> Dep2
    '''
    three_tok_treelets = []
    for n in dg.nodes:
        node = dg.get_by_address(n)
        node_word = node['word']
        node_pos = node['tag']
        node_deps = node['deps']
        node_rel = node['rel']
        new_deps = flat_dep_list(node_deps)

        # not taking into account the fake root added
        if len(new_deps) > 1 and node_word != None:
            new_deps = sorted(new_deps, key=lambda x: x[1])
            # --> head dominating two nodes
            for i, (deprel1, dep_add1) in enumerate(new_deps[:-1]):
                for (deprel2, dep_add2) in new_deps[i+1:]:
                    three_tok_treelets.append(dg.get_by_address(dep_add1)[
                                              'tag']+'->'+deprel1+'->'+node_pos+'<-'+deprel2+'<-'+dg.get_by_address(dep_add2)['tag'])

            # --> chains: head -> dep1 -> dep2
            for i, (deprel1, dep_add1) in enumerate(new_deps):
                # Look for dependents of dep1
                dependents = get_deps(dg, dep_add1)
                if len(dependents) > 0:
                    for (deprel2, dep_add2) in dependents:
                        three_tok_treelets.append(node_pos+'->'+deprel1+'->'+dg.get_by_address(
                            dep_add1)['tag']+'->'+deprel2+'->'+dg.get_by_address(dep_add2)['tag'])
    return three_tok_treelets

def flat_dep_list(node_deps):
    new_deps = []
    for (deprel, dep_add) in node_deps.items():
        for a in dep_add:
            new_deps.append((deprel, a))
    return new_deps

def get_deps(dg, add):
    '''
    Look for dependents of node at the address add
    '''
    deps = []
    for n in dg.nodes:
        if n == add:
            node = dg.get_by_address(n)
            node_deps = node['deps']
            new_deps = flat_dep_list(node_deps)
            return new_deps
    return []