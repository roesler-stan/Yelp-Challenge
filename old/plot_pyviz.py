#!/usr/bin/env python
"""
References.
-----------
[1] Donald E. Knuth,
    "The Stanford GraphBase: A Platform for Combinatorial Computing",
    ACM Press, New York, 1993.
[2] http://www-cs-faculty.stanford.edu/~knuth/sgb.html
[3] https://github.com/pygraphviz
"""

import warnings
import pygraphviz as pgv
import math
import re
import gzip

def main():
    # ignore Graphviz warning messages
    warnings.simplefilter('ignore', RuntimeWarning)

    infile = '../Data/edges1k.csv'
    G=graph(infile)
    print("Loaded data file.")

    G.write("miles.dot")
    print("Wrote dot file.")
    
    G.draw("miles.png",prog='neato',args='-n2')
    print("Wrote png file".)

def graph(infile):
    """Return a graph."""
    G = pgv.AGraph(name='words')
    G.node_attr['shape']='circle'
    G.node_attr['fixedsize']='true'
    G.node_attr['fontsize']='8'
    G.node_attr['style']='filled'
    G.graph_attr['outputorder']='edgesfirst'
    G.graph_attr['label']="words"
    G.graph_attr['ratio']='1.0'
    G.edge_attr['color']='#1100FF'
    G.edge_attr['style']='setlinewidth(2)'

    nodes=[]
    with open(infile, 'r') as f:
        for line in f:
            if line.startswith("*"): # skip comments
                continue
            numfind=re.compile("^\d+") 

            if numfind.match(line): # this line is distances
                dist=line.split()
                for d in dist:
                    if float(d) < 300: # connect if closer then 300 miles
                        G.add_edge(city,nodes[i])
                    i=i+1
            else: # this line is a city, position, population
                i=1
                (city,coordpop)=line.split("[")
                nodes.insert(0,city)
                (coord,pop)=coordpop.split("]")
                (y,x)=coord.split(",")
                G.add_node(city)
                n=G.get_node(city)
                # assign positions, scale to be something reasonable in points
                n.attr['pos']="%f,%f)"%(-(float(x)-7000)/10.0,(float(y)-2000)/10.0)
                # assign node size, in sqrt of 1,000,000's of people 
                d=math.sqrt(float(pop)/1000000.0)
                n.attr['height']="%s"%(d/2)
                n.attr['width']="%s"%(d/2)
                # assign node color
                n.attr['fillcolor']="#0000%2x"%(int(d*256))
                # empty labels
                n.attr['label']=' '

    return G

if __name__ == '__main__':
    main()