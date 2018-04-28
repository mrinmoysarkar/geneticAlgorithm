#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 10:15:47 2018

@author: mrinmoy
"""

from graph_tool.all import *

g = Graph()
v1 = g.add_vertex()
v2 = g.add_vertex()

e = g.add_edge(v1, v2)

graph_draw(g, vertex_text=g.vertex_index, vertex_font_size=18,\
           output_size=(200, 200), output="two-nodes.png")