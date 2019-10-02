# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 15:56:45 2019

@author: victoria.lockridge
"""

import panel as pn
import hvplot.pandas

from bokeh.sampledata.autompg import autompg_clean

autompg_clean.columns = ['pulse_width', 'cyl', 'PRI','true_bearing','frequency','elevation','yr','origin', 'name','mfr']
quant = ['frequency','pulse_width', 'PRI','true_bearing','elevation']
cat = [None, 'yr', 'cyl', 'yr']
combined = quant+cat[1:]

x = pn.widgets.Select(name='x', value='PRI', options=combined)
y = pn.widgets.Select(name='y', value='pulse_width', options=combined)
color = pn.widgets.Select(name='color', options=combined)
facet = pn.widgets.Select(name='facet', options=cat)
ev = pn.widgets.Select(name='event', options=['Formidable Shield'])
ag = pn.widgets.Select(name='AD algorithm', options=['Proximity-based'])

@pn.depends(x.param.value, y.param.value, color.param.value, facet.param.value)
def plot(x, y, color, facet):
    cmap = 'Category10' if color in cat else 'viridis'
    return autompg_clean.hvplot.scatter(
        x, y, color=color, by=facet, subplots=True, padding=0.1, width=700, height=None, cmap=cmap).opts(
        'Scatter', min_height=300, responsive=True)

pn.Column(
    pn.layout.Spacer(height = 20),
    plot,
    pn.Row(pn.WidgetBox(x, y, color), pn.WidgetBox(facet, ev, ag)),
    width_policy='max'
).show()
#).servable()