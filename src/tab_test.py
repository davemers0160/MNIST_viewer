# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 16:04:11 2019

@author: victoria.lockridge
"""

import panel as pn
import numpy as np
import pandas as pd
import holoviews as hv
hv.extension('matplotlib')
from holoviews import opts

distributions = {
    'all logs': np.random.normal,
    'inlier': np.random.lognormal,
    'outlier': np.random.exponential
}

checkboxes = pn.widgets.ToggleGroup(options=distributions, behavior='radio')
slider = pn.widgets.IntSlider(name='Track quality', value=50, start=0, end=100)

@pn.depends(checkboxes.param.value, slider.param.value)
def tabs(distribution, n):
    values = hv.Dataset(distribution(size=n), 'Threat level')
    return pn.Tabs(
        ('Plot', values.hist(adjoin=False).opts(
            #responsive=True,
            # max_height=500,
            padding=0.1
        )),
        ('Summary', values.dframe().describe().T),
        ('Table', hv.Table(values)),
    )

pn.Row(pn.Column('### Emitter RF Values', checkboxes, slider), tabs).show() #.servable()