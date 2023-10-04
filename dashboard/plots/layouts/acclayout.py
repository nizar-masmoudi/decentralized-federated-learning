import plotly.graph_objects as go
import numpy as np


class AccLayout(go.Layout):
    def __init__(self):
        super().__init__()
        self.autosize = True
        self.paper_bgcolor = 'rgba(0, 0, 0, 0)'
        self.plot_bgcolor = 'rgba(0, 0, 0, 0)'
        self.margin = dict(l=0, r=0, t=0, b=0)
        self.title = dict(
            text='Accuracy',
            x=.5,
            y=.95,
            font=dict(
                color='#adbac7',
                family='Poppins, sans-serif',
                size=18,
            ),
        )
        self.xaxis = dict(
            color='rgba(68, 76, 86, 1)',
            tickfont=dict(
                color='#adbac7',
                family='Poppins, sans-serif'
            ),
            gridcolor='rgba(68, 76, 86, 1)',
            griddash='solid',
            zeroline=False,
            fixedrange=True,
            mirror=True,
            showspikes=False,
            ticklabelposition='inside right',
        )
        self.yaxis = dict(
            color='rgba(68, 76, 86, 1)',
            tickfont=dict(
                color='#adbac7',
                family='Poppins, sans-serif'
            ),
            gridcolor='rgba(68, 76, 86, 1)',
            griddash='solid',
            zeroline=False,
            fixedrange=True,
            mirror=True,
            ticklabelposition='inside top',
            range=[-.2, 1.2],
            tickvals=list(np.arange(0, 1.2, .2))
        )
        self.hoverlabel = dict(
            bgcolor='#1a2226',
            bordercolor='#1a2226',
            font_family='Poppins, sans-serif',
            font_color='white',
            font_size=12,
        )
        self.hovermode = 'x unified'
        self.legend = dict(
            font=dict(
                color='#adbac7',
                family='Poppins, sans-serif',
            ),
            bgcolor='#1c2128',
            bordercolor='#444c56',
            borderwidth=1,
            x=.85,
            y=.2
        )
