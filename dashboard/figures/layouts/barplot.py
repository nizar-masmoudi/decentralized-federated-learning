import plotly.graph_objects as go


class BarPlotLayout(go.Layout):
    def __init__(self):
        super().__init__()
        self.autosize = True
        self.paper_bgcolor = 'rgba(0, 0, 0, 0)'
        self.plot_bgcolor = 'rgba(0, 0, 0, 0)'
        self.margin = dict(l=0, r=0, t=0, b=0)
        self.xaxis = dict(
            color='rgba(68, 76, 86, 1)',
            tickfont=dict(
                color='#7B91B0',
                family='Poppins, sans-serif'
            ),
            showgrid=False,
            zeroline=False,
            showspikes=False,

        )
        self.yaxis = dict(
            color='rgba(68, 76, 86, 1)',
            tickfont=dict(
                color='#444c56',
                family='Poppins, sans-serif'
            ),
            showgrid=False,
            zeroline=False,
            fixedrange=True,
            mirror=True,
            showticklabels=False,
        )
        self.hoverlabel = dict(
            bgcolor='white',
            bordercolor='#EFF1F3',
            font_family='Poppins, sans-serif',
            font_color='#444A6D',
            font_size=12,
        )
        self.hovermode = 'x unified'
