import plotly.graph_objects as go


class ClassDistLayout(go.Layout):
    def __init__(self):
        super().__init__()
        self.autosize = True
        self.paper_bgcolor = 'rgba(0, 0, 0, 0)'
        self.plot_bgcolor = 'rgba(0, 0, 0, 0)'
        self.margin = dict(l=0, r=0, t=0, b=0)
        self.xaxis = dict(
            color='rgba(68, 76, 86, 1)',
            tickfont=dict(
                color='#444c56',
                family='Poppins, sans-serif'
            ),
            showgrid=False,
            zeroline=False,
            showspikes=False,
            showticklabels=False,
            ticklabelposition='inside top',
            range=[0, 11],
        )
        self.yaxis = dict(
            color='rgba(68, 76, 86, 1)',
            tickfont=dict(
                color='#444c56',
                family='Poppins, sans-serif'
            ),
            gridcolor='rgba(68, 76, 86, 1)',
            griddash='solid',
            zeroline=False,
            fixedrange=True,
            mirror=True,
            showticklabels=False,
            range=[0, 3500],
        )
        self.hoverlabel = dict(
            bgcolor='#1a2226',
            bordercolor='#1a2226',
            font_family='Poppins, sans-serif',
            font_color='white',
            font_size=12,
        )
        self.hovermode = 'x unified'
