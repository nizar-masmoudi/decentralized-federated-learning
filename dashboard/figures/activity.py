import plotly.graph_objects as go
from dashboard.figures.layouts import LinePlotLayout


class ActivityFigure(go.Figure):
    def __init__(self, x: list = None, ys: dict = None):
        if x is None:
            x = []
        if ys is None:
            ys = {}
        data = []
        if ys:
            for name, y in ys.items():
                data.append(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode='markers+lines',
                        hovertemplate='Clients = %{y}<extra></extra>',
                        name=name,
                        fill='tonexty',
                        fillcolor='rgba(100, 204, 197, .2)' if name == 'Active clients' else 'rgba(255, 105, 105, .2)',
                        line=dict(color='#FF6969' if name == 'Total clients' else '#64CCC5')
                    )
                )

        layout = LinePlotLayout()
        layout.legend.x = .5
        layout.legend.y = 0
        layout.legend.entrywidth = 0
        layout.legend.entrywidthmode = 'pixels'
        layout.legend.xanchor = 'center'
        layout.legend.orientation = 'h'
        super().__init__(data, layout)
