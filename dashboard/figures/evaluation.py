import plotly.graph_objects as go
from dashboard.figures.layouts import LinePlotLayout
import numpy as np


class EvaluationFigure(go.Figure):
    def __init__(self, x: list = None, ys: dict = None):
        if x is None:
            x = []
        if ys is None:
            ys = {}
        data = []
        if ys:
            for id_, y in ys.items():
                data.append(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode='markers+lines',
                        hovertemplate='Loss = %{y:.3f}<extra></extra>',
                        name=f'Client {id_}',
                    )
                )

        layout = LinePlotLayout()
        layout.yaxis.range = [-.5, 3]
        layout.yaxis.tickvals = list(np.arange(0, 3, .5))
        layout.legend.x = .5
        layout.legend.y = 0
        layout.legend.entrywidth = 0
        layout.legend.entrywidthmode = 'pixels'
        layout.legend.xanchor = 'center'
        layout.legend.orientation = 'h'

        super().__init__(data, layout)
