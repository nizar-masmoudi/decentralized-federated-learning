import plotly.graph_objects as go
from dashboard.plots.layouts import LossLayout


class LossPlot(go.Figure):
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
                        hovertemplate='%{y:.3f}<extra></extra>',
                        name=f'Client {id_}'
                    )
                )
        super().__init__(data, LossLayout())
