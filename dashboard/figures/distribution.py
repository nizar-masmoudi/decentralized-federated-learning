import plotly.graph_objects as go
from dashboard.figures.layouts import BarPlotLayout


class DistributionFigure(go.Figure):
    def __init__(self, x: list = None, y: list = None):
        if x is None:
            x = []
        if y is None:
            y = []

        data = go.Bar(
            x=x,
            y=y,
            marker=dict(
                color='#FFCF00',
                line=dict(width=0),
            ),
            hovertemplate='Count = %{y}<extra></extra>',
        )

        layout = BarPlotLayout()
        layout.xaxis.range = [-1, 10]
        layout.yaxis.range = [0, 1500]
        layout.xaxis.tickvals = list(range(0, 10))
        super().__init__(data, layout)
