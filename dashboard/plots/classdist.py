import plotly.graph_objects as go
from dashboard.plots.layouts import ClassDistLayout


class ClassDistPlot(go.Figure):
    def __init__(self, x: list = None, y: list = None):
        if x is None:
            x = []
        if y is None:
            y = []

        data = go.Bar(
            x=x,
            y=y,
            text=x,
            textfont=dict(
                family='Poppins, sans-serif',
                color='#adbac7'
            ),
            marker=dict(
                color='#176B87',
                line=dict(width=0),
            ),
            hovertemplate='Count = %{y}<extra></extra>',
        )
        super().__init__(data, ClassDistLayout())
