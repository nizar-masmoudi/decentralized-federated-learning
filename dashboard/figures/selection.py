import plotly.graph_objects as go
from dashboard.figures.layouts import LinePlotLayout


class SelectionFigure(go.Figure):
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
                        hovertemplate='Rate = %{y}<extra></extra>',
                        name=name,
                        line=dict(color='#0095FF')
                    )
                )

        layout = LinePlotLayout()
        layout.showlegend = False
        super().__init__(data, layout)
