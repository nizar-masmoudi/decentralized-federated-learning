import plotly.graph_objects as go
from dashboard.figures.layouts import LinePlotLayout


class CommunicationEnergyFigure(go.Figure):
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
                        customdata=[v*1e6 for v in y],
                        mode='markers+lines',
                        hovertemplate='Energy = %{customdata:.0f} µJ<extra></extra>',
                    )
                )

        layout = LinePlotLayout()

        super().__init__(data, layout)
