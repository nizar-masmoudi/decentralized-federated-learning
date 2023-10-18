import plotly.graph_objects as go
from dashboard.figures.layouts import LinePlotLayout
import numpy as np
from typing import Literal


class EvaluationFigure(go.Figure):
    def __init__(self, x: list = None, ys: dict = None, aggregate: Literal['mean', 'median'] = None):
        if x is None:
            x = []
        if ys is None:
            ys = {}
        data = []

        if aggregate:
            if aggregate == 'Mean':
                agg = np.array(list(ys.values())).mean(0)
                lower = agg - np.array(list(ys.values())).std(0)
                upper = agg + np.array(list(ys.values())).std(0)
                customdata = ['Std = {:.3f}'.format(std) for std in upper - agg]
            else:
                agg = np.median(np.array(list(ys.values())), 0)
                lower = np.array(list(ys.values())).min(0)
                upper = np.array(list(ys.values())).max(0)
                customdata = ['Min = {:.3f}<br>Max = {:.3f}'.format(lower[i], upper[i]) for i in range(len(agg))]

            ys = {
                'agg': agg,
                'lower': lower,
                'upper': upper,
            }

            if ys:
                data = [
                    go.Scatter(
                        x=x,
                        y=ys['upper'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip',
                    ),
                    go.Scatter(
                        x=x,
                        y=ys['agg'],
                        mode='markers+lines',
                        fill='tonexty',
                        fillcolor='rgba(20, 108, 148, .2)',
                        line=dict(color='#146C94'),
                        customdata=customdata,
                        hovertemplate='Aggregate = %{y:.3f}<br>'
                                      '%{customdata}'
                                      '<extra></extra>',
                        name='Aggregate',
                    ),
                    go.Scatter(
                        x=x,
                        y=ys['lower'],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(20, 108, 148, .2)',
                        name='Deviation',
                        hoverinfo='skip',
                    )
                ]

        else:
            if ys:
                for name, y in ys.items():
                    data.append(
                        go.Scatter(
                            x=x,
                            y=y,
                            mode='markers+lines',
                            hovertemplate='Loss = %{y:.3f}<extra></extra>',
                            name=name,
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
