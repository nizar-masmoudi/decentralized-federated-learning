import json
import os

import numpy as np
import plotly.graph_objects as go

from dashboard.figures.layouts import LinePlotLayout


class AggEvalFigure(go.Figure):
    def __init__(self):

        lines = []
        files = [os.path.join('logs/', file) for file in os.listdir('logs/')]
        for file in files:
            with open(file) as json_file:
                data = json.load(json_file)
                avg_loss = np.array([client['test/loss'] for client in data['clients']])
                lines.append(
                    go.Scatter(
                        x=list(range(1, len(data['clients'][0]['activity']) + 1)),
                        y=avg_loss.mean(0),
                        mode='markers+lines',
                        hovertemplate='Loss = %{y:.3f}<extra></extra>',
                        name=data['name'],
                    )
                )

        layout = LinePlotLayout()
        layout.yaxis.range = [0, 1]
        layout.yaxis.tickvals = list(np.arange(.2, 1, .2))
        layout.legend.x = .5
        layout.legend.y = 0
        layout.legend.entrywidth = 0
        layout.legend.entrywidthmode = 'pixels'
        layout.legend.xanchor = 'center'
        layout.legend.orientation = 'h'
        super().__init__(lines, layout)
