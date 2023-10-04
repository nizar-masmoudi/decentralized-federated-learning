from dash import html, dcc
import uuid
from dashboard.plots import LossPlot


class LossPlotAIO(html.Div):
    class ID:
        graph = lambda aio_id: {
            'component': 'LossPlotAIO',
            'subcomponent': 'graph',
            'aio_id': aio_id
        }
    ID = ID

    def __init__(self, aio_id: str = None):
        if aio_id is None:
            aio_id = str(uuid.uuid4())

        super().__init__([
            dcc.Graph(
                id=self.ID.graph(aio_id),
                figure=LossPlot(),
                animate=True,
                style={'width': '100%', 'height': '100%'}
            ),
        ], className='col-span-1 h-96 bg-transparent border-2 border-[#444c56] rounded-lg p-4')
