from dash import html, dcc
import uuid
from dashboard.plots import ClassDistPlot


class ClassDistAIO(html.Div):
    class ID:
        parent = lambda aio_id: {
            'component': 'ClassDistAIO',
            'subcomponent': 'parent',
            'aio_id': aio_id
        }
        graph = lambda aio_id: {
            'component': 'ClassDistAIO',
            'subcomponent': 'graph',
            'aio_id': aio_id
        }
    ID = ID

    def __init__(self, aio_id: str = None):
        if aio_id is None:
            aio_id = str(uuid.uuid4())

        super().__init__([
            dcc.Graph(
                self.ID.graph(aio_id),
                figure=ClassDistPlot(),
                animate=True,
                style={'width': '100%', 'height': '100%'}
            )
        ], self.ID.parent(aio_id), className='row-span-1 col-span-1 bg-transparent border-2 '
                                             'border-[#444c56] rounded-lg p-4')
