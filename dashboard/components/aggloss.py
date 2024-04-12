import uuid

from dash import html, dcc

from dashboard.figures import AggEvalFigure


# noinspection PyMethodParameters
class AggLossAIO(html.Div):
    class ID:
        graph = lambda aio_id: {
            'component': 'AggLossAIO',
            'subcomponent': 'graph',
            'aio_id': aio_id
        }
    ID = ID

    def __init__(self, aio_id: str = None):
        if aio_id is None:
            aio_id = str(uuid.uuid4())

        super().__init__([
            html.H2('Global Accuracy', className='font-semibold text-lg'),
            dcc.Graph(
                id=self.ID.graph(aio_id),
                figure=AggEvalFigure(),
                animate=True,
                style={'width': '100%', 'height': '100%'}
            ),
        ], className='relative flex flex-col w-1/2 p-7 h-96 bg-white rounded-lg '
                     'shadow-[0px_4px_20px_rgba(237,237,237,0.5)]')
