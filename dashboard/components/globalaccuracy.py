from dash import html, dcc, callback, Input, Output, MATCH
from dashboard.figures import EvaluationFigure
import numpy as np
import uuid


# noinspection PyMethodParameters
class GlobalAccuracyAIO(html.Div):
    class ID:
        graph = lambda aio_id: {
            'component': 'GlobalAccuracyAIO',
            'subcomponent': 'graph',
            'aio_id': aio_id
        }
        aggregate = lambda aio_id: {
            'component': 'GlobalAccuracyAIO',
            'subcomponent': 'aggregate',
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
                figure=EvaluationFigure(),
                animate=True,
                style={'width': '100%', 'height': '100%'}
            ),
            html.Span([
                dcc.Dropdown(
                    id=self.ID.aggregate(aio_id),
                    options=['None', 'Mean', 'Median'],
                    value='None',
                    className='w-28 focus:outline-none focus-visible:outline-none '
                              'border border-[#EFF1F3] rounded-md',
                    searchable=False,
                    clearable=False,
                ),
            ], className='flex space-x-2 absolute right-7 top-7 text-xs'),
        ], className='relative flex flex-col w-1/2 p-7 h-96 bg-white rounded-lg '
                     'shadow-[0px_4px_20px_rgba(237,237,237,0.5)]')

    @callback(
        Output(ID.graph(MATCH), 'figure'),
        Input('local-storage', 'data'),
        Input(ID.aggregate(MATCH), 'value'),
        prevent_initial_callbacks=True
    )
    def update_global_accuracy(data: dict, aggregate: str):
        if data == {}:
            return EvaluationFigure()

        x = list(range(1, len(data['clients'][0]['activity']) + 1))
        ys = {'Client {}'.format(client['id']): client['test/accuracy'] for client in data['clients']}

        if aggregate == 'None':
            aggregate = None
        figure = EvaluationFigure(x, ys, aggregate)
        figure.layout.yaxis.range = [-.2, 1.2]
        figure.layout.yaxis.tickvals = list(np.arange(0, 1.2, .2))

        return figure
