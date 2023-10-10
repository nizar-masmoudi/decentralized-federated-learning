from dash import html, dcc, callback, Input, Output, MATCH
from dashboard.figures import EvaluationFigure
import numpy as np
import uuid


# noinspection PyMethodParameters
class LocalAccuracyAIO(html.Div):
    class ID:
        graph = lambda aio_id: {
            'component': 'LocalAccuracyAIO',
            'subcomponent': 'graph',
            'aio_id': aio_id
        }
        dropdown = lambda aio_id: {
            'component': 'LocalAccuracyAIO',
            'subcomponent': 'dropdown',
            'aio_id': aio_id
        }
    ID = ID

    def __init__(self, aio_id: str = None):
        if aio_id is None:
            aio_id = str(uuid.uuid4())

        super().__init__([
            html.H2('Local Accuracy', className='font-semibold text-lg'),
            dcc.Graph(
                id=self.ID.graph(aio_id),
                figure=EvaluationFigure(),
                animate=True,
                style={'width': '100%', 'height': '100%'}
            ),
            dcc.Dropdown(
                id=self.ID.dropdown(aio_id),
                options=['Training', 'Validation'],
                value='Training',
                className='absolute right-7 top-7 text-xs w-28 focus:outline-none focus-visible:outline-none '
                          'border border-[#EFF1F3] rounded-md',
                searchable=False,
                clearable=False,
            )
        ], className='relative flex flex-col w-1/2 p-7 h-96 bg-white rounded-lg '
                     'shadow-[0px_4px_20px_rgba(237,237,237,0.5)]')

    @callback(
        Output(ID.graph(MATCH), 'figure'),
        Input('local-storage', 'data'),
        Input(ID.dropdown(MATCH), 'value'),
    )
    def update_local_accuracy(data: dict, subset: str):
        if not data:
            return EvaluationFigure()

        key = 'tacc' if subset == 'Training' else 'vacc'
        x = list(range(1, len(data['1']['tacc']) + 1))
        ys = {id_: data[str(id_)][key] for id_ in range(1, len(data) + 1)}

        figure = EvaluationFigure(x, ys)
        figure.layout.yaxis.range = [-.2, 1.2]
        figure.layout.yaxis.tickvals = list(np.arange(0, 1.2, .2))

        return figure
