from dash import html, dcc, callback, Input, Output, MATCH
from dashboard.figures import EvaluationFigure
import uuid


# noinspection PyMethodParameters
class GlobalLossAIO(html.Div):
    class ID:
        graph = lambda aio_id: {
            'component': 'GlobalLossAIO',
            'subcomponent': 'graph',
            'aio_id': aio_id
        }
    ID = ID

    def __init__(self, aio_id: str = None):
        if aio_id is None:
            aio_id = str(uuid.uuid4())

        super().__init__([
            html.H2('Global Loss', className='font-semibold text-lg'),
            dcc.Graph(
                id=self.ID.graph(aio_id),
                figure=EvaluationFigure(),
                animate=True,
                style={'width': '100%', 'height': '100%'}
            ),
        ], className='relative flex flex-col w-1/2 p-7 h-96 bg-white rounded-lg '
                     'shadow-[0px_4px_20px_rgba(237,237,237,0.5)]')

    @callback(
        Output(ID.graph(MATCH), 'figure'),
        Input('local-storage', 'data'),
    )
    def update_globaloss(data: dict):
        if not data:
            return EvaluationFigure()

        x = list(range(1, len(data['1']['activity']) + 1))
        ys = {id_: data[str(id_)]['sloss'] for id_ in range(1, len(data) + 1)}

        figure = EvaluationFigure(x, ys)

        return figure
