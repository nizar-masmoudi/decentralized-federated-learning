from dash import html, dcc, callback, Input, Output, MATCH
from dashboard.figures import ActivityFigure
import uuid


# noinspection PyMethodParameters
class ActivityAIO(html.Div):
    class ID:
        graph = lambda aio_id: {
            'component': 'ActivityAIO',
            'subcomponent': 'graph',
            'aio_id': aio_id
        }
    ID = ID

    def __init__(self, aio_id: str = None):
        if aio_id is None:
            aio_id = str(uuid.uuid4())

        super().__init__([
            html.H2('Client Activity Rate', className='font-semibold text-lg'),
            dcc.Graph(
                id=self.ID.graph(aio_id),
                figure=ActivityFigure(),
                animate=True,
                style={'width': '100%', 'height': '100%'}
            )
        ], className='relative flex flex-col w-1/3 p-7 h-96 bg-white rounded-lg '
                     'shadow-[0px_4px_20px_rgba(237,237,237,0.5)]')

    @callback(
        Output(ID.graph(MATCH), 'figure'),
        Input('local-storage', 'data'),
        prevent_initial_callbacks=True
    )
    def update_activity(data: dict):
        if data == {}:
            return ActivityFigure()

        x = list(range(1, len(data['clients'][0]['activity']) + 1))
        ys = {
            'Active clients': [],
            'Total clients': [len(data['clients'])] * len(x)
        }
        for i in x:
            ys['Active clients'].append(sum([client['activity'][i - 1] for client in data['clients']]))

        figure = ActivityFigure(x, ys)
        figure.layout.yaxis.range = [-1, len(data['clients']) + 6]
        figure.layout.yaxis.tickvals = list(range(0, len(data['clients']) + 2, 5))
        figure.layout.xaxis.tickvals = list(range(1, len(x), 1))

        return figure
