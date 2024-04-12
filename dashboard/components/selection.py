import uuid

import numpy as np
from dash import html, dcc, callback, Input, Output, MATCH

from dashboard.figures import SelectionFigure


# noinspection PyMethodParameters
class SelectionAIO(html.Div):
    class ID:
        graph = lambda aio_id: {
            'component': 'SelectionAIO',
            'subcomponent': 'graph',
            'aio_id': aio_id
        }
    ID = ID

    def __init__(self, aio_id: str = None):
        if aio_id is None:
            aio_id = str(uuid.uuid4())

        super().__init__([
            html.H2('Peer Selection Rate', className='font-semibold text-lg'),
            dcc.Graph(
                id=self.ID.graph(aio_id),
                figure=SelectionFigure(),
                animate=True,
                style={'width': '100%', 'height': '100%'}
            ),
        ], className='relative flex flex-col w-[calc(100%-25%-33%)] p-7 h-96 bg-white rounded-lg '
                     'shadow-[0px_4px_20px_rgba(237,237,237,0.5)]')

    @callback(
        Output(ID.graph(MATCH), 'figure'),
        Input('local-storage', 'data'),
        prevent_initial_callbacks=True
    )
    def update_selection(data: dict):
        if data == {}:
            return SelectionFigure()

        x = list(range(1, len(data['clients'][0]['activity']) + 1))
        ys = {'Rate': []}
        rates = []
        for client in data['clients']:
            # Filter neighbors according to client activity
            activity = np.array(client['activity'])
            neighbors_count = np.array(list(map(lambda l: len(l), client['neighbors'])))[activity]
            peers_count = np.array(list(map(lambda l: len(l), client['peers'])))

            rates.append(np.divide(peers_count[neighbors_count != 0], neighbors_count[neighbors_count != 0]))

        rates = np.array(rates)
        ys['Rate'] = rates.mean(0)

        figure = SelectionFigure(x, ys)
        figure.layout.xaxis.range = [1, len(x)]
        figure.layout.yaxis.range = [-.2, 1.2]
        figure.layout.yaxis.tickvals = list(np.arange(0, 1.2, .2))
        figure.layout.xaxis.tickvals = list(range(1, len(x), 1))

        return figure
