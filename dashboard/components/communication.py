from dash import html, dcc, callback, Input, Output, MATCH
from dashboard.figures import CommunicationEnergyFigure
import numpy as np
import uuid


# noinspection PyMethodParameters
class CommunicationEnergyAIO(html.Div):
    class ID:
        graph = lambda aio_id: {
            'component': 'CommunicationEnergyAIO',
            'subcomponent': 'graph',
            'aio_id': aio_id
        }
    ID = ID

    def __init__(self, aio_id: str = None):
        if aio_id is None:
            aio_id = str(uuid.uuid4())

        super().__init__([
            html.H2('Communication Energy', className='font-semibold text-lg'),
            dcc.Graph(
                id=self.ID.graph(aio_id),
                figure=CommunicationEnergyFigure(),
                animate=True,
                style={'width': '100%', 'height': '100%'}
            )
        ], className='relative flex flex-col w-1/3 p-7 h-96 bg-white rounded-lg '
                     'shadow-[0px_4px_20px_rgba(237,237,237,0.5)]')

    @callback(
        Output(ID.graph(MATCH), 'figure'),
        Input('local-storage', 'data'),
    )
    def update_communication_energy(data: dict):
        if not data:
            return CommunicationEnergyFigure()

        x = list(range(1, len(data['1']['activity']) + 1))

        energy_arr = np.array([data[id_]['comm_energy'] for id_ in data.keys()])
        avg_energy = energy_arr.mean(axis=0)
        std_energy = energy_arr.std(axis=0)
        ys = {
            'Mean': avg_energy,
            # 'Standard deviation': std_energy
        }

        figure = CommunicationEnergyFigure(x, ys)
        return figure
