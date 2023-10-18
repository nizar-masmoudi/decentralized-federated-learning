from dash import html, dcc, callback, Input, Output, MATCH
from dashboard.figures import DistributionFigure
import uuid


# noinspection PyMethodParameters
class DistributionAIO(html.Div):
    class ID:
        graph = lambda aio_id: {
            'component': 'DistributionAIO',
            'subcomponent': 'graph',
            'aio_id': aio_id
        }
        dropdown = lambda aio_id: {
            'component': 'DistributionAIO',
            'subcomponent': 'dropdown',
            'aio_id': aio_id
        }
    ID = ID

    def __init__(self, aio_id: str = None):
        if aio_id is None:
            aio_id = str(uuid.uuid4())

        super().__init__([
            html.H2('Dataset Distribution', className='font-semibold text-lg'),
            dcc.Graph(
                self.ID.graph(aio_id),
                figure=DistributionFigure(),
                animate=True,
                style={'width': '100%', 'height': '100%'}
            ),
            dcc.Dropdown(
                id=self.ID.dropdown(aio_id),
                options=[f'Client {i}' for i in range(1, 21)],
                value='Client 1',
                className='absolute right-7 top-7 text-xs w-24 focus:outline-none focus-visible:outline-none '
                          'border border-[#EFF1F3] rounded-md',
                searchable=False,
                clearable=False,
            )
        ], className='relative flex flex-col w-1/4 p-7 h-96 bg-white rounded-lg '
                     'shadow-[0px_4px_20px_rgba(237,237,237,0.5)]')

    @callback(
        Output(ID.dropdown(MATCH), 'options'),
        Input('local-storage', 'data'),
        prevent_initial_callbacks=True
    )
    def update_dropdown_options(data: dict):
        if data == {}:
            return []
        return [f'Client {id_}' for id_ in range(1, len(data['clients']) + 1)]

    @callback(
        Output(ID.graph(MATCH), 'figure'),
        Input('local-storage', 'data'),
        Input(ID.dropdown(MATCH), 'value'),
        prevent_initial_callbacks=True
    )
    def update_distribution(data: dict, value: str):
        if not data:
            return DistributionFigure()

        id_ = 1 if value is None else int(value.split(' ')[-1])
        client = next((item for item in data['clients'] if item['id'] == id_), None)

        return DistributionFigure(
            x=list(range(0, 10)),
            y=client['dataset']['distribution'],
        )
