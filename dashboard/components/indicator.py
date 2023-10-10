from dash import html, callback, Input, Output, MATCH
import uuid


# noinspection PyMethodParameters
class IndicatorAIO(html.Div):
    class ID:
        communication = lambda aio_id: {
            'component': 'IndicatorAIO',
            'subcomponent': 'communication',
            'aio_id': aio_id
        }
        computation = lambda aio_id: {
            'component': 'IndicatorAIO',
            'subcomponent': 'computation',
            'aio_id': aio_id
        }
        total = lambda aio_id: {
            'component': 'IndicatorAIO',
            'subcomponent': 'total',
            'aio_id': aio_id
        }
    ID = ID

    def __init__(self, aio_id: str = None):
        if aio_id is None:
            aio_id = str(uuid.uuid4())

        super().__init__([
            html.H2('Total Energy Consumption', className='font-semibold text-lg'),
            html.Span([
                html.Div([
                    html.H3('Communication', className='text-xs'),
                    html.H3(
                        '-',
                        self.ID.communication(aio_id),
                        className='text-center text-[#FA5A7D] text-4xl font-semibold'
                    ),
                ], className='flex flex-col items-center justify-between w-52 h-28 p-4 bg-[#FFE2E5] rounded-md'),
                html.Div([
                    html.H3(
                        'Computation',
                        className='text-xs'
                    ),
                    html.H3(
                        '-',
                        self.ID.computation(aio_id),
                        className='text-center text-[#BF83FF] text-4xl font-semibold'),
                ], className='flex flex-col items-center justify-between w-52 h-28 p-4 bg-[#F3E8FF] rounded-md'),
            ], className='flex items-center justify-evenly w-full h-1/2'),
            html.Div([
                html.H3(
                    'Total',
                    className='text-xs'
                ),
                html.H3(
                    '10 J',
                    self.ID.total(aio_id),
                    className='text-center text-[#FF947A] text-4xl font-semibold'),
            ], className='flex flex-col items-center justify-between w-52 h-28 p-4 mx-auto bg-[#FFF4DE] rounded-md'),
        ], className='flex flex-col w-1/3 h-96 p-7 bg-white rounded-lg '
                     'shadow-[0px_4px_20px_rgba(237,237,237,0.5)]')

    @callback(
        Output(ID.communication(MATCH), 'children'),
        Output(ID.computation(MATCH), 'children'),
        Output(ID.total(MATCH), 'children'),
        Input('local-storage', 'data'),
    )
    def upodate_indicators(data: dict):
        if not data:
            return '-', '-', '-'

        comm_energy = 0
        comp_energy = 0
        for id_ in data.keys():
            comm_energy += sum(data[id_]['comm_energy'])
        for id_ in data.keys():
            comp_energy += sum(data[id_]['comp_energy'])
        return ('{:.2f} mJ'.format(comm_energy * 1e3),
                '{:.2f} J'.format(comp_energy),
                '{:.2f} J'.format(comm_energy + comp_energy))
