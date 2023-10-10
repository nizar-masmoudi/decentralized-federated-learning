from dash import html, callback, Output, Input, State, ctx, ALL, MATCH, clientside_callback
from utils import process_data
import os as os
import uuid
import json


# noinspection PyMethodParameters
class ModalAIO(html.Div):
    class ID:
        store = lambda aio_id: {
            'component': 'ModalAIO',
            'subcomponent': 'store',
            'aio_id': aio_id
        }
        parent = lambda aio_id: {
            'component': 'ModalAIO',
            'subcomponent': 'parent',
            'aio_id': aio_id
        }
        li = lambda index, value: {
            'type': 'li',
            'index': index,
            'value': value
        }
    ID = ID

    def __init__(self, aio_id: str = None):
        if aio_id is None:
            aio_id = str(uuid.uuid4())

        super().__init__(id=self.ID.parent(aio_id), children=[
            html.Span(id='dummy', className='hidden'),
            html.Div([
                html.H1('Pick a configuration file'),
                html.Div(className='w-full h-px bg-[#EFF1F3]'),
                html.Ul(children=[
                    html.Li(
                        id=self.ID.li(index, file),
                        children=file,
                        n_clicks=0,
                        className='w-full px-2 py-2 rounded-sm cursor-pointer hover:bg-[#ECECEC]'
                    ) for index, file in enumerate(os.listdir('dashboard')) if file.endswith('.json')
                ], className='text-sm font-light space-y-1'),
            ], className='bg-white w-1/4 border border-[#EFF1F3] rounded-md rounded-lg p-7 space-y-4'),

        ], className='duration-200 absolute inset-0 w-screen h-screen z-50 flex items-center justify-center '
                     'bg-black/20 backdrop-blur-sm overflow-hidden visible opacity-100')

    @callback(
        Output('local-storage', 'data'),
        Input(ID.li(ALL, ALL), 'n_clicks'),
    )
    def select_file(n_clicks):
        data = {}
        if any(n_clicks):
            # Read json file
            clicked = ctx.triggered_id
            file = clicked['value']
            with open(os.path.join('dashboard', file)) as json_file:
                data = json.load(json_file)
                data = process_data(data)
        return data

    # Disable body scrolling when Modal is open
    clientside_callback(
        """
        function setVisible(clicks, className) {
            if (clicks.some(e => e != 0)) {
                document.body.style.overflow = 'visible'
                className = className.replace('opacity-100', 'opacity-0')
                className = className.replace('visible', 'invisible')
            }
            return className
        }
        """,
        Output(ID.parent(MATCH), 'className'),
        Input(ID.li(ALL, ALL), 'n_clicks'),
        State(ID.parent(MATCH), 'className'),
    )
