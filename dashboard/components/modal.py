from dash import html, callback, Output, Input, State, ctx, ALL, clientside_callback
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
                html.Div(className='w-full h-px bg-[#444c56]'),
                html.Ul(children=[
                    html.Li(
                        id=self.ID.li(index, file),
                        children=file,
                        n_clicks=0,
                        className='w-full px-2 py-2 rounded-sm cursor-pointer hover:bg-[#444c56]/50'
                    ) for index, file in enumerate(os.listdir('dashboard')) if file.endswith('.json')
                ], className='text-sm font-light space-y-1'),
            ], className='bg-[#1c2128] w-1/4 border-2 border-[#444c56] rounded-lg p-7 space-y-4'),

        ], className='duration-200 absolute inset-0 w-screen h-screen z-50 flex items-center justify-center '
                     'bg-black/20 backdrop-blur-sm overflow-hidden visible opacity-100')

    @callback(
        Output('local-storage', 'data'),
        Output(ID.parent('modal'), 'className'),
        Input(ID.li(ALL, ALL), 'n_clicks'),
        State(ID.parent('modal'), 'className'),
    )
    def select_file(n_clicks, parent_style):
        data = {}
        if any(n_clicks):
            # Read json file
            clicked = ctx.triggered_id
            file = clicked['value']
            with open(os.path.join('dashboard', file)) as json_file:
                data = json.load(json_file)
            # Hide modal
            classes = parent_style.split(' ')
            classes[-1] = 'opacity-0'
            classes[-2] = 'invisible'
            parent_style = ' '.join(classes)
        return data, parent_style

    # Disable body scrolling when Modal is open
    clientside_callback(
        """
        function setVisible(clicks) {
            if (clicks.some(e => e != 0)) {
                console.log(clicks.some(e => e != 0))
                document.body.style.overflow = 'visible'
            }
            return []
        }
        """,
        Output('dummy', 'children'),
        Input(ID.li(ALL, ALL), 'n_clicks'),
    )
