from dash import html, callback, Input, Output, State, ALL, ctx
from dash_svg import Svg, Path
import json
import uuid
import os
from dashboard.utils import process_data


# noinspection PyMethodParameters
class SidebarAIO(html.Div):
    class ID:
        runs = lambda index, file: {
            'type': 'runs',
            'index': index,
            'file': file
        }
    ID = ID

    def __init__(self, aio_id: str = None):
        if aio_id is None:
            aio_id = str(uuid.uuid4())

        super().__init__([
            html.P('Runs', className='ml-2 text-[#737791]'),
            html.Div([
                html.Span([
                    html.Div(
                        [
                            html.P(file.split('.')[0]),
                            Svg([
                                Path(d='M8.25 4.5l7.5 7.5-7.5 7.5', strokeLinecap='round', strokeLinejoin='round')
                            ], fill='none', viewBox='0 0 24 24', className='w-4 h-4 stroke-[#444A6D] stroke-2')
                        ],
                        SidebarAIO.ID.runs(i, os.path.join('logs/', file)),
                        className='flex items-center justify-between p-2 w-full rounded-lg '
                                  'cursor-pointer bg-transparent'
                    ),
                    html.Div(className='w-full h-px bg-[#EFF1F3] my-1'),
                ])
                for i, file in enumerate(os.listdir('logs/'))
            ], className='mt-4')
        ], className='fixed inset-0 bg-white w-60 h-screen p-2 pt-28 text-sm text-[#444A6D]')

    @callback(
        Output('local-storage', 'data'),
        Output(ID.runs(ALL, ALL), 'className'),
        Input(ID.runs(ALL, ALL), 'n_clicks'),
        State(ID.runs(ALL, ALL), 'className'),
    )
    def update_run(n_clicks: list, styles: list):
        data = {}
        if any(n_clicks):
            # Read json file
            clicked = ctx.triggered_id

            file = clicked['file']
            with open(file) as json_file:
                data = json.load(json_file)
            data = process_data(data)

            index = clicked['index']

            for i, style in enumerate(styles):
                if i == index:
                    styles[i] = style.replace('bg-transparent', 'bg-[#444A6D]/5')
                else:
                    styles[i] = style.replace('bg-[#444A6D]/5', 'bg-transparent')

        return data, styles
