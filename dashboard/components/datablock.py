from dash import html
import uuid


class DataBlockAIO(html.Div):
    class ID:
        config = lambda index: {
            'type': 'config',
            'index': index,
        }
    ID = ID

    def __init__(self, aio_id: str = None):
        if aio_id is None:
            aio_id = str(uuid.uuid4())

        super().__init__([
            html.Div([
                html.Span([
                    html.P('Client'),
                    html.P('-', self.ID.config(0)),
                ], className='flex justify-between'),
                html.Div(className='w-full h-px bg-[#444c56]'),
                html.Div([
                    html.Span([
                        html.P('Latitude'),
                        html.P('-', self.ID.config(1)),
                    ], className='flex justify-between'),
                    html.Span([
                        html.P('Longitude'),
                        html.P('-', self.ID.config(2)),
                    ], className='flex justify-between'),
                ], className='space-y-2 text-sm'),
                html.Div(className='w-full h-px bg-[#444c56]'),
                html.Div([
                    html.Span([
                        html.P('FLOPs per CPU cycle'),
                        html.P('-', self.ID.config(3)),
                    ], className='flex justify-between'),
                    html.Span([
                        html.P('CPU clock frequency'),
                        html.P('-', self.ID.config(4)),
                    ], className='flex justify-between'),
                    html.Span([
                        html.P("CPU's effective capacitance"),
                        html.P('-', self.ID.config(5)),
                    ], className='flex justify-between'),
                ], className='space-y-2 text-sm'),
                html.Div(className='w-full h-px bg-[#444c56]'),
                html.Div([
                    html.Span([
                        html.P('Transmission power'),
                        html.P('-', self.ID.config(6)),
                    ], className='flex justify-between'),
                    html.Span([
                        html.P('Signal bandwidth'),
                        html.P('-', self.ID.config(7)),
                    ], className='flex justify-between'),
                    html.Span([
                        html.P('Signal frequency'),
                        html.P('-', self.ID.config(8)),
                    ], className='flex justify-between'),
                ], className='space-y-2'),
            ], className='space-y-2 text-sm'),
        ], className='row-span-1 col-span-1 bg-transparent border-2 border-[#444c56] rounded-lg p-4')
