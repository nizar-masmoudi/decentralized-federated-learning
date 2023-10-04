from dash import html, dcc, Input, Output, State, ctx, callback, MATCH, ALL
from dash_daq.BooleanSwitch import BooleanSwitch
import uuid


# noinspection PyMethodParameters
class SideBarAIO(html.Div):
    class ID:
        plus = lambda aio_id: {
            'component': 'SideBarAIO',
            'subcomponent': 'plus',
            'aio_id': aio_id
        }
        minus = lambda aio_id: {
            'component': 'SideBarAIO',
            'subcomponent': 'minus',
            'aio_id': aio_id
        }
        round = lambda aio_id: {
            'component': 'SideBarAIO',
            'subcomponent': 'round',
            'aio_id': aio_id
        }
        neighbors = lambda aio_id: {
            'component': 'SideBarAIO',
            'subcomponent': 'neighbors',
            'aio_id': aio_id
        }
        peers = lambda aio_id: {
            'component': 'SideBarAIO',
            'subcomponent': 'peers',
            'aio_id': aio_id
        }
        dropdown_button = lambda aio_id: {
            'component': 'DropdownAIO',
            'subcomponent': 'dropdown_button',
            'aio_id': aio_id
        }
        dropdown = lambda aio_id: {
            'component': 'DropdownAIO',
            'subcomponent': 'parent',
            'aio_id': aio_id
        }
    ID = ID

    def __init__(self, aio_id: str = None):
        if aio_id is None:
            aio_id = str(uuid.uuid4())

        super().__init__([
            # Configuration
            html.P('Configuration', className='font-light mb-4'),
            html.Div(className='w-full h-px bg-[#444c56] rounded-full my-3'),

            # Layout controls
            html.P('Controls', className='font-light mt-10'),
            html.Div(className='w-full h-px bg-[#444c56] rounded-full my-3'),
            html.Div([
                # Round
                html.Span([
                    html.P('Round'),
                    html.Div([
                        html.Button('-', self.ID.minus(aio_id), className='text-center px-1.5'),
                        html.Div(className='h-full w-px bg-[#adbac7]/50'),
                        html.Div(1, self.ID.round(aio_id), className='text-center w-10'),
                        html.Div(className='h-full w-px bg-[#adbac7]/50'),
                        html.Button('+', self.ID.plus(aio_id), className='text-center px-1.5'),
                    ], className='flex items-center justify-between w-18 border-2 border-[#444c56] rounded-sm py-0.5')
                ], className='flex justify-between'),

                # Display neighbors
                html.Span([
                    html.P('Display neighbors', className='text-sm'),
                    BooleanSwitch(id=self.ID.neighbors(aio_id), on=False, color='#22272e')
                ], className='flex justify-between'),

                # Display peers
                html.Span([
                    html.P('Display peers', className='text-sm'),
                    BooleanSwitch(id=self.ID.peers(aio_id), on=False, color='#22272e')
                ], className='flex justify-between'),

                # Train, validation and test dropdown
                html.Span([
                    html.P('Data split', className='text-sm'),
                    html.Div(id=self.ID.dropdown(aio_id), children=[
                        html.Button(
                            id=self.ID.dropdown_button(aio_id),
                            children='Training',
                            className='absolute top-0 flex items-center justify-between h-7 w-full',
                            n_clicks=0,
                        ),
                        html.Ul([
                            html.Div(className='w-full h-px bg-[#adbac7]/50'),
                            html.Li('Training', {'type': 'option', 'index': 0}, className='cursor-pointer'),
                            html.Div(className='w-full h-px bg-[#adbac7]/50'),
                            html.Li('Validation', {'type': 'option', 'index': 1}, className='cursor-pointer'),
                            html.Div(className='w-full h-px bg-[#adbac7]/50'),
                            html.Li('Test', {'type': 'option', 'index': 2}, className='cursor-pointer'),
                        ], className='absolute top-7 right-0 space-y-2 px-2 w-full')
                    ], className='transition-all duration-300 z-20 relative border-2 border-[#444c56] rounded-sm px-2 '
                                 'h-8 min-h-8 w-1/3 overflow-hidden')
                ], className='flex justify-between'),

            ], className='space-y-3 text-sm')
        ], className='fixed inset-0 bg-[#22272e] w-1/5 h-screen px-4 py-24 border-r-2 border-[#444c56] font-light')

    @callback(
        Output(ID.round(MATCH), 'children'),
        Input(ID.plus(MATCH), 'n_clicks'),
        Input(ID.minus(MATCH), 'n_clicks'),
        State(ID.round(MATCH), 'children'),
        State('local-storage', 'data'),
        prevent_initial_call=True
    )
    def update_round(_1, _2, round_: int, data: dict):
        max_rounds = len(data['1']['activity'])
        clicked = ctx.triggered_id
        if clicked['subcomponent'] == 'plus':
            round_ = min(max_rounds, round_ + 1)
        elif clicked['subcomponent'] == 'minus':
            round_ = max(1, round_ - 1)
        return round_

    @callback(
        Output(ID.dropdown_button(MATCH), 'children'),
        Output(ID.dropdown(MATCH), 'className'),
        Input(ID.dropdown_button(MATCH), 'n_clicks'),
        Input({'type': 'option', 'index': ALL}, 'n_clicks'),
        State(ID.dropdown(MATCH), 'className'),
        State(ID.dropdown_button(MATCH), 'children'),
        prevent_initial_call=True
    )
    def update_value(button_clicks: int, _1, _2, value: str):
        values = ['Training', 'Validation', 'Test']
        if button_clicks:
            parent_style = ('transition-all duration-300 z-20 relative border-2 border-[#444c56] rounded-md px-2 '
                            'h-36 min-h-36 w-1/3 overflow-hidden')
        else:
            parent_style = ('transition-all duration-300 z-20 relative border-2 border-[#444c56] rounded-md px-2 '
                            'h-8 min-h-8 w-1/3 overflow-hidden')
        clicked = ctx.triggered_id
        if 'index' in clicked.keys():
            value = values[clicked['index']]
            parent_style = ('transition-all duration-300 z-20 relative border-2 border-[#444c56] rounded-md px-2 '
                            'h-8 min-h-8 w-1/3 overflow-hidden')
        return value, parent_style
