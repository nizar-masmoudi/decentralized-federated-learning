import dash
from dash import html, dcc, callback, Input, Output
from dashboard.utils import process_data
import os.path as osp
import json
from dashboard.components import (SidebarAIO, HeaderAIO, TopologyAIO, DistributionAIO, ActivityAIO, SelectionAIO,
                                  LocalLossAIO, GlobalLossAIO, LocalAccuracyAIO, GlobalAccuracyAIO, IndicatorAIO)

dash.register_page(__name__, path_template='/run/<run_name>', name='Run')


def layout(run_name: str = None):

    return html.Div([
        HeaderAIO(),
        SidebarAIO(),
        html.Div([
            TopologyAIO(),
            html.Span([
                DistributionAIO(),
                ActivityAIO(),
                SelectionAIO(),
            ], className='flex space-x-7'),
            html.Span([
                LocalLossAIO(),
                GlobalLossAIO()
            ], className='flex space-x-7'),
            html.Span([
                LocalAccuracyAIO(),
                GlobalAccuracyAIO()
            ], className='flex space-x-7'),
            html.Span([
                IndicatorAIO(),
            ], className='flex space-x-7')
        ], className='flex flex-col space-y-7 w-full min-h-screen p-7 pl-[268px]')
    ], className='relative w-full h-full overflow-hidden')


@callback(
    Output('local-storage', 'data'),
    Input('location', 'pathname'),
    prevent_initial_call=True
)
def update_data(pathname: str):
    if pathname == '/':
        return {}
    file = '{}.json'.format(osp.basename(pathname))
    with open(osp.join('logs/', file)) as json_file:
        data = json.load(json_file)
    data = process_data(data)
    return data
