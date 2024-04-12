import dash
from dash import html

from dashboard.components import SidebarAIO, HeaderAIO, TableAIO, AggLossAIO

dash.register_page(__name__, path_template='/', name='Dashboard')

layout = html.Div([
    HeaderAIO(),
    SidebarAIO(),
    html.Div([
        TableAIO(),
        AggLossAIO()
    ], className='flex flex-col space-y-7 w-full min-h-screen p-7 pl-[268px]')
], className='relative w-full h-full overflow-hidden')
