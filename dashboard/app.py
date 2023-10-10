from dash import Dash, html, dcc
from components import (SidebarAIO, HeaderAIO, TopologyAIO, DistributionAIO, ActivityAIO, SelectionAIO,
                        LocalLossAIO, GlobalLossAIO, LocalAccuracyAIO, GlobalAccuracyAIO, ModalAIO,
                        CommunicationEnergyAIO)


app = Dash(__name__, external_scripts=['https://cdn.tailwindcss.com'])

app.layout = html.Div([
    dcc.Store('local-storage', 'memory'),
    ModalAIO(),
    html.Div([
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
                CommunicationEnergyAIO()
            ], className='flex space-x-7')
        ], className='flex flex-col space-y-7 w-full min-h-screen p-7 pl-[268px]')
    ], className='relative w-full h-full overflow-hidden')
], className='w-screen min-h-screen overflow-hidden')

if __name__ == '__main__':
    app.run_server(debug=True)
