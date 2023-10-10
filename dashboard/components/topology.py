from dash import html, callback, Input, Output, State, ALL
from dash_daq.NumericInput import NumericInput
import dash_cytoscape as cyto
import uuid
from utils import format_size, geo_distance


# noinspection PyMethodParameters
class TopologyAIO(html.Div):
    class ID:
        client_cfg = lambda index: {
            'type': 'client_cfg',
            'index': index,
        }
        channel_cfg = lambda index: {
            'type': 'channel_cfg',
            'index': index,
        }

    ID = ID

    def __init__(self, aio_id: str = None):
        if aio_id is None:
            aio_id = str(uuid.uuid4())

        super().__init__([
            html.Div([
                html.H2('Network Topolgy', className='font-semibold text-lg'),

                # Client configuration
                html.Div([
                    html.Span([
                        html.P('Client'),
                        html.P('-', self.ID.client_cfg(0)),
                    ], className='flex justify-between'),
                    html.Div(className='w-full h-px bg-[#EFF1F3]'),
                    html.Div([
                        html.Span([
                            html.P('Latitude'),
                            html.P('-', self.ID.client_cfg(1)),
                        ], className='flex justify-between'),
                        html.Span([
                            html.P('Longitude'),
                            html.P('-', self.ID.client_cfg(2)),
                        ], className='flex justify-between'),
                    ], className='space-y-2'),
                    html.Div(className='w-full h-px bg-[#EFF1F3]'),
                    html.Div([
                        html.Span([
                            html.P('FLOPs per CPU cycle'),
                            html.P('-', self.ID.client_cfg(3)),
                        ], className='flex justify-between'),
                        html.Span([
                            html.P('CPU clock frequency'),
                            html.P('-', self.ID.client_cfg(4)),
                        ], className='flex justify-between'),
                        html.Span([
                            html.P("CPU's effective capacitance"),
                            html.P('-', self.ID.client_cfg(5)),
                        ], className='flex justify-between'),
                    ], className='space-y-2'),
                    html.Div(className='w-full h-px bg-[#EFF1F3]'),
                    html.Div([
                        html.Span([
                            html.P('Transmission power'),
                            html.P('-', self.ID.client_cfg(6)),
                        ], className='flex justify-between'),
                        html.Span([
                            html.P('Signal bandwidth'),
                            html.P('-', self.ID.client_cfg(7)),
                        ], className='flex justify-between'),
                        html.Span([
                            html.P('Signal frequency'),
                            html.P('-', self.ID.client_cfg(8)),
                        ], className='flex justify-between'),
                    ], className='space-y-2'),
                ], className='w-full my-7 pr-7 space-y-2'),

                # Channel configuration
                html.Div([
                    html.Span([
                        html.P('Channel'),
                        html.P('-', self.ID.channel_cfg(0)),
                    ], className='flex justify-between'),
                    html.Div(className='w-full h-px bg-[#EFF1F3]'),
                    html.Div([
                        html.Span([
                            html.P('Distance'),
                            html.P('-', self.ID.channel_cfg(1)),
                        ], className='flex justify-between'),
                        html.Span([
                            html.P('Channel gain'),
                            html.P('-', self.ID.channel_cfg(2)),
                        ], className='flex justify-between'),
                        html.Span([
                            html.P('Transmission rate'),
                            html.P('-', self.ID.channel_cfg(3)),
                        ], className='flex justify-between'),
                    ], className='space-y-2'),
                ], className='w-full my-7 pr-7 space-y-2'),

                # Legend
                html.Div([
                    html.Span([
                        html.Div(className='w-3 h-3 bg-[#EF4444] rounded-full'),
                        html.P('Inactive client')
                    ], className='flex items-center space-x-2 text-xs'),
                    html.Span([
                        html.Div(className='w-3 h-3 bg-[#00E096] rounded-full'),
                        html.P('Active client')
                    ], className='flex items-center space-x-2 text-xs'),
                    html.Span([
                        html.Div(className='border-t border-[#444A6D] border-dashed w-3 h-px'),
                        html.P('Neighbor link')
                    ], className='flex items-center space-x-2 text-xs'),
                    html.Span([
                        html.Div(className='border-t border-[#444A6D] border-solid w-3 h-px'),
                        html.P('Peer link')
                    ], className='flex items-center space-x-2 text-xs'),
                ], className='absolute bottom-0 left-0 flex justify-evenly w-full pr-7'),
            ], className='relative w-1/3 h-full text-[#444A6D] text-sm'),

            # Network
            html.Div([
                cyto.Cytoscape(
                    id='cytoscape',
                    layout={
                        'name': 'preset',
                        'animate': True,
                        'animationDuration': 500,
                    },
                    userZoomingEnabled=False,
                    userPanningEnabled=False,
                    zoomingEnabled=False,
                    panningEnabled=False,
                    pan={'x': 0, 'y': 0},
                    zoom=1,
                    style={'width': '100%', 'height': '100%'},
                    elements=[],
                    stylesheet=[
                        {'selector': 'node', 'style': {
                            'background-color': '#EF4444',
                            'content': 'data(label)',
                            'text-margin-y': '-10px',
                            'width': 20,
                            'height': 20,
                            'color': '#444A6D',
                            'font-size': '12px',
                        }},
                        {'selector': 'node[?active]', 'style': {
                            'background-color': '#00E096',
                        }},
                        {'selector': 'edge', 'style': {
                            'line-color': '#444A6D',
                            'width': 1,
                            'line-style': 'dashed',
                            'curve-style': 'bezier',
                            'target-arrow-color': '#444A6D',
                            'target-arrow-shape': 'triangle',
                            'arrow-scale': .7,
                            'text-halign': 'center',
                            'text-valign': 'center',
                            'color': '#444A6D',
                            'font-size': '12px',
                            'text-background-color': '#FAFBFC',
                            'text-background-opacity': 1,
                        }},
                        {'selector': '[?peer]', 'style': {
                            'line-style': 'solid',
                        }},
                    ]
                ),
                html.Div([
                    html.P('Round'),
                    NumericInput('round', 1, 40, 1, 10)
                ], className='absolute right-4 top-4 flex items-center justify-between bg-white '
                             'rounded-lg w-36 h-12 p-2 border border-[#EFF1F3]'),
            ], className='relative w-2/3 h-full bg-[#FAFBFC] rounded-lg'),
        ], className='flex items-center justify-center p-7 w-full h-[650px] bg-white '
                     'rounded-lg shadow-[0px_4px_20px_rgba(237,237,237,0.5)]')

    @callback(
        Output(ID.client_cfg(ALL), 'children'),
        Output('round', 'max'),
        Input('local-storage', 'data'),
        Input('cytoscape', 'tapNodeData'),
        Input('round', 'value'),
        State(ID.client_cfg(ALL), 'children'),
    )
    def update_client_config(data: dict, node: dict, round_: int, configs: list):
        max_round = 10
        round_ = round_ - 1

        if not data:
            return configs, max_round
        if node is None:
            node_id = '1'
        else:
            node_id = node['id']

        max_round = len(data['1']['activity'])
        configs = [
            'Client {}'.format(node['id'] if node else '1'),
            round(data[node_id]['locations'][round_ - 1][0], 7),
            round(data[node_id]['locations'][round_ - 1][1], 7),
            data[node_id]['cpu']['fpc'],
            format_size(data[node_id]['cpu']['frequency'], 'hertz'),
            data[node_id]['cpu']['kappa'],
            '{} dBm'.format(data[node_id]['transmitter']['power']),
            format_size(data[node_id]['transmitter']['bandwidth']),
            format_size(data[node_id]['transmitter']['signal_frequency'], 'hertz'),
        ]
        return configs, max_round

    @callback(
        Output(ID.channel_cfg(ALL), 'children'),
        Input('local-storage', 'data'),
        Input('cytoscape', 'tapEdgeData'),
        Input('round', 'value'),
        State(ID.channel_cfg(ALL), 'children'),
    )
    def update_client_config(data: dict, edge: dict, round_: int, configs: list):
        if not data or edge is None:
            return configs
        else:
            distance = geo_distance(
                data[edge['source']]['locations'][round_ - 1],
                data[edge['target']]['locations'][round_ - 1]
            )
            configs = [
                'Client {} - Client {}'.format(edge['source'], edge['target']),
                '{:.3f} Km'.format(distance),
                '-',
                '-'
            ]
        return configs

    @callback(
        Output('cytoscape', 'elements'),
        Input('local-storage', 'data'),
        Input('cytoscape', 'tapNodeData'),
        State('round', 'value'),
        Input('cytoscape', 'layout'),
    )
    def update_cytoscape_elements(data: dict, node: dict, round_: int, _):
        if not data:
            return []
        if node is None:
            node_id = '1'
        else:
            node_id = node['id']

        round_ -= 1
        (min_lat, min_lon), (max_lat, max_lon) = ((36.897092, 10.152086), (36.870453, 10.219636))
        # Set nodes
        nodes = [{
            'data': {'id': id_, 'label': f'Client {id_}',  'active': data[id_]['activity'][round_]},
            'position': {
                'y': 594 * (data[id_]['locations'][round_][0] - min_lat) / (max_lat - min_lat),
                'x': 1040 * (data[id_]['locations'][round_][1] - min_lon) / (max_lon - min_lon)},
            'grabbable': False} for id_ in data.keys()]
        # Set edges
        edges = []
        edges += [
            {'data': {
                'id': node_id + neighbor,
                'source': node_id,
                'target': neighbor,
                'peer': neighbor in data[node_id]['peers'][round_],
            }} for neighbor in data[node_id]['neighbors'][round_]
        ]
        return nodes + edges

    @callback(
        Output('cytoscape', 'layout'),
        Input('round', 'value'),
        State('local-storage', 'data'),
        State('cytoscape', 'layout'),
    )
    def update_cytoscape_layout(round_: int, data: dict, layout: dict):
        if not data:
            return layout

        (min_lat, min_lon), (max_lat, max_lon) = ((36.897092, 10.152086), (36.870453, 10.219636))
        layout = {
            'name': 'preset',
            'animate': True,
            'animationDuration': 500,
            'positions': {
                id_: {
                    'y': 594 * (data[id_]['locations'][round_ - 1][0] - min_lat) / (max_lat - min_lat),
                    'x': 1040 * (data[id_]['locations'][round_ - 1][1] - min_lon) / (max_lon - min_lon),
                } for id_ in data.keys()
            }
        }
        return layout
