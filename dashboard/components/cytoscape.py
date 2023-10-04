from dash import html, dcc
import dash_cytoscape as cyto
import uuid


# noinspection PyMethodParameters
class CytoscapeAIO(html.Div):

    class ID:
        cytoscape = lambda aio_id: {
            'component': 'NetworkAIO',
            'subcomponent': 'cytoscape',
            'aio_id': aio_id
        }
        store = lambda aio_id: {
            'component': 'NetworkAIO',
            'subcomponent': 'store',
            'aio_id': aio_id
        }
    ID = ID

    def __init__(self, aio_id: str = None):
        if aio_id is None:
            aio_id = str(uuid.uuid4())

        super().__init__([
            cyto.Cytoscape(
                id=self.ID.cytoscape(aio_id),
                layout={
                    'name': 'preset',
                    'animate': True,
                    'animationDuration': 500,
                },
                # userZoomingEnabled=False,
                userPanningEnabled=False,
                style={'width': '100%', 'height': '100%'},
                elements=[],
                stylesheet=[
                    {'selector': 'node', 'style': {
                        'background-color': '#BB2525',
                        'content': 'data(label)',
                        'text-margin-y': '-10px',
                        'width': 40,
                        'height': 40,
                        'color': '#ADBAC7',
                        'font-size': '20px',
                    }},
                    {'selector': 'node[?active]', 'style': {
                        'background-color': '#557A46',
                    }},
                    {'selector': 'edge', 'style': {
                        'line-color': '#ADBAC7',
                        'curve-style': 'bezier',
                        'target-arrow-color': '#ADBAC7',
                        'target-arrow-shape': 'triangle',
                        'arrow-scale': 2,
                        'content': 'data(distance)',
                        'text-halign': 'center',
                        'text-valign': 'center',
                        'color': '#ADBAC7',
                        'font-size': '20px',
                        'text-background-color': '#1C2128',
                        'text-background-opacity': 1,
                    }},
                    {'selector': '[?opaque]', 'style': {
                        'opacity': .3
                    }},
                ]
            ),
            dcc.Store(self.ID.store(aio_id), 'local')
        ], className='row-span-2 col-span-3 bg-transparent border-2 border-[#444c56] rounded-lg')
