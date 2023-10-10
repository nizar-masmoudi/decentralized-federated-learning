from dash import html
import uuid


class SidebarAIO(html.Div):
    class ID:
        pass
    ID = ID

    def __init__(self, aio_id: str = None):
        if aio_id is None:
            aio_id = str(uuid.uuid4())

        super().__init__([], className='fixed inset-0 bg-white w-60 h-screen')
