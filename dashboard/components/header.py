from dash import html
import uuid


class HeaderAIO(html.Div):
    class ID:
        pass
    ID = ID

    def __init__(self, aio_id: str = None):
        if aio_id is None:
            aio_id = str(uuid.uuid4())

        super().__init__([
            html.H1('Dashboard', className='text-[#151D48] font-semibold text-2xl'),
            html.Button('Upload', className='py-2 px-4 border border-[#EFF1F3] rounded-md '
                                            'text-[#151D48] cursor-pointer')
        ], className='flex items-center justify-between bg-white w-full h-20 pl-72 pr-12 flex items-center')
