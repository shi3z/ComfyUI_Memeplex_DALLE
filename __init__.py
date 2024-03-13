from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
WEB_DIRECTORY = "./js"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS','WEB_DIRECTORY']
import server

from aiohttp import web
@server.PromptServer.instance.routes.post('/memeplex/update_text')
async def update_text(request):
    data = await request.json()
    text = data["text"]
    server.PromptServer.instance.send_sync("update_text", data)
    server.PromptServer.instance.send_sync("run_workflow", {})
    return web.Response()
