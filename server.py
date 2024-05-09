from common import Common

import fastapi
import starlette
import uvicorn


class Server:
    def __init__(self):
        self.common = Common()

        self.host = "0.0.0.0"
        self.port = 60000

    def run(self):
        app = fastapi.FastAPI()

        from fastapi.templating import Jinja2Templates
        from fastapi.staticfiles import StaticFiles
        templates = Jinja2Templates(directory="/hky3535/api/templates") # 引入html位置
        app.mount("/static", StaticFiles(directory="/hky3535/api/static"), name="static") # 引入css与js位置

        @app.get("/")
        async def root(request: fastapi.Request):
            return templates.TemplateResponse("index.html", {"request": request})

        def error_handler(request, exc): # 处理内部报错
            return fastapi.responses.JSONResponse(status_code=500, content={"ret": False, "response": str(exc)})

        app.add_exception_handler(Exception, handler=error_handler)
        app.add_exception_handler(fastapi.exceptions.HTTPException, handler=error_handler)
        app.add_exception_handler(fastapi.exceptions.RequestValidationError, handler=error_handler)
        app.add_exception_handler(starlette.exceptions.HTTPException, handler=error_handler)

        @app.route("/usage", methods=["GET", "POST"])
        def usage(request: fastapi.Request): # 获取帮助文档
            usage = open("/hky3535/api/usage.py", "r").read()
            return fastapi.responses.PlainTextResponse(usage)

        @app.route("/status", methods=["GET", "POST"])
        def status(request: fastapi.Request): # 查询加载状态（保障调用）
            status = self.common.status()
            return fastapi.responses.JSONResponse(status_code=200, content={"ret": True, "response": status})

        @app.post("/load")
        def load(data: dict): # 动态加载静态框架 加载（保障调用）
            self.common.load(data=data)
            return fastapi.responses.JSONResponse(status_code=200, content={"ret": True, "response": ""})

        @app.post("/infer")
        def infer(data: dict): # 动态加载静态框架 推理（保障调用）
            results = self.common.infer(data=data)
            return fastapi.responses.JSONResponse(status_code=200, content={"ret": True, "response": results})

        uvicorn.run(app, host="0.0.0.0", port=60000)


if __name__ == "__main__":
    Server().run()
