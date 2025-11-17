import azure.functions as func
from azure.functions import AsgiMiddleware
from training_api.app.main import app


def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    return AsgiMiddleware(app).handle(req, context)
