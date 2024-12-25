from fastapi import APIRouter, UploadFile, File, HTTPException, FastAPI, Request
from typing import List
import pandas as pd
from pydantic import BaseModel
import tempfile
import sweetviz as sv
from io import StringIO
import pickle

app = FastAPI()

df = pd.DataFrame()

class ColumnsRequest(BaseModel):
    columns: List[str]

class DataFramePayload(BaseModel):
    data: dict

@app.post("/upload_dataframe")
async def upload_dataframe(request: Request):
    try:
        content = await request.body()

        dataframe = pickle.loads(content)

        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Данные не являются объектом DataFrame")

        return {"message": "DataFrame успешно получен"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка обработки DataFrame: {str(e)}")

@app.post("/get_columns/")
async def get_columns(request: ColumnsRequest):
    global df
    if df.empty:
        return {"error": "DataFrame пуст"}
    
    try:
        result_df = df[request.columns]
    except KeyError as e:
        return {"error": f"Столбец {str(e)} не найден"}
    return result_df.to_dict(orient='records')

@app.get("/get_profile/")
async def get_profile():
    global df
    if df.empty:
        return {"error": "DataFrame пуст"}

    report = sv.analyze(df)
    html_profile = report.show_html(open_browser=False)
    with open(html_profile, "r") as file:
        html_content = file.read()

    return {"profile_html": html_content}
