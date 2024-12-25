from fastapi import HTTPException, FastAPI, Request
from fastapi.responses import StreamingResponse
from io import BytesIO
from starlette.responses import FileResponse
from pandas_profiling import ProfileReport
import pandas as pd
from pydantic import BaseModel
import sweetviz as sv
import pickle

app = FastAPI()

df = pd.DataFrame()

class ColumnsRequest(BaseModel):
    columns: list

@app.post("/upload_dataframe")
async def upload_dataframe(request: Request):
    global df
    try:
        content = await request.body()
        dataframe = pickle.loads(content)

        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Данные не являются объектом DataFrame")

        df = dataframe

        return {"message": "DataFrame успешно получен"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка обработки DataFrame: {str(e)}")


@app.post("/get_columns")
async def get_columns(request: ColumnsRequest):
    global df
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="DataFrame пуст или не инициализирован")

    try:
        result_df = df[request.columns]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Column {str(e)} not found")

    pickle_data = pickle.dumps(result_df)
    return StreamingResponse(
        BytesIO(pickle_data),
        media_type="application/octet-stream"
    )


@app.get("/get_profile")
async def get_profile():
    global df
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="DataFrame пуст или не инициализирован")

    try:
        profile = ProfileReport(df, title="ProfileReport", explorative=True)
        profile_path = "profiling_report.html"
        profile.to_file(profile_path)

        return FileResponse(
            path=profile_path,
            media_type='text/html',
            filename="profiling_report.html"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при создании профиля: {str(e)}")
