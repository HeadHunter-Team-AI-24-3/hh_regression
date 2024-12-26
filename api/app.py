from fastapi import HTTPException, FastAPI, Request
from fastapi.responses import StreamingResponse
from io import BytesIO
from starlette.responses import FileResponse
from pandas_profiling import ProfileReport
import pandas as pd
from pydantic import BaseModel
import pickle
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from preprocess import *
from typing import Dict, Any

app = FastAPI()
df = pd.DataFrame()
models = {}

class ColumnsRequest(BaseModel):
    columns: list

class TrainModelRequest(BaseModel):
    model_id: str
    model_name: str
    hyperparameters: Dict[str, Any]

@app.post("/upload_dataframe")
async def upload_dataframe(request: Request):
    global df
    try:
        content = await request.body()
        dataframe = pickle.loads(content)

        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Данные не являются объектом DataFrame")

        df = dataframe
        print(type(df))

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
    

@app.post("/train_model")
async def train_model(request: TrainModelRequest):
    global df
    global models

    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="DataFrame is empty or not initialized")

    try:
        data = preprocess_data(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during preprocessing: {str(e)}")

    try:
        X_train, X_test, y_train, y_test = preprocess_data_for_model(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during preprocessing data for model: {str(e)}")

    try:
        model = CatBoostRegressor(**request.hyperparameters)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid hyperparameters: {str(e)}")

    try:
        model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during model training: {str(e)}")

    try:
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during model evaluation: {str(e)}")

    model_info = {
        "id": request.model_id,
        "name": request.model_name,
        "model": model,
        "status": "trained",
        "metrics": {
            "RMSE": rmse,
            "R2": r2
        }
    }

    models[request.model_id] = model_info
    response = {
        "id": model_info["id"],
        "name": model_info["name"],
        "status": model_info["status"],
        "metrics": model_info["metrics"]
    }
    return response

@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    global models
    if model_id in models:
        del models[model_id]
        return {"message": f"Model with ID '{model_id}' has been deleted."}
    else:
        raise HTTPException(status_code=404, detail=f"Model with ID '{model_id}' not found.")

@app.delete("/models")
async def delete_all_models():
    global models
    models.clear()
    return {"message": "All models have been deleted."}
