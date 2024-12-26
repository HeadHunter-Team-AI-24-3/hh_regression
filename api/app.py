import logging
from fastapi import HTTPException, FastAPI, Request
from fastapi.responses import StreamingResponse
from io import BytesIO
from starlette.responses import FileResponse
from pandas_profiling import ProfileReport
import pandas as pd
from pydantic import BaseModel
import pickle
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error
from preprocess import *
from typing import Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/fastapi.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

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
        logger.info("Вызов /upload_dataframe")
        content = await request.body()
        logger.info(f"Получен контент размером: {len(content)} байт")

        dataframe = pickle.loads(content)

        if not isinstance(dataframe, pd.DataFrame):
            logger.error("Данные не являются объектом DataFrame")
            raise ValueError("Данные не являются объектом DataFrame")

        df = dataframe

        logger.info("DataFrame успешно получен")
        return {"message": "DataFrame успешно получен"}
    except Exception as e:
        logger.error(f"Ошибка обработки DataFrame: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Ошибка обработки DataFrame: {str(e)}")


@app.post("/get_columns")
async def get_columns(request: ColumnsRequest):
    global df
    logger.info("Вызов /get_columns")
    if df is None or df.empty:
        logger.error("DataFrame пуст или не инициализирован")
        raise HTTPException(status_code=404, detail="DataFrame пуст или не инициализирован")

    try:
        logger.info(f"Колонки датасета по которым берем срез: {str(request.columns)}")
        result_df = df[request.columns]
    except KeyError as e:
        logger.error(f"Столбец {str(e)} не найден")
        raise HTTPException(status_code=400, detail=f"Столбец {str(e)} не найден")

    pickle_data = pickle.dumps(result_df)
    return StreamingResponse(
        BytesIO(pickle_data),
        media_type="application/octet-stream"
    )


@app.get("/get_profile")
async def get_profile():
    global df
    logger.info("Вызов /get_profile")
    if df is None or df.empty:
        logger.error("DataFrame пуст или не инициализирован")
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
        logger.error(f"Ошибка при создании профиля: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при создании профиля: {str(e)}")


@app.post("/train_model")
async def train_model(request: TrainModelRequest):
    global df
    global models

    logger.info("Вызов /train_model")
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="DataFrame is empty or not initialized")

    try:
        data = preprocess_data(df)
        logger.info("Обработка DF прошла успешно")
    except Exception as e:
        logger.info("Обработка DF прошла неуспешно")
        raise HTTPException(status_code=500, detail=f"Error during preprocessing: {str(e)}")

    try:
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data_for_model(data, is_trained = True)
        logger.info("Разбиение на train/val/test прошло успешно")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during preprocessing data for model: {str(e)}")

    try:
        logger.info("Начало обучение модели")
        model = CatBoostRegressor(**request.hyperparameters)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid hyperparameters: {str(e)}")

    try:
        model.fit(
            X_train, 
            y_train, 
            eval_set=(X_val, y_val), 
            verbose=True, 
            plot=False
        )
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
        "hyperparameters": request.hyperparameters,
        "metrics": {
            "RMSE": rmse,
            "R2": r2
        },
        "learning_curves": {
            "iterations": list(range(len(model.evals_result_['validation']['RMSE']))),
            "train_rmse": model.evals_result_['learn']['RMSE'],
            "test_rmse": model.evals_result_['validation']['RMSE']
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


@app.get("/get_model_info/{model_id}")
async def get_model_info(model_id: str):
    logger.info("Вызов /get_model_info")
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model with id {model_id} not found")
    
    model_info = models[model_id]
    return {
        "model_id": model_id,
        "model_name": model_info["name"],
        "hyperparameters": model_info["model"].get_params(),
        "metrics": model_info["metrics"]
    }

@app.get("/get_learning_curves/{model_id}")
async def get_learning_curves(model_id: str):
    logger.info("Вызов /get_learning_curves")
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model with id {model_id} not found")
    
    model_info = models[model_id]
    
    return {
        'learning_curves': model_info["learning_curves"]
    }

@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    logger.info(f"Удаление модели {model_id}")
    global models
    if model_id in models:
        del models[model_id]
        return {"message": f"Model with ID '{model_id}' has been deleted."}
    else:
        raise HTTPException(status_code=404, detail=f"Model with ID '{model_id}' not found.")

@app.delete("/models")
async def delete_all_models():
    logger.info("Удаление всех моделей")
    global models
    models.clear()
    return {"message": "All models have been deleted."}

@app.post("/predict/{model_id}")
async def predict(model_id: str, request: Request):
    logger.info("Инференс модели")
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Модель с ID '{model_id}' не найдена.")
    try:
        dataframe_bytes = await request.body()
        input_data = pickle.loads(dataframe_bytes)
        if not isinstance(input_data, pd.DataFrame):
            raise ValueError("Переданный объект не является DataFrame")
        model_info = models[model_id]
        model = model_info["model"]
        processed_data = preprocess_data(input_data)
        X = preprocess_data_for_model(processed_data, is_trained=False)
        predictions = model.predict(X)
        return {"predictions": predictions.tolist(), "model_id": model_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка обработки данных или выполнения инференса: {str(e)}")