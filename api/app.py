import logging
import pickle
from contextlib import asynccontextmanager
from io import BytesIO
from json import JSONDecodeError
from typing import Annotated, Any, Dict, List

import pandas as pd
from catboost import CatBoostRegressor
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from preprocess import preprocess_data, preprocess_data_for_model
from pydantic import BaseModel
from sklearn.metrics import r2_score, root_mean_squared_error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/fastapi.log"), logging.StreamHandler()],
)


logger = logging.getLogger(__name__)

df = pd.DataFrame()
models = {}


class ColumnsRequest(BaseModel):
    columns: list


class TrainModelRequest(BaseModel):
    model_id: str
    model_name: str
    hyperparameters: Dict[str, Any]


class SuccessResponse(BaseModel):
    message: str


class ErrorResponse(BaseModel):
    detail: str


class DataFrameResponse(BaseModel):
    df: str


class ModelResponse(BaseModel):
    id: str
    name: str
    status: str
    metrics: Dict[str, float]


class ModelInfo(BaseModel):
    model_id: str
    model_name: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]


class LearningCurves(BaseModel):
    learning_curves: Dict[str, List[Any]]


class DeleteModelResponse(BaseModel):
    message: str


class PredictionResponse(BaseModel):
    predictions: List[float]
    model_id: str


class LearningCurvesComparisonResponse(BaseModel):
    learning_curves_comparison: Dict[str, Dict[str, List[float]]]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global models
    logging.info("Running lifespan...")

    try:
        logging.info("Upload model...")

        model = CatBoostRegressor()
        model.load_model("pretrained_model.cbm")

        hyperparameters = {"iterations": 100, "learning_rate": 0.1, "depth": 6}

        rmse = 72576.98492
        r2 = 0.4749209

        learning_curves = {
            "iterations": [],
            "train_rmse": [],
            "test_rmse": [],
        }

        model_info = {
            "id": "default_model",
            "name": "Pretrained CatBoost",
            "model": model,
            "status": "loaded",
            "hyperparameters": hyperparameters,
            "metrics": {"RMSE": rmse, "R2": r2},
            "learning_curves": learning_curves,
        }

        models["default_model"] = model_info
        logging.info("Model uploaded.")
        yield
    finally:
        models.clear()
        logging.info("Stop lifespan.")


app = FastAPI(lifespan=lifespan)


@app.post("/upload_dataframe", response_model=SuccessResponse, responses={400: {"model": ErrorResponse}})
async def upload_dataframe(request: Annotated[Request, Request]) -> SuccessResponse:
    global df
    try:
        logger.info("Call to /upload_dataframe")
        content = await request.body()
        logger.info(f"Content received with size: {len(content)} bytes")

        dataframe = pickle.loads(content)

        if not isinstance(dataframe, pd.DataFrame):
            logger.error("Data is not a DataFrame object")
            raise ValueError("Data is not a DataFrame object")

        df = dataframe

        logger.info("DataFrame successfully received")
        return {"message": "DataFrame successfully received"}

    except Exception as e:
        logger.error(f"Error processing DataFrame: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing DataFrame: {str(e)}")


@app.get("/get_dataframe", response_model=DataFrameResponse, responses={400: {"model": ErrorResponse}})
async def get_dataframe() -> DataFrameResponse:
    global df
    try:
        logger.info("Call to /get_dataframe")

        if df is None or df.empty:
            logger.error("DataFrame is empty or not initialized")
            raise HTTPException(status_code=400, detail="DataFrame is empty or not initialized")

        df_serialized = pickle.dumps(df)

        logger.info("DataFrame successfully sent")
        return {"df": df_serialized.hex()}
    except Exception as e:
        logger.error(f"Error getting DataFrame: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error getting DataFrame: {str(e)}")


@app.post(
    "/get_columns",
    responses={
        200: {"content": {"application/octet-stream": {}}},
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def get_columns(request: Annotated[ColumnsRequest, Request]) -> StreamingResponse:
    global df
    logger.info("Call to /get_columns")

    if df is None or df.empty:
        logger.error("DataFrame is empty or not initialized")
        raise HTTPException(status_code=404, detail="DataFrame is empty or not initialized")

    try:
        logger.info(f"Columns for slicing: {request.columns}")
        result_df = df[request.columns]
    except KeyError as e:
        logger.error(f"Column {str(e)} not found")
        raise HTTPException(status_code=400, detail=f"Column {str(e)} not found")

    pickle_data = pickle.dumps(result_df)
    return StreamingResponse(BytesIO(pickle_data), media_type="application/octet-stream")


@app.post(
    "/train_model",
    response_model=ModelResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def train_model(request: Annotated[TrainModelRequest, BaseModel]) -> ModelResponse:
    global df
    global models

    logger.info("Call to /train_model")
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="DataFrame is empty or not initialized")

    try:
        data = preprocess_data(df)
        logger.info("DataFrame preprocessing succeeded")
    except Exception as e:
        logger.info("DataFrame preprocessing failed")
        raise HTTPException(status_code=500, detail=f"Error during preprocessing: {str(e)}")

    try:
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data_for_model(data, is_trained=True)
        logger.info("Train/val/test split succeeded")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during preprocessing data for model: {str(e)}",
        )

    try:
        logger.info("Beginning model training")
        model = CatBoostRegressor(**request.hyperparameters)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid hyperparameters: {str(e)}")

    try:
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=True, plot=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during model training: {str(e)}")

    try:
        y_pred = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during model evaluation: {str(e)}")

    model_info = {
        "id": request.model_id,
        "name": request.model_name,
        "model": model,
        "status": "trained",
        "hyperparameters": request.hyperparameters,
        "metrics": {"RMSE": rmse, "R2": r2},
        "learning_curves": {
            "iterations": list(range(len(model.evals_result_["validation"]["RMSE"]))),
            "train_rmse": model.evals_result_["learn"]["RMSE"],
            "test_rmse": model.evals_result_["validation"]["RMSE"],
        },
    }

    models[request.model_id] = model_info

    response = {
        "id": model_info["id"],
        "name": model_info["name"],
        "status": model_info["status"],
        "metrics": model_info["metrics"],
    }
    return response


@app.get("/get_model_info/{model_id}", response_model=ModelInfo, responses={404: {"model": ErrorResponse}})
async def get_model_info(model_id: Annotated[str, Any]) -> ModelInfo:
    logger.info("Call to /get_model_info")
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model with id {model_id} not found")

    model_info = models[model_id]
    return {
        "model_id": model_id,
        "model_name": model_info["name"],
        "hyperparameters": model_info["model"].get_params(),
        "metrics": model_info["metrics"],
    }


@app.get("/get_models_info", response_model=List[ModelInfo], responses={404: {"model": ErrorResponse}})
async def get_models_info() -> List[ModelInfo]:
    logger.info("Call to /get_models_info")
    if not models:
        raise HTTPException(status_code=404, detail="No models found")
    result = []
    for model_id, model_info in models.items():
        result.append(
            {
                "model_id": model_id,
                "model_name": model_info["name"],
                "hyperparameters": model_info["model"].get_params(),
                "metrics": model_info["metrics"],
            }
        )
    return result


@app.get("/get_learning_curves/{model_id}", response_model=LearningCurves, responses={404: {"model": ErrorResponse}})
async def get_learning_curves(model_id: Annotated[str, Any]) -> LearningCurves:
    logger.info("Call to /get_learning_curves")
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model with id {model_id} not found")

    model_info = models[model_id]

    return {"learning_curves": model_info["learning_curves"]}


@app.delete("/models/{model_id}", response_model=DeleteModelResponse, responses={404: {"model": ErrorResponse}})
async def delete_model(model_id: Annotated[str, Any]) -> DeleteModelResponse:
    logger.info(f"Deleting model {model_id}")
    global models
    if model_id in models:
        del models[model_id]
        return {"message": f"Model with ID '{model_id}' has been deleted."}
    else:
        raise HTTPException(status_code=404, detail=f"Model with ID '{model_id}' not found.")


@app.delete("/models", response_model=DeleteModelResponse)
async def delete_all_models() -> DeleteModelResponse:
    logger.info("Deleting all models")
    global models
    models.clear()
    return {"message": "All models have been deleted."}


@app.post(
    "/predict/{model_id}",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def predict(model_id: Annotated[str, Any], request: Annotated[Request, Any]) -> PredictionResponse:
    logger.info("Model inference")
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model with ID '{model_id}' not found.")
    try:
        dataframe_bytes = await request.body()
        input_data = pickle.loads(dataframe_bytes)
        if not isinstance(input_data, pd.DataFrame):
            raise ValueError("The passed object is not a DataFrame")
        model_info = models[model_id]
        model = model_info["model"]
        processed_data = preprocess_data(input_data)
        X = preprocess_data_for_model(processed_data, is_trained=False)
        predictions = model.predict(X)
        return {"predictions": predictions.tolist(), "model_id": model_id}
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing data or performing inference: {str(e)}",
        )


@app.post(
    "/compare_learning_curves/",
    response_model=LearningCurvesComparisonResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def compare_learning_curves(request: Annotated[Request, Any]) -> LearningCurvesComparisonResponse:
    logger.info("Comparing learning curves for multiple experiments")
    try:
        request_data = await request.json()
        model_ids = request_data.get("model_ids", [])

        if not model_ids:
            raise HTTPException(status_code=400, detail="No model IDs provided")

        if len(model_ids) > 5:
            raise HTTPException(status_code=400, detail="Cannot compare more than 5 models at once")

        learning_curves = {}
        for model_id in model_ids:
            if model_id not in models:
                raise HTTPException(status_code=404, detail=f"Model with ID '{model_id}' not found")
            learning_curves[model_id] = models[model_id]["learning_curves"]

        return {"learning_curves_comparison": learning_curves}

    except JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
