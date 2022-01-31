from fastapi import FastAPI
import numpy as np
from joblib import dump, load
from dinamic_optimization.temp import *
from typing import Optional
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None
    #class Config:
    #     orm_mode=True
    


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {'item_id':item_id}


@app.get("/model_reg/")
async def predict(x1,x2):
    reg = load('./model/reg_model_1.joblib') 
    res = reg.predict(np.array([[x1,x2]]))[0]
    return {'item_id':res}


@app.get("/optimizacion_lineal/")
async def optimizacion_lineal(M=1400000, iter_500=1000, iter_1000=1000, nu_100_min:Optional[int] = None, nu_200_min:Optional[int] = None, nu_500_min:Optional[int] = None, nu_1000_min:Optional[int] = None, nu_100_eq:Optional[int] = None, nu_200_eq:Optional[int] = None, nu_500_eq:Optional[int] = None, nu_1000_eq:Optional[int] = None):
    r_gavetas_min = {'100':nu_100_min,'200':nu_200_min,'500':nu_500_min,'1000':nu_1000_min}
    r_gavetas_eq = {'100':nu_100_eq,'200':nu_200_eq,'500':nu_500_eq,'1000':nu_1000_eq}
    
    return optimizer_1(M,r_gavetas_min=r_gavetas_min,r_gavetas_eq=r_gavetas_eq)




@app.post("/items_post/")
async def create_item(item: Item):
    return item.dict()