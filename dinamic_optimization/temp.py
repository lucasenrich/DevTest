import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import random
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.optimize import linprog
from itertools import combinations_with_replacement
import numpy as np
#import seaborn as sns
from collections import defaultdict
from sklearn.metrics import average_precision_score
    
class CajeroVirtual:
    def __init__(self, nu_100,nu_200,nu_500,nu_1000):
        self.nu_100 = nu_100
        self.nu_200 = nu_200
        self.nu_500 = nu_500
        self.nu_1000 = nu_1000
    def get_total(self):
        return self.nu_100*100+self.nu_200*200+self.nu_500*500+self.nu_1000*1000
        
    def get_billetes(self):
        return f'Billete de 100: {self.nu_100}\n Billete de 200: {self.nu_200}\n Billete de 500: {self.nu_500}\n Billete de 1000: {self.nu_1000}'
    
    def extraer(self, nu):
        billetes_extraidos=0
        if int(np.floor(nu/1000))>=1:
            if self.nu_1000>=int(np.floor(nu/1000)):            
                self.nu_1000 = self.nu_1000-int(np.floor(nu/1000))
                billetes_extraidos=int(np.floor(nu/1000))+billetes_extraidos
                nu = nu-int(np.floor(nu/1000))*1000
            else:
                billetes_extraidos=self.nu_1000
                nu = nu-self.nu_1000*1000
                self.nu_1000 = 0
                
            
        if int(np.floor(nu/500))>=1:
            if self.nu_500>=int(np.floor(nu/500)): 
                self.nu_500 = self.nu_500-int(np.floor(nu/500))
                billetes_extraidos=int(np.floor(nu/500))+billetes_extraidos
                nu = nu-int(np.floor(nu/500))*500
            else:
                billetes_extraidos=self.nu_500
                nu = nu-self.nu_500*500
                self.nu_500 = 0
                
        if int(np.floor(nu/200))>=1:
            if self.nu_200>=int(np.floor(nu/200)): 
                self.nu_200 = self.nu_200-int(np.floor(nu/200))
                billetes_extraidos=int(np.floor(nu/200))+billetes_extraidos
                nu = nu-int(np.floor(nu/200))*200
            else:
                billetes_extraidos=self.nu_200
                nu = nu-self.nu_200*200
                self.nu_200 = 0
                
        if int(np.floor(nu/100))>=1:
            if self.nu_100>=int(np.floor(nu/100)): 
                self.nu_100 = self.nu_100-int(np.floor(nu/100))
                billetes_extraidos=int(np.floor(nu/100))+billetes_extraidos
                nu = nu-int(np.floor(nu/100))*100
            else:
                billetes_extraidos=self.nu_100
                nu = nu-self.nu_100*100
                self.nu_100 = 0
        
        if nu == 0 and billetes_extraidos<40:
            return True
        else:
            return False
            

#%%
def minmax(x):
    x = np.array(x)
    return (x-min(x))/(max(x)-min(x))

def optimizer_1(M, iter_500=1000, iter_1000=1000, r_gavetas_min = defaultdict(lambda: None), r_gavetas_eq = defaultdict(lambda: None), max_distribucion=1):
    
    A_ineq = np.array([[1,1,1,1]])
    b_ineq = np.array([8000])
    
    A_eq = np.array([[100,200,500,1000]])
    b_eq = np.array([M])
    
    
    c = np.array([0.2,0.1,1,0.75])

    #arr = np.array([np.power(n,1),np.power(n,2),np.power(n,3),np.power(n,4)])
    #c = np.array(list((1/arr)*10))    
    print(c)
    resultados_exito = []
    resultado_p100200 =[]
    extracciones_valor = []
    extracciones_resultado = []
    perm = combinations_with_replacement([100, 200, 500, 1000], 4)
    perms = np.array(list(perm))
    perms_n = perms
    
    r_gavetas_min = defaultdict(lambda:None,r_gavetas_min)
    if r_gavetas_min['100'] is not None:    
        perms_n = []
        for k in range(len(perms)):
            gav_100 = np.sum(perms[k]==100)
            if r_gavetas_min['100'] <= gav_100:
                perms_n.append(perms[k])
    perms = perms_n
    
    if r_gavetas_min['200'] is not None:    
        perms_n = []
        for k in range(len(perms)):
            gav_200 = np.sum(perms[k]==200)
            if r_gavetas_min['200'] <= gav_200:
                perms_n.append(perms[k])
    perms = perms_n
    if r_gavetas_min['500'] is not None:    
        perms_n = []
        for k in range(len(perms)):
            gav_500 = np.sum(perms[k]==500)
            if r_gavetas_min['500'] <= gav_500:
                perms_n.append(perms[k])
    perms = perms_n
    if r_gavetas_min['1000'] is not None:    
        perms_n = []
        for k in range(len(perms)):
            gav_1000 = np.sum(perms[k]==1000)
            if r_gavetas_min['1000'] <= gav_1000:
                perms_n.append(perms[k])
            
    
    perms = perms_n
    
    
    
    r_gavetas_eq = defaultdict(lambda:None,r_gavetas_eq)
    if r_gavetas_eq['100'] is not None:    
        perms_n = []
        for k in range(len(perms)):
            gav_100 = np.sum(perms[k]==100)
            if r_gavetas_eq['100'] == gav_100:
                perms_n.append(perms[k])
                
    perms = perms_n
    
    if r_gavetas_eq['200'] is not None:    
        perms_n = []
        for k in range(len(perms)):
            gav_200 = np.sum(perms[k]==200)
            if r_gavetas_eq['200'] == gav_200:
                perms_n.append(perms[k])
                
    if r_gavetas_eq['500'] is not None:    
        perms_n = []
        for k in range(len(perms)):
            gav_500 = np.sum(perms[k]==500)
            if r_gavetas_eq['500'] == gav_500:
                perms_n.append(perms[k])
                
    if r_gavetas_eq['1000'] is not None:    
        perms_n = []
        for k in range(len(perms)):
            gav_1000 = np.sum(perms[k]==1000)
            if r_gavetas_eq['1000'] == gav_1000:
                perms_n.append(perms[k])
                
    
    
    perms = perms_n
    print(perms)
    
    extracciones_exitosas_p=[]
    distrs_k = []
    distrs_p = []
    e_sim=[]
    costo_oportunidad_iter = []
    
    for i_500 in range(0,8000,iter_500):
        print(i_500/8000)
        for i_1000 in range(0,8000,iter_1000):
        #i_1000 = i_500
            nu_100_min,nu_200_min,nu_500_min,nu_1000_min = 0,0,i_500,i_1000
            for k in range(len(perms)):
                extracciones_exitosas_k=[]
    
                nu_100_max,nu_200_max,nu_500_max,nu_1000_max = np.sum(perms[k]==100)*2000,np.sum(perms[k]==200)*2000,np.sum(perms[k]==500)*2000,np.sum(perms[k]==1000)*2000
    
                if nu_100_max<=nu_100_min:
                    nu_100_min = nu_100_max        
                if nu_200_max<=nu_200_min:
                    nu_200_min = nu_200_max
                if nu_500_max<=nu_500_min:
                    nu_500_min = nu_500_max
                if nu_1000_max<=nu_1000_min:
                    nu_1000_min = nu_1000_max
    
                bounds = [(nu_100_min,nu_100_max),(nu_200_min,nu_200_max),(nu_500_min,nu_500_max),(nu_1000_min,nu_1000_max)] 
                res = linprog(c, A_ub=A_ineq, b_ub=b_ineq, A_eq=A_eq, b_eq=b_eq,method = 'revised simplex', bounds = bounds)
                res_bill = np.floor(res.x)
    
                c1 = CajeroVirtual(res_bill[0],res_bill[1],res_bill[2],res_bill[3])
                while c1.get_total()>100:
                    #e = random.choices(population=df_means.importe_transaccion_vl.tolist(),
                    #    weights=df_means.prop_dia.tolist(),
                    #    k=1)[0]
                    e = np.round(np.exp(np.random.normal(8,1.5))/100,0)*100
                    #print(e)
                    if e>100 and e<50000:
                        e_sim.append(e)
                        extracciones_exitosas_k.append(c1.extraer(e))
    
                extracciones_exitosas_p.append(np.mean(extracciones_exitosas_k))
                
                costo_oportunidad_iter.append((c1.nu_500*500+c1.nu_1000*1000)*(0.37/365))
    
                distrs_k.append(res_bill)
                p_ok = sum(res_bill[0:2])/sum(res_bill)
                distrs_p.append(p_ok)
        
    distrs_p_norm = minmax(np.array(distrs_p))
    extracciones_exitosas_p_norm = minmax(np.array(extracciones_exitosas_p))
    dict_res = {'distribucion':distrs_p,'extracciones_exitosas':extracciones_exitosas_p}
#        return dict_res
    #print(distrs_k)
    res_comb = pd.DataFrame(distrs_k,columns = ["nu_100","nu_200","nu_500","nu_1000"])
    res_df = pd.concat([pd.DataFrame(dict_res),res_comb],axis=1)
    
    res_df['distribucion'] = res_df['distribucion'].round(2)
    res_df['extracciones_exitosas'] = res_df['extracciones_exitosas'].round(2)
    res_df['ratio'] = res_df['distribucion']/res_df['extracciones_exitosas']
    res_df = res_df.groupby(['nu_100','nu_200','nu_500','nu_1000'], as_index=False).agg({'distribucion':'mean','extracciones_exitosas':'mean'}).reset_index(drop=True)
                   
    model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    x_tr = np.array(dict_res['distribucion']).reshape(-1, 1)
    y_tr = np.array(dict_res['extracciones_exitosas']).reshape(-1, 1)
    model.fit(x_tr, y_tr)
    

    
    y_pred = model.predict(x_tr).reshape(1,-1)[0]
    y_eval = y_tr.reshape(1,-1)[0]
    mAP=1-np.mean(abs(y_pred-y_eval)/y_eval)
    if mAP<0.725:
        print("Modelo Cuadratico entre %[100&200] y % de extracciones fallidas incorrecto, mAP: ",mAP)
    y1 = model.predict(np.linspace(0,1,100).reshape(-1, 1)).reshape(1, -1)[0]
    x1 = np.linspace(0,1,100)
    res_df['dif_teo'] = (res_df['distribucion']-x1[np.argmax(y1)]).abs()
    res_df.sort_values('dif_teo',ascending = True).reset_index(drop=True)
    res_df['distribucion'] = res_df['distribucion'].round(2)
    #res_df = res_df[res_df.distribucion<=max_distribucion].reset_index(drop=True)
    
    #res_df = res_df[res_df.extracciones_exitosas>0.8].reset_index(drop=True)
    
    res_df = res_df[res_df.dif_teo==min(res_df.dif_teo)].reset_index(drop=True)
    res = res_df.groupby(['distribucion','dif_teo','nu_100','nu_200','nu_500','nu_1000'],as_index=False).agg({'extracciones_exitosas':'mean'})
    res['mAP_teorico'] = mAP
    return res.iloc[0].to_dict()#,x1[np.argmax(y1)],y1[np.argmax(y1)]
