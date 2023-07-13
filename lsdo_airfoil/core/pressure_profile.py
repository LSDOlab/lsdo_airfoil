import m3l
import csdl
import numpy as np
from lsdo_airfoil.core.airfoil_model_csdl import AirfoilModelCSDL, CpModelCSDL


class PressureProfile(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('component', default=None)
        self.parameters.declare('airfoil_name')
        self.parameters.declare('compute_control_points', default=False)
        self.parameters.declare('num_nodes', types=int, default=1)
        self.parameters.declare('use_inverse_cl_map', types=bool)
        self.m3l_var_list = None
    
    def compute(self) -> csdl.Model:
        num_nodes = self.parameters['num_nodes']
        airfoil_name = self.parameters['airfoil_name']
        compute_control_points = self.parameters['compute_control_points']
        use_inverse_cl_map = self.parameters['use_inverse_cl_map']
        
        csdl_model = CpModelCSDL(
            num_nodes=num_nodes,
            airfoil_name=airfoil_name,
            compute_control_points=compute_control_points,
            use_inverse_cl_map=use_inverse_cl_map,
            m3l_var_list=self.m3l_var_list,
        )

        return csdl_model
        
        # return super().compute()
    
    def evaluate(self, cl_list:list=[]) -> tuple:
        self.name = 'airfoil_ml_model'
        self.arguments = {}
        self.m3l_var_list = cl_list
   

        cp_upper_list = []
        cp_lower_list = []
        cd_list = []

        for cl in cl_list:
            self.arguments[cl.name] = cl
   
            shape = cl.shape
            num_eval = shape[0] * shape[1]
            
            cp_upper = m3l.Variable(name=f'{cl.name.split("_")[0]}_cp_upper', shape=(num_eval, 100), operation=self)
            cp_lower = m3l.Variable(name=f'{cl.name.split("_")[0]}_cp_lower', shape=(num_eval, 100), operation=self)

            cd = m3l.Variable(name=f'{cl.name.split("_")[0]}_cd', shape=(num_eval, 100), operation=self)

            cp_upper_list.append(cp_upper)
            cp_lower_list.append(cp_lower)
            cd_list.append(cd)

        return cp_upper_list, cp_lower_list, cd_list