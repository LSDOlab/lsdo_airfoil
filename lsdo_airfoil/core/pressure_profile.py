import m3l
import csdl
import numpy as np
from lsdo_airfoil.core.airfoil_model_csdl import AirfoilModelCSDL, CpModelCSDL
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL


class PressureProfile(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('component', default=None)
        self.parameters.declare('airfoil_name')
        self.parameters.declare('compute_control_points', default=False)
        self.parameters.declare('num_nodes', types=int, default=1)
        self.parameters.declare('use_inverse_cl_map', types=bool)
        self.m3l_var_list_cl = None
        self.m3l_var_list_re = None
    
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
            m3l_var_list_cl=self.m3l_var_list_cl,
            m3l_var_list_re=self.m3l_var_list_re,
        )

        return csdl_model
        
        # return super().compute()
    
    def evaluate(self, cl_list:list=[], re_list:list=[]) -> tuple:
        self.name = 'airfoil_ml_model'
        self.arguments = {}
        self.m3l_var_list_cl = cl_list
        self.m3l_var_list_re = re_list
   

        cp_upper_list = []
        cp_lower_list = []
        cd_list = []

        counter = 0
        for cl in cl_list:
            self.arguments[cl.name] = cl
            
            Re = re_list[counter]
            self.arguments[Re.name] = Re
            counter += 1
   
            shape = cl.shape
            num_eval = shape[0] * shape[1]
            
            cp_upper = m3l.Variable(name=f'{cl.name.split("_")[0]}_cp_upper', shape=(num_eval, 100), operation=self)
            cp_lower = m3l.Variable(name=f'{cl.name.split("_")[0]}_cp_lower', shape=(num_eval, 100), operation=self)

            cd = m3l.Variable(name=f'{cl.name.split("_")[0]}_cd', shape=(num_eval, 100), operation=self)

            cp_upper_list.append(cp_upper)
            cp_lower_list.append(cp_lower)
            cd_list.append(cd)

        return cp_upper_list, cp_lower_list, cd_list
    
class NodalPressureProfile(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
    
    def compute(self):
        surface_names = self.parameters['surface_names']  
        surface_shapes = self.parameters['surface_shapes']

        csdl_module = ModuleCSDL()

        for i in range(len(surface_names)):
            name = surface_names[i]
            if 'tail' in name:
                num_nodes = surface_shapes[i][1] - 1
            else:
                num_nodes = surface_shapes[i][1]
            ml_pressures_upper = csdl_module.register_module_input(f'{name.split("_")[0]}_cp_upper', shape=(num_nodes, 100))
            ml_pressures_lower = csdl_module.register_module_input(f'{name.split("_")[0]}_cp_lower', shape=(num_nodes, 100))
    
            # Identity map
            csdl_module.register_module_output(f'{name.split("_")[0]}_oml_cp_upper', ml_pressures_upper * 1)
            csdl_module.register_module_output(f'{name.split("_")[0]}_oml_cp_lower', ml_pressures_lower * 1)

        return csdl_module

    def evaluate(self, ml_pressure_upper, ml_pressure_lower, nodal_pressure_mesh) -> tuple:
        surface_names = self.parameters['surface_names']  
        surface_shapes = self.parameters['surface_shapes']
        
        self.nodal_forces_meshes = nodal_pressure_mesh
        self.name = f"{''.join(surface_names)}_ml_pressure_mapping_model"
        self.arguments = {}

        for i in range(len(surface_names)):
            surface_name = surface_names[i].split("_")[0]
            self.arguments[surface_name + '_cp_upper'] = ml_pressure_upper[i]
            self.arguments[surface_name + '_cp_lower'] = ml_pressure_lower[i]

        oml_pressures_upper = []
        oml_pressures_lower = []
        shapes = [(100, surface_shapes[0][1])]
        for i in range(len(surface_names)):
            surface_name = surface_names[i].split("_")[0]
            # shape = (surface_shapes[i][0], surface_shapes[i][1])
            shape = shapes[i]
            print(shape)
            oml_pressure_upper = m3l.Variable(name=f'{surface_name}_oml_cp_upper', shape=shape, operation=self)
            oml_pressure_lower = m3l.Variable(name=f'{surface_name}_oml_cp_lower', shape=shape, operation=self)

            oml_pressures_upper.append(oml_pressure_upper)
            oml_pressures_lower.append(oml_pressure_lower)
        
        # exit()
        return oml_pressures_upper, oml_pressures_lower
    
class NodalForces(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        pass

    def evaluate(self, oml_pressures_upper:m3l.Variable, oml_pressures_lower:m3l.Variable, normals_upper:m3l.Variable, normals_lower:m3l.Variable, upper_ml_mesh, lower_ml_mesh, upper_ml_vlm_mesh, lower_ml_vlm_mesh) -> tuple:
        self.name = 'ml_nodal_force_evaluation'
        
        self.upper_ml_mesh = upper_ml_mesh
        self.lower_ml_mesh = lower_ml_mesh
        self.upper_ml_vlm_mesh = upper_ml_vlm_mesh
        self.lower_ml_vlm_mesh = lower_ml_vlm_mesh
        
        self.arguments = {}
        self.arguments['oml_pressures_upper'] = oml_pressures_upper
        self.arguments['oml_pressures_lower'] = oml_pressures_lower
        self.arguments['normals_upper'] = normals_upper
        self.arguments['normals_lower'] = normals_lower

        ml_f_upper = m3l.Variable('ml_f_upper', shape=oml_pressures_upper.shape, operation=self)
        ml_f_lower = m3l.Variable('ml_f_lower', shape=oml_pressures_lower.shape, operation=self)
        return ml_f_upper, ml_f_lower

    def compute(self):
        # just going to use static values for now, switch to get_map etc later
        upper_ml_mesh = self.upper_ml_mesh.value
        lower_ml_mesh = self.lower_ml_mesh.value
        upper_ml_vlm_mesh = self.upper_ml_vlm_mesh.value
        lower_ml_vlm_mesh = self.lower_ml_vlm_mesh.value

        csdl_module = ModuleCSDL()

        pressures_upper_csdl = csdl_module.register_module_input('oml_pressures_upper', shape=self.arguments['oml_pressures_upper'].shape)
        pressures_lower_csdl = csdl_module.register_module_input('oml_pressures_lower', shape=self.arguments['oml_pressures_lower'].shape)
        normals_upper_csdl = csdl_module.register_module_input('normals_upper', shape=self.arguments['normals_upper'].shape)
        normals_lower_csdl = csdl_module.register_module_input('normals_lower', shape=self.arguments['normals_lower'].shape)

        # upper_ml_mesh_csdl = csdl_module.register_module_input('upper_ml_mesh', shape=upper_ml_mesh.shape, val=upper_ml_mesh)
        # lower_ml_mesh_csdl = csdl_module.register_module_input('lower_ml_mesh', shape=lower_ml_mesh.shape, val=lower_ml_mesh)
        # upper_ml_vlm_mesh_csdl = csdl_module.register_module_input('upper_ml_vlm_mesh', shape=upper_ml_vlm_mesh.shape, val=upper_ml_vlm_mesh)
        # lower_ml_vlm_mesh_csdl = csdl_module.register_module_input('lower_ml_vlm_mesh', shape=lower_ml_vlm_mesh.shape, val=lower_ml_vlm_mesh)

        upper_full_distances = np.linalg.norm(upper_ml_mesh[0:-1,:,:]-upper_ml_mesh[1:,:,:], axis=2)
        dist_upper = np.zeros(upper_ml_mesh.shape[0:2])
        dist_upper[0:-1,:] = upper_full_distances/2
        dist_upper[1:,:] += upper_full_distances/2
        
        lower_full_distances = np.linalg.norm(lower_ml_mesh[0:-1,:,:]-lower_ml_mesh[1:,:,:], axis=2)
        dist_lower = np.zeros(lower_ml_mesh.shape[0:2])
        dist_lower[0:-1,:] = lower_full_distances/2
        dist_lower[1:,:] += lower_full_distances/2

        width_upper = np.linalg.norm(upper_ml_vlm_mesh[:,0:-1,:]-upper_ml_vlm_mesh[:,1:,:], axis=2)
        width_lower = np.linalg.norm(lower_ml_vlm_mesh[:,0:-1,:]-lower_ml_vlm_mesh[:,1:,:], axis=2)

        areas_upper = np.multiply(dist_upper,width_upper)
        areas_lower = np.multiply(dist_lower,width_lower)

        areas_upper_csdl = csdl_module.register_module_input('areas_upper', shape=areas_upper.shape, val=areas_upper)
        areas_lower_csdl = csdl_module.register_module_input('areas_lower', shape=areas_lower.shape, val=areas_lower)

        forces_upper = csdl.expand(areas_upper_csdl*pressures_upper_csdl,(100,40,3),'ij->ijk')*csdl.reshape(normals_upper_csdl, (100,40,3))
        forces_lower = csdl.expand(areas_lower_csdl*pressures_lower_csdl,(100,40,3),'ij->ijk')*csdl.reshape(normals_lower_csdl, (100,40,3))

        csdl_module.register_module_output('ml_f_upper', forces_upper)
        csdl_module.register_module_output('ml_f_lower', forces_lower)

        return csdl_module















