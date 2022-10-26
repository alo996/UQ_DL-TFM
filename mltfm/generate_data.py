import numpy as np
import pandas as pd
from tqdm import tqdm
import tables
from scipy.special import ellipk, ellipe

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)
        
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def inCircle(X,Y,x,y,r):
    return (X-x)**2 +(Y-y)**2 <= r**2
        
class EventGenerator:
    def __init__(self, params):
        self.params = params
        self.displacement_mesh = self.generate_mesh(0, 1, 0, 1, params['resolutionX'])
        self.force_mesh = self.generate_mesh(0,1,0,1,params['resolutionY']) 
        self.E = 1000
        self.f_res = 1/(self.params['resolutionY']-1)

    def generate_mesh(self, left, right, front, back, resolution):
        X, Y = np.meshgrid(np.linspace(left,right,resolution), np.linspace(front,back,resolution))
        mesh = np.array([[(X[i][j],Y[i][j]) for j in range(len(X[i]))] for i in range(len(X))])
        return mesh

    def write_PointForces(self, PointForces):
        self.data_force.append(np.array([PointForces]))

    def write_Displacement(self, displacement):
        self.data_disp.append(np.array([displacement]))
      
    def min_dist(self,PF,point,R):
        return np.min(np.sqrt((PF.x_coord-point[0])**2 + (PF.y_coord-point[1])**2)-2*R)
 
    def generate_PointForces(self):
        num_i = self.params['resolutionY']
        counter = 0
        PointForcemesh = np.zeros((self.params['resolutionY'],self.params['resolutionY'],2))
        PointForces = pd.DataFrame({"x_coord":[],"y_coord":[],"force":[],"gamma":[],"radius":[]})
        while counter < np.random.uniform(10, 50):
            R = np.random.uniform(0.01,0.05)
            point = np.random.uniform(0+R+0.05,1-R-0.05,2)
            force = np.random.uniform(self.params['traction_min'],self.params['traction_max'])
            force = force/self.E
            gamma = np.random.uniform(0,2*np.pi)
            if counter == 0 or self.min_dist(PointForces,point,R) > 0.001:
                PointForces = PointForces.append({"x_coord":point[0],
                                                  "y_coord":point[1],
                                                  "force":force,
                                                  "gamma":gamma,
                                                  "radius":R         },
                                                  ignore_index=True    )
                x_f,y_f = pol2cart(force,gamma)
                PointForcemesh[inCircle(self.force_mesh[:,:,0],self.force_mesh[:,:,1],point[0],point[1],R)]+=np.array([x_f,y_f])
                counter+=1

        return PointForcemesh, PointForces
   
    def generate_displacement(self, PointForces):
        raise NotImplementedError
    
    def generate(self, event_num):
        f_data_disp = tables.open_file('displacements_3000.h5', mode='w')
        atom = tables.Float64Atom()
        self.data_disp = f_data_disp.create_earray(f_data_disp.root,'data',atom,(0,self.params['resolutionX'],self.params['resolutionX'],2))
        f_data_force = tables.open_file('tractions_3000.h5', mode='w')
        self.data_force = f_data_force.create_earray(f_data_force.root,'data',atom,(0,self.params['resolutionY'],self.params['resolutionY'],2))
        for i in tqdm(range(event_num)):
            PointForcesmesh, PointForces = self.generate_PointForces()
            displacement = self.generate_displacement(PointForces)
            self.write_PointForces(PointForcesmesh)
            self.write_Displacement(displacement)
        f_data_disp.close()
        f_data_force.close()

class AnalyticalEventGenerator(EventGenerator):

    def analytical(self, point, traction,R):        
        p0=traction[0]
        gamma=traction[1] 
        r,theta = cart2pol(point[0],point[1])
        if r<R:
            if r < 1e-4:
                N1 = 2*np.pi
                N2 = np.pi
                N3 = 0
                N4 = np.pi
            zeta1 = r**2/R**2
            E0 = ellipe(zeta1)
            K0 = ellipk(zeta1)
            N1 = 4*E0
            N2 = (4*np.cos(2*theta)*((r**2+R**2)*E0 + (r**2-R**2)*K0))/(3*r**2) + 4*np.sin(theta)**2*E0
            N3 = (2*np.sin(2*theta)*((r**2-2*R**2)*E0 + 2*(R**2-r**2)*K0))/(3*r**2)
            N4 = 4*np.cos(theta)**2*E0 - (4*np.cos(2*theta)*( (r**2+R**2)*E0 + (r**2-R**2)*K0))/(3*r**2)
        else:
            zeta2 = R**2/r**2
            E0 = ellipe(zeta2)
            K0 = ellipk(zeta2)
            N1 = (4*(r**2*E0 + (R**2-r**2)*K0))/(r*R)
            N2 = ((6*r**2 - 2*(r**2-2*R**2)*np.cos(2*theta))*E0 + 2*(r**2-R**2)*(np.cos(2*theta)-3)*K0)/(3*r*R)
            N3 = (2*np.sin(2*theta)*((r**2-2*R**2)*E0 + (R**2-r**2)*K0))/(3*r*R)
            N4 = ((6*r**2 + 2*(r**2-2*R**2)*np.cos(2*theta))*E0 - 2*(r**2-R**2)*(np.cos(2*theta)+3)*K0)/(3*r*R)
        ux = R*(1+self.params['nu'])/(np.pi) * (( (1-self.params['nu'])*N1 + self.params['nu']*N2)*p0*np.cos(gamma) - self.params['nu']*N3*p0*np.sin(gamma))
        uy = R*(1+self.params['nu'])/(np.pi) * (-self.params['nu']*N3*p0*np.cos(gamma) + ((1-self.params['nu'])*N1 + self.params['nu']*N4)*p0*np.sin(gamma))
        return ux,uy

    def generate_displacement(self, PointForces):
        displacement = np.zeros((len(self.displacement_mesh),len(self.displacement_mesh[0]),2))
        for index, row in PointForces.iterrows():
            trafo = np.array([-row.x_coord,-row.y_coord])
            force = np.array([row.force,row.gamma])
            displacement += np.array([[self.analytical(self.displacement_mesh[i][j]+trafo, force, row.radius)
                                       if self.analytical(self.displacement_mesh[i][j]+trafo, force, row.radius) is not np.nan
                                       else self.analytical(self.displacement_mesh[i][(j-1)%len(self.displacement_mesh[i])]+trafo, force, row.radius)
                                       for j in range(len(self.displacement_mesh[i]))
                                      ]
                                      for i in range(len(self.displacement_mesh))
                                     ])
        return displacement


Gen = AnalyticalEventGenerator({'resolutionX':104,
                                'resolutionY':104,
                                'traction_min':0,
                                'traction_max':500,
                                'nu':0.49          })

count = 25000
Gen.generate(count)
