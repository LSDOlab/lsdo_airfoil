import pickle
from lsdo_airfoil import UIUC_INTERPOLATION
from lsdo_airfoil.utils.compute_b_spline_mat import get_bspline_mtx
from lsdo_airfoil.utils.make_skewed_distribution import make_skewed_normalized_distribution
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import itertools
from torch.autograd import Variable


with open(UIUC_INTERPOLATION / 'airfoil_interpolation_7.pickle', 'rb') as handle:
    airfoil_interp_dict = pickle.load(handle)


num_dense_pts = 150
num_ctrl_pts = 145
# x_interp = make_skewed_normalized_distribution(num_dense_pts)
B_dense = get_bspline_mtx(num_cp=num_ctrl_pts, num_pt=num_dense_pts, order=4).todense().astype(np.float32)
control_points = make_skewed_normalized_distribution(num_ctrl_pts, half_cos=False, power=1.25)
x_interp =  make_skewed_normalized_distribution(num_dense_pts, half_cos=False, power=1.25)#  np.linspace(0, 1, num_dense_pts) #
interpolation_error_lower = 0
ax = plt.gca()

error_lower = []
error_upper = []
counter = 0 

control_points_array = np.zeros((1441, 290)).astype(np.float32)
dense_points_array = np.zeros((1441, 300)).astype(np.float32)

for airfoil, airfoil_interp in airfoil_interp_dict.items():
    if airfoil_interp['max_b_spline_error_lower'] > 0.05:# or airfoil_interp['max_b_spline_error_lower'] > 100:
        if airfoil in []: ##['rc0864c.dat', 'lnv109a.dat', 'e662.dat','naca6412.dat', 'hq2010.dat', 'rg15.dat', 'stf86361.dat', 'glennmartin2.dat',
                       #'isa960.dat','rg1495.dat','sc20403.dat','goe238.dat','cap21c.dat','fx80080.dat']: #['eiffel428.dat', 'coanda1.dat', 'isa571.dat']:
            pass
        else:
            pass
            print(airfoil, airfoil_interp['max_b_spline_error_upper'], airfoil_interp['max_b_spline_error_lower'])
            
            error_upper.append(airfoil_interp['max_b_spline_error_upper'])
            error_lower.append(airfoil_interp['max_b_spline_error_lower'])

            x_upper = airfoil_interp['x_upper_raw']
            x_lower = airfoil_interp['x_lower_raw']
            y_upper = airfoil_interp['y_upper_raw']
            y_lower = airfoil_interp['y_lower_raw']

            cy_upper = airfoil_interp['upper_ctr_pts']
            cy_lower = airfoil_interp['lower_ctr_pts']

            # print(cy_upper.shape)
            
            # print(cy_lower.shape)

            cy_upper[0] = 0
            cy_upper[-1] = 0
            
            control_points_array[counter, 0:145] = cy_upper
            control_points_array[counter, 145:] = cy_lower


            counter += 1

            camber = B_dense @ cy_upper
            thickness = B_dense @ cy_lower

            dense_points_array[counter, 0:150] = camber + 0.5*thickness
            dense_points_array[counter, 150:] = camber - 0.5*thickness

            
            color = next(ax._get_lines.prop_cycler)['color']
            
            plt.scatter(x_upper, y_upper, color=color, s=5)
            plt.scatter(x_lower, y_lower, color=color, s=5)
            plt.plot(x_interp, camber + 0.5*thickness, color=color, label=airfoil)
            plt.plot(x_interp, camber - 0.5*thickness, color=color)
            # plt.plot(x_interp, camber, color=color, label=airfoil)
            # plt.plot(x_interp, thickness, color=color)
            
            plt.axis('equal')
            plt.legend()
    else:
            x_upper = airfoil_interp['x_upper_raw']
            x_lower = airfoil_interp['x_lower_raw']
            y_upper = airfoil_interp['y_upper_raw']
            y_lower = airfoil_interp['y_lower_raw']

            cy_upper = airfoil_interp['upper_ctr_pts']
            cy_lower = airfoil_interp['lower_ctr_pts']

            cy_upper[0] = 0
            cy_upper[-1] = 0

            # print(cy_upper)
            
            # print(cy_lower)
            
            control_points_array[counter, 0:145] = cy_upper
            control_points_array[counter, 145:] = cy_lower

            camber = B_dense @ cy_upper
            thickness = B_dense @ cy_lower

            dense_points_array[counter, 0:150] = camber + 0.5*thickness
            dense_points_array[counter, 150:] = camber - 0.5*thickness

            counter += 1

            # if counter %50 ==0:
            #     print(airfoil)
            #     # print(cy_upper)
            #     camber = B_dense @ cy_upper
            #     thickness = B_dense @ cy_lower
            #     color = next(ax._get_lines.prop_cycler)['color']
                
            #     # plt.scatter(x_upper, y_upper, color=color, s=5)
            #     # plt.scatter(x_lower, y_lower, color=color, s=5)
            #     plt.plot(x_interp, camber + 0.5*thickness, color=color, label=airfoil)
            #     plt.plot(x_interp, camber - 0.5*thickness, color=color)

            #     # plt.scatter(control_points, cy_upper, color=color, label=airfoil, s=5)
            #     # plt.scatter(control_points, cy_lower, color=color, s=5)
            #     # plt.plot(x_interp, camber, color=color, label=airfoil)
            #     # plt.plot(x_interp, thickness, color=color)
                
            #     plt.axis('equal')
            #     plt.legend()

        # cy_upper = airfoil_interp['upper_ctr_pts']
        # cy_lower = airfoil_interp['lower_ctr_pts']

        # # control_points[counter, 0:250] = cy_upper
        # # control_points[counter, 250:] = cy_lower

        # # dense_points[counter, 0:300] = B_dense @ cy_upper
        # # dense_points[counter, 300:] = B_dense @ cy_lower

        # error_upper.append(airfoil_interp['max_b_spline_error_upper'])
        # error_lower.append(airfoil_interp['max_b_spline_error_lower'])
        # counter += 1

if False:
    # print(dense_points_array)
    print(counter)
    plt.show()
    # exit()

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

noise_dim = 12
latent_dim = 20
G_input_dim = noise_dim + latent_dim
latent_parameters = np.random.randn(1441, G_input_dim)
G_output_dim = control_points_array.shape[1]
D_input_dim = dense_points_array.shape[1]


data_curve_y = torch.from_numpy(dense_points_array).to(device)
data_latent_parameters = torch.from_numpy(latent_parameters).to(device)

batch_size = 50

dataset = torch.utils.data.TensorDataset(*(data_curve_y, data_latent_parameters))
data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, drop_last=False)

B_dense = torch.tensor(B_dense).to(device=device)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.G_net = nn.Sequential(nn.Linear(G_input_dim, 120), nn.ReLU(), 
                    #   nn.Linear(120, 120), nn.ReLU(),
                    #   nn.Linear(120, 120), nn.ReLU(),
                    #   nn.Linear(120, 120), nn.ReLU(), 
                    #   nn.Linear(120, G_output_dim))
                    # nn.Linear(120, 150), nn.ReLU(), # deeper architecture 1
                    # nn.Linear(150, 200), nn.ReLU(),
                    # nn.Linear(200, 150), nn.ReLU(), 
                    # nn.Linear(150, 120), nn.ReLU(), 
                    # nn.Linear(120, G_output_dim))

                    nn.Linear(120, 150), nn.ReLU(), # deeper architecture 2
                    nn.Linear(150, 200), nn.ReLU(),
                    nn.Linear(200, 150), nn.ReLU(), 
                    nn.Linear(150, 150), nn.ReLU(), 
                    nn.Linear(150, 120), nn.ReLU(), 
                    nn.Linear(120, 100), nn.ReLU(), 
                    nn.Linear(100, G_output_dim))

        # self.G_net = nn.Sequential(nn.Linear(G_input_dim, 80), nn.ReLU(), 
        #               nn.Linear(80, 120), nn.ReLU(),
        #               nn.Linear(120, 120), nn.ReLU(),
        #               nn.Linear(120, 120), nn.ReLU(), 
        #               nn.Linear(120, 80), nn.ReLU(),
        #               nn.Linear(80, 60), nn.ReLU(), 
        #               nn.Linear(60, G_output_dim))

    def forward(self, zc):
        control_camber_thickness_fake = self.G_net(zc)
        control_camber_thickness_fake[:,0] = 0
        control_camber_thickness_fake[:,144] = 0
        control_camber_thickness_fake[:,145] = 0
        control_camber_thickness_fake[:,289] = 0
        curve_camber_fake = ((torch.matmul(B_dense,control_camber_thickness_fake[:,0:145].T)).T)
        curve_thickness_fake = ((torch.matmul(B_dense,control_camber_thickness_fake[:,145:].T)).T)

        # print('max_camber_loss',max_camber_loss)
        # print('max_thickness_loss',max_thickness_loss)
        # print('min_camber_loss',min_camber_loss)
        # max_thickness_fake = torch.max()

        curve_camber_thickness_fake = torch.cat((curve_camber_fake, curve_thickness_fake),-1)
        

        cp_adjacent_camber = torch.norm(control_camber_thickness_fake[:,1:145] - control_camber_thickness_fake[:,:144], dim = 1)
        cp_adjacent_thickness = torch.norm(control_camber_thickness_fake[:,146:] - control_camber_thickness_fake[:,145:-1], dim = 1)
        r_adjacent_thickness = torch.mean(cp_adjacent_thickness)
        r_adjacent_camber = torch.mean(cp_adjacent_camber)
        # print('r_adjacent_thickness',r_adjacent_thickness)
        # print('r_adjacent_camber',r_adjacent_camber)

        # cp_nonintersection = control_camber_thickness_fake[:,1:144] -control_camber_thickness_fake[:,146:289] 
        cp_nonintersection = control_camber_thickness_fake[:,146:289] 
        r_nonintersection = torch.mean(torch.maximum(torch.zeros(cp_nonintersection.shape).to(device=device), -1*cp_nonintersection))
        
        curve_upper_fake = (curve_camber_fake) + 0.5*curve_thickness_fake
        curve_lower_fake = (curve_camber_fake) - 0.5*curve_thickness_fake
        curve_y_coords_fake = torch.cat((curve_upper_fake, curve_lower_fake),-1)

        # p_non_interseaction = curve_lower_fake - curve_upper_fake
        # r_nonintersection = torch.mean(torch.maximum(torch.zeros(p_non_interseaction.shape).to(device=device), p_non_interseaction))

        return curve_camber_thickness_fake , curve_y_coords_fake , r_adjacent_thickness, r_adjacent_camber, r_nonintersection, control_camber_thickness_fake

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.D_net = nn.Sequential(nn.Linear(D_input_dim,300), nn.ReLU(), 
                    #   nn.Linear(300, 400), nn.ReLU(),
                    #   nn.Linear(400, 300), nn.ReLU(),
                    #   nn.Linear(300, 300), nn.ReLU(),
                    #   nn.Linear(300, 100), nn.ReLU())
                    nn.Linear(300, 300), nn.ReLU(),
                    nn.Linear(300, 300), nn.ReLU(),
                    nn.Linear(300, 300), nn.ReLU(),
                    nn.Linear(300, 200), nn.ReLU(),
                    nn.Linear(200, 100), nn.ReLU())
        self.adv_layer = nn.Sequential(nn.Linear(100,1))
        # self.adv_layer = nn.Sequential(nn.Sigmoid())
        self.latent_layer = nn.Sequential(nn.Linear(100, latent_dim))
    def forward(self, curve_y_coords_fake):
        out = self.D_net(curve_y_coords_fake)
        validity = self.adv_layer(out)
        latent_code = self.latent_layer(out)
        return validity, latent_code


loss_G_list = []
loss_D_list = []
loss_info_list = []
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
net_G = Generator().to(device)
net_D = Discriminator().to(device)

adversarial_loss = torch.nn.MSELoss()
adversarial_loss_2 = torch.nn.MSELoss()
adversarial_loss_svd = torch.nn.MSELoss()

# adversarial_loss = torch.nn.BCELoss()
# adversarial_loss_2 = torch.nn.BCELoss()
continuous_loss = torch.nn.MSELoss()
continuous_loss_2 = torch.nn.MSELoss()

reconstruction_loss = torch.nn.MSELoss() 
reconstruction_loss_val = 0

lambda_con = 2.
lambda_svd = 0.01


test_size = 5000

def train(
        net_D, net_G, 
        data_iter, num_epochs, 
        lr_D, lr_G, lr_info, 
        noise_dim, latent_dim,
        lambda_nonintersection,
        lambda_adjacent,
        lambda_adjacent_2,

    ):
    for w in net_D.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in net_G.parameters():
        nn.init.normal_(w, 0, 0.02)
    trainer_D = torch.optim.Adam(net_D.parameters(), lr=lr_D)#betas=(opt.b1, opt.b2)
    scheduler_D = torch.optim.lr_scheduler.StepLR(trainer_D, step_size=100, gamma=0.8)
    trainer_G = torch.optim.Adam(net_G.parameters(), lr=lr_G)
    scheduler_G = torch.optim.lr_scheduler.StepLR(trainer_G, step_size=100, gamma=0.8)
    trainer_info = torch.optim.Adam(
        itertools.chain(net_G.parameters(), net_D.parameters()), lr=lr_info)
    reconstruction_loss_updated = 0
    
    for epoch in range(num_epochs):
        for i, (Xy,Xc) in enumerate(data_iter):
            # print(i)
            batch_size = Xy.shape[0]
            Z = torch.normal(0, 1, size=(batch_size, noise_dim)).to(device=device)
            C = torch.normal(0, 1, size=(batch_size, latent_dim)).to(device=device)

            
            if (epoch>0) and (epoch % 50 == 0):
                dv = torch.cat((Z,C),-1).to(device=device)
                dv.requires_grad_(True)
                num_steps = 100
                k = 99
                reconstruction_loss_list = []
                # for j in range(batch_size):
                optimizer = torch.optim.Adam([dv], lr=0.025)
                Gen = Generator().to(device=device)
                Gen.load_state_dict(net_G.state_dict())
                Gen.requires_grad_(False)
                for num in range(num_steps):
                    optimizer.zero_grad()
                    _, upper_lower, _, _,_,_ = Gen(dv)
                    reconstruction_loss_eval = reconstruction_loss(upper_lower, Xy)
                    reconstruction_loss_eval.backward(retain_graph=True)
                    if num % k == 0:
                        print('Reconstruction loss for batch {}'.format(i) + 
                            ':MSE loss: {}'.format(reconstruction_loss_eval))
                    optimizer.step()
               
                #  Train Generator
                    trainer_G.zero_grad()
                    ones = torch.ones((batch_size,), device=Z.device)
                    _, G_Xy,r_adjacent,r_adjacent_camber, r_non_intersection, _ = net_G(dv)
                    validity, _ = net_D(G_Xy)
                    loss_G = adversarial_loss(validity, ones.reshape(validity.shape)) + 6 * reconstruction_loss_eval + lambda_nonintersection * r_non_intersection + lambda_adjacent * r_adjacent + lambda_adjacent_2 * r_adjacent_camber
                    loss_G.backward(retain_graph=True)
                    # scheduler_G.step()
                    
                
                    #  Train Discriminator
                    trainer_D.zero_grad()
                    ones = torch.ones((batch_size,), device=torch.cat((Xy,Xc),-1).device)
                    zeros = torch.zeros((batch_size,), device=torch.cat((Xy,Xc),-1).device)            
                    real_pred,_ = net_D(Xy)
                    fake_pred,_ = net_D(G_Xy)
                    loss_D = (adversarial_loss(real_pred, ones.reshape(real_pred.shape)) +
                            adversarial_loss(fake_pred, zeros.reshape(fake_pred.shape))) + 6 * reconstruction_loss_eval + lambda_nonintersection * r_non_intersection+ lambda_adjacent * r_adjacent + lambda_adjacent_2 * r_adjacent_camber
                    loss_D.backward(retain_graph=True)
                    scheduler_D.step()
                    

                    trainer_G.step()
                    trainer_D.step()

                    # Information Loss
                    # trainer_info.zero_grad()
                    _, G_Xy,r_adjacent,r_adjacent_camber,r_non_intersection, _ = net_G(dv)
                    _, pred_c = net_D(G_Xy)
                    loss_info = lambda_con * continuous_loss(pred_c, C) + 6 * reconstruction_loss_eval +  lambda_nonintersection * r_non_intersection + lambda_adjacent * r_adjacent + lambda_adjacent_2 * r_adjacent_camber
                    loss_info.backward(retain_graph=True)
                    trainer_info.step() 

                    reconstruction_loss_list.append(reconstruction_loss_eval)
                # if i == 29:
                #     print('Batch mean MSE reconstruction loss {}'.format(torch.mean(torch.tensor(reconstruction_loss_list))))

                # reconstruction_loss_updated = copy.copy(10 * reconstruction_loss_eval)
            

            # Additional training objective:
            # Every 25 major training epochs, we use the generator and compute 5000 random camber + thickness control points 
            # Then we compute the SVD on those control points and compute the MSE loss of the singular values between the fake and real SVD
            # We add that loss to the loss of the generator, discriminator and info loss (last two I'm not sure if it makes sense)
            # elif (epoch>0) and (epoch % 25 == 0) and True:  
            #     for i in range(2):
            #         C_test = torch.normal(0,1,size = (test_size,latent_dim))
            #         Z_test = torch.normal(0, 1, size=(test_size, noise_dim))
            #         _,_,_,_,_,control_camber_thickness_fake,_,_,_ = net_G(torch.cat((Z_test,C_test),-1))
                    
            #         # _, S_fake,_ = torch.linalg.svd(control_camber_thickness_fake, full_matrices=False)
            #         # # print(S_fake)
            #         # # print(S_fake.shape)

            #         # loss_G_SVD = adversarial_loss_svd(S[0:5],S_fake[0:5])

            #         #  Train Generator
            #         trainer_G.zero_grad()
            #         ones = torch.ones((batch_size,), device=Z.device)
            #         _, G_Xy,r_nonintersection,r_adjacent, r_adjacent_camber,_,_,_,_ = net_G(torch.cat((Z,C),-1))
            #         validity, _ = net_D(G_Xy)
            #         loss_G = adversarial_loss_2(validity, ones.reshape(validity.shape)) + lambda_nonintersection * r_nonintersection + lambda_adjacent * r_adjacent + lambda_adjacent_2 * r_adjacent_camber
            #         # loss_G = adversarial_loss_2(validity, ones.reshape(validity.shape)) + lambda_svd * loss_G_SVD + lambda_nonintersection * r_nonintersection + lambda_adjacent * r_adjacent + lambda_adjacent_2 * r_adjacent_camber
            #         # print('Loss SVD: %f',  loss_G_SVD)
                    
                    
            #         # trainer_G.step()

            #         #  Train Discriminator
            #         trainer_D.zero_grad()
            #         ones = torch.ones((batch_size,), device=torch.cat((Xy,Xc),-1).device)
            #         zeros = torch.zeros((batch_size,), device=torch.cat((Xy,Xc),-1).device)            
            #         real_pred,_ = net_D(Xy)
            #         fake_pred,_ = net_D(G_Xy.detach())
            #         loss_D = (adversarial_loss_2(real_pred, ones.reshape(real_pred.shape)) +
            #                 adversarial_loss_2(fake_pred, zeros.reshape(fake_pred.shape))) + lambda_nonintersection * r_nonintersection + lambda_adjacent * r_adjacent + lambda_adjacent_2 * r_adjacent_camber
            #         # adversarial_loss_2(fake_pred, zeros.reshape(fake_pred.shape))) + lambda_svd * loss_G_SVD + lambda_nonintersection * r_nonintersection + lambda_adjacent * r_adjacent + lambda_adjacent_2 * r_adjacent_camber
                    

            #         # Information Loss
            #         trainer_info.zero_grad()
            #         _, G_Xy,r_nonintersection,r_adjacent,r_adjacent_camber,_,_,_,_ = net_G(torch.cat((Z,C),-1))
            #         _, pred_c = net_D(G_Xy)
            #         loss_info = lambda_con * continuous_loss_2(pred_c, C) + lambda_nonintersection * r_nonintersection + lambda_adjacent * r_adjacent + lambda_adjacent_2 * r_adjacent_camber
            #         # loss_info = lambda_con * continuous_loss_2(pred_c, C) + lambda_svd * loss_G_SVD + lambda_nonintersection * r_nonintersection + lambda_adjacent * r_adjacent + lambda_adjacent_2 * r_adjacent_camber
                    
            #         loss_G.backward(retain_graph=True)
            #         loss_D.backward(retain_graph=True)
            #         loss_info.backward(retain_graph=True)
                    
                    
                    
            #         trainer_G.step()
            #         scheduler_G.step()
            #         trainer_D.step()
            #         scheduler_D.step()
            #         trainer_info.step()


            # else:
                #  Train Generator
            trainer_G.zero_grad()
            ones = torch.ones((batch_size,), device=Z.device)
            _, G_Xy, r_adjacent, r_adjacent_camber, r_non_intersection, _ = net_G(torch.cat((Z,C),-1))

            validity, _ = net_D(G_Xy)
            loss_G = adversarial_loss_2(validity, ones.reshape(validity.shape)) + lambda_nonintersection * r_non_intersection + lambda_adjacent * r_adjacent + lambda_adjacent_2 * r_adjacent_camber #+ max_camber_loss + max_thickness_loss + min_camber_loss
            
            
            # trainer_G.step()

                #  Train Discriminator
            trainer_D.zero_grad()
                       
            real_pred,_ = net_D(Xy)
            fake_pred,_ = net_D(G_Xy.detach())


            ones = torch.ones((batch_size,), device=torch.cat((Xy,Xc),-1).device)
            zeros = torch.zeros((batch_size,), device=torch.cat((Xy,Xc),-1).device) 

            loss_D = (adversarial_loss_2(real_pred, ones.reshape(real_pred.shape)) + 
                    adversarial_loss_2(fake_pred, zeros.reshape(fake_pred.shape))) + lambda_nonintersection * r_non_intersection + lambda_adjacent * r_adjacent + lambda_adjacent_2 * r_adjacent_camber
            

            # Information Loss
            trainer_info.zero_grad()
            _, G_Xy, r_adjacent,r_adjacent_camber, r_non_intersection, _ = net_G(torch.cat((Z,C),-1))
            _, pred_c = net_D(G_Xy)
            loss_info = lambda_con * continuous_loss_2(pred_c, C) + lambda_nonintersection * r_non_intersection + lambda_adjacent * r_adjacent + lambda_adjacent_2 * r_adjacent_camber #+ max_camber_loss + min_camber_loss
            
            loss_G.backward(retain_graph=True)
            loss_D.backward(retain_graph=True)
            loss_info.backward(retain_graph=True)
            
            
            
            trainer_G.step()
            scheduler_G.step()
            trainer_D.step()
            scheduler_D.step()
            trainer_info.step() 
        
            
            if i % 30 == 0:           
                print( "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]"
                % (epoch, num_epochs, i, len(data_iter), loss_D.item(), loss_G.item(), loss_info.item()))    
                loss_D_list.append(loss_D.item())
                loss_G_list.append(loss_G.item())
                loss_info_list.append(loss_info.item())




lr_D, lr_G,lr_info, num_epochs = 0.00002, 0.00002, 0.00003, 2500 #50
test_str = '_new_method_1st_attempt_camber_thickness_' + "%s"%num_epochs

train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, lr_info,noise_dim, latent_dim, 0.25, 1.5, 1.5)

plt.figure(figsize=(8,6))
plt.plot(range(num_epochs), loss_G_list)
plt.plot(range(num_epochs), loss_D_list)
plt.plot(range(num_epochs), loss_info_list)
plt.legend(['Generator', 'Discriminator','Infomation'])
plt.savefig('Loss'+test_str)

# # Saving trained generator: IF YOU JUST WANT TO RUN A TRAINED GENERATOR, COMMENT OUT LINE 192! 
generator_model_path =  'GAN_GENERATOR_coordinates_deeper_architecture' +test_str
torch.save(net_G.state_dict(),generator_model_path)

# Gen = Generator()
# Gen.load_state_dict(torch.load(generator_model_path))

test_size = 10
Z = torch.zeros(test_size, noise_dim).to(device=device)
C = torch.zeros(test_size, latent_dim).to(device=device)
latent_code  = np.linspace(-2,2,test_size)
fig, axs = plt.subplots(5, 4, figsize=(15,7.2))#
axs = axs.flatten()
for latent_c in range(latent_dim):
    for i in range(test_size):
        C[i,latent_c] = latent_code[i]
    fake_data_test,_,_,_,_,_ = net_G(torch.cat((Z,C),-1))
    fake_data_test = fake_data_test.cpu().detach().numpy()
    color = plt.cm.rainbow(np.linspace(0, 1, test_size))
    for i, c in zip(range(test_size), color):
        axs[latent_c].plot(x_interp, fake_data_test[i,0:150] + 0.5 * fake_data_test[i, 150:], c = c, label='C_'+' = {0}'.format(round(latent_code[i],2)))
        axs[latent_c].plot(x_interp, fake_data_test[i,0:150] - 0.5 * fake_data_test[i, 150:], c = c)     
        # axs[latent_c].plot(curve_x_coords[0:301,0], fake_data_test[i,0:301]+0.5*fake_data_test[i,301:602], c = c, label='C_'+' = {0}'.format(round(latent_code[i],2)))
        # axs[latent_c].plot(curve_x_coords[0:301,0], fake_data_test[i,0:301]-0.5*fake_data_test[i,301:602], c = c)     

fig.tight_layout()
plt.savefig('Fake_airfoil_deeper_architecture_2'+test_str)

