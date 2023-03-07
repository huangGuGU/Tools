import numpy as np
import cmath
import math
import time
import numpy as np
import matplotlib.pyplot as plt  # import the package of imaging.
import tensorflow as tf
import datetime


#############################
#   calculation function    #
#############################
def aFunction(al, belta):
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # This is a code of Eqs (1) to (4) in
    # Enderlein_Theoretical study of detection of a dipole emitter through
    #   an objective with high numerical aperture_OL25(2000)634
    # This code can show defocus of a dipole from an object medium with high
    #   refractive index to an image medium with low refrative index
    # Dilope location is at (0,0,0) which is also the focus of the lens
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    alpha = math.pi * (al / 180);  # variable that can be changed
    beta = math.pi * (belta / 180);  # variable that can be changed
    pdip = 1;  # dipole strength
    px_dip = pdip * math.sin(beta) * math.cos(alpha)  # x-component of dipole polarisation
    py_dip = pdip * math.sin(beta) * math.sin(alpha)  # y-component of dipole polarisation
    pz_dip = pdip * math.cos(beta)  # z-component of dipole polarisation
    n_obj = 1.33;  # refractive indx of the object medium
    k_obj = 1;  # *10^-6;  wavenumber of the object medium
    # normal water immersion objectives
    # %%definition of effective pixel size based on a paper in OE, pixel
    # %%size/magnification
    # %%pixel size =effective pixel size=50nm
    Mag = 10;  # magnification size
    NA = 1.2;  # numerical aperture of the lens
    foc = 1;  # focal length of the lens
    n_img = 1.0;  # refractive index of the image medium
    k_img = k_obj / n_obj * n_img;  # wavenumber of the image medium

    x = np.linspace(-40, 40, 20 + 1);  # *10^-6;%*10^-6;   % x posiition on image
    y = np.linspace(-40, 40, 20 + 1);  # y posiition on image
    X, Y = np.meshgrid(x, y);  # mesh
    # output calculation to see if it is correct
    eta_img_max = math.atan(NA / (Mag * math.sqrt(n_obj ** 2 - NA ** 2)));  # max eta in image medium in Eq (3)
    numeta_img = 21;  # small angle, a big number is not needed here
    numeta_img = numeta_img + 1;  # number of nodes for numerical integration over eta_img
    eta_img_step = eta_img_max / (numeta_img - 1);  # step for numerical integration over eta_img

    psi_obj_max = 2 * math.pi;  # integration over psi from 0 to 2pi
    numpsi_obj = 360;
    numpsi_obj = numpsi_obj + 1;  # number of nodes for numerical integration over psi
    psi_obj_step = psi_obj_max / (numpsi_obj - 1);  # step for numerical integration over psi

    numx_img = len(x);
    numy_img = len(x);
    Ex_nd = np.zeros([numx_img, numy_img], dtype='complex128');  # x component of electric field on image surface
    Ey_nd = np.zeros([numx_img, numy_img], dtype='complex128');  # y component of electric field on image surface
    Ez_nd = np.zeros([numx_img, numy_img], dtype='complex128');  # z component of electric field on image surface
    Iz_nd = np.zeros([numx_img, numy_img],
                     dtype='float64');  # x component of Poynting vector on image surface (our results)
    for i in range(numx_img):
        for j in range(numx_img):
            rx_img = X[i, j];  # x coordinate of image location
            ry_img = Y[i, j];  # y coordinate of image location
            rz_img = foc + Mag * foc;  # z coordinate of image location = focal length*(Mag+1)
            eex_eta_int = np.zeros([1, numeta_img], dtype='complex128');
            eey_eta_int = np.zeros([1, numeta_img], dtype='complex128');
            eez_eta_int = np.zeros([1, numeta_img], dtype='complex128');
            for ii in range(numeta_img):
                eta_img = (ii) * eta_img_step;  # eta_img
                eta_obj = math.atan(Mag * math.tan(eta_img));  # tan(eta_obj) = Mag*tan(eta_img)
                R_obj = foc / math.cos(eta_obj);  # R = foc/cos(eta_obj)

                ztp1 = cmath.sqrt(math.cos(eta_img) / math.cos(eta_obj));
                ztp2 = cmath.exp(complex(0, 1) * k_obj * R_obj) / R_obj;  # exp(i k_obj R)/R
                eex_psi_int = np.zeros([1, numpsi_obj], dtype='complex128');
                eey_psi_int = np.zeros([1, numpsi_obj], dtype='complex128');
                eez_psi_int = np.zeros([1, numpsi_obj], dtype='complex128');
                for jj in range(numpsi_obj):
                    psi_obj = (jj) * psi_obj_step;
                    ex_hat_per_obj = math.cos(psi_obj) * math.cos(eta_obj);
                    ey_hat_per_obj = math.sin(psi_obj) * math.cos(eta_obj);
                    ez_hat_per_obj = -math.sin(eta_obj);
                    ex_hat_pll_obj = -math.sin(psi_obj);
                    ey_hat_pll_obj = math.cos(psi_obj);
                    ez_hat_pll_obj = 0;
                    ex_hat_per_img = math.cos(psi_obj) * math.cos(eta_img);
                    ey_hat_per_img = math.sin(psi_obj) * math.cos(eta_img);
                    ez_hat_per_img = math.sin(eta_img);
                    ex_hat_pll_img = -math.cos(psi_obj) * math.sin(eta_img);
                    ey_hat_pll_img = -math.sin(psi_obj) * math.sin(eta_img);
                    ez_hat_pll_img = math.cos(eta_img);

                    tp = ex_hat_pll_img * rx_img + ey_hat_pll_img * ry_img + ez_hat_pll_img * rz_img;  # s*r
                    ztp3 = cmath.exp(complex(0, 1) * k_img * tp);  # exp(i k_img s*r)
                    ztp = ztp1 * ztp2 * ztp3;

                    ztp4 = px_dip * ex_hat_per_obj + py_dip * ey_hat_per_obj + pz_dip * ez_hat_per_obj;  # p e^hat_per
                    ztp5 = px_dip * ex_hat_pll_obj + py_dip * ey_hat_pll_obj + pz_dip * ez_hat_pll_obj;  # p e^hat_pll

                    eex_psi_int[0, jj] = ztp * (
                                ztp4 * ex_hat_per_img + ztp5 * ex_hat_pll_obj);  # x component of inner integrand in Eq (1)
                    eey_psi_int[0, jj] = ztp * (
                                ztp4 * ey_hat_per_img + ztp5 * ey_hat_pll_obj);  # y component of inner integrand in Eq (1)
                    eez_psi_int[0, jj] = ztp * (
                                ztp4 * ez_hat_per_img + ztp5 * ez_hat_pll_obj);  # z component of inner integrand in Eq (1)

                    # numerical integral over psi (inner part) by using the trapezoidal rule
                for jj in range(numpsi_obj - 1):
                    eex_eta_int[0, ii] = eex_eta_int[0, ii] + 0.5 * psi_obj_step * (
                                eex_psi_int[0, jj] + eex_psi_int[0, jj + 1]);
                    eey_eta_int[0, ii] = eey_eta_int[0, ii] + 0.5 * psi_obj_step * (
                                eey_psi_int[0, jj] + eey_psi_int[0, jj + 1]);
                    eez_eta_int[0, ii] = eez_eta_int[0, ii] + 0.5 * psi_obj_step * (
                                eez_psi_int[0, jj] + eez_psi_int[0, jj + 1]);

                eex_eta_int[0, ii] = eex_eta_int[0, ii] * math.sin(eta_img);  # x component of outer integrand in Eq (1)
                eey_eta_int[0, ii] = eey_eta_int[0, ii] * math.sin(eta_img);  # y component of outer integrand in Eq (1)
                eez_eta_int[0, ii] = eez_eta_int[0, ii] * math.sin(eta_img);  # z component of outer integrand in Eq (1)

                # numerical integral over eta_img (outer part) by using the trapezoidal rule
                for ii in range(numeta_img - 1):
                    Ex_nd[i][j] = Ex_nd[i, j] + 0.5 * eta_img_step * (eex_eta_int[0, ii] + eex_eta_int[0, ii + 1]);
                    Ey_nd[i][j] = Ey_nd[i, j] + 0.5 * eta_img_step * (eey_eta_int[0, ii] + eey_eta_int[0, ii + 1]);
                    Ez_nd[i][j] = Ez_nd[i, j] + 0.5 * eta_img_step * (eez_eta_int[0, ii] + eez_eta_int[0, ii + 1]);

            Iz_nd[i, j] = np.real(Ex_nd[i, j] * np.conj(Ex_nd[i, j]) + Ey_nd[i, j] * np.conj(Ey_nd[i, j]));
    return Iz_nd


#############################
#  create sample_random     #
#############################
dataset_total_num = 600  # total number of samples(changeable)
dataset_now_num = 0  # now number(not change)

y_train = np.zeros([dataset_total_num, 2], dtype='float64')
x_train = np.zeros([dataset_total_num, 21, 21], dtype='float64')

starttime = time.time()

# create 0,0
alpha_angles = 0
belta_angles = 0
y_train[(dataset_now_num, 0)] = alpha_angles
y_train[(dataset_now_num, 1)] = belta_angles
x_train[dataset_now_num] = aFunction(alpha_angles, belta_angles)

endtime = time.time()
print("%.2s s" % (endtime - starttime))
print('zero ok')

for dataset_now_num in range(dataset_total_num - 1):
    for bignum in range(500000):  # set a big num to avoid mistake
        alpha_angles = np.random.randint(0, 181, 1)  # should calculate 0 - 180
        belta_angles = np.random.randint(0, 91, 1)  # should calculate 0 - 90
        if [alpha_angles, belta_angles] in y_train:
            if bignum > 10000:
                print('something wrong')
                break
            else:
                pass
        else:
            break

    y_train[(dataset_now_num + 1, 0)] = alpha_angles
    y_train[(dataset_now_num + 1, 1)] = belta_angles
    x_train[dataset_now_num + 1] = aFunction(alpha_angles, belta_angles)

    if (dataset_now_num + 1) % 1 == 0:  ### change here to change show
        endtime = time.time()
        costtime_minute = round((endtime - starttime) / 60, 1)
        date_and_time = datetime.datetime.now().strftime('%Y%m%d%H%M')
        print(str(dataset_now_num + 2) + '/' + str(dataset_total_num) + ' time:' + str(
            costtime_minute) + 'min' + "   " + str(date_and_time))

    if (dataset_now_num + 1) % 500 == 0:  ### change here to save sample
        print('saving dataset')
        date_and_time = datetime.datetime.now().strftime('%Y%m%d%H%M')
        np.save(file="x-data" + date_and_time + '_samplenum_' + str(dataset_now_num + 1) + ".npy", arr=x_train)
        np.save(file="y-data" + date_and_time + '_samplenum_' + str(dataset_now_num + 1) + ".npy", arr=y_train)
        print('saving dataset ok')
    else:
        pass

    dataset_now_num = dataset_now_num + 1

date_and_time = datetime.datetime.now().strftime('%Y%m%d%H%M')
np.save(file="x-test-data" + date_and_time + '_samplenum_' + str(dataset_now_num + 1) + ".npy", arr=x_train)
np.save(file="y-test-data" + date_and_time + '_samplenum_' + str(dataset_now_num + 1) + ".npy", arr=y_train)

# #############################
# #   save_name add time      #
# #############################
# date_and_time = datetime.datetime.now().strftime('%Y%m%d%H%M')
# np.save(file="x-data"+date_and_time+".npy", arr=x_train)
# np.save(file="y-data"+date_and_time+".npy", arr=y_train)


# #############################
# #            load           #
# #############################
# xx_train= np.load(file="x-data202104261613.npy")
# yy_train= np.load(file="y-data202104261613.npy")


# #############################
# #         read npy          #
# #############################
# print(len(xx_train))
# print(len(yy_train))

# plt.imshow(x_train[0])
# print(y_train[0])

