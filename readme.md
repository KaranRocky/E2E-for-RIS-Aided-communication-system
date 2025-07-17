This simulation code package is mainly used to reproduce the results of the following paper [1]:

[1] H. Jiang, L. Dai, M. Hao and R. Mackenzie, "End-to-End Learning for RIS-Aided Communication Systems," IEEE Trans. Veh. Technol., vol. 71, no. 6, pp. 6778-6783, Jun. 2022.

*********************************************************************************************************************************
If you use this simulation code package in any way, please cite the original paper [1] above. 
 
The author in charge of this simulation code pacakge is: Hao Jiang (email: jiang-h18@mails.tsinghua.edu.cn).

Reference: We highly respect reproducible research, so we try to provide the simulation codes for our published papers (more information can be found at: 
http://oa.ee.tsinghua.edu.cn/dailinglong/publications/publications.html). 

Please note that the pytorch 1.0.0 in Python3.6 is used for this simulation code package,  and there may be some imcompatibility problems among different python or pytorch versions. 

Copyright reserved by the Broadband Communications and Signal Processing Laboratory (led by Dr. Linglong Dai), Beijing National Research Center for Information Science and Technology (BNRist), Department of Electronic Engineering, Tsinghua University, Beijing 100084, China. 

*********************************************************************************************************************************

Abstract of the paper: 
From adapting to the propagation environment to reconstructing the propagation environment, reconfigurable
intelligent surface (RIS) changes the design paradigm of wireless communications. To reconstruct the propagation environment,
joint beamforming of RIS and multi-input multi-output (MIMO) is crucial. Unfortunately, due to the coupling effect of the
active beamforming of MIMO and passive beamforming of RIS, it is difficult to find the optimal solution to the joint
beamforming problem, so a serious performance loss will be caused. In this paper, inspired by the end-to-end (E2E) learning of
communication system, we propose the E2E learning based RIS aided communication system to mitigate the performance loss
via deep learning techniques. The key idea is to simultaneously optimize the signal processing functions at base station (BS), RIS,
and user, including active beamforming for BS and user, passive beamforming for RIS. This way is able to avoid the performance
loss caused by alternately optimizing each function of the RIS aided system. Specifically, we firstly utilize a deep neural network
(DNN) to realize the modulation and beamforming for BS and utilize another DNN to realize the demodulation and combining
for user. Then, the RIS passive beamforming is also represented by trainable parameters, which could be simultaneously optimized
 with the DNNs at BS and user. Simulation results show that the proposed E2E learning based RIS-aided communication
system could achieve the better bit error rate (BER) performance than traditional RIS-aided communication systems.

******************
How to use this simulation code?

1. You can run the code called as 'figure_picture.m' to get the simulation results about:
    1) BER performance against user location (L m,0)
    2) BER performance agsinst SNR.

2. You can run the the code called as "figure_learning_curve.m" to get the simulation results about:
    1) Learning curve of different training methods.

3. Note that the DNN weights of BS and user, and RIS weights are well trained in last simulation. 
    0）The channel saved in "channle_&_model" is generated according to the rayleigh fading channle model.
    1）Of course, you can train the RIS-aided from zero by run the code called as "main_RIS.py". The different elements number of RIS are setted in line 44,  variable 'self.Num_RIS_Element'.
    2）Maybe you want to generate the BER results from traditional RIS-aided communication systemIf according to [6]. If so, you can run the code called as "main_tradition.m" to get the BER results.
    3）At last, run the code called as figure_picture.m' to get the simulation figures.

[6] Z. Zhang and L. Dai, “A joint precoding framework for wideband reconfigurable intelligent surface-aided cell-free network,” IEEE Trans. Signal Process., vol. 69, pp. 4085-4101, Aug. 2021.

*********************************************************************************************************************************
Enjoy the reproducible research!













